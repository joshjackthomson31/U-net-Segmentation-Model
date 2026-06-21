"""
unet.py — U-Net architecture for FloodNet semantic segmentation

Architecture: Standard U-Net (Ronneberger et al. 2015) adapted for:
  - 10-class FloodNet segmentation
  - 512x512 input
  - ~6M parameters (matching paper Table II: GEO-U-Net = 5,988,837)
  - Configurable dropout (tuned by HHO)

Structure:
  Input (3, 512, 512)
      ↓ Encoder: 4 levels, each doubles filters and halves spatial size
      ↓ Bottleneck: deepest representation
      ↑ Decoder: 4 levels, each halves filters and doubles spatial size
      ↑ Skip connections: concat encoder features into decoder (preserves detail)
  Output (NUM_CLASSES, 512, 512) — raw logits, one per class per pixel

Filter progression: 32 → 64 → 128 → 256 → 512
This gives ~6M parameters (use base_filters=64 for ~31M standard U-Net).

NOTE: Output is raw logits (no softmax). CrossEntropyLoss in train.py
      applies softmax internally — do NOT add softmax here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from src.config import NUM_CLASSES


# ─────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive: Conv2d → BatchNorm → ReLU blocks.

    This is the core repeated unit of U-Net.

    Think of it like: look at the image twice in a row,
    each time learning slightly more complex patterns.

    Example (encoder level 1):
      Input:  (B, 3,  512, 512)   ← RGB image
      After first conv:  (B, 32, 512, 512)
      After second conv: (B, 32, 512, 512)  ← same size, more features

    Args:
        in_ch  : number of input channels
        out_ch : number of output channels
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    One encoder step: DoubleConv → MaxPool(2x2).

    MaxPool halves spatial size (512→256→128→64→32).
    Returns BOTH the conv output (for skip connection) and pooled output.

    Simple analogy:
      - Conv output = "keep a copy of this detail" (skip connection)
      - Pooled output = "compressed version, pass down" (to next level)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)       # full resolution features (for skip connection)
        down = self.pool(skip)    # halved resolution (passed to next encoder level)
        return skip, down


class DecoderBlock(nn.Module):
    """
    One decoder step: Upsample → concat skip → DoubleConv → optional Dropout.

    Upsample doubles spatial size back up.
    Concatenating skip connection restores fine spatial details lost during pooling.

    Simple analogy:
      Encoder: took a photo, then zoomed out 4 times to understand "big picture"
      Decoder: zooms back in step-by-step, using saved copies to restore details

    Args:
        in_ch   : channels coming from previous decoder level
        skip_ch : channels from the corresponding encoder skip connection
        out_ch  : output channels
        dropout : dropout probability (0 = no dropout, HHO tunes this)
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        # ConvTranspose2d doubles spatial size (32→64→128→256→512)
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # After concat: channels = (in_ch // 2) + skip_ch
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch (can occur with non-power-of-2 inputs)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)   # concat along channel dimension
        x = self.conv(x)
        x = self.drop(x)
        return x


# ─────────────────────────────────────────────
# U-NET
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for semantic segmentation of FloodNet aerial images.

    Args:
        num_classes  : number of output classes (10 for FloodNet)
        base_filters : filters in first encoder block (32 → ~6M params, 64 → ~31M)
        dropout      : dropout probability in decoder (0.1–0.5, tuned by HHO)

    Input:  (B, 3, 512, 512)   — batch of RGB images
    Output: (B, 10, 512, 512)  — raw logit scores per class per pixel
                                  (NOT softmax — CrossEntropyLoss handles that)
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 base_filters: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        f = base_filters   # shorthand: f=32, 2f=64, 4f=128, 8f=256, 16f=512

        # ── Encoder ──────────────────────────────────
        # Each level: doubles filters, halves spatial size via MaxPool
        self.enc1 = EncoderBlock(3,    f)     # 512→256,  3→32
        self.enc2 = EncoderBlock(f,    f*2)   # 256→128, 32→64
        self.enc3 = EncoderBlock(f*2,  f*4)   # 128→64,  64→128
        self.enc4 = EncoderBlock(f*4,  f*8)   # 64→32,  128→256

        # ── Bottleneck ───────────────────────────────
        # Deepest level — no pooling, maximum filter count
        self.bottleneck = DoubleConv(f*8, f*16)   # 32×32, 256→512

        # ── Decoder ──────────────────────────────────
        # Each level: halves filters, doubles spatial size via ConvTranspose2d
        # skip_ch must match corresponding encoder's output channels
        self.dec4 = DecoderBlock(f*16, f*8,  f*8,  dropout)  # 32→64
        self.dec3 = DecoderBlock(f*8,  f*4,  f*4,  dropout)  # 64→128
        self.dec2 = DecoderBlock(f*4,  f*2,  f*2,  dropout)  # 128→256
        self.dec1 = DecoderBlock(f*2,  f,    f,    dropout)  # 256→512

        # ── Output head ──────────────────────────────
        # 1×1 conv: maps f channels → num_classes (no activation — raw logits)
        self.head = nn.Conv2d(f, num_classes, kernel_size=1)

        # ── Weight initialization ─────────────────────
        self._init_weights()

        # Print parameter count for verification
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[UNet] Parameters: {n_params:,}  (base_filters={base_filters}, dropout={dropout})")

    def _init_weights(self):
        """
        Kaiming (He) initialization for Conv layers.
        Helps with training stability — avoids vanishing/exploding gradients.
        Standard practice for ReLU-based networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the full U-Net.

        Data flow:
          x → enc1 → enc2 → enc3 → enc4 → bottleneck
                ↓       ↓      ↓       ↓
              skip1   skip2  skip3   skip4     (saved for decoder)
                                       ↑
                                      dec4 ← (bottleneck + skip4)
                                      dec3 ← (dec4 + skip3)
                                      dec2 ← (dec3 + skip2)
                                      dec1 ← (dec2 + skip1)
                                       ↓
                                      head → output logits
        """
        # Encoder: save skip connections at each level
        skip1, x = self.enc1(x)   # skip1: (B, f,    512, 512)
        skip2, x = self.enc2(x)   # skip2: (B, f*2,  256, 256)
        skip3, x = self.enc3(x)   # skip3: (B, f*4,  128, 128)
        skip4, x = self.enc4(x)   # skip4: (B, f*8,   64,  64)

        # Bottleneck
        x = self.bottleneck(x)    # x:     (B, f*16,  32,  32)

        # Decoder: restore spatial resolution using skip connections
        x = self.dec4(x, skip4)   # x:     (B, f*8,   64,  64)
        x = self.dec3(x, skip3)   # x:     (B, f*4,  128, 128)
        x = self.dec2(x, skip2)   # x:     (B, f*2,  256, 256)
        x = self.dec1(x, skip1)   # x:     (B, f,    512, 512)

        # Output head: one score per class per pixel
        return self.head(x)       # out:   (B, 10,   512, 512)


# ─────────────────────────────────────────────
# RESNET BACKBONE U-NET
# ─────────────────────────────────────────────

class ResNetDecoderBlock(nn.Module):
    """
    One decoder step for the ResNet-encoder U-Net.

    Upsamples `x` to match `skip` spatial size (bilinear), concatenates
    along channel dim, then applies DoubleConv to mix features.

    Args:
        in_ch   : channels coming from the previous decoder level
        skip_ch : channels from the corresponding ResNet encoder skip
        out_ch  : output channels after DoubleConv
        dropout : dropout probability (0 = off)
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Bilinear upsample to match skip's spatial dims (handles non-power-of-2 safely)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.drop(x)
        return x


class ResNetUNet(nn.Module):
    """
    U-Net with a pre-trained ResNet encoder (ImageNet weights).

    Why pre-trained?
      1,445 training images is far too few for a scratch encoder to learn
      low-level edges, textures, and shapes from zero. ImageNet weights
      provide a rich prior — the encoder already "knows" what grass, roads,
      and rooftops look like in natural aerial imagery.

    Architecture (ResNet-34 default):
      Encoder (frozen or fine-tuned):
        enc0 = conv1+bn1+relu  → (B, 64,  256, 256)  ← skip s0
        pool = maxpool          → (B, 64,  128, 128)
        enc1 = layer1           → (B, 64,  128, 128)  ← skip s1
        enc2 = layer2           → (B, 128,  64,  64)  ← skip s2
        enc3 = layer3           → (B, 256,  32,  32)  ← skip s3
        enc4 = layer4           → (B, 512,  16,  16)  ← bottleneck

      Decoder (randomly initialized, trained from scratch):
        dec4: up(512) + cat(s3:256) → DoubleConv(768→256) → (B,256, 32, 32)
        dec3: up(256) + cat(s2:128) → DoubleConv(384→128) → (B,128, 64, 64)
        dec2: up(128) + cat(s1: 64) → DoubleConv(192→64)  → (B, 64,128,128)
        dec1: up( 64) + cat(s0: 64) → DoubleConv(128→64)  → (B, 64,256,256)
        up×2 → DoubleConv(64→32)                          → (B, 32,512,512)
        head: Conv2d(32→10)                                → (B, 10,512,512)

    Args:
        num_classes : output classes (10 for FloodNet)
        backbone    : 'resnet34' or 'resnet50'
        dropout     : dropout probability in decoder (HHO tunes this)
    """

    # ResNet channel dimensions per backbone
    _ENC_CH = {
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
    }

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone:    str = "resnet34",
        dropout:     float = 0.0,
    ):
        super().__init__()

        if backbone not in self._ENC_CH:
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose from {list(self._ENC_CH)}")

        ec = self._ENC_CH[backbone]   # encoder channel sizes at each stage

        # ── Load pre-trained encoder ──────────────────────────────
        if backbone == "resnet34":
            resnet = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)

        # Split ResNet into encoder stages (matching U-Net skip-connection points)
        self.enc0  = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 512→256
        self.pool  = resnet.maxpool                                          # 256→128
        self.enc1  = resnet.layer1    # 128×128
        self.enc2  = resnet.layer2    #  64×64
        self.enc3  = resnet.layer3    #  32×32
        self.enc4  = resnet.layer4    #  16×16  (bottleneck)

        # ── Decoder ───────────────────────────────────────────────
        # in_ch  = channels from previous decoder output
        # skip_ch = channels from corresponding encoder stage
        # out_ch  = channels after DoubleConv
        self.dec4 = ResNetDecoderBlock(ec[4], ec[3], 256, dropout)   # 16→32
        self.dec3 = ResNetDecoderBlock(256,   ec[2], 128, dropout)   # 32→64
        self.dec2 = ResNetDecoderBlock(128,   ec[1],  64, dropout)   # 64→128
        self.dec1 = ResNetDecoderBlock(64,    ec[0],  64, dropout)   # 128→256

        # Final upsample: 256×256 → 512×512
        self.final_conv = DoubleConv(64, 32)
        self.head       = nn.Conv2d(32, num_classes, kernel_size=1)

        # Initialize decoder + head weights (encoder keeps ImageNet weights)
        self._init_decoder_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ResNetUNet] Parameters: {n_params:,}  (backbone={backbone}, dropout={dropout})")

    def _init_decoder_weights(self):
        """Kaiming init for all decoder and head layers (not the encoder)."""
        decoder_parts = [self.dec4, self.dec3, self.dec2, self.dec1,
                         self.final_conv, self.head]
        for module in decoder_parts:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 3,  512, 512) — ImageNet-normalized RGB
        Output: (B, 10, 512, 512) — raw logits per class per pixel
        """
        # ── Encoder ────────────────────────────────────────────
        s0 = self.enc0(x)      # (B,  64, 256, 256)
        x  = self.pool(s0)     # (B,  64, 128, 128)
        s1 = self.enc1(x)      # (B,  64, 128, 128)
        s2 = self.enc2(s1)     # (B, 128,  64,  64)
        s3 = self.enc3(s2)     # (B, 256,  32,  32)
        x  = self.enc4(s3)     # (B, 512,  16,  16)

        # ── Decoder ────────────────────────────────────────────
        x = self.dec4(x, s3)   # (B, 256,  32,  32)
        x = self.dec3(x, s2)   # (B, 128,  64,  64)
        x = self.dec2(x, s1)   # (B,  64, 128, 128)
        x = self.dec1(x, s0)   # (B,  64, 256, 256)

        # Final upsample 256→512, refine, classify
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.final_conv(x) # (B,  32, 512, 512)
        return self.head(x)    # (B,  10, 512, 512)


# ─────────────────────────────────────────────
# FACTORY — used by HHO and training scripts
# ─────────────────────────────────────────────

def build_unet(
    dropout:     float = 0.0,
    base_filters: int  = 32,
    backbone:    str   = "scratch",
) -> nn.Module:
    """
    Build a U-Net with given hyperparameters and backbone choice.

    Args:
        dropout      : dropout probability in decoder (HHO tunes this)
        base_filters : filter multiplier for scratch U-Net (ignored for resnet*)
        backbone     : 'scratch' → original U-Net (~7M params)
                       'resnet34' → pre-trained ResNet-34 encoder (~24M params)
                       'resnet50' → pre-trained ResNet-50 encoder (~33M params)

    Returns:
        nn.Module (not yet moved to device)
    """
    if backbone == "scratch":
        return UNet(num_classes=NUM_CLASSES, base_filters=base_filters, dropout=dropout)
    else:
        return ResNetUNet(num_classes=NUM_CLASSES, backbone=backbone, dropout=dropout)


# ─────────────────────────────────────────────
# QUICK SANITY CHECK (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from src.config import DEVICE, NUM_CLASSES

    print(f"Device: {DEVICE}")

    # Build model with paper-matched settings
    model = build_unet(dropout=0.3, base_filters=32).to(DEVICE)

    # Dummy forward pass: batch of 2 images, 3 channels, 512x512
    dummy_input = torch.randn(2, 3, 512, 512).to(DEVICE)
    output = model(dummy_input)

    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, NUM_CLASSES, 512, 512), \
        f"Expected (2, {NUM_CLASSES}, 512, 512), got {output.shape}"
    print("Sanity check PASSED.")
