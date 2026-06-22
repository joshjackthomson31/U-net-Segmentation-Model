# src/unet.py — Explained Simply

---

## What is this file for?

This file defines the **U-Net model** — the deep learning architecture that looks at aerial images and labels every single pixel with a class (Background, Building Flooded, Grass, etc.).

It now contains **two model variants:**
- `UNet` — original from-scratch U-Net (~7.8M params)
- `ResNetUNet` — pre-trained ResNet-34 encoder + U-Net decoder (~24.5M params) ← **currently active**

The `build_unet()` factory function selects which one to build based on `BACKBONE` in `config.py`.

---

## Why two variants?

The original from-scratch `UNet` was our first attempt. With only 1,445 training images it struggled to learn meaningful features, reaching ~26–41% mIoU.

`ResNetUNet` replaces the encoder with **ResNet-34 pre-trained on ImageNet** (1.2M images). The encoder already knows what edges, textures, roads, buildings, and water look like — we only need to fine-tune it for aerial flood imagery. This is the primary path to matching the paper's 67.97% mIoU.

---

## What is ResNet? (The "Res" in ResNet)

ResNet = **Residual Network**. The key idea is a **residual (skip) connection** inside each block:

```
Regular layer:   output = F(input)
ResNet layer:    output = F(input) + input    ← adds input back directly
```

**Why this matters:** In very deep networks (34 layers), gradients vanish as they travel backward. The `+ input` shortcut gives gradients a direct path to skip layers — so early layers still learn. Without this trick, 34-layer networks were impossible to train reliably.

**Analogy:** Instead of redrawing a painting from a blank canvas every layer, each ResNet block only adds the *difference* (the residual). It's editing, not repainting.

---

## Why pre-training on ImageNet helps so much

ResNet-34 was trained on **1.2 million photos across 1,000 categories** before we ever touch it. After that training:
- Early layers already detect edges, corners, and basic textures
- Middle layers already recognise shapes — rectangular rooftops, curved roads, irregular water surfaces
- Deep layers already understand objects — buildings, vehicles, vegetation

When we use this as our U-Net encoder, we are not starting from zero. We plug in this knowledge and only need 1,445 FloodNet images to teach the decoder how to turn these rich features into 10-class pixel labels.

**The scratch U-Net encoder needed those same 1,445 images to simultaneously learn what edges are, what buildings look like, AND how to label them.** That is too much to ask.

---

## The U-Shape — Architecture Overview

### Original UNet (backbone='scratch')

```
Input Image (B, 3, 512, 512)
       |
   [Encoder Level 1] → skip1 (B, 32,  512, 512)
       |  MaxPool
   [Encoder Level 2] → skip2 (B, 64,  256, 256)
       |  MaxPool
   [Encoder Level 3] → skip3 (B, 128, 128, 128)
       |  MaxPool
   [Encoder Level 4] → skip4 (B, 256,  64,  64)
       |  MaxPool
   [Bottleneck]       → x    (B, 512,  32,  32)
       |
   [Decoder Level 4] ← skip4
   [Decoder Level 3] ← skip3
   [Decoder Level 2] ← skip2
   [Decoder Level 1] ← skip1
       |
   [Output Head]  →  (B, 10, 512, 512)
```

Parameters: ~7.8M | Encoder: random init

---

### ResNetUNet (backbone='resnet34') ← currently active

```
Input Image (B, 3, 512, 512)   ← ImageNet-normalized
       |
   ResNet conv1+bn1+relu → s0 (B,  64, 256, 256)   ← skip s0
       |  MaxPool
   ResNet layer1          → s1 (B,  64, 128, 128)   ← skip s1
   ResNet layer2          → s2 (B, 128,  64,  64)   ← skip s2
   ResNet layer3          → s3 (B, 256,  32,  32)   ← skip s3
   ResNet layer4          → x  (B, 512,  16,  16)   ← bottleneck
       |
   [ResNetDecoderBlock 4] ← s3  →  (B, 256, 32, 32)
   [ResNetDecoderBlock 3] ← s2  →  (B, 128, 64, 64)
   [ResNetDecoderBlock 2] ← s1  →  (B,  64,128,128)
   [ResNetDecoderBlock 1] ← s0  →  (B,  64,256,256)
       |  bilinear upsample ×2
   [DoubleConv 64→32]     →  (B,  32, 512, 512)
   [Output Head]          →  (B,  10, 512, 512)
```

Parameters: ~24.5M | Encoder: ImageNet pre-trained | Decoder: random init

---

## Building Blocks — The Small Pieces

### Class: `DoubleConv`

**What it does:** Applies two rounds of "look at the image and learn patterns."

Each round = `Conv2d → BatchNorm → ReLU`

- **Conv2d:** A sliding 3×3 window that learns to detect patterns (edges, colors, shapes)
- **BatchNorm:** Normalizes the values to keep training stable
- **ReLU:** Sets negative values to 0 — adds non-linearity so the model can learn complex things

```
Input: (B, in_ch, H, W)
→ Conv(3×3) → BatchNorm → ReLU
→ Conv(3×3) → BatchNorm → ReLU
Output: (B, out_ch, H, W)   ← same spatial size, more features
```

Used by BOTH UNet and ResNetUNet.

---

### Class: `EncoderBlock` (UNet only)

**What it does:** One step DOWN the scratch U-Net.

```
Input → DoubleConv → [skip connection] → MaxPool(2×2) → output
                          ↓
                    (saved for decoder)
```

Not used by ResNetUNet — ResNet-34 layers replace this.

---

### Class: `DecoderBlock` (UNet only)

**What it does:** One step UP the scratch U-Net. Uses `ConvTranspose2d` to upsample.

Not used by ResNetUNet — see `ResNetDecoderBlock` below.

---

### Class: `ResNetDecoderBlock` (ResNetUNet only)

**What it does:** One step UP the ResNet U-Net. Uses bilinear interpolation to upsample (smoother than ConvTranspose2d, avoids checkerboard artifacts).

```python
def forward(self, x, skip):
    x = interpolate(x, size=skip.shape[2:], bilinear)  # upsample to match skip size
    x = concat([x, skip], dim=1)                        # combine: current + saved details
    x = DoubleConv(x)                                   # refine
    x = Dropout(x)                                      # regularize
    return x
```

**Parameters:**
- `in_ch`: channels from previous decoder output (e.g. 512 from bottleneck)
- `skip_ch`: channels from matching ResNet encoder stage (e.g. 256 from layer3)
- `out_ch`: output channels (e.g. 256)
- `dropout`: dropout probability (HHO tunes this 0.1–0.5)

---

### Class: `UNet`

Original from-scratch model. Still present as fallback (`backbone='scratch'`). See previous architecture table. ~7.8M params.

---

### Class: `ResNetUNet`

**What it does:** Assembles ResNet-34 encoder + U-Net-style decoder.

**Encoder (frozen or fine-tuned — ImageNet weights):**
```python
self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 512→256
self.pool  = resnet.maxpool                                         # 256→128
self.enc1  = resnet.layer1    # 128×128 → 64 channels
self.enc2  = resnet.layer2    #  64×64  → 128 channels
self.enc3  = resnet.layer3    #  32×32  → 256 channels
self.enc4  = resnet.layer4    #  16×16  → 512 channels (bottleneck)
```

**Decoder (randomly initialized, trained on FloodNet):**
```python
self.dec4 = ResNetDecoderBlock(512, 256, 256, dropout)
self.dec3 = ResNetDecoderBlock(256, 128, 128, dropout)
self.dec2 = ResNetDecoderBlock(128,  64,  64, dropout)
self.dec1 = ResNetDecoderBlock( 64,  64,  64, dropout)
self.final_conv = DoubleConv(64, 32)
self.head       = nn.Conv2d(32, 10, kernel_size=1)   # 10 class scores per pixel
```

**Weight initialization:**
- Encoder: keeps ImageNet weights (do NOT re-initialize)
- Decoder + head: Kaiming initialization (standard for ReLU networks)

**Why the decoder is randomly initialized?**
The ImageNet decoder head predicts 1,000 ImageNet categories — completely different from our 10 FloodNet flood classes. We discard it and replace it with our own decoder that learns to produce pixel-level flood labels.

---

### Method: `forward(x)` — ResNetUNet

```python
# Encoder: extract features at multiple scales
s0 = self.enc0(x)      # (B,  64, 256, 256)  ← first skip
x  = self.pool(s0)     # (B,  64, 128, 128)
s1 = self.enc1(x)      # (B,  64, 128, 128)  ← second skip
s2 = self.enc2(s1)     # (B, 128,  64,  64)  ← third skip
s3 = self.enc3(s2)     # (B, 256,  32,  32)  ← fourth skip
x  = self.enc4(s3)     # (B, 512,  16,  16)  ← bottleneck

# Decoder: upsample + combine with encoder skips
x = self.dec4(x, s3)   # (B, 256,  32,  32)
x = self.dec3(x, s2)   # (B, 128,  64,  64)
x = self.dec2(x, s1)   # (B,  64, 128, 128)
x = self.dec1(x, s0)   # (B,  64, 256, 256)

# Final upsample 256→512 + output
x = interpolate(x, scale_factor=2)   # (B,  64, 512, 512)
x = self.final_conv(x)               # (B,  32, 512, 512)
return self.head(x)                  # (B,  10, 512, 512)
```

---

## Factory Function: `build_unet`

```python
def build_unet(dropout=0.0, base_filters=32, backbone='scratch') -> nn.Module:
```

**What it does:** Builds the correct model based on `backbone`.

| `backbone` | Model built | Params | Notes |
|---|---|---|---|
| `'scratch'` | `UNet` | ~7.8M | Random init, no ImageNet |
| `'resnet34'` | `ResNetUNet` | ~24.5M | ImageNet pre-trained encoder |
| `'resnet50'` | `ResNetUNet` | ~33M | Larger ImageNet encoder |

`BACKBONE` is set in `config.py` and passed by `train.py` automatically. You never need to call `build_unet` directly — it's done for you by proxy_train, full_train, and evaluate.

**Called by:**
- `proxy_train` — builds fresh model for each HHO hawk evaluation
- `full_train` — builds final model for 20-epoch training
- `run_evaluation` — rebuilds model to load checkpoint weights
- `cmd_sanity` in `main.py` — verifies architecture works

---

## Key facts summary

| Property | Scratch UNet | ResNetUNet (active) |
|---|---|---|
| Total parameters | ~7.8M | ~24.5M |
| Encoder | Random init | ImageNet pre-trained |
| Decoder | Random init | Random init |
| Input | (B, 3, 512, 512) | (B, 3, 512, 512) normalized |
| Output | (B, 10, 512, 512) logits | (B, 10, 512, 512) logits |
| Expected mIoU | ~26–41% | ~55–67% (target) |
| Backbone config | `BACKBONE='scratch'` | `BACKBONE='resnet34'` |
