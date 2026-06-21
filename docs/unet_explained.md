# src/unet.py — Explained Simply

---

## What is this file for?

This file defines the **U-Net model** — the deep learning architecture that looks at aerial images and labels every single pixel with a class (Background, Building Flooded, Grass, etc.).

Think of U-Net as a **zoom-out then zoom-in process:**
- **Zoom out (Encoder):** Understand what's in the image at a big-picture level
- **Zoom in (Decoder):** Restore precise pixel locations using what was learned

---

## Why U-Net?

U-Net was invented in 2015 by Ronneberger et al. originally for medical image segmentation. It's excellent for aerial image segmentation because of its **skip connections** — it "remembers" fine details from early layers and reuses them later when drawing precise boundaries.

---

## The U-Shape — Architecture Overview

```
Input Image (B, 3, 512, 512)
       |
   [Encoder Level 1] → skip1 (B, 32,  512, 512)
       |
   [Encoder Level 2] → skip2 (B, 64,  256, 256)
       |
   [Encoder Level 3] → skip3 (B, 128, 128, 128)
       |
   [Encoder Level 4] → skip4 (B, 256,  64,  64)
       |
   [Bottleneck]       → x    (B, 512,  32,  32)   ← deepest point
       |
   [Decoder Level 4] ← skip4
       |
   [Decoder Level 3] ← skip3
       |
   [Decoder Level 2] ← skip2
       |
   [Decoder Level 1] ← skip1
       |
   [Output Head]
       |
Output Logits (B, 10, 512, 512)   ← one score per class per pixel
```

`B` = batch size (number of images processed at once)

---

## Filter Progression (base_filters=32)

| Level | Filters (channels) | Spatial size | What the model "sees" |
|---|---|---|---|
| Input | 3 (RGB) | 512×512 | Raw pixels |
| Encoder 1 | 32 | 512×512 | Edges, basic textures |
| Encoder 2 | 64 | 256×256 | Simple shapes |
| Encoder 3 | 128 | 128×128 | Complex patterns |
| Encoder 4 | 256 | 64×64 | Object parts |
| Bottleneck | 512 | 32×32 | Full scene understanding |
| Decoder 4 | 256 | 64×64 | Rough object outlines |
| Decoder 3 | 128 | 128×128 | Refined shapes |
| Decoder 2 | 64 | 256×256 | Fine boundaries |
| Decoder 1 | 32 | 512×512 | Pixel-precise labels |
| Output | 10 | 512×512 | Class scores per pixel |

This gives **7,763,338 parameters** (~7.8M) with base_filters=32.

---

## Building Blocks — The Small Pieces

### Class: `DoubleConv`

**What it does:** Applies two rounds of "look at the image and learn patterns."

Each round = `Conv2d → BatchNorm → ReLU`

- **Conv2d:** A sliding 3×3 window that learns to detect patterns (edges, colors, shapes)
- **BatchNorm:** Normalizes the values to keep training stable (like keeping scores on the same scale)
- **ReLU:** Sets negative values to 0 — adds non-linearity so the model can learn complex things

```
Input: (B, in_ch, H, W)
→ Conv(3×3) → BatchNorm → ReLU
→ Conv(3×3) → BatchNorm → ReLU
Output: (B, out_ch, H, W)   ← same spatial size, more features
```

**Simple analogy:** Look at a photo twice through different lenses — each time spotting more detail.

**Parameters:**
- `in_ch`: number of input channels (e.g., 3 for RGB)
- `out_ch`: number of output channels (how many patterns to detect)

---

### Class: `EncoderBlock`

**What it does:** One step DOWN the U. Learns features, then compresses the image.

```
Input → DoubleConv → [skip connection] → MaxPool(2×2) → output to next level
                          ↓
                    (saved for later use in decoder)
```

- **DoubleConv:** Learns patterns at current resolution
- **Skip:** Saves a copy of those patterns before compression
- **MaxPool:** Halves the spatial size (512→256, 256→128, etc.) — like zooming out

**Returns TWO things:**
1. `skip` — the full-resolution feature map (saved, sent to decoder later)
2. `down` — the compressed, halved-resolution output (sent to next encoder level)

**Simple analogy:** Take a photo, keep a copy, then shrink it for further analysis.

---

### Class: `DecoderBlock`

**What it does:** One step UP the U. Enlarges the image back and combines with saved details.

```
Input → ConvTranspose2d (upsample 2×) → concat with skip → DoubleConv → Dropout
```

- **ConvTranspose2d:** Doubles the spatial size (32→64, 64→128, etc.) — like zooming in
- **Concat with skip:** Adds back the fine details saved during encoding
- **DoubleConv:** Refines the combined features
- **Dropout:** Randomly zeros some neurons during training to prevent overfitting (HHO tunes this probability)

**The skip connection is critical:** Without it, the decoder would only have the compressed bottleneck features — it would lose all the fine boundary information from earlier layers.

**Parameters:**
- `in_ch`: channels coming in from previous decoder level
- `skip_ch`: channels from the matching encoder skip connection
- `out_ch`: output channels
- `dropout`: probability to zero neurons (0.0 = off, 0.5 = 50% chance per neuron)

---

## Main Class: `UNet`

**What it does:** Assembles all the blocks into the complete model.

```python
def __init__(self, num_classes=10, base_filters=32, dropout=0.0):
```

**Parameters:**
- `num_classes=10` — FloodNet has 10 classes (Background, Building Flooded, etc.)
- `base_filters=32` — Starting filter count. 32 gives ~7.8M params (paper-matched)
- `dropout=0.0` — How much dropout in decoder. HHO searches 0.1–0.5

**Encoder (4 levels):**
```python
self.enc1 = EncoderBlock(3,    32)   # RGB → 32 filters
self.enc2 = EncoderBlock(32,   64)   # 32 → 64 filters
self.enc3 = EncoderBlock(64,  128)   # 64 → 128 filters
self.enc4 = EncoderBlock(128, 256)   # 128 → 256 filters
```

**Bottleneck:**
```python
self.bottleneck = DoubleConv(256, 512)  # 256 → 512 filters, 32×32 spatial
```
This is the deepest point. The model has compressed everything to 32×32 pixels but with 512 feature channels — it "sees" the whole scene but abstractly.

**Decoder (4 levels, mirror of encoder):**
```python
self.dec4 = DecoderBlock(512, 256, 256, dropout)  # 32×32 → 64×64
self.dec3 = DecoderBlock(256, 128, 128, dropout)  # 64×64 → 128×128
self.dec2 = DecoderBlock(128,  64,  64, dropout)  # 128×128 → 256×256
self.dec1 = DecoderBlock( 64,  32,  32, dropout)  # 256×256 → 512×512
```

**Output head:**
```python
self.head = nn.Conv2d(32, 10, kernel_size=1)  # 32 channels → 10 class scores
```
A 1×1 convolution that maps the final 32 features to 10 class scores per pixel. No activation — raw numbers (logits). CrossEntropyLoss in train.py handles softmax internally.

---

### Method: `_init_weights`

**What it does:** Sets starting values for all weights using Kaiming initialization.

**Why needed?** If weights start at 0, no learning happens. If they start too large, training explodes. Kaiming initialization sets them to just the right scale for ReLU networks — proven to work better than random normal or uniform.

---

### Method: `forward(x)`

**What it does:** Defines the actual computation path — how data flows from input to output.

```python
# Go down (encoder)
skip1, x = self.enc1(x)   # save skip1, compress x
skip2, x = self.enc2(x)   # save skip2, compress x more
skip3, x = self.enc3(x)   # save skip3, compress x more
skip4, x = self.enc4(x)   # save skip4, compress x to 32×32

# Bottleneck — full scene understanding
x = self.bottleneck(x)

# Come back up (decoder), using saved skip connections
x = self.dec4(x, skip4)   # upsample + combine with skip4
x = self.dec3(x, skip3)   # upsample + combine with skip3
x = self.dec2(x, skip2)   # upsample + combine with skip2
x = self.dec1(x, skip1)   # upsample + combine with skip1 → back to 512×512

# Produce class scores
return self.head(x)        # (B, 10, 512, 512)
```

---

## Factory Function: `build_unet`

```python
def build_unet(dropout=0.0, base_filters=32) -> UNet:
```

**What it does:** A convenience function that builds a fresh U-Net.

Called by:
- HHO for each hawk evaluation (different dropout per hawk)
- `full_train` to build the final model
- `cmd_sanity` in main.py to verify the model works

Returns an uninitialized model (not yet moved to device, not yet trained).

---

## Key facts for your supervisor

| Property | Value |
|---|---|
| Total parameters | 7,763,338 |
| Input | Batch of RGB aerial images (B, 3, 512, 512) |
| Output | Raw logit scores per class per pixel (B, 10, 512, 512) |
| Architecture | Standard U-Net (Ronneberger 2015) |
| Hyperparameter tuned by HHO | `dropout` (how much to regularize the decoder) |
| No softmax in model | CrossEntropyLoss applies softmax internally |
