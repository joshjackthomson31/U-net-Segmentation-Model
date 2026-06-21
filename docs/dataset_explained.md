# src/dataset.py — Explained Simply

---

## What is this file for?

`dataset.py` is the **data pipeline** — it loads aerial images and their label masks from disk, prepares them into the correct format, and hands them to the model for training.

Think of it as the **kitchen prep cook** in a restaurant: the model (chef) doesn't go shopping for ingredients — the dataset file does all the preparation (washing, chopping, measuring) and hands ready-to-use ingredients to the model.

---

## What does it do?

1. Finds all image-mask pairs in the FloodNet folders
2. Resizes images to 512×512
3. Applies data augmentation during training (random flips/rotations)
4. Converts images to normalized tensors
5. Converts masks to class index tensors
6. Provides a DataLoader that batches multiple images together for efficient training

---

## Key Discovery About FloodNet Masks

FloodNet label masks are **grayscale PNG files**, not color images. Each pixel's value IS the class index directly:
- Pixel value `0` = Background
- Pixel value `1` = Building Flooded
- Pixel value `5` = Water
- ...and so on up to `9` = Grass

Some pixels have values 250-255 from the labeling tool's boundary artifacts. These get clamped to 0 (Background).

File naming convention:
- Image: `10165.jpg`
- Mask: `10165_lab.png` (same name + `_lab` suffix)

---

## Image Normalization

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
```

**Why normalize?** Raw pixel values are 0–255. Neural networks train faster when values are small and centered around 0. Normalization transforms pixel values to roughly the range [-2, 2].

**Formula:** `normalized = (pixel/255 - mean) / std`

**Example:** Red channel pixel = 200 (raw)
→ 200/255 = 0.784
→ (0.784 - 0.485) / 0.229 = **1.31** (normalized, small number, centered)

**Why ImageNet stats?** These are the standard stats from the ImageNet dataset (1.2M natural images). Even though our images are aerial photos, they're still natural images, so these stats work well.

---

## Class: `FloodNetDataset`

**What it does:** A custom PyTorch Dataset — a class that knows how to load and return one image-mask pair at a time.

PyTorch requires any dataset to implement three methods:
- `__init__`: set up, find all files
- `__len__`: how many items are in the dataset?
- `__getitem__`: give me item number N

### `__init__(img_dir, mask_dir, split, augment)`

**Parameters:**
- `img_dir`: folder path containing the aerial images
- `mask_dir`: folder path containing the label masks
- `split`: `"train"`, `"val"`, or `"test"` — used only for logging
- `augment`: `True` during training, `False` during validation/testing

**What it does:**
1. Lists all `.jpg`/`.jpeg`/`.png` files in `img_dir`
2. For each image file (e.g., `10165.jpg`), looks for its mask (`10165_lab.png`)
3. Only keeps pairs where BOTH image AND mask exist
4. Stores valid pairs in `self.pairs` list
5. Raises an error if no pairs found (catches wrong path issues immediately)

**Example output:**
```
[Dataset] train: 1444 image-mask pairs loaded.
```
(1 missing because `9033.jpg` failed to download — mask exists but image doesn't)

---

### `__len__()`

Returns `len(self.pairs)` — simply the count of image-mask pairs.

Used by PyTorch's DataLoader to know how many items exist and how to split them into batches.

---

### `__getitem__(idx)`

**What it does:** Returns ONE prepared image-mask pair at index `idx`.

**Step by step:**

**1. Load image:**
```python
image = Image.open(img_path).convert("RGB")
```
Opens the JPEG as a 3-channel (Red, Green, Blue) PIL Image.

**2. Load mask:**
```python
mask = Image.open(mask_path).convert("L")
```
Opens the PNG as a grayscale image. "L" mode = single channel, values 0-255.

**3. Resize to 512×512:**
```python
image = image.resize(IMAGE_SIZE, Image.BILINEAR)  # smooth interpolation
mask  = mask.resize(IMAGE_SIZE,  Image.NEAREST)   # no interpolation — preserves exact class values
```
**Why BILINEAR for image but NEAREST for mask?**
- BILINEAR blends pixel values during resize → smooth, natural-looking image
- NEAREST copies the nearest pixel value → a class index of 5 stays 5 (not averaged to 4.7 or 5.3)

**4. Augment (training only):**
```python
if self.augment:
    image, mask = self._augment(image, mask)
```
Random flips and rotations to make the model more robust.

**5. Clamp mask values:**
```python
mask_np = np.array(mask, dtype=np.int64)
mask_np[mask_np >= NUM_CLASSES] = 0  # values 10-255 → 0 (Background)
```
Boundary artifact pixels (250, 255) become Background (0). This is safe because those pixels are on label boundaries, not meaningful content.

**6. Convert to tensors:**
```python
image_tensor = TF.to_tensor(image)     # (3, H, W), float32, values 0.0-1.0
image_tensor = normalize(image_tensor)  # centered, standard deviation ~1
mask_tensor  = torch.from_numpy(mask_np)  # (H, W), int64, values 0-9
```

**Returns:** `(image_tensor, mask_tensor)` — a tuple ready for the model.

---

### `_augment(image, mask)`

**What it does:** Applies random geometric transforms identically to both image and mask.

**Critical rule:** Image and mask must receive the EXACT same transform. If we flip the image left-right, the mask must also flip left-right — otherwise "Building Flooded" labels end up on the wrong side of the buildings.

**Three augmentations (each 50% chance):**

1. **Horizontal flip** — mirror the image left ↔ right
   ```python
   if random.random() > 0.5:
       image = TF.hflip(image)
       mask  = TF.hflip(mask)
   ```

2. **Vertical flip** — mirror the image top ↔ bottom
   ```python
   if random.random() > 0.5:
       image = TF.vflip(image)
       mask  = TF.vflip(mask)
   ```

3. **90°/180°/270° rotation** — rotate the scene
   ```python
   if random.random() > 0.5:
       angle = random.choice([90, 180, 270])
       image = TF.rotate(image, angle)
       mask  = TF.rotate(mask,  angle)
   ```

**Why augment?** Aerial images can come from any direction. A flooded road looks the same whether the camera faces north or east. Augmentation teaches the model to recognize objects regardless of orientation, making it more general and robust.

**Why only during training?** Val and test sets measure real performance — augmenting them would give different results each time, making metrics unreliable.

---

## Function: `get_dataloaders(batch_size, include_test=True)`

**What it does:** Creates three PyTorch DataLoaders (train, val, test) ready for the training loop.

**What a DataLoader does:**
- Wraps a Dataset
- Batches multiple images together (batch_size=4 → 4 images at once)
- Shuffles training data each epoch (so the model doesn't memorize order)
- Loads data in parallel using multiple CPU workers (NUM_WORKERS=4)

**Parameters:**
- `batch_size`: how many images per batch (HHO tunes this: 2, 3, 4, or 8)
- `include_test`: if `False`, skips creating the test DataLoader — used in proxy_train to save a few seconds

**Key settings:**
```python
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,      # shuffle each epoch
    num_workers=4,     # 4 CPU processes loading in parallel
    drop_last=True,    # discard last incomplete batch (prevents BatchNorm issues)
    generator=g,       # seeded for reproducibility
)
val_loader = DataLoader(
    val_ds,
    shuffle=False,     # always same order for consistent metrics
)
```

**Why `drop_last=True` for training?**
BatchNorm computes statistics (mean, variance) per batch. If the last batch has only 1-2 images (smaller than batch_size), these statistics are unreliable. Dropping it avoids this issue.

---

## Function: `get_class_weights(batch_size=4)`

**What it does:** Scans the entire training set, counts how many pixels belong to each class, and computes inverse-frequency weights.

**Why needed?** FloodNet is severely imbalanced:
- Grass: 56.45% of pixels (very common)
- Vehicle: 0.18% of pixels (very rare)

Without weights, the model learns to predict Grass everywhere because it's right 56% of the time. With inverse-frequency weights, rare classes get higher loss penalty — the model is forced to learn them.

**Step by step:**

1. **Scan all training images:**
   ```python
   for _, masks in loader:
       for c in range(10):
           counts[c] += (masks == c).sum().item()
   ```
   After scanning 1444 images: `counts = [pixels_in_class_0, ..., pixels_in_class_9]`

2. **Compute inverse frequency:**
   ```python
   weights = 1.0 / counts
   ```
   Classes with more pixels get lower weight, rare classes get higher weight.

3. **Normalize so sum = NUM_CLASSES:**
   ```python
   weights = weights / weights.sum() * NUM_CLASSES
   ```
   Keeps the loss scale similar to unweighted training.

4. **Clamp minimum to 0.1:**
   ```python
   weights = weights.clamp(min=0.1)
   weights = weights / weights.sum() * NUM_CLASSES  # re-normalize
   ```
   Prevents any class (like Grass at 56%) from getting so low a weight that the model ignores it completely. Results from our experiments:
   - Grass (56.45%): weight = 0.0983
   - Vehicle (0.18%): weight = 4.3307

**Returns:** A float tensor of shape (10,) — one weight per class.

**Why computed here, not inside proxy_train?**
Scanning 1444 images takes ~53 seconds. If computed inside proxy_train, it would run for every hawk evaluation (up to 1000 times) = ~14 hours of wasted computation. Computed once and passed as an argument.
