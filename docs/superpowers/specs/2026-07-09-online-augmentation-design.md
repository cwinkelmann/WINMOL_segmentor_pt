# Online Augmentation (albumentations) — Design Spec

**Date:** 2026-07-09
**Status:** Approved (FEATURES.md #3)
**Repo:** `WINMOL_segmentor_pt` (`training/` package)

## 1. Goal

Replace the trainer's torchvision-based augmentation with an **albumentations** pipeline,
applied on-the-fly per epoch, with augmentation types + probabilities configurable via CLI
flags. Keeps the shared `winmol_unet.preprocess` resize and the resize cache.

## 2. Scope (locked with user)

- **Bounded CLI knobs** (not a YAML pipeline file): a curated set of transforms, each with a
  probability + strength.
- **Resize stays** in `winmol_unet.preprocess` (bicubic image / nearest mask); albumentations
  runs *after*, on the cached 512×512 arrays.
- Replace `StemDataset(augment: bool, ...)` with `StemDataset(transform: A.Compose | None, ...)`.
- Val set gets no augmentation.
- Add `albumentations` (newest) to the `[train]` extra.

## 3. Curated transforms (config knobs)

| Transform (albumentations) | Applies to | Config knobs (defaults) |
|---|---|---|
| `HorizontalFlip` | image+mask | `aug_hflip_p=0.5` |
| `VerticalFlip` | image+mask | `aug_vflip_p=0.5` |
| `Rotate` | image+mask (mask nearest, zero-fill) | `aug_rotate_p=0.0`, `aug_rotate_limit=15` |
| `RandomBrightnessContrast` | image only | `aug_bc_p=0.5`, `aug_brightness_limit=0.2`, `aug_contrast_limit=0.2` |
| `HueSaturationValue` | image only | `aug_hsv_p=0.5` |

Rotation is **off by default** (`p=0.0`) — it introduces border artifacts; opt in via CLI.

## 4. Components

- **`training/augment.py`** — `build_augmentation(cfg) -> albumentations.Compose`: assembles
  the pipeline from the knobs above; a transform is included only when its probability > 0.
  Geometric transforms carry image+mask; photometric carry image only (albumentations default).
- **`training/dataset.py`** — `StemDataset(image_dir, mask_dir, img_size=512, transform=None,
  seed=1, ids=None)`. Cache holds **resized numpy** arrays (`image` HWC float32 [0,1], `mask`
  HW float32 {0,1}). `__getitem__`: pull from cache → if `transform`, `transform(image=img,
  mask=mask)` → to torch (CHW image, `[1,H,W]` mask, re-binarize mask ≥0.5). Cache not mutated.
  `train_val_split(..., transform=None)` passes `transform` to the train ds, `None` to val.
- **`training/config.py`** — add the `aug_*` fields (Section 3).
- **`training/run_train.py`** — `run_training` builds `transform = build_augmentation(cfg)` and
  passes it to `train_val_split`; `config_from_args` adds `--aug-*` flags; seed numpy for
  reproducibility (num_workers=0 already).

## 5. Data flow

jpeg/gif → resize (preprocess) → **cache (numpy)** → `A.Compose` transform (train only) →
torch tensors → `UNet`. Val: cache → torch (no transform).

## 6. Testing (hermetic)

1. `build_augmentation` — only-hflip config → Compose with exactly `HorizontalFlip`; knobs
   (limits/probabilities) pass through to the transforms.
2. Paired geometric — left-bright image + left-half mask, `HorizontalFlip(p=1)` → bright pixels
   and mask==1 stay aligned (bright region mean > dark region mean where mask==1).
3. Photometric image-only — `RandomBrightnessContrast(p=1)` changes the image but the returned
   mask is identical to input.
4. Shapes/ranges — dataset item: image CHW float32 in [0,1], mask `[1,512,512]` in {0,1}.
5. CLI — `--aug-rotate-p 0.3 --aug-rotate-limit 20` → `cfg.aug_rotate_p==0.3`,
   `cfg.aug_rotate_limit==20`; e2e run still trains and exports.

## 7. Constraints

- Python 3.9; `albumentations` newest; opencv pulled transitively.
- Resize unchanged (train/inference consistency); cache still requires `num_workers=0`.
- Mask stays binary {0,1} after geometric transforms (nearest interpolation + re-threshold).
- Determinism: seed numpy RNG (albumentations uses it); single-process loader.

## 8. Out of scope (YAGNI)

- YAML/JSON augmentation-pipeline files (bounded CLI knobs only).
- Non-geometric/photometric transforms beyond the curated five.
- GPU augmentation (kornia — dropped per updated FEATURES).
