# Online Augmentation (albumentations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the torchvision augmentation with a configurable albumentations pipeline applied online per epoch; augmentation types + probabilities set via CLI flags.

**Architecture:** `build_augmentation(cfg)` assembles an `albumentations.Compose` from curated config knobs. `StemDataset` takes a `transform` object (not an `augment` bool), caches resized numpy arrays, and applies the transform per `__getitem__`. Resize stays in `winmol_unet.preprocess`; val gets no transform.

**Tech Stack:** Python 3.9, albumentations 2.0.8 (verified), numpy, PyTorch, pytest.

## Global Constraints

- Python 3.9; `albumentations` (newest, 2.0.8 verified) added to the `[train]` extra.
- Resize stays in `winmol_unet.preprocess.resize_batch` (bicubic image, nearest mask); the
  albumentations pipeline runs AFTER resize, on cached 512×512 arrays. Cache still requires
  `num_workers=0`.
- Geometric transforms apply to image+mask together (albumentations default); photometric to
  the image only. Mask stays binary {0,1} (re-threshold ≥0.5 after transforms).
- Verified albumentations 2.0.8 API: `A.HorizontalFlip(p)`, `A.VerticalFlip(p)`,
  `A.Rotate(limit, p, border_mode=0, fill=0, fill_mask=0)`, `A.RandomBrightnessContrast(
  brightness_limit, contrast_limit, p)`, `A.HueSaturationValue(p)`, `A.Compose([...])(
  image=HWC_float32, mask=HW_float32)`. All accept float32 [0,1] images.
- `StemDataset` signature becomes `(image_dir, mask_dir, img_size=512, transform=None, seed=1,
  ids=None)` — the `augment` bool is REMOVED. Default `transform=None` = no augmentation
  (existing callers that constructed `StemDataset(img, mask)` keep working).
- Val dataset always gets `transform=None`.

---

### Task 1: albumentations dep, config knobs, `build_augmentation`

**Files:**
- Modify: `pyproject.toml` (add `albumentations` to the `train` extra)
- Modify: `training/config.py` (add `aug_*` fields)
- Create: `training/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Produces: `TrainConfig` gains `aug_hflip_p=0.5`, `aug_vflip_p=0.5`, `aug_rotate_p=0.0`,
  `aug_rotate_limit=15.0`, `aug_bc_p=0.5`, `aug_brightness_limit=0.2`, `aug_contrast_limit=0.2`,
  `aug_hsv_p=0.5` (all `float`). `build_augmentation(cfg) -> albumentations.Compose`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import numpy as np
import albumentations as A

from training.config import TrainConfig
from training.augment import build_augmentation


def _cfg(**kw):
    base = dict(data_dir="d", checkpoint_dir="c", log_dir="l", hdf5_out="h", onnx_out="o")
    base.update(kw)
    return TrainConfig(**base)


def test_only_enabled_transforms_included():
    cfg = _cfg(aug_hflip_p=0.5, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=0.0, aug_hsv_p=0.0)
    names = [type(t).__name__ for t in build_augmentation(cfg).transforms]
    assert names == ["HorizontalFlip"]


def test_all_transforms_when_enabled():
    cfg = _cfg(aug_hflip_p=0.5, aug_vflip_p=0.5, aug_rotate_p=0.3,
               aug_bc_p=0.5, aug_hsv_p=0.5)
    names = {type(t).__name__ for t in build_augmentation(cfg).transforms}
    assert names == {"HorizontalFlip", "VerticalFlip", "Rotate",
                     "RandomBrightnessContrast", "HueSaturationValue"}


def test_paired_geometric_and_photometric_image_only():
    # Left-bright image + left-half mask; forced hflip must move both together.
    img = np.zeros((512, 512, 3), np.float32); img[:, :256, :] = 1.0
    mask = np.zeros((512, 512), np.float32); mask[:, :256] = 1.0
    cfg = _cfg(aug_hflip_p=1.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=0.0, aug_hsv_p=0.0)
    out = build_augmentation(cfg)(image=img, mask=mask)
    # after hflip the bright half and the mask==1 half are both on the right, still aligned
    assert out["image"][:, 256:, :].mean() > out["image"][:, :256, :].mean()
    assert out["mask"][:, 256:].mean() > out["mask"][:, :256].mean()
    assert np.allclose((out["image"].mean(axis=2) > 0.5), out["mask"] > 0.5)

    # photometric changes the image but never the mask
    cfg2 = _cfg(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=1.0, aug_hsv_p=0.0)
    out2 = build_augmentation(cfg2)(image=img, mask=mask)
    assert not np.allclose(out2["image"], img)
    assert np.allclose(out2["mask"], mask)


def test_rotate_keeps_mask_binary():
    img = np.random.rand(512, 512, 3).astype(np.float32)
    mask = np.zeros((512, 512), np.float32); mask[100:400, 100:400] = 1.0
    cfg = _cfg(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=1.0, aug_rotate_limit=30,
               aug_bc_p=0.0, aug_hsv_p=0.0)
    out = build_augmentation(cfg)(image=img, mask=mask)
    assert set(np.unique(out["mask"]).tolist()) <= {0.0, 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_augment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.augment'` (and unknown `aug_*`
kwargs). Install first: add `albumentations` to `pyproject.toml`'s `train` extra and run
`.venv/bin/pip install -e ".[train]"` (albumentations 2.0.8 is already present in this venv).

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`: append `"albumentations>=2.0"` to the `train` extra list.

`training/config.py`: add after `wandb_run_name` (or the pt/keras fields):
```python
    aug_hflip_p: float = 0.5
    aug_vflip_p: float = 0.5
    aug_rotate_p: float = 0.0
    aug_rotate_limit: float = 15.0
    aug_bc_p: float = 0.5
    aug_brightness_limit: float = 0.2
    aug_contrast_limit: float = 0.2
    aug_hsv_p: float = 0.5
```

```python
# training/augment.py
"""Build an albumentations pipeline from TrainConfig knobs (online augmentation).

Geometric transforms carry image+mask; photometric carry the image only. Applied
after the winmol_unet.preprocess resize, on 512x512 float32 [0,1] arrays.
border_mode=0 is cv2.BORDER_CONSTANT; fill/fill_mask=0 pad rotated borders with 0
(no phantom stems). A transform is included only when its probability > 0.
"""
import albumentations as A


def build_augmentation(cfg):
    transforms = []
    if cfg.aug_hflip_p > 0:
        transforms.append(A.HorizontalFlip(p=cfg.aug_hflip_p))
    if cfg.aug_vflip_p > 0:
        transforms.append(A.VerticalFlip(p=cfg.aug_vflip_p))
    if cfg.aug_rotate_p > 0:
        transforms.append(A.Rotate(limit=cfg.aug_rotate_limit, p=cfg.aug_rotate_p,
                                   border_mode=0, fill=0, fill_mask=0))
    if cfg.aug_bc_p > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=cfg.aug_brightness_limit,
            contrast_limit=cfg.aug_contrast_limit, p=cfg.aug_bc_p))
    if cfg.aug_hsv_p > 0:
        transforms.append(A.HueSaturationValue(p=cfg.aug_hsv_p))
    return A.Compose(transforms)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_augment.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml training/config.py training/augment.py tests/test_augment.py
git commit -m "feat: add albumentations build_augmentation + config knobs"
```

---

### Task 2: StemDataset uses a transform (numpy cache + albumentations)

**Files:**
- Modify: `training/dataset.py`
- Test: `tests/test_train_dataset.py` (update to the new `transform` API)

**Interfaces:**
- Consumes: `winmol_unet.preprocess.resize_batch`, `winmol_unet.preprocess.to_float01`, an
  `albumentations.Compose` (or None).
- Produces: `StemDataset(image_dir, mask_dir, img_size=512, transform=None, seed=1, ids=None)`;
  `__getitem__` → `(image CHW float32 [0,1], mask [1,H,W] {0,1})`; cache holds resized numpy
  arrays. `train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512, transform=
  None)` → train ds gets `transform`, val ds gets `None`.

- [ ] **Step 1: Write the failing test** (replace the augmentation-related tests)

```python
# tests/test_train_dataset.py
import numpy as np
import torch
from PIL import Image
import albumentations as A
from training.dataset import StemDataset, train_val_split


def _make_pair(img_dir, mask_dir, n, size=40):
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgb = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(img_dir / f"train{n}.jpeg")
    m = (np.random.rand(size, size) > 0.5).astype(np.uint8) * 255
    Image.fromarray(m, mode="L").save(mask_dir / f"mask{n}.gif")


def test_getitem_shapes_ranges_and_pairing(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    for n in (1, 2, 10):
        _make_pair(img_dir, mask_dir, n)
    ds = StemDataset(str(img_dir), str(mask_dir))       # transform=None
    assert len(ds) == 3
    img, mask = ds[0]
    assert img.shape == (3, 512, 512) and img.dtype == torch.float32
    assert mask.shape == (1, 512, 512)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_transform_paired_flip_keeps_alignment(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    rgb = np.zeros((40, 40, 3), np.uint8); rgb[:, :20, :] = 255
    Image.fromarray(rgb, "RGB").save(img_dir / "train1.jpeg")
    m = np.zeros((40, 40), np.uint8); m[:, :20] = 255
    Image.fromarray(m, "L").save(mask_dir / "mask1.gif")
    transform = A.Compose([A.HorizontalFlip(p=1.0)])    # always flip
    ds = StemDataset(str(img_dir), str(mask_dir), transform=transform)
    img, mask = ds[0]
    bright = img[:, mask[0] == 1].mean()
    dark = img[:, mask[0] == 0].mean()
    assert bright > dark        # image and mask flipped together

def test_transform_photometric_leaves_mask(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    _make_pair(img_dir, mask_dir, 1)
    base = StemDataset(str(img_dir), str(mask_dir))[0][1]        # mask, no transform
    tf = A.Compose([A.RandomBrightnessContrast(p=1.0)])
    _, mask = StemDataset(str(img_dir), str(mask_dir), transform=tf)[0]
    assert torch.equal(base, mask)                              # mask unchanged by photometric


def test_cache_not_mutated_by_transform(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    _make_pair(img_dir, mask_dir, 1)
    tf = A.Compose([A.HorizontalFlip(p=1.0)])
    ds = StemDataset(str(img_dir), str(mask_dir), transform=tf)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)      # deterministic (p=1 flip) -> cache base intact each call


def test_split_disjoint_and_val_has_no_transform(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    for n in range(1, 11):
        _make_pair(img_dir, mask_dir, n)
    tf = A.Compose([A.HorizontalFlip(p=1.0)])
    tr, va = train_val_split(str(img_dir), str(mask_dir), 0.2, 1, transform=tf)
    assert len(tr) == 8 and len(va) == 2
    assert set(tr.ids).isdisjoint(set(va.ids))
    assert tr.transform is tf and va.transform is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train_dataset.py -v`
Expected: FAIL (`StemDataset` has no `transform` param / `augment` removed).

- [ ] **Step 3: Write minimal implementation**

Rewrite `training/dataset.py`'s imports and class (drop torchvision; cache numpy):
```python
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from winmol_unet.preprocess import resize_batch, to_float01


def _index_by_n(names, prefix, ext):
    out = {}
    for name in names:
        if name.startswith(prefix) and name.endswith(ext):
            stem = name[len(prefix):-len(ext)]
            if stem.isdigit():
                out[int(stem)] = name
    return out


def _paired_ids(image_dir, mask_dir):
    imgs = _index_by_n(os.listdir(image_dir), "train", ".jpeg")
    masks = _index_by_n(os.listdir(mask_dir), "mask", ".gif")
    return sorted(set(imgs) & set(masks))


class StemDataset(Dataset):
    """Paired jpeg-image / gif-mask dataset. Resizes to img_size (bicubic image,
    nearest mask) via winmol_unet.preprocess and caches the resized numpy arrays
    (~4 MB/pair) so the skimage resize runs once; the optional albumentations
    `transform` runs per __getitem__ on the cached arrays. Requires
    DataLoader(num_workers=0) for the cache to persist across epochs.
    """

    def __init__(self, image_dir, mask_dir, img_size=512, transform=None, seed=1, ids=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        self.ids = ids if ids is not None else _paired_ids(image_dir, mask_dir)
        self._rng = random.Random(seed)
        self._cache = {}   # n -> (image HWC float32 [0,1], mask HW float32 {0,1})

    def __len__(self):
        return len(self.ids)

    def _load_image(self, n):
        im = Image.open(os.path.join(self.image_dir, f"train{n}.jpeg")).convert("RGB")
        arr = to_float01(np.asarray(im))                       # HWC [0,1]
        arr = resize_batch(arr[None], size=self.img_size, mode="bicubic")[0]
        return np.ascontiguousarray(arr, dtype=np.float32)     # HWC

    def _load_mask(self, n):
        mk = Image.open(os.path.join(self.mask_dir, f"mask{n}.gif"))
        mk.seek(0)
        arr = to_float01(np.asarray(mk.convert("L")))[..., None]   # HW1 [0,1]
        arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]  # HW1
        return (arr[..., 0] >= 0.5).astype(np.float32)             # HW binary

    def __getitem__(self, i):
        n = self.ids[i]
        if n not in self._cache:
            self._cache[n] = (self._load_image(n), self._load_mask(n))
        img, mask = self._cache[n]
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)   # albumentations returns new arrays
            img, mask = out["image"], out["mask"]
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return img_t.float(), (mask_t >= 0.5).float()


def train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512, transform=None):
    ids = _paired_ids(image_dir, mask_dir)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    train_ds = StemDataset(image_dir, mask_dir, img_size, transform=transform, seed=seed, ids=train_ids)
    val_ds = StemDataset(image_dir, mask_dir, img_size, transform=None, seed=seed, ids=val_ids)
    return train_ds, val_ds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train_dataset.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py tests/test_train_dataset.py
git commit -m "refactor: StemDataset takes an albumentations transform (numpy cache)"
```

---

### Task 3: Wire augmentation into training + CLI flags

**Files:**
- Modify: `training/run_train.py`
- Test: `tests/test_aug_wiring.py`

**Interfaces:**
- Consumes: `training.augment.build_augmentation`, `training.dataset.train_val_split`.
- Produces: `run_training` builds `transform = build_augmentation(cfg)` and passes it to
  `train_val_split(..., transform=transform)`; seeds numpy. `config_from_args` adds `--aug-*`
  flags mapping to the config fields.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aug_wiring.py
import os
import numpy as np
from PIL import Image

from training.run_train import config_from_args, run_training
from training.config import TrainConfig


def test_cli_parses_aug_flags():
    cfg = config_from_args(["--data-dir", "d", "--aug-rotate-p", "0.3",
                            "--aug-rotate-limit", "20", "--aug-hflip-p", "0.25"])
    assert cfg.aug_rotate_p == 0.3
    assert cfg.aug_rotate_limit == 20
    assert cfg.aug_hflip_p == 0.25


def test_aug_flags_default_preserved():
    cfg = config_from_args(["--data-dir", "d"])
    assert cfg.aug_hflip_p == 0.5 and cfg.aug_rotate_p == 0.0


def test_run_training_with_rotation_enabled(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, 7):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"), hdf5_out=str(out / "m.hdf5"),
        onnx_out=str(out / "m.onnx"), epochs=1, batch_size=2, patience=999,
        device="cpu", aug_rotate_p=1.0, aug_rotate_limit=25, aug_hsv_p=1.0,
    )
    metrics = run_training(cfg)          # exercises the full albumentations path
    assert set(metrics) == {"loss", "precision", "recall", "f1"}
    assert os.path.exists(cfg.hdf5_out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_aug_wiring.py -v`
Expected: FAIL — unknown `--aug-*` flags / augmentation not wired.

- [ ] **Step 3: Write minimal implementation**

In `training/run_train.py`: add `import numpy as np` and
`from .augment import build_augmentation`. In `run_training`, after `torch.manual_seed`:
```python
    np.random.seed(cfg.seed)                  # albumentations uses numpy RNG
    transform = build_augmentation(cfg)
    train_ds, val_ds = train_val_split(
        cfg.image_dir, cfg.mask_dir, cfg.val_fraction, cfg.seed, cfg.img_size,
        transform=transform)
```
(remove the old `train_val_split(...)` call without transform).

In `config_from_args`, add the flags and pass them through:
```python
    p.add_argument("--aug-hflip-p", type=float, default=0.5)
    p.add_argument("--aug-vflip-p", type=float, default=0.5)
    p.add_argument("--aug-rotate-p", type=float, default=0.0)
    p.add_argument("--aug-rotate-limit", type=float, default=15.0)
    p.add_argument("--aug-bc-p", type=float, default=0.5)
    p.add_argument("--aug-brightness-limit", type=float, default=0.2)
    p.add_argument("--aug-contrast-limit", type=float, default=0.2)
    p.add_argument("--aug-hsv-p", type=float, default=0.5)
```
and in the `TrainConfig(...)` construction add:
```python
        aug_hflip_p=a.aug_hflip_p, aug_vflip_p=a.aug_vflip_p,
        aug_rotate_p=a.aug_rotate_p, aug_rotate_limit=a.aug_rotate_limit,
        aug_bc_p=a.aug_bc_p, aug_brightness_limit=a.aug_brightness_limit,
        aug_contrast_limit=a.aug_contrast_limit, aug_hsv_p=a.aug_hsv_p,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_aug_wiring.py -v`
Expected: PASS (3 tests). Also run the full training suite to confirm no regression:
`.venv/bin/pytest tests/ -q`

- [ ] **Step 5: Commit**

```bash
git add training/run_train.py tests/test_aug_wiring.py
git commit -m "feat: wire albumentations augmentation + --aug-* CLI flags into training"
```

---

## Self-Review

**Spec coverage:** albumentations dep + knobs + `build_augmentation` §3,§4→T1; StemDataset
`transform` + numpy cache + val-no-transform §4→T2; run_train wiring + CLI + seed §4→T3;
tests (build/paired/photometric/rotate-binary/shapes/CLI/e2e) §6→T1-T3. ✓
**Placeholder scan:** none. ✓
**Type consistency:** `build_augmentation(cfg)->A.Compose` (T1) used in T3; `StemDataset(...,
transform=...)` / `train_val_split(..., transform=...)` (T2) used in T3; `aug_*` config fields
(T1) consumed by `build_augmentation` (T1) and `config_from_args` (T3). Backward-compat:
`StemDataset(img,mask)` default `transform=None` keeps `test_train_loop`/`test_wandb_wiring`/
`test_multiformat_e2e` (which build datasets without augmentation) working. ✓
