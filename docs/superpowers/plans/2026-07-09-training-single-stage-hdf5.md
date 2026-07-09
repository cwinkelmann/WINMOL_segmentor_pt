# Single-Stage PyTorch Training + HDF5 Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `winmol_unet.UNet` on a real jpeg-image/gif-mask dataset and export the trained model as a Keras `.hdf5` that drops into `WINMOL_Analyzer` unmodified (plus ONNX for free).

**Architecture:** A `training/` package (never imported by the analyzer) consumes `winmol_unet`'s model, preprocess, and exporters. Single-stage 80/20 train/val on one dataset; loss is `BCEWithLogits + (1 − soft_F1)` on logits; metrics are hard-rounded F1/precision/recall. The entry point reloads the best checkpoint and exports HDF5 + ONNX.

**Tech Stack:** Python 3.9, PyTorch 2.8, torchvision (photometric aug), Pillow (image IO), tensorboard, numpy, pytest. Depends on `winmol_unet`.

## Global Constraints

- Python floor **3.9**; depends on `winmol_unet` (`pip install -e ".[train]"`).
- Geometry fixed by the contract: **512×512×3 → 512×512×1**; resize tiles to 512 via `winmol_unet.preprocess.resize_batch` (**bicubic** image, **nearest** mask).
- `UNet.forward` returns **logits**; loss consumes logits (`BCEWithLogitsLoss`); metrics apply sigmoid + 0.5 threshold.
- Data convention (verified, `/Users/christian/data/Winmol/data/TestDS`): images `train/trainN.jpeg` (RGB), masks `mask/maskN.gif` (mode L, frame 0, binary {0,255}); pair by integer **N**; normalize image `/255`→[0,1]; mask `/255`→binarize {0,1}.
- Augmentation: geometric (random horizontal + vertical flip) applied **paired** to image+mask (same decision); photometric (brightness/contrast/saturation/hue) on the **image only**.
- `training/` must **never** be imported by the analyzer; it only depends on `winmol_unet`.
- Committed tests must be **hermetic** — build tiny synthetic jpeg/gif tiles in `tmp_path`; never depend on the absolute TestDS path. Filter only third-party deprecation noise, scoped.
- `winmol_unet` APIs available: `winmol_unet.model.UNet(in_channels=3, out_channels=1, dropout=0.1)` → logits `[N,1,512,512]`; `winmol_unet.preprocess.resize_batch(batch_nhwc, size=512, mode="bicubic"|"nearest")`; `winmol_unet.preprocess.to_float01(arr)`; `winmol_unet.export_keras.export_to_keras_hdf5(torch_model, path, dropout=0.1) -> path`; `winmol_unet.export.export_to_onnx(model, path) -> path`; `winmol_unet.contract.IMG_SIZE == 512`.

---

### Task 1: Training config

**Files:**
- Create: `training/__init__.py` (empty)
- Create: `training/config.py`
- Test: `tests/test_train_config.py`
- Modify: `pyproject.toml` (add `torchvision` to the `train` extra; add `training` to packages)

**Interfaces:**
- Produces: `TrainConfig` dataclass with fields `data_dir: str`, `checkpoint_dir: str`, `log_dir: str`, `hdf5_out: str`, `onnx_out: str`, `batch_size: int = 4`, `epochs: int = 100`, `lr: float = 1e-3`, `dropout: float = 0.1`, `img_size: int = 512`, `val_fraction: float = 0.2`, `patience: int = 5`, `seed: int = 1`; property `image_dir` → `<data_dir>/train`, `mask_dir` → `<data_dir>/mask`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_config.py
import os
from training.config import TrainConfig


def test_config_defaults_and_derived_dirs():
    c = TrainConfig(
        data_dir="/data/TestDS", checkpoint_dir="ck", log_dir="log",
        hdf5_out="out/model.hdf5", onnx_out="out/model.onnx",
    )
    assert c.batch_size == 4
    assert c.epochs == 100
    assert c.lr == 1e-3
    assert c.img_size == 512
    assert c.val_fraction == 0.2
    assert c.patience == 5
    assert c.seed == 1
    assert c.image_dir == os.path.join("/data/TestDS", "train")
    assert c.mask_dir == os.path.join("/data/TestDS", "mask")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training'`

- [ ] **Step 3: Write minimal implementation**

Create `training/__init__.py` (empty file).

```python
# training/config.py
"""Hyperparameters + paths for single-stage training (mirrors R controlling.R)."""
import os
from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str          # contains train/ (jpeg) and mask/ (gif)
    checkpoint_dir: str
    log_dir: str
    hdf5_out: str
    onnx_out: str
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-3
    dropout: float = 0.1
    img_size: int = 512
    val_fraction: float = 0.2
    patience: int = 5
    seed: int = 1

    @property
    def image_dir(self) -> str:
        return os.path.join(self.data_dir, "train")

    @property
    def mask_dir(self) -> str:
        return os.path.join(self.data_dir, "mask")
```

In `pyproject.toml`: change the `train` extra to
`train = ["torch>=2.0", "torchvision>=0.15", "pillow>=9", "tensorboard>=2.10"]`
and change `[tool.setuptools] packages` to `["winmol_unet", "training"]`. Then run
`.venv/bin/pip install -e ".[train]"` (installs torchvision). If torchvision cannot install
on this platform/Python, report BLOCKED with the exact pip error — do not substitute.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/__init__.py training/config.py tests/test_train_config.py pyproject.toml
git commit -m "feat: add TrainConfig + training package scaffold"
```

---

### Task 2: Loss (BCE + soft-F1) and metrics

**Files:**
- Create: `training/losses.py`
- Create: `training/metrics.py`
- Test: `tests/test_train_losses.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `bce_soft_f1_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor` (scalar, differentiable; `target` is {0,1} float, same shape as `logits`). `precision(logits, target) -> float`, `recall(logits, target) -> float`, `f1(logits, target) -> float` (hard-rounded via sigmoid + 0.5 threshold).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_losses.py
import torch
from training.losses import bce_soft_f1_loss
from training.metrics import precision, recall, f1


def _logits_for(target, strong=12.0):
    # logits whose sigmoid ~ target (near 0 or 1)
    return (target * 2 - 1) * strong


def test_perfect_prediction_low_loss_high_f1():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = _logits_for(target)
    loss = bce_soft_f1_loss(logits, target)
    assert loss.item() < 1e-2
    assert f1(logits, target) == 1.0
    assert precision(logits, target) == 1.0
    assert recall(logits, target) == 1.0


def test_loss_is_differentiable_through_soft_f1():
    target = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.zeros_like(target, requires_grad=True)
    bce_soft_f1_loss(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_metrics_match_hand_computed():
    # pred: [1,1,0,0]  target: [1,0,0,0]  -> tp=1 fp=1 fn=0 tn=2
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    logits = _logits_for(torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]))
    assert abs(precision(logits, target) - 0.5) < 1e-6   # tp/(tp+fp)=1/2
    assert abs(recall(logits, target) - 1.0) < 1e-6      # tp/(tp+fn)=1/1
    assert abs(f1(logits, target) - (2 / 3)) < 1e-6      # 2*.5*1/(1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.losses'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/losses.py
"""BCE + (1 - soft_F1) on logits. Soft (un-thresholded) F1 keeps the term
differentiable, unlike the R F1 which rounds y_pred (see design spec §8)."""
import torch
import torch.nn.functional as F


def bce_soft_f1_loss(logits, target, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum()
    fp = (probs * (1 - target)).sum()
    fn = ((1 - probs) * target).sum()
    soft_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return bce + (1 - soft_f1)
```

```python
# training/metrics.py
"""Hard-rounded precision/recall/F1 for reporting (sigmoid + 0.5 threshold)."""
import torch


def _counts(logits, target):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return tp, fp, fn


def precision(logits, target):
    tp, fp, _ = _counts(logits, target)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(logits, target):
    tp, _, fn = _counts(logits, target)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(logits, target):
    p, r = precision(logits, target), recall(logits, target)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train_losses.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add training/losses.py training/metrics.py tests/test_train_losses.py
git commit -m "feat: add BCE+soft-F1 loss and hard-rounded metrics"
```

---

### Task 3: Dataset (jpeg images + gif masks) with augmentation

**Files:**
- Create: `training/dataset.py`
- Test: `tests/test_train_dataset.py`

**Interfaces:**
- Consumes: `winmol_unet.preprocess.resize_batch`, `winmol_unet.preprocess.to_float01`.
- Produces:
  - `StemDataset(image_dir, mask_dir, img_size=512, augment=False, seed=1)` — a
    `torch.utils.data.Dataset`; `__len__` = number of paired samples; `__getitem__(i)` →
    `(image, mask)` where `image` is float32 CHW `[3,512,512]` in [0,1] and `mask` is
    float32 `[1,512,512]` in {0,1}. Pairs `trainN.jpeg` ↔ `maskN.gif` by integer N.
  - `train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512)` →
    `(train_ds, val_ds)` with augmentation ON for train, OFF for val, deterministic split.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_dataset.py
import numpy as np
import torch
from PIL import Image
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
    ds = StemDataset(str(img_dir), str(mask_dir))
    assert len(ds) == 3
    img, mask = ds[0]
    assert img.shape == (3, 512, 512) and img.dtype == torch.float32
    assert mask.shape == (1, 512, 512)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_paired_flip_keeps_alignment(tmp_path):
    # A mask that is all-ones on the left half; after any flip, image and mask
    # transform together, so correlation of a constant-structured pair is preserved.
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    rgb = np.zeros((40, 40, 3), np.uint8); rgb[:, :20, :] = 255
    Image.fromarray(rgb, "RGB").save(img_dir / "train1.jpeg")
    m = np.zeros((40, 40), np.uint8); m[:, :20] = 255
    Image.fromarray(m, "L").save(mask_dir / "mask1.gif")
    ds = StemDataset(str(img_dir), str(mask_dir), augment=True, seed=7)
    for _ in range(5):
        img, mask = ds[0]
        # where mask==1, the image (bright side) should be brighter than where mask==0
        bright = img[:, mask[0] == 1].mean()
        dark = img[:, mask[0] == 0].mean()
        assert bright > dark


def test_split_is_deterministic_and_disjoint(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    for n in range(1, 11):
        _make_pair(img_dir, mask_dir, n)
    tr, va = train_val_split(str(img_dir), str(mask_dir), val_fraction=0.2, seed=1)
    tr2, va2 = train_val_split(str(img_dir), str(mask_dir), val_fraction=0.2, seed=1)
    assert len(tr) == 8 and len(va) == 2
    assert tr.ids == tr2.ids and va.ids == va2.ids       # deterministic
    assert set(tr.ids).isdisjoint(set(va.ids))           # disjoint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/dataset.py
"""Paired jpeg-image / gif-mask dataset with shared-seed geometric augmentation
(flips) on both and photometric augmentation on the image only. Resize to 512
via winmol_unet.preprocess to eliminate train/inference skew."""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as TF

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
    def __init__(self, image_dir, mask_dir, img_size=512, augment=False, seed=1, ids=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.ids = ids if ids is not None else _paired_ids(image_dir, mask_dir)
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.ids)

    def _load_image(self, n):
        im = Image.open(os.path.join(self.image_dir, f"train{n}.jpeg")).convert("RGB")
        arr = to_float01(np.asarray(im))                       # HWC [0,1]
        arr = resize_batch(arr[None], size=self.img_size, mode="bicubic")[0]
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))  # CHW

    def _load_mask(self, n):
        mk = Image.open(os.path.join(self.mask_dir, f"mask{n}.gif"))
        mk.seek(0)
        arr = to_float01(np.asarray(mk.convert("L")))[..., None]   # HW1 [0,1]
        arr = resize_batch(arr, size=self.img_size, mode="nearest")[0]  # 512x512x1
        m = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return (m >= 0.5).float()

    def __getitem__(self, i):
        n = self.ids[i]
        img, mask = self._load_image(n), self._load_mask(n)
        if self.augment:
            if self._rng.random() < 0.5:                       # horizontal flip (paired)
                img, mask = TF.hflip(img), TF.hflip(mask)
            if self._rng.random() < 0.5:                       # vertical flip (paired)
                img, mask = TF.vflip(img), TF.vflip(mask)
            # photometric on image only
            img = TF.adjust_brightness(img, 1.0 + self._rng.uniform(-0.2, 0.2))
            img = TF.adjust_contrast(img, 1.0 + self._rng.uniform(-0.2, 0.2))
            img = TF.adjust_saturation(img, 1.0 + self._rng.uniform(-0.2, 0.2))
            img = TF.adjust_hue(img, self._rng.uniform(-0.05, 0.05))
            img = img.clamp(0.0, 1.0)
        return img, mask


def train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512):
    ids = _paired_ids(image_dir, mask_dir)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    train_ds = StemDataset(image_dir, mask_dir, img_size, augment=True, seed=seed, ids=train_ids)
    val_ds = StemDataset(image_dir, mask_dir, img_size, augment=False, seed=seed, ids=val_ids)
    return train_ds, val_ds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train_dataset.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py tests/test_train_dataset.py
git commit -m "feat: add paired jpeg/gif dataset with augmentation + deterministic split"
```

---

### Task 4: Train loop + evaluate

**Files:**
- Create: `training/train.py`
- Create: `training/evaluate.py`
- Test: `tests/test_train_loop.py`

**Interfaces:**
- Consumes: `training.losses.bce_soft_f1_loss`, `training.metrics.{precision,recall,f1}`,
  `winmol_unet.model.UNet`.
- Produces:
  - `train_one_run(model, train_loader, val_loader, cfg) -> model` — Adam(`cfg.lr`);
    per-epoch train + val loss; checkpoints the best-`val_loss` `state_dict` to
    `os.path.join(cfg.checkpoint_dir, "best.pt")`; early-stops after `cfg.patience` epochs
    without val-loss improvement; logs scalars to TensorBoard in `cfg.log_dir`; on return,
    the model holds the best weights.
  - `evaluate(model, loader) -> dict` with keys `loss, precision, recall, f1` (means).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_loop.py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.dataset import StemDataset
from training.train import train_one_run
from training.evaluate import evaluate
from winmol_unet.model import UNet


def _tiny_dataset(tmp_path, n_items=2):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for n in range(1, n_items + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{n}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{n}.gif")
    return StemDataset(str(img_dir), str(mask_dir), augment=False)


def test_overfit_loss_decreases(tmp_path):
    torch.manual_seed(0)
    ds = _tiny_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=2)
    cfg = TrainConfig(data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
                      log_dir=str(tmp_path / "log"), hdf5_out="x.hdf5", onnx_out="x.onnx",
                      epochs=8, lr=1e-2, patience=999)
    model = UNet(dropout=0.0)
    before = evaluate(model, loader)["loss"]
    train_one_run(model, loader, loader, cfg)
    after = evaluate(model, loader)["loss"]
    assert after < before


def test_evaluate_returns_metric_keys(tmp_path):
    ds = _tiny_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=2)
    out = evaluate(UNet(dropout=0.0), loader)
    assert set(out) == {"loss", "precision", "recall", "f1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/evaluate.py
"""Mean loss + hard-rounded metrics over a loader (no grad)."""
import torch

from .losses import bce_soft_f1_loss
from .metrics import precision, recall, f1


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = 0
    for img, mask in loader:
        logits = model(img)
        tot["loss"] += bce_soft_f1_loss(logits, mask).item()
        tot["precision"] += precision(logits, mask)
        tot["recall"] += recall(logits, mask)
        tot["f1"] += f1(logits, mask)
        n += 1
    return {k: (v / n if n else 0.0) for k, v in tot.items()}
```

```python
# training/train.py
"""Single-stage train/validate loop: Adam, best-val-loss checkpoint, early stop,
TensorBoard logging."""
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from .evaluate import evaluate
from .losses import bce_soft_f1_loss


def train_one_run(model, train_loader, val_loader, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(cfg.checkpoint_dir, "best.pt")
    writer = SummaryWriter(cfg.log_dir)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    since_improve = 0
    for epoch in range(cfg.epochs):
        model.train()
        for img, mask in train_loader:
            opt.zero_grad()
            loss = bce_soft_f1_loss(model(img), mask)
            loss.backward()
            opt.step()

        val = evaluate(model, val_loader)
        writer.add_scalar("val/loss", val["loss"], epoch)
        writer.add_scalar("val/f1", val["f1"], epoch)

        if val["loss"] < best_val:
            best_val = val["loss"]
            since_improve = 0
            torch.save(model.state_dict(), ckpt)
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                break

    writer.close()
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train_loop.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add training/train.py training/evaluate.py tests/test_train_loop.py
git commit -m "feat: add train loop (best-ckpt, early stop, tensorboard) + evaluate"
```

---

### Task 5: CLI entry point + end-to-end HDF5 export/parity

**Files:**
- Create: `training/run_train.py`
- Test: `tests/test_train_e2e.py`

**Interfaces:**
- Consumes: `training.config.TrainConfig`, `training.dataset.train_val_split`,
  `training.train.train_one_run`, `training.evaluate.evaluate`, `winmol_unet.model.UNet`,
  `winmol_unet.export_keras.export_to_keras_hdf5`, `winmol_unet.export.export_to_onnx`.
- Produces: `run_training(cfg) -> dict` — builds loaders from `cfg`, trains, reloads best,
  exports `cfg.hdf5_out` (Keras) and `cfg.onnx_out` (ONNX), returns final val metrics dict.
  Also a `__main__` block parsing `--data-dir/--out-dir/--epochs/--batch-size` into a
  `TrainConfig` and calling `run_training`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_e2e.py
import os
import numpy as np
import torch
from PIL import Image

from training.config import TrainConfig
from training.run_train import run_training
from winmol_unet.model import UNet


def _make_ds(tmp_path, n=6):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")


def test_end_to_end_train_export_hdf5_dropin(tmp_path):
    _make_ds(tmp_path)
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"), hdf5_out=str(out / "m.hdf5"),
        onnx_out=str(out / "m.onnx"), epochs=2, batch_size=2, patience=999,
    )
    metrics = run_training(cfg)
    assert set(metrics) == {"loss", "precision", "recall", "f1"}
    assert os.path.exists(cfg.hdf5_out) and os.path.exists(cfg.onnx_out)

    # Drop-in: load HDF5 the analyzer's way and predict on NHWC tiles.
    from tensorflow import keras
    km = keras.models.load_model(cfg.hdf5_out, compile=False)
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)
    pred = np.asarray(km.predict_on_batch(x))
    assert pred.shape == (2, 512, 512, 1)
    assert (pred >= 0).all() and (pred <= 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train_e2e.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.run_train'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/run_train.py
"""CLI: train on one dataset, export HDF5 (analyzer drop-in) + ONNX."""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from winmol_unet.export import export_to_onnx
from winmol_unet.export_keras import export_to_keras_hdf5
from winmol_unet.model import UNet

from .config import TrainConfig
from .dataset import train_val_split
from .evaluate import evaluate
from .train import train_one_run


def run_training(cfg):
    torch.manual_seed(cfg.seed)
    train_ds, val_ds = train_val_split(
        cfg.image_dir, cfg.mask_dir, cfg.val_fraction, cfg.seed, cfg.img_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = UNet(dropout=cfg.dropout)
    train_one_run(model, train_loader, val_loader, cfg)

    os.makedirs(os.path.dirname(cfg.hdf5_out) or ".", exist_ok=True)
    export_to_keras_hdf5(model, cfg.hdf5_out, dropout=cfg.dropout)
    export_to_onnx(model, cfg.onnx_out)

    return evaluate(model, val_loader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    cfg = TrainConfig(
        data_dir=a.data_dir,
        checkpoint_dir=os.path.join(a.out_dir, "checkpoints"),
        log_dir=os.path.join(a.out_dir, "logs"),
        hdf5_out=os.path.join(a.out_dir, "model.hdf5"),
        onnx_out=os.path.join(a.out_dir, "model.onnx"),
        epochs=a.epochs, batch_size=a.batch_size,
    )
    print(run_training(cfg))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/run_train.py tests/test_train_e2e.py
git commit -m "feat: add training CLI with HDF5 + ONNX export and drop-in e2e test"
```

---

## Post-plan validation (controller, not a committed test)

After all tasks pass, run the trainer on the **real** dataset to satisfy "test it with this
prediction" on actual data:

```bash
.venv/bin/python -m training.run_train \
  --data-dir /Users/christian/data/Winmol/data/TestDS \
  --out-dir output/TestDS --epochs 15 --batch-size 4
```

Confirm: printed val F1/precision/recall are meaningfully > 0 (model learned), and
`output/TestDS/model.hdf5` exists. Then verify the drop-in on the trained file with the
scratchpad `verify_dropin.py` (load `compile=False` → `predict_on_batch` → parity <1e-4).
This is the `main_prediction.R` stand-in: a trained model, exported to HDF5, loaded exactly
as the analyzer loads it, producing predictions on held-out tiles.

## Self-Review

**Spec coverage:** config §4→T1; loss/metrics §4→T2; dataset+split+aug §3,§4→T3; train
loop/checkpoint/early-stop/tensorboard + evaluate §4→T4; CLI + HDF5/ONNX export + drop-in
validation §4,§5,§6→T5 + post-plan run. ✓
**Placeholder scan:** none — all steps carry full code. ✓
**Type consistency:** `TrainConfig` fields/props (T1) used unchanged in T4/T5; `StemDataset`
/`train_val_split` signatures (T3) match T4/T5 usage; `bce_soft_f1_loss`, `precision/recall/f1`
(T2) used in T4; `train_one_run(model, train_loader, val_loader, cfg)` and `evaluate(model,
loader)->dict` consistent across T4/T5. ✓
