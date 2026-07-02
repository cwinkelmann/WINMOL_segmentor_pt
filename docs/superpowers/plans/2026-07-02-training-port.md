# Training Pipeline Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the R segmentor's full training pipeline (two-stage GenDS→SpecDS training, jpeg/gif data pipeline, augmentation, F1 loss + precision/recall/F1 metrics, checkpoint/early-stop/TensorBoard) to PyTorch, producing ONNX models via `winmol_unet.export`.

**Architecture:** A `training/` package (not imported by the analyzer) consumes the `winmol_unet` model + preprocess + export. Data are jpeg images + gif masks; geometric augmentation is applied paired to image and mask, photometric augmentation to the image only. Loss is `BCEWithLogits + (1 − soft_F1)` on logits; metrics use hard-rounded F1. Training runs in two stages and exports ONNX at the end.

**Tech Stack:** Python 3.9, PyTorch, torchvision/albumentations, Pillow, tensorboard, numpy, pytest. Depends on `winmol_unet` (Plan 1).

**Prerequisite:** Plan 1 (Inference Bridge) is complete — `winmol_unet.model.UNet`, `winmol_unet.preprocess`, and `winmol_unet.export.export_to_onnx` exist and pass their tests.

## Global Constraints

- Python floor **3.9**; depends on `winmol_unet` (install `pip install -e ".[train]"`).
- Model input/output geometry is fixed by the contract: **512×512×3 → 512×512×1**; the training pipeline must resize tiles to 512 via `winmol_unet.preprocess.resize_batch`.
- Model `forward` returns **logits**; loss consumes logits (`BCEWithLogitsLoss`); metrics apply sigmoid + 0.5 threshold.
- Data convention (verbatim from R): images are **`.jpeg`**, masks are **`.gif`** (first frame, single channel), values normalized `/255`; mask is binary {0,1}.
- Two-stage training: **stage 1 = GenDS**, 80/20 train/val split, early-stop patience **3**; **stage 2 = SpecDS** fine-tune, early-stop patience **5**; up to **100** epochs; Adam `lr=1e-3`.
- Augmentation: geometric (random up/down + left/right flips) applied **paired** to image+mask with a shared seed; photometric (brightness/contrast/saturation/hue) on the **image only**.

---

### Task 1: Training config

**Files:**
- Create: `training/__init__.py`
- Create: `training/config.py`
- Test: `tests/test_train_config.py`

**Interfaces:**
- Produces: `TrainConfig` dataclass with fields: `gen_train_dir`, `gen_mask_dir`, `spec_train_dir`, `spec_mask_dir`, `checkpoint_dir`, `log_dir`, `onnx_out`, `batch_size=4`, `epochs=100`, `lr=1e-3`, `dropout=0.1`, `img_size=512`, `patience_stage1=3`, `patience_stage2=5`, `val_fraction=0.2`, `seed=1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_config.py
from training.config import TrainConfig

def test_config_defaults():
    c = TrainConfig(
        gen_train_dir="g/train", gen_mask_dir="g/mask",
        spec_train_dir="s/train", spec_mask_dir="s/mask",
        checkpoint_dir="ck", log_dir="log", onnx_out="out/model.onnx",
    )
    assert c.batch_size == 4
    assert c.epochs == 100
    assert c.lr == 1e-3
    assert c.img_size == 512
    assert c.patience_stage1 == 3
    assert c.patience_stage2 == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_config.py -v`
Expected: FAIL with `ModuleNotFoundError: training.config`

- [ ] **Step 3: Write minimal implementation**

```python
# training/__init__.py
```

```python
# training/config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    gen_train_dir: str
    gen_mask_dir: str
    spec_train_dir: str
    spec_mask_dir: str
    checkpoint_dir: str
    log_dir: str
    onnx_out: str
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-3
    dropout: float = 0.1
    img_size: int = 512
    patience_stage1: int = 3
    patience_stage2: int = 5
    val_fraction: float = 0.2
    seed: int = 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/__init__.py training/config.py tests/test_train_config.py
git commit -m "feat: add training config dataclass"
```

---

### Task 2: Losses (soft-F1 + BCE) and metrics

**Files:**
- Create: `training/losses.py`
- Create: `training/metrics.py`
- Test: `tests/test_losses_metrics.py`

**Interfaces:**
- Consumes: logits from `UNet`.
- Produces:
  - `f1_bce_loss(logits, target, eps=1e-7) -> torch.Tensor` — `BCEWithLogits + (1 − soft_F1)`.
  - `precision_recall_f1(logits, target, threshold=0.5) -> dict` — hard-rounded metrics.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_losses_metrics.py
import torch
from training.losses import f1_bce_loss
from training.metrics import precision_recall_f1

def test_loss_lower_when_prediction_matches():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :] = 1.0
    good = torch.where(target > 0, 6.0, -6.0)   # confident-correct logits
    bad = -good
    assert f1_bce_loss(good, target).item() < f1_bce_loss(bad, target).item()

def test_metrics_perfect_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :] = 1.0
    logits = torch.where(target > 0, 6.0, -6.0)
    m = precision_recall_f1(logits, target)
    assert m["precision"] > 0.99 and m["recall"] > 0.99 and m["f1"] > 0.99

def test_loss_is_differentiable():
    logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    target = (torch.rand(1, 1, 8, 8) > 0.5).float()
    f1_bce_loss(logits, target).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_losses_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# training/losses.py
"""Loss = BCEWithLogits + (1 - soft_F1). soft_F1 uses probabilities (no rounding),
so the F1 term contributes a real gradient -- unlike the R version whose k_round
zeroed it (see design spec section 6)."""
import torch
import torch.nn.functional as F


def soft_f1(logits, target, eps=1e-7):
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum()
    precision = tp / (prob.sum() + eps)
    recall = tp / (target.sum() + eps)
    return (2 * precision * recall) / (precision + recall + eps)


def f1_bce_loss(logits, target, eps=1e-7):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + (1.0 - soft_f1(logits, target, eps))
```

```python
# training/metrics.py
"""Hard-rounded precision/recall/F1 for reporting (matches R controlling.R metrics)."""
import torch


def precision_recall_f1(logits, target, threshold=0.5, eps=1e-7):
    pred = (torch.sigmoid(logits) >= threshold).float()
    tp = (pred * target).sum()
    precision = (tp / (pred.sum() + eps)).item()
    recall = (tp / (target.sum() + eps)).item()
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_losses_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/losses.py training/metrics.py tests/test_losses_metrics.py
git commit -m "feat: add soft-F1+BCE loss and hard-rounded metrics"
```

---

### Task 3: Dataset (jpeg images + gif masks) with augmentation

**Files:**
- Create: `training/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `winmol_unet.preprocess.to_float01`, `resize_batch`.
- Produces:
  - `StemDataset(img_dir, mask_dir, img_size=512, train=False, seed=1)` — a `torch.utils.data.Dataset` yielding `(img_chw_float01, mask_1hw_binary)` tensors.
  - Pairs files by sorted order; images `.jpeg`, masks `.gif` (first frame, channel 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_datasets.py
import numpy as np
from PIL import Image
import torch
from training.datasets import StemDataset

def _make_pair(img_dir, mask_dir, name):
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)).save(
        img_dir / f"{name}.jpeg")
    m = (np.random.rand(40, 50) > 0.5).astype(np.uint8) * 255
    Image.fromarray(m, mode="L").save(mask_dir / f"{name}.gif")

def test_dataset_item_shapes(tmp_path):
    img_dir, mask_dir = tmp_path / "img", tmp_path / "mask"
    _make_pair(img_dir, mask_dir, "a")
    ds = StemDataset(str(img_dir), str(mask_dir), img_size=512, train=False)
    img, mask = ds[0]
    assert img.shape == (3, 512, 512) and img.dtype == torch.float32
    assert mask.shape == (1, 512, 512)
    assert float(img.max()) <= 1.0
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

def test_dataset_len(tmp_path):
    img_dir, mask_dir = tmp_path / "img", tmp_path / "mask"
    _make_pair(img_dir, mask_dir, "a")
    _make_pair(img_dir, mask_dir, "b")
    assert len(StemDataset(str(img_dir), str(mask_dir))) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# training/datasets.py
"""jpeg image + gif mask dataset. Geometric aug is paired (img+mask); photometric
aug is image-only (matches R input_pipeline.R)."""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from winmol_unet.preprocess import resize_batch, to_float01


def _list_sorted(directory, ext):
    return sorted(glob.glob(os.path.join(directory, f"*.{ext}")))


class StemDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, train=False, seed=1):
        self.imgs = _list_sorted(img_dir, "jpeg")
        self.masks = _list_sorted(mask_dir, "gif")
        if len(self.imgs) != len(self.masks):
            raise ValueError("image/mask counts differ")
        self.img_size = img_size
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.imgs)

    def _photometric(self, pil_img):
        for enhancer, lo, hi in (
            (ImageEnhance.Brightness, 0.9, 1.1),
            (ImageEnhance.Contrast, 0.95, 1.05),
            (ImageEnhance.Color, 0.95, 1.05),   # saturation
        ):
            pil_img = enhancer(pil_img).enhance(self.rng.uniform(lo, hi))
        return pil_img

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        mask.seek(0)                       # first gif frame
        mask = mask.convert("L")

        if self.train:
            img = self._photometric(img)
            if self.rng.random() < 0.5:    # paired flips
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rng.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        img_arr = to_float01(np.asarray(img))                       # HWC [0,1]
        img_arr = resize_batch(img_arr[None], self.img_size)[0]     # 512x512x3
        mask_arr = (np.asarray(mask) > 127).astype(np.float32)[..., None]
        mask_arr = resize_batch(mask_arr, self.img_size, mode="nearest")[0]
        mask_arr = (mask_arr > 0.5).astype(np.float32)

        img_t = torch.from_numpy(np.transpose(img_arr, (2, 0, 1)).copy())
        mask_t = torch.from_numpy(np.transpose(mask_arr, (2, 0, 1)).copy())
        return img_t, mask_t
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_datasets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/datasets.py tests/test_datasets.py
git commit -m "feat: add jpeg/gif StemDataset with paired+photometric augmentation"
```

---

### Task 4: Training callbacks (checkpoint / early-stop / reduce-LR)

**Files:**
- Create: `training/callbacks.py`
- Test: `tests/test_callbacks.py`

**Interfaces:**
- Produces:
  - `EarlyStopping(patience)` with `.step(val_loss) -> bool` (True == stop).
  - `BestCheckpoint(dirpath, prefix)` with `.step(val_loss, model) -> bool` (True == saved new best), writes `{prefix}_best.pt`.
  - (LR scheduling uses stdlib `torch.optim.lr_scheduler.ReduceLROnPlateau` directly in `train.py`.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_callbacks.py
import torch
import torch.nn as nn
from training.callbacks import EarlyStopping, BestCheckpoint

def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=2)
    assert es.step(1.0) is False   # improve
    assert es.step(1.1) is False   # 1 stale
    assert es.step(1.2) is True    # 2 stale -> stop

def test_best_checkpoint_saves_on_improvement(tmp_path):
    ck = BestCheckpoint(str(tmp_path), prefix="unet")
    model = nn.Conv2d(1, 1, 1)
    assert ck.step(1.0, model) is True     # first is best
    assert ck.step(2.0, model) is False    # worse -> no save
    assert (tmp_path / "unet_best.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_callbacks.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# training/callbacks.py
import os
import torch


class EarlyStopping:
    def __init__(self, patience, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.stale = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.stale = 0
            return False
        self.stale += 1
        return self.stale >= self.patience


class BestCheckpoint:
    def __init__(self, dirpath, prefix):
        os.makedirs(dirpath, exist_ok=True)
        self.path = os.path.join(dirpath, f"{prefix}_best.pt")
        self.best = float("inf")

    def step(self, val_loss, model):
        if val_loss < self.best:
            self.best = val_loss
            torch.save(model.state_dict(), self.path)
            return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_callbacks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/callbacks.py tests/test_callbacks.py
git commit -m "feat: add early-stopping and best-checkpoint callbacks"
```

---

### Task 5: Single-stage train/validate loop

**Files:**
- Create: `training/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `UNet`, `f1_bce_loss`, `precision_recall_f1`, `EarlyStopping`, `BestCheckpoint`, `TrainConfig`.
- Produces: `run_stage(model, train_loader, val_loader, cfg, patience, prefix, device="cpu") -> str` — trains, checkpoints best, early-stops, returns best checkpoint path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from winmol_unet.model import UNet
from training.loop import run_stage
from training.config import TrainConfig

def _cfg(tmp_path):
    return TrainConfig(
        gen_train_dir="", gen_mask_dir="", spec_train_dir="", spec_mask_dir="",
        checkpoint_dir=str(tmp_path), log_dir=str(tmp_path / "log"),
        onnx_out=str(tmp_path / "m.onnx"), epochs=2, img_size=32,
    )

def test_run_stage_overfits_tiny_batch(tmp_path):
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    y = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = UNet()
    path = run_stage(model, loader, loader, _cfg(tmp_path),
                     patience=5, prefix="stage_test", device="cpu")
    assert path.endswith("stage_test_best.pt")
    import os
    assert os.path.exists(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: training.loop`

- [ ] **Step 3: Write minimal implementation**

```python
# training/loop.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from training.callbacks import BestCheckpoint, EarlyStopping
from training.losses import f1_bce_loss
from training.metrics import precision_recall_f1


def _run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        with torch.set_grad_enabled(train):
            logits = model(img)
            loss = f1_bce_loss(logits, mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * img.size(0)
        n += img.size(0)
    return total / max(n, 1)


def run_stage(model, train_loader, val_loader, cfg, patience, prefix, device="cpu"):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    stopper = EarlyStopping(patience=patience)
    checkpoint = BestCheckpoint(cfg.checkpoint_dir, prefix)
    writer = SummaryWriter(log_dir=f"{cfg.log_dir}/{prefix}")

    for epoch in range(cfg.epochs):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        val_loss = _run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_loss)
        checkpoint.step(val_loss, model)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        if stopper.step(val_loss):
            break
    writer.close()
    return checkpoint.path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/loop.py tests/test_loop.py
git commit -m "feat: add single-stage train/val loop with scheduler, checkpoint, early-stop"
```

---

### Task 6: Two-stage training entry point + ONNX export

**Files:**
- Create: `training/train.py`
- Test: `tests/test_train_entry.py`

**Interfaces:**
- Consumes: `StemDataset`, `run_stage`, `winmol_unet.export.export_to_onnx`, `TrainConfig`.
- Produces:
  - `split_loaders(img_dir, mask_dir, cfg, train) -> (train_loader, val_loader)`.
  - `train_two_stage(cfg, device="cpu") -> str` — runs stage 1 (GenDS), stage 2 (SpecDS fine-tune from stage-1 best), exports `cfg.onnx_out`, returns the ONNX path.
  - `main()` — argparse CLI (`--gen-train`, `--gen-mask`, `--spec-train`, `--spec-mask`, `--out`, ...).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_entry.py
import numpy as np
from PIL import Image
import onnx
from training.train import train_two_stage
from training.config import TrainConfig
from winmol_unet import contract

def _pair(img_dir, mask_dir, name):
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
        img_dir / f"{name}.jpeg")
    Image.fromarray(((np.random.rand(32, 32) > 0.5) * 255).astype(np.uint8), "L").save(
        mask_dir / f"{name}.gif")

def test_train_two_stage_produces_valid_onnx(tmp_path):
    for stage in ("gen", "spec"):
        for i in range(4):
            _pair(tmp_path / stage / "train", tmp_path / stage / "mask", f"{i}")
    cfg = TrainConfig(
        gen_train_dir=str(tmp_path / "gen/train"), gen_mask_dir=str(tmp_path / "gen/mask"),
        spec_train_dir=str(tmp_path / "spec/train"), spec_mask_dir=str(tmp_path / "spec/mask"),
        checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "log"),
        onnx_out=str(tmp_path / "model.onnx"),
        epochs=1, batch_size=2, img_size=512, val_fraction=0.5,
    )
    out = train_two_stage(cfg, device="cpu")
    contract.validate_onnx_model(onnx.load(out))   # exported model honors contract
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_entry.py -v`
Expected: FAIL with `ModuleNotFoundError: training.train`

- [ ] **Step 3: Write minimal implementation**

```python
# training/train.py
import argparse

import torch
from torch.utils.data import DataLoader, random_split

from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet
from training.config import TrainConfig
from training.datasets import StemDataset
from training.loop import run_stage


def split_loaders(img_dir, mask_dir, cfg, train):
    full = StemDataset(img_dir, mask_dir, img_size=cfg.img_size, train=train, seed=cfg.seed)
    n_val = max(1, int(len(full) * cfg.val_fraction))
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False),
    )


def train_two_stage(cfg, device="cpu"):
    model = UNet(dropout=cfg.dropout)

    tr1, va1 = split_loaders(cfg.gen_train_dir, cfg.gen_mask_dir, cfg, train=True)
    best1 = run_stage(model, tr1, va1, cfg, cfg.patience_stage1, "stage1_gen", device)
    model.load_state_dict(torch.load(best1, map_location=device))

    tr2, va2 = split_loaders(cfg.spec_train_dir, cfg.spec_mask_dir, cfg, train=True)
    best2 = run_stage(model, tr2, va2, cfg, cfg.patience_stage2, "stage2_spec", device)
    model.load_state_dict(torch.load(best2, map_location=device))

    return export_to_onnx(model, cfg.onnx_out)


def main():
    p = argparse.ArgumentParser(description="Two-stage WINMOL segmentor training")
    p.add_argument("--gen-train", required=True)
    p.add_argument("--gen-mask", required=True)
    p.add_argument("--spec-train", required=True)
    p.add_argument("--spec-mask", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg = TrainConfig(
        gen_train_dir=a.gen_train, gen_mask_dir=a.gen_mask,
        spec_train_dir=a.spec_train, spec_mask_dir=a.spec_mask,
        checkpoint_dir=a.checkpoint_dir, log_dir=a.log_dir, onnx_out=a.out,
        epochs=a.epochs,
    )
    out = train_two_stage(cfg, device=a.device)
    print(f"Exported ONNX model to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_entry.py -v`
Expected: PASS (this is the training smoke test — spec §8.5 — plus contract check on the trained export)

- [ ] **Step 5: Commit**

```bash
git add training/train.py tests/test_train_entry.py
git commit -m "feat: add two-stage training entry point with ONNX export"
```

---

### Task 7: Evaluation + visual comparison

**Files:**
- Create: `training/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `UNet`, `precision_recall_f1`, `StemDataset`.
- Produces:
  - `evaluate(model, loader, device="cpu") -> dict` — mean precision/recall/F1 over the loader.
  - `save_comparison(model, dataset, out_png, n=4, device="cpu") -> str` — writes a stacked mask/image/prediction PNG (matches R evaluation.R visual).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from winmol_unet.model import UNet
from training.datasets import StemDataset
from training.evaluate import evaluate, save_comparison

def _pair(img_dir, mask_dir, name):
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
        img_dir / f"{name}.jpeg")
    Image.fromarray(((np.random.rand(32, 32) > 0.5) * 255).astype(np.uint8), "L").save(
        mask_dir / f"{name}.gif")

def test_evaluate_returns_metrics(tmp_path):
    _pair(tmp_path / "img", tmp_path / "mask", "a")
    ds = StemDataset(str(tmp_path / "img"), str(tmp_path / "mask"), img_size=512)
    m = evaluate(UNet().eval(), DataLoader(ds, batch_size=1), device="cpu")
    assert {"precision", "recall", "f1"} <= set(m)

def test_save_comparison_writes_png(tmp_path):
    _pair(tmp_path / "img", tmp_path / "mask", "a")
    ds = StemDataset(str(tmp_path / "img"), str(tmp_path / "mask"), img_size=512)
    out = save_comparison(UNet().eval(), ds, str(tmp_path / "cmp.png"), n=1, device="cpu")
    import os
    assert os.path.exists(out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# training/evaluate.py
import numpy as np
import torch
from PIL import Image

from training.metrics import precision_recall_f1


def evaluate(model, loader, device="cpu"):
    model.to(device).eval()
    agg = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = 0
    with torch.no_grad():
        for img, mask in loader:
            m = precision_recall_f1(model(img.to(device)), mask.to(device))
            for k in agg:
                agg[k] += m[k]
            n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def save_comparison(model, dataset, out_png, n=4, device="cpu"):
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            img, mask = dataset[i]
            prob = torch.sigmoid(model(img[None].to(device)))[0, 0].cpu().numpy()
            img_np = (np.transpose(img.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
            mask_np = (mask[0].numpy() * 255).astype(np.uint8)
            pred_np = (prob * 255).astype(np.uint8)
            mask_rgb = np.stack([mask_np] * 3, axis=-1)
            pred_rgb = np.stack([pred_np] * 3, axis=-1)
            rows.append(np.concatenate([mask_rgb, img_np, pred_rgb], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_png)
    return out_png
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation metrics and visual comparison export"
```

---

## Self-Review

**Spec coverage (spec §4.2 training-only components):**
- `config.py` → Task 1 ✓
- `losses.py` / `metrics.py` (soft-F1 loss + hard metrics, spec §6) → Task 2 ✓
- `datasets.py` (jpeg/gif, paired + photometric aug) → Task 3 ✓
- `callbacks.py` (checkpoint / early-stop / reduce-LR) → Task 4 (+ scheduler in Task 5) ✓
- `train.py` two-stage GenDS→SpecDS + ONNX export → Tasks 5–6 ✓
- `evaluate.py` (eval + visual) → Task 7 ✓
- Training smoke test (spec §8.5) → Task 6 test ✓

**Placeholder scan:** No TBDs; every step shows complete code. TensorBoard logging uses stdlib `torch.utils.tensorboard`.

**Type consistency:** `f1_bce_loss(logits, target)`, `precision_recall_f1(logits, target)->dict`, `StemDataset(img_dir, mask_dir, img_size, train, seed)`, `run_stage(model, train_loader, val_loader, cfg, patience, prefix, device)->path`, `train_two_stage(cfg, device)->onnx_path`, `export_to_onnx(model, path)->path` are used consistently across Tasks 2–7 and align with Plan 1.

**Note on augmentation fidelity:** the R pipeline also applied random hue; Pillow has no single hue enhancer, so Task 3 covers brightness/contrast/saturation. If hue jitter proves important, add an HSV-shift step in `_photometric` (documented follow-up, not a blocker).
