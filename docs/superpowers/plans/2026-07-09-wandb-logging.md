# Optional wandb Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in Weights & Biases scalar logging to the single-stage trainer, alongside the existing TensorBoard logging (metrics only, no artifact uploads).

**Architecture:** A `RunLogger` wraps the TensorBoard `SummaryWriter` and, when `use_wandb=True`, also logs to wandb. `train_one_run` logs via `RunLogger` and stays clean; wandb is off unless `--wandb` is passed and is an optional dependency.

**Tech Stack:** Python 3.9, PyTorch, tensorboard, wandb (optional), python-dotenv (optional), pytest. Depends on the existing `training/` package.

## Global Constraints

- Python floor 3.9; use `Optional[...]` (no `X | None` syntax).
- **Metrics only** — no artifact/checkpoint/model uploads to wandb.
- **Off unless `--wandb`**; default training behavior unchanged; TensorBoard always on.
- `wandb` + `python-dotenv` are an **optional** `[wandb]` extra; training runs without them.
- If `use_wandb=True` and `wandb`/`python-dotenv` cannot be imported → raise `RuntimeError`
  naming the fix (`pip install '.[wandb]'`). Never silent-skip when explicitly enabled.
- Tests must be **hermetic**: no real wandb account, no network — fake `wandb`/`dotenv` via
  `monkeypatch.setitem(sys.modules, ...)`. Do NOT install wandb to run the tests.
- Scalars logged per epoch: `train/loss`, `val/loss`, `val/precision`, `val/recall`, `val/f1`.
- Existing `winmol_unet` + `training` APIs unchanged except as specified here.

---

### Task 1: `[wandb]` extra, TrainConfig fields, and RunLogger

**Files:**
- Modify: `pyproject.toml` (add `wandb` extra)
- Modify: `training/config.py` (add 3 fields)
- Create: `training/run_logger.py`
- Test: `tests/test_run_logger.py`

**Interfaces:**
- Consumes: `torch.utils.tensorboard.SummaryWriter`.
- Produces:
  - `TrainConfig` gains `wandb: bool = False`, `wandb_project: Optional[str] = None`,
    `wandb_run_name: Optional[str] = None`.
  - `RunLogger(log_dir, use_wandb=False, project=None, run_name=None)` with
    `log_scalars(scalars: dict, step: int)` and `close()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_logger.py
import sys
import types
import glob
import pytest


def _fake_wandb():
    calls = {}
    m = types.ModuleType("wandb")
    m.init = lambda **kw: calls.__setitem__("init", kw)
    m.log = lambda d, step=None: calls.__setitem__("log", (dict(d), step))
    m.finish = lambda: calls.__setitem__("finish", True)
    return m, calls


def _fake_dotenv():
    m = types.ModuleType("dotenv")
    m.load_dotenv = lambda *a, **k: True
    return m


def test_tensorboard_only_when_wandb_off(tmp_path, monkeypatch):
    # Ensure wandb is never needed when off.
    monkeypatch.setitem(sys.modules, "wandb", None)   # import wandb -> ImportError if touched
    from training.run_logger import RunLogger
    lg = RunLogger(str(tmp_path / "log"), use_wandb=False)
    lg.log_scalars({"val/f1": 0.5}, 1)
    lg.close()
    assert glob.glob(str(tmp_path / "log" / "events*"))   # TB wrote something


def test_wandb_init_log_finish(tmp_path, monkeypatch):
    fake, calls = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "dotenv", _fake_dotenv())
    from training.run_logger import RunLogger
    lg = RunLogger(str(tmp_path / "log"), use_wandb=True, project="P", run_name="R")
    assert calls["init"] == {"project": "P", "name": "R"}
    lg.log_scalars({"val/f1": 0.5}, 3)
    assert calls["log"] == ({"val/f1": 0.5}, 3)
    lg.close()
    assert calls["finish"] is True


def test_missing_wandb_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)   # forces ImportError on `import wandb`
    from training.run_logger import RunLogger
    with pytest.raises(RuntimeError, match=r"\[wandb\]"):
        RunLogger(str(tmp_path / "log"), use_wandb=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_run_logger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.run_logger'`

- [ ] **Step 3: Write minimal implementation**

In `pyproject.toml` `[project.optional-dependencies]` add:
```toml
wandb = ["wandb>=0.16", "python-dotenv>=1.0"]
```
(Do NOT install it — the tests are hermetic and fake these modules.)

In `training/config.py`: add `from typing import Optional` at the top, and inside the
dataclass (after `device`):
```python
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
```

```python
# training/run_logger.py
"""Scalar logging to TensorBoard, plus optional Weights & Biases (opt-in, metrics only).

wandb is imported lazily and only when use_wandb=True, so it stays an optional dependency.
"""
from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, log_dir, use_wandb=False, project=None, run_name=None):
        self.writer = SummaryWriter(log_dir)
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                from dotenv import load_dotenv
            except ImportError as e:
                raise RuntimeError(
                    "wandb logging requires the optional extra: "
                    "pip install '.[wandb]'") from e
            load_dotenv()                      # picks up WANDB_API_KEY from .env
            wandb.init(project=project, name=run_name)
            self._wandb = wandb

    def log_scalars(self, scalars, step):
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        if self._wandb is not None:
            self._wandb.log(dict(scalars), step=step)

    def close(self):
        self.writer.close()
        if self._wandb is not None:
            self._wandb.finish()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_run_logger.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml training/config.py training/run_logger.py tests/test_run_logger.py
git commit -m "feat: add RunLogger (TensorBoard + optional wandb) and config fields"
```

---

### Task 2: Wire RunLogger into training + CLI flags

**Files:**
- Modify: `training/train.py` (use RunLogger; log train/loss + val scalars)
- Modify: `training/run_train.py` (CLI flags → TrainConfig; testable arg builder)
- Test: `tests/test_wandb_wiring.py`

**Interfaces:**
- Consumes: `training.run_logger.RunLogger`, `training.config.TrainConfig`,
  `training.device.resolve_device`, `training.losses.bce_soft_f1_loss`,
  `training.evaluate.evaluate`.
- Produces: `run_train.config_from_args(argv=None) -> TrainConfig` (builds the argparse
  parser and returns a TrainConfig; `main()` calls it). `train_one_run` unchanged signature,
  now logs `train/loss` plus `val/loss,precision,recall,f1` per epoch via `RunLogger`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wandb_wiring.py
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.dataset import StemDataset
from training.train import train_one_run
from training.run_train import config_from_args
from winmol_unet.model import UNet


def test_cli_parses_wandb_flags():
    cfg = config_from_args(["--data-dir", "d", "--wandb",
                            "--wandb-project", "P", "--wandb-run-name", "R"])
    assert cfg.wandb is True
    assert cfg.wandb_project == "P"
    assert cfg.wandb_run_name == "R"


def test_cli_wandb_off_by_default():
    cfg = config_from_args(["--data-dir", "d"])
    assert cfg.wandb is False


def test_train_logs_train_loss_scalar(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for n in (1, 2):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{n}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{n}.gif")
    ds = StemDataset(str(img_dir), str(mask_dir))
    loader = DataLoader(ds, batch_size=2)
    cfg = TrainConfig(data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
                      log_dir=str(tmp_path / "log"), hdf5_out="x", onnx_out="x",
                      epochs=1, patience=999)
    train_one_run(UNet(dropout=0.0), loader, loader, cfg)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(sorted(glob.glob(str(tmp_path / "log" / "events*")))[-1]); ea.Reload()
    assert "train/loss" in ea.Tags().get("scalars", [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wandb_wiring.py -v`
Expected: FAIL with `ImportError: cannot import name 'config_from_args'`

- [ ] **Step 3: Write minimal implementation**

In `training/train.py`: replace the `SummaryWriter` usage. Change the imports
`from torch.utils.tensorboard import SummaryWriter` → `from .run_logger import RunLogger`,
and rewrite the body of `train_one_run` so the epoch loop accumulates mean train loss and
logs all scalars through the logger:

```python
def train_one_run(model, train_loader, val_loader, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(cfg.checkpoint_dir, "best.pt")
    logger = RunLogger(cfg.log_dir, cfg.wandb, cfg.wandb_project, cfg.wandb_run_name)
    device = resolve_device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    since_improve = 0
    for epoch in range(cfg.epochs):
        model.train()
        running, nb = 0.0, 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            loss = bce_soft_f1_loss(model(img), mask)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        train_loss = running / nb if nb else 0.0

        val = evaluate(model, val_loader)
        logger.log_scalars({
            "train/loss": train_loss,
            "val/loss": val["loss"],
            "val/precision": val["precision"],
            "val/recall": val["recall"],
            "val/f1": val["f1"],
        }, epoch)

        if val["loss"] < best_val:
            best_val = val["loss"]
            since_improve = 0
            torch.save(model.state_dict(), ckpt)
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                break

    logger.close()
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model
```

In `training/run_train.py`: extract the parser into `config_from_args` and add the flags:

```python
def config_from_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    a = p.parse_args(argv)
    return TrainConfig(
        data_dir=a.data_dir,
        checkpoint_dir=os.path.join(a.out_dir, "checkpoints"),
        log_dir=os.path.join(a.out_dir, "logs"),
        hdf5_out=os.path.join(a.out_dir, "model.hdf5"),
        onnx_out=os.path.join(a.out_dir, "model.onnx"),
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
        wandb=a.wandb, wandb_project=a.wandb_project, wandb_run_name=a.wandb_run_name,
    )


def main():
    print(run_training(config_from_args()))
```

Remove the old inline parser from `main()`. Keep the existing imports; the top of
`run_train.py` no longer needs its own `argparse` handling beyond `config_from_args`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_wandb_wiring.py -v`
Expected: PASS (3 tests). Also run the existing loop test to confirm no regression:
`.venv/bin/pytest tests/test_train_loop.py -q`

- [ ] **Step 5: Commit**

```bash
git add training/train.py training/run_train.py tests/test_wandb_wiring.py
git commit -m "feat: wire RunLogger + --wandb CLI flags; log per-epoch train/loss"
```

---

## Self-Review

**Spec coverage:** `[wandb]` extra §3→T1; TrainConfig fields §3→T1; RunLogger + TB/wandb/
fail-loud §3,§5→T1; train.py wiring + train/loss + val scalars §3,§4→T2; CLI flags §3→T2;
hermetic tests §6→T1(fake wandb/dotenv, missing-raises) + T2(CLI + train/loss). ✓
**Placeholder scan:** none — full code in every step. ✓
**Type consistency:** `RunLogger(log_dir, use_wandb, project, run_name)` / `log_scalars(dict,
step)` / `close()` identical in T1 and T2; `config_from_args(argv)->TrainConfig` used in T2
test + `main`; new TrainConfig fields (T1) consumed by `train_one_run` and `config_from_args`
(T2). ✓
