# Multi-format Export (pt + hdf5 + keras + onnx) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export a trained UNet in four formats — PyTorch `.pt`, Keras legacy `.hdf5`, Keras native `.keras`, and `.onnx`. `.hdf5`/`.onnx` exist; add `.pt` and native `.keras`.

**Architecture:** `export_keras.py` gains a shared `_build_and_transfer` helper feeding both `export_to_keras_hdf5` (legacy) and a new `export_to_keras` (native). `export.py` gains `export_to_pt`. The trainer wires all four; `.pt`/`.keras` are optional in `TrainConfig` (default None) so existing callers are unaffected.

**Tech Stack:** Python 3.9, PyTorch, TensorFlow/Keras 3 (TF 2.20), onnx, pytest.

## Global Constraints

- Python floor 3.9; native `.keras` needs Keras 3 (TF 2.20 present); legacy `.hdf5` still supported.
- Export runs on CPU (weights read via numpy); the trainer already moves the model to CPU first.
- Keras mirror stays standard-layers-only so `.hdf5` and `.keras` load without `custom_objects`.
- `.keras` and `.hdf5` must remain numerically equal to the PyTorch model (parity atol=1e-4).
- New `TrainConfig` fields `keras_out`, `pt_out` must be **Optional[str] = None** (do NOT make
  them required — existing `TrainConfig(...)` calls and tests pass only hdf5_out/onnx_out).
- `run_training` exports `.hdf5` + `.onnx` always (unchanged); exports `.pt`/`.keras` only when
  their config paths are set.
- Existing `winmol_unet` weight-transfer conventions unchanged (Conv `[out,in,kh,kw]`→
  `[kh,kw,in,out]`; ConvTranspose `[in,out,kh,kw]`→`[kh,kw,out,in]`; BN eps 1e-5; fail loud
  on layer-count/shape mismatch).

---

### Task 1: `export_to_keras` (native) + `export_to_pt`

**Files:**
- Modify: `winmol_unet/export_keras.py` (extract shared helper; add `export_to_keras`; drop unused `UNet` import)
- Modify: `winmol_unet/export.py` (add `export_to_pt`)
- Test: `tests/test_export_multiformat.py`

**Interfaces:**
- Consumes: `winmol_unet.keras_model.build_keras_unet`, existing `_transfer`,
  `winmol_unet.model.UNet`.
- Produces:
  - `winmol_unet.export_keras._build_and_transfer(torch_model, dropout=0.1) -> keras.Model`
    (eval + build mirror + transfer weights).
  - `winmol_unet.export_keras.export_to_keras(torch_model, path, dropout=0.1) -> path`
    (native `.keras`). `export_to_keras_hdf5` unchanged in behavior (now uses the helper).
  - `winmol_unet.export.export_to_pt(model, path) -> path`
    (`torch.save(model.state_dict(), path)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_multiformat.py
import numpy as np
import torch

from winmol_unet.model import UNet
from winmol_unet.export import export_to_pt
from winmol_unet.export_keras import export_to_keras, export_to_keras_hdf5


def _torch_probs(model, x_nhwc):
    with torch.no_grad():
        nchw = torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2))
        return torch.sigmoid(model(nchw)).numpy().transpose(0, 2, 3, 1)


def test_export_pt_roundtrip(tmp_path):
    m = UNet().eval()
    path = export_to_pt(m, str(tmp_path / "m.pt"))
    m2 = UNet()
    m2.load_state_dict(torch.load(path))
    m2.eval()
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-6)


def test_export_keras_native_parity(tmp_path):
    model = UNet().eval()
    path = export_to_keras(model, str(tmp_path / "m.keras"))
    assert path.endswith(".keras")
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)
    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)   # native-format load
    out = loaded.predict(x, verbose=0)
    assert out.shape == (2, 512, 512, 1)
    assert np.allclose(_torch_probs(model, x), out, rtol=0.0, atol=1e-4)


def test_export_hdf5_still_works(tmp_path):
    model = UNet().eval()
    path = export_to_keras_hdf5(model, str(tmp_path / "m.hdf5"))
    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)
    x = np.random.rand(1, 512, 512, 3).astype(np.float32)
    assert np.allclose(_torch_probs(model, x), loaded.predict(x, verbose=0), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_export_multiformat.py -v`
Expected: FAIL with `ImportError: cannot import name 'export_to_keras'` (and `export_to_pt`).

- [ ] **Step 3: Write minimal implementation**

In `winmol_unet/export_keras.py`: remove the unused `from .model import UNet` import, and
replace `export_to_keras_hdf5` with the shared helper + both exporters:

```python
def _build_and_transfer(torch_model, dropout=0.1):
    torch_model = torch_model.eval()
    keras_model = build_keras_unet(dropout=dropout)
    _transfer(torch_model, keras_model)
    return keras_model


def export_to_keras_hdf5(torch_model, path, dropout=0.1):
    """Legacy HDF5 (analyzer's current load path)."""
    _build_and_transfer(torch_model, dropout).save(path, save_format="h5")
    return path


def export_to_keras(torch_model, path, dropout=0.1):
    """Native Keras 3 format (path should end in .keras)."""
    _build_and_transfer(torch_model, dropout).save(path)
    return path
```

In `winmol_unet/export.py`, add (uses the already-imported `torch`):
```python
def export_to_pt(model, path):
    """Save the trained PyTorch weights (state_dict); reload into UNet().load_state_dict()."""
    torch.save(model.state_dict(), path)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_export_multiformat.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/export_keras.py winmol_unet/export.py tests/test_export_multiformat.py
git commit -m "feat: add native .keras export + .pt export (shared build+transfer helper)"
```

---

### Task 2: Config fields + trainer wiring for all four formats

**Files:**
- Modify: `training/config.py` (add `keras_out`, `pt_out` Optional fields)
- Modify: `training/run_train.py` (export pt/hdf5/keras/onnx; CLI derives all paths)
- Test: `tests/test_multiformat_e2e.py`

**Interfaces:**
- Consumes: `winmol_unet.export.export_to_pt`, `winmol_unet.export_keras.export_to_keras`,
  existing `export_to_keras_hdf5`, `export_to_onnx`.
- Produces: `TrainConfig` gains `keras_out: Optional[str] = None`, `pt_out: Optional[str] =
  None`. `run_training` writes `.pt`/`.keras` when those paths are set (always `.hdf5`/`.onnx`).
  `config_from_args` derives `model.pt` and `model.keras` under `--out-dir`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiformat_e2e.py
import os
import numpy as np
from PIL import Image

from training.config import TrainConfig
from training.run_train import run_training, config_from_args


def _make_ds(tmp_path, n=6):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")


def test_cli_derives_all_four_paths():
    cfg = config_from_args(["--data-dir", "d", "--out-dir", "o"])
    assert cfg.pt_out == os.path.join("o", "model.pt")
    assert cfg.keras_out == os.path.join("o", "model.keras")
    assert cfg.hdf5_out == os.path.join("o", "model.hdf5")
    assert cfg.onnx_out == os.path.join("o", "model.onnx")


def test_run_training_writes_all_four(tmp_path):
    _make_ds(tmp_path)
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"),
        pt_out=str(out / "m.pt"), hdf5_out=str(out / "m.hdf5"),
        keras_out=str(out / "m.keras"), onnx_out=str(out / "m.onnx"),
        epochs=1, batch_size=2, patience=999, device="cpu",
    )
    run_training(cfg)
    for f in ("m.pt", "m.hdf5", "m.keras", "m.onnx"):
        assert os.path.exists(out / f), f"missing {f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_multiformat_e2e.py -v`
Expected: FAIL — `TypeError` (unknown `keras_out`/`pt_out` kwargs) or missing files.

- [ ] **Step 3: Write minimal implementation**

In `training/config.py`, add after `wandb_run_name` (keep `Optional` already imported):
```python
    keras_out: Optional[str] = None
    pt_out: Optional[str] = None
```

In `training/run_train.py`: add imports
`from winmol_unet.export import export_to_onnx, export_to_pt` and
`from winmol_unet.export_keras import export_to_keras, export_to_keras_hdf5`, then replace the
export block in `run_training`:
```python
    model.cpu()                                 # exporters read weights via CPU numpy
    os.makedirs(os.path.dirname(cfg.hdf5_out) or ".", exist_ok=True)
    if cfg.pt_out:
        export_to_pt(model, cfg.pt_out)
    export_to_keras_hdf5(model, cfg.hdf5_out, dropout=cfg.dropout)
    if cfg.keras_out:
        export_to_keras(model, cfg.keras_out, dropout=cfg.dropout)
    export_to_onnx(model, cfg.onnx_out)

    return val_metrics
```
And in `config_from_args`, add the two derived paths to the `TrainConfig(...)` construction:
```python
        pt_out=os.path.join(a.out_dir, "model.pt"),
        keras_out=os.path.join(a.out_dir, "model.keras"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_multiformat_e2e.py -v`
Expected: PASS (2 tests). Also run `.venv/bin/pytest tests/test_train_e2e.py -q` to confirm the
existing e2e (hdf5+onnx only, no pt/keras set) still passes.

- [ ] **Step 5: Commit**

```bash
git add training/config.py training/run_train.py tests/test_multiformat_e2e.py
git commit -m "feat: trainer exports pt + hdf5 + keras + onnx (keras/pt optional in config)"
```

---

## Self-Review

**Spec coverage:** native `.keras` §3→T1; `.pt` §3→T1; shared helper/no-hdf5-regression §3,§5→
T1; config `keras_out`/`pt_out` optional §3→T2; run_train four-format wiring + CLI paths §3→T2;
e2e all-four §5→T2. ✓
**Placeholder scan:** none. ✓
**Type consistency:** `export_to_keras(torch_model, path, dropout)`, `export_to_pt(model,
path)`, `_build_and_transfer(torch_model, dropout)->keras.Model` consistent T1↔T2; new optional
`TrainConfig` fields consumed by `run_training`/`config_from_args` (T2). Backward-compat:
`.pt`/`.keras` gated on truthy config so existing `test_train_e2e` (hdf5+onnx only) still passes. ✓
