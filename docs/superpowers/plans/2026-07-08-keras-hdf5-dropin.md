# Keras HDF5 Drop-in Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a PyTorch → Keras `.hdf5` export path so a PyTorch-trained U-Net can be loaded by the **completely unmodified** analyzer (`keras.models.load_model(path, compile=False)`), numerically matching the PyTorch model.

**Architecture:** A mirror Keras U-Net with architecture identical to the PyTorch `winmol_unet.model.UNet` is built in TensorFlow/Keras. Trained PyTorch weights are copied into it layer-by-layer (conv kernels transposed for the NHWC/`[kh,kw,in,out]` convention; BatchNorm epsilon pinned to PyTorch's 1e-5), the sigmoid head is included, and the model is saved as HDF5. The exported file loads through the analyzer's existing `keras.models.load_model` path with no code change.

**Tech Stack:** Python 3.9, PyTorch, TensorFlow/Keras, numpy, pytest. Depends on `winmol_unet` (model + contract from the inference-bridge plan).

**Prerequisite:** `winmol_unet.model.UNet` and `winmol_unet.contract` exist (inference-bridge Tasks 2 & 4, done).

## Global Constraints

- Exported file must load via **`keras.models.load_model(path, compile=False)`** with the analyzer UNMODIFIED — that call is the drop-in contract.
- The Keras mirror must use **standard Keras layers only** (no custom layers) so it loads without `custom_objects`.
- **BatchNorm epsilon = 1e-5** in the Keras model, to match PyTorch `nn.BatchNorm2d` default (Keras default 1e-3 would break parity).
- Architecture must mirror `winmol_unet.model.UNet` exactly: encoder 64→128→256→512, bottleneck 1024, decoder 512→256→128→64, `_double_conv` = conv-bn-relu → dropout → conv-bn-relu, `bias=False` on conv/deconv (BN follows), 1×1 sigmoid head. Concatenation order is **[skip, upsampled]** (same as PyTorch `torch.cat([cN, upN], dim=1)`).
- Keras I/O is **NHWC**: input `(512,512,3)`, output `(512,512,1)` sigmoid.
- Parity: `torch.sigmoid(unet(x))` vs the loaded Keras model output must agree within **atol=1e-4**. Do not loosen.
- Python floor **3.9**; no 3.10+-only syntax. Test output must be pristine (filter only third-party deprecation noise, scoped).
- Weight transfer must FAIL LOUDLY (raise) on any layer-count or per-layer shape mismatch — never assign silently misaligned weights.

---

### Task 1: Keras mirror U-Net

**Files:**
- Create: `winmol_unet/keras_model.py`
- Test: `tests/test_keras_model.py`
- Modify: `pyproject.toml` (add a `keras` extra with `tensorflow`)

**Interfaces:**
- Consumes: `contract.IMG_SIZE`, `contract.IN_CHANNELS`, `contract.OUT_CHANNELS`.
- Produces: `build_keras_unet(dropout=0.1) -> tf.keras.Model` — NHWC `(N,512,512,3)` → `(N,512,512,1)` sigmoid; standard layers; BN epsilon 1e-5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_keras_model.py
import numpy as np
from winmol_unet.keras_model import build_keras_unet

def test_keras_unet_output_shape_and_range():
    model = build_keras_unet()
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 512, 512, 1)
    assert (y >= 0).all() and (y <= 1).all()   # sigmoid head

def test_keras_unet_uses_torch_bn_epsilon():
    from tensorflow.keras.layers import BatchNormalization
    model = build_keras_unet()
    bns = [l for l in model.layers if isinstance(l, BatchNormalization)]
    assert bns and all(abs(l.epsilon - 1e-5) < 1e-12 for l in bns)

def test_keras_unet_roundtrips_via_load_model(tmp_path):
    from tensorflow import keras
    path = str(tmp_path / "m.hdf5")
    build_keras_unet().save(path, save_format="h5")
    loaded = keras.models.load_model(path, compile=False)   # analyzer's load path
    assert loaded.predict(np.zeros((1, 512, 512, 3), np.float32), verbose=0).shape == (1, 512, 512, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_keras_model.py -v`
Expected: FAIL with `ModuleNotFoundError: winmol_unet.keras_model` (after installing tensorflow — see Step 3).

- [ ] **Step 3: Write minimal implementation**

Add a `keras` extra to `pyproject.toml` `[project.optional-dependencies]`:
```toml
keras = ["tensorflow>=2.13"]
```
Install it: `uv pip install --python .venv -e ".[torch,keras,dev]"`. (If plain `tensorflow` cannot install on this platform/Python, report BLOCKED with the exact pip error — do not substitute a different package silently.)

```python
# winmol_unet/keras_model.py
"""Keras mirror of winmol_unet.model.UNet, for PyTorch->HDF5 export.

Standard layers only (loads without custom_objects). BatchNorm epsilon is
pinned to 1e-5 to match PyTorch nn.BatchNorm2d (Keras default is 1e-3).
"""
from tensorflow.keras import Input, Model, layers

from .contract import IMG_SIZE, IN_CHANNELS, OUT_CHANNELS

BN_EPS = 1e-5


def _double_conv(x, filters, dropout, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def _up(x, filters, name):
    return layers.Conv2DTranspose(
        filters, 2, strides=2, padding="same", use_bias=False, name=name)(x)


def build_keras_unet(dropout=0.1):
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, IN_CHANNELS), name="input")
    c1 = _double_conv(inp, 64, dropout, "enc1")
    c2 = _double_conv(layers.MaxPool2D(2, name="pool1")(c1), 128, dropout, "enc2")
    c3 = _double_conv(layers.MaxPool2D(2, name="pool2")(c2), 256, dropout, "enc3")
    c4 = _double_conv(layers.MaxPool2D(2, name="pool3")(c3), 512, dropout, "enc4")
    b = _double_conv(layers.MaxPool2D(2, name="pool4")(c4), 1024, dropout, "bottleneck")

    d4 = _double_conv(
        layers.Concatenate(name="cat4")([c4, _up(b, 512, "up4")]), 512, dropout, "dec4")
    d3 = _double_conv(
        layers.Concatenate(name="cat3")([c3, _up(d4, 256, "up3")]), 256, dropout, "dec3")
    d2 = _double_conv(
        layers.Concatenate(name="cat2")([c2, _up(d3, 128, "up2")]), 128, dropout, "dec2")
    d1 = _double_conv(
        layers.Concatenate(name="cat1")([c1, _up(d2, 64, "up1")]), 64, dropout, "dec1")

    out = layers.Conv2D(OUT_CHANNELS, 1, activation="sigmoid", name="head")(d1)
    return Model(inp, out, name="winmol_unet_keras")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_keras_model.py -v`
Expected: PASS (3 tests). If `save_format="h5"` emits a Keras-3 deprecation warning, add a scoped `filterwarnings` entry in `pyproject.toml` (matching only that message) to keep output pristine.

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/keras_model.py tests/test_keras_model.py pyproject.toml
git commit -m "feat: add Keras mirror U-Net for HDF5 export (BN eps matched to torch)"
```

---

### Task 2: PyTorch → Keras HDF5 weight transfer + export + drop-in parity

**Files:**
- Create: `winmol_unet/export_keras.py`
- Test: `tests/test_export_keras.py`

**Interfaces:**
- Consumes: `winmol_unet.model.UNet`, `build_keras_unet`.
- Produces: `export_to_keras_hdf5(torch_model, path, dropout=0.1) -> str` — builds the mirror, transfers weights, saves HDF5, returns path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_keras.py
import numpy as np
import torch

from winmol_unet.model import UNet
from winmol_unet.export_keras import export_to_keras_hdf5

def _torch_probs(model, x_nhwc):
    with torch.no_grad():
        nchw = torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2))
        return torch.sigmoid(model(nchw)).numpy().transpose(0, 2, 3, 1)

def test_export_parity_torch_vs_keras(tmp_path):
    model = UNet().eval()
    path = export_to_keras_hdf5(model, str(tmp_path / "m.hdf5"))
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)

    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)   # unmodified-analyzer load path
    keras_out = loaded.predict(x, verbose=0)

    assert keras_out.shape == (2, 512, 512, 1)
    assert np.allclose(_torch_probs(model, x), keras_out, rtol=0.0, atol=1e-4)

def test_transfer_fails_loudly_on_mismatch(tmp_path):
    # A model with a different architecture must not silently mis-map.
    import torch.nn as nn
    class Wrong(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 1, 1)
        def forward(self, x):
            return self.c(x)
    import pytest
    with pytest.raises((ValueError, AssertionError)):
        export_to_keras_hdf5(Wrong().eval(), str(tmp_path / "bad.hdf5"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_export_keras.py -v`
Expected: FAIL with `ModuleNotFoundError: winmol_unet.export_keras`

- [ ] **Step 3: Write minimal implementation**

```python
# winmol_unet/export_keras.py
"""Copy trained PyTorch UNet weights into the Keras mirror and save HDF5.

The exported file loads via keras.models.load_model(path, compile=False) — the
analyzer's existing load path — with no analyzer code change.

Conventions:
- Conv2d kernel: torch [out,in,kh,kw] -> keras [kh,kw,in,out]  (transpose 2,3,1,0)
- ConvTranspose2d kernel: torch [in,out,kh,kw] -> keras [kh,kw,out,in] (transpose 2,3,1,0)
- BatchNorm: torch (weight,bias,running_mean,running_var)
             -> keras [gamma,beta,moving_mean,moving_variance]
Weighted layers are collected in construction order from both models and zipped;
counts and per-layer shapes are asserted so a mismatch raises instead of
silently misaligning.
"""
import numpy as np
import torch.nn as nn
from tensorflow.keras import layers as klayers

from .keras_model import build_keras_unet
from .model import UNet


def _torch_weighted(model):
    return [m for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d))]


def _keras_weighted(model):
    return [l for l in model.layers
            if isinstance(l, (klayers.Conv2D, klayers.Conv2DTranspose,
                              klayers.BatchNormalization))]


def _transfer(torch_model, keras_model):
    t_layers = _torch_weighted(torch_model)
    k_layers = _keras_weighted(keras_model)
    if len(t_layers) != len(k_layers):
        raise ValueError(
            f"weighted-layer count mismatch: torch {len(t_layers)} vs "
            f"keras {len(k_layers)} — architectures differ")

    for t, k in zip(t_layers, k_layers):
        if isinstance(t, nn.Conv2d):
            if not isinstance(k, klayers.Conv2D):
                raise ValueError(f"type mismatch: {type(t)} vs {type(k)}")
            w = t.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
            weights = [w]
            if t.bias is not None:
                weights.append(t.bias.detach().cpu().numpy())
        elif isinstance(t, nn.ConvTranspose2d):
            if not isinstance(k, klayers.Conv2DTranspose):
                raise ValueError(f"type mismatch: {type(t)} vs {type(k)}")
            w = t.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
            weights = [w]
            if t.bias is not None:
                weights.append(t.bias.detach().cpu().numpy())
        else:  # nn.BatchNorm2d
            if not isinstance(k, klayers.BatchNormalization):
                raise ValueError(f"type mismatch: {type(t)} vs {type(k)}")
            weights = [
                t.weight.detach().cpu().numpy(),
                t.bias.detach().cpu().numpy(),
                t.running_mean.detach().cpu().numpy(),
                t.running_var.detach().cpu().numpy(),
            ]

        expected = [wgt.shape for wgt in k.get_weights()]
        got = [wgt.shape for wgt in weights]
        if expected != got:
            raise ValueError(
                f"weight-shape mismatch for {k.name}: keras {expected} vs "
                f"transferred {got}")
        k.set_weights(weights)


def export_to_keras_hdf5(torch_model, path, dropout=0.1):
    torch_model = torch_model.eval()
    keras_model = build_keras_unet(dropout=dropout)
    _transfer(torch_model, keras_model)
    keras_model.save(path, save_format="h5")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_export_keras.py -v`
Expected: PASS (both tests). If parity fails narrowly on the transposed-conv path, the fallback knob is `padding="valid"` in `keras_model._up` (for kernel=stride=2 it is numerically identical to `"same"`); investigate before changing, and do not touch the tolerance.

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/export_keras.py tests/test_export_keras.py
git commit -m "feat: add PyTorch->Keras HDF5 export with drop-in parity test"
```

---

### Task 3 (optional, deferred): Real-analyzer-env smoke

**Files:**
- Create: `WINMOL_Analyzer/tests/test_hdf5_dropin_smoke.py`

Runs in the analyzer's conda env (with TensorFlow installed) and loads an exported `.hdf5` through the analyzer's ACTUAL `utils.IO.load_model_from_path`, then `predict_on_batch` on a synthetic NHWC batch, asserting shape `(N,512,512,1)`. Guard with `pytest.importorskip("tensorflow")`. This is the end-to-end proof in the real analyzer; defer until the analyzer env has TF installed.

---

## Self-Review

**Spec coverage:**
- Zero-analyzer-change load (`keras.models.load_model`) → Task 1 roundtrip test + Task 2 parity test use exactly that call ✓
- Standard-layers-only / no custom_objects → Task 1 (build_keras_unet uses only stock layers) ✓
- BN epsilon 1e-5 → Task 1 (`BN_EPS`, asserted by `test_keras_unet_uses_torch_bn_epsilon`) ✓
- Architecture mirror + concat order [skip, up] → Task 1 ✓
- Weight-transpose conventions + fail-loud on mismatch → Task 2 (`_transfer`, `test_transfer_fails_loudly_on_mismatch`) ✓
- atol=1e-4 parity → Task 2 ✓
- Real-analyzer end-to-end → Task 3 (deferred) ✓

**Placeholder scan:** none. All code shown in full.

**Type consistency:** `build_keras_unet(dropout)->Model`, `export_to_keras_hdf5(torch_model, path, dropout)->path`, weighted-layer collectors consistent across Tasks 1–2 and align with `winmol_unet.model.UNet`.
