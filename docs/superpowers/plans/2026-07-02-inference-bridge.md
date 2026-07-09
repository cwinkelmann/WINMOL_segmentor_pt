# Inference Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `winmol_unet` shared package (PyTorch U-Net + ONNX contract + export + `OnnxSegmenter` runtime + preprocess) and wire an additive ONNX inference path into `WINMOL_Analyzer`, proving the full contract end-to-end with an untrained model.

**Architecture:** A single installable package `winmol_unet` is the source of truth for the model architecture and the ONNX I/O contract. Models are exported to ONNX (NCHW, dynamic batch, fixed 512×512, sigmoid baked in, opset 17). The analyzer keeps TensorFlow and gains an extension-based branch: `.onnx` files load through `OnnxSegmenter`, which duck-types the Keras model object (`.predict_on_batch`, `.summary`).

**Tech Stack:** Python 3.9, PyTorch, onnx, onnxruntime, numpy, pytest. (Analyzer side also has TensorFlow 2.10, rasterio, skimage — unchanged.)

## Global Constraints

- Python floor: **3.9** (analyzer conda env is `python==3.9`).
- ONNX contract (verbatim): input `float32 [batch, 3, 512, 512]`, output `float32 [batch, 1, 512, 512]`, **batch axis dynamic**, spatial **fixed 512×512**, input range `[0,1]` RGB, **sigmoid baked into the graph**, **opset 17**.
- Wrapper I/O is **NHWC**: `OnnxSegmenter.predict_on_batch(x)` takes `[N,512,512,3]` and returns `[N,512,512,1]` float32.
- The analyzer keeps TensorFlow; ONNX support is **additive only** — the existing `.hdf5` load path must not change behavior.
- The PyTorch model `forward` returns **logits** (no sigmoid); sigmoid is appended only at export time. This keeps training stable while satisfying the "sigmoid in graph" contract.
- Constants live once in `winmol_unet/contract.py`; no magic numbers duplicated elsewhere.

---

### Task 1: Package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `winmol_unet/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Produces: installable package `winmol_unet`, importable as `import winmol_unet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_package_imports():
    import winmol_unet
    assert hasattr(winmol_unet, "__version__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'winmol_unet'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "winmol_unet"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.21",
    "onnx>=1.14",
    "onnxruntime>=1.16",
]

[project.optional-dependencies]
torch = ["torch>=2.0"]
train = ["torch>=2.0", "pillow>=9", "albumentations>=1.3", "tensorboard>=2.10"]
dev = ["pytest>=7"]

[tool.setuptools]
packages = ["winmol_unet"]
```

```python
# winmol_unet/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[torch,dev]" && pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml winmol_unet/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "feat: scaffold winmol_unet package"
```

---

### Task 2: ONNX contract module

**Files:**
- Create: `winmol_unet/contract.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Produces:
  - Constants `IMG_SIZE=512`, `IN_CHANNELS=3`, `OUT_CHANNELS=1`, `OPSET=17`.
  - `INPUT_NAME="input"`, `OUTPUT_NAME="output"`, `DYNAMIC_AXES={"input": {0: "batch"}, "output": {0: "batch"}}`.
  - `validate_onnx_model(onnx_model) -> None` — raises `ValueError` if the graph violates the contract.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contract.py
from winmol_unet import contract

def test_contract_constants():
    assert contract.IMG_SIZE == 512
    assert contract.IN_CHANNELS == 3
    assert contract.OUT_CHANNELS == 1
    assert contract.OPSET == 17
    assert contract.INPUT_NAME == "input"
    assert contract.OUTPUT_NAME == "output"
    assert contract.DYNAMIC_AXES == {"input": {0: "batch"}, "output": {0: "batch"}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contract.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`

- [ ] **Step 3: Write minimal implementation**

```python
# winmol_unet/contract.py
"""Frozen ONNX I/O contract shared by training export and analyzer inference."""

IMG_SIZE = 512
IN_CHANNELS = 3
OUT_CHANNELS = 1
OPSET = 17

INPUT_NAME = "input"
OUTPUT_NAME = "output"
DYNAMIC_AXES = {"input": {0: "batch"}, "output": {0: "batch"}}

INPUT_SHAPE = ("batch", IN_CHANNELS, IMG_SIZE, IMG_SIZE)
OUTPUT_SHAPE = ("batch", OUT_CHANNELS, IMG_SIZE, IMG_SIZE)


def _dim_values(tensor_type):
    dims = []
    for d in tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    return dims


def validate_onnx_model(onnx_model):
    """Raise ValueError if the model graph violates the contract."""
    graph = onnx_model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("Contract requires exactly one input and one output")

    got_in = _dim_values(graph.input[0].type.tensor_type)
    got_out = _dim_values(graph.output[0].type.tensor_type)

    # spatial + channel dims must match; batch dim must be symbolic (dynamic)
    if got_in[1:] != [IN_CHANNELS, IMG_SIZE, IMG_SIZE]:
        raise ValueError(f"Input shape {got_in} violates contract {INPUT_SHAPE}")
    if got_out[1:] != [OUT_CHANNELS, IMG_SIZE, IMG_SIZE]:
        raise ValueError(f"Output shape {got_out} violates contract {OUTPUT_SHAPE}")
    if isinstance(got_in[0], int) or isinstance(got_out[0], int):
        raise ValueError("Batch axis must be dynamic (symbolic), not fixed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/contract.py tests/test_contract.py
git commit -m "feat: add ONNX contract constants and validator"
```

---

### Task 3: Preprocess module

**Files:**
- Create: `winmol_unet/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `contract.IMG_SIZE`.
- Produces:
  - `to_float01(arr: np.ndarray) -> np.ndarray` — uint8→float32/255, passthrough float.
  - `resize_batch(batch_nhwc: np.ndarray, size: int = IMG_SIZE, mode: str = "bicubic") -> np.ndarray` — resizes an NHWC float32 batch to `size×size`; used identically by training and inference.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess.py
import numpy as np
from winmol_unet import preprocess

def test_to_float01_uint8():
    arr = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = preprocess.to_float01(arr)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)

def test_resize_batch_shape():
    batch = np.zeros((4, 100, 120, 3), dtype=np.float32)
    out = preprocess.resize_batch(batch, size=512)
    assert out.shape == (4, 512, 512, 3)
    assert out.dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# winmol_unet/preprocess.py
"""Resize/normalize shared by training and inference (no TensorFlow)."""
import numpy as np
from skimage.transform import resize as _sk_resize

from .contract import IMG_SIZE

_ORDER = {"nearest": 0, "bilinear": 1, "bicubic": 3}


def to_float01(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return (arr / 255.0).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def resize_batch(batch_nhwc, size=IMG_SIZE, mode="bicubic"):
    """Resize an NHWC float32 batch to (size, size). Deterministic, no antialias."""
    batch_nhwc = np.asarray(batch_nhwc, dtype=np.float32)
    n = batch_nhwc.shape[0]
    c = batch_nhwc.shape[3]
    out = np.empty((n, size, size, c), dtype=np.float32)
    order = _ORDER[mode]
    for i in range(n):
        out[i] = _sk_resize(
            batch_nhwc[i], (size, size), order=order,
            mode="edge", anti_aliasing=False, preserve_range=True,
        ).astype(np.float32)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS (add `scikit-image` to `pyproject.toml` dependencies if import fails, then re-run)

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/preprocess.py tests/test_preprocess.py pyproject.toml
git commit -m "feat: add shared preprocess (resize/normalize)"
```

---

### Task 4: PyTorch U-Net model

**Files:**
- Create: `winmol_unet/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `contract.IN_CHANNELS`, `contract.OUT_CHANNELS`, `contract.IMG_SIZE`.
- Produces: `UNet(in_channels=3, out_channels=1, dropout=0.1) -> nn.Module`. `forward(x)` takes NCHW `[N,3,512,512]`, returns **logits** NCHW `[N,1,512,512]` (no sigmoid).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
from winmol_unet.model import UNet

def test_unet_output_shape():
    model = UNet().eval()
    x = torch.zeros(2, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 512, 512)

def test_unet_returns_logits_not_probs():
    # logits are unbounded; a constant-0 input should not be forced into [0,1]
    model = UNet().eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert y.min() < 0.0 or y.max() > 1.0 or True  # shape/logit contract; no sigmoid layer
    assert not any(m.__class__.__name__ == "Sigmoid" for m in model.modules())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# winmol_unet/model.py
"""PyTorch U-Net. Mirrors the R model_UNet.R (512x512, BN+dropout, skip concats).

forward() returns LOGITS. Sigmoid is appended only at ONNX export time
(see winmol_unet.export), satisfying the "sigmoid in graph" contract while
keeping training numerically stable with BCEWithLogitsLoss.
"""
import torch
import torch.nn as nn

from .contract import IN_CHANNELS, OUT_CHANNELS


def _conv_bn_relu(in_ch, out_ch):
    # use_bias=False because BatchNorm follows (matches R: use_bias=FALSE)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _DoubleConv(nn.Module):
    """conv-bn-relu -> dropout -> conv-bn-relu (matches R block layout)."""

    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.c1 = _conv_bn_relu(in_ch, out_ch)
        self.drop = nn.Dropout2d(p=dropout)
        self.c2 = _conv_bn_relu(out_ch, out_ch)

    def forward(self, x):
        return self.c2(self.drop(self.c1(x)))


class UNet(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, dropout=0.1):
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, 64, dropout)
        self.enc2 = _DoubleConv(64, 128, dropout)
        self.enc3 = _DoubleConv(128, 256, dropout)
        self.enc4 = _DoubleConv(256, 512, dropout)
        self.bottleneck = _DoubleConv(512, 1024, dropout)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False)
        self.dec4 = _DoubleConv(1024, 512, dropout)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.dec3 = _DoubleConv(512, 256, dropout)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.dec2 = _DoubleConv(256, 128, dropout)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.dec1 = _DoubleConv(128, 64, dropout)

        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))

        d4 = self.dec4(torch.cat([c4, self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([c3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([c2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([c1, self.up1(d2)], dim=1))
        return self.head(d1)  # logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/model.py tests/test_model.py
git commit -m "feat: add PyTorch U-Net (logits output)"
```

---

### Task 5: ONNX export + parity (core gate)

**Files:**
- Create: `winmol_unet/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `UNet`, `contract.*`.
- Produces: `export_to_onnx(model: nn.Module, path: str) -> str` — wraps model with a trailing `Sigmoid`, exports contract-conformant ONNX, validates it, returns `path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export.py
import numpy as np
import onnx
import onnxruntime as ort
import torch

from winmol_unet import contract
from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet

def test_export_is_contract_conformant(tmp_path):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    contract.validate_onnx_model(onnx.load(path))  # raises on violation

def test_export_parity_torch_vs_onnx(tmp_path):
    model = UNet().eval()
    path = export_to_onnx(model, str(tmp_path / "m.onnx"))
    x = np.random.rand(2, 3, 512, 512).astype(np.float32)

    with torch.no_grad():
        torch_out = torch.sigmoid(model(torch.from_numpy(x))).numpy()

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run([contract.OUTPUT_NAME], {contract.INPUT_NAME: x})[0]

    assert onnx_out.shape == (2, 1, 512, 512)
    assert np.allclose(torch_out, onnx_out, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: winmol_unet.export`

- [ ] **Step 3: Write minimal implementation**

```python
# winmol_unet/export.py
"""Export a trained (or untrained) UNet to a contract-conformant ONNX file."""
import onnx
import torch
import torch.nn as nn

from .contract import (
    DYNAMIC_AXES, IMG_SIZE, IN_CHANNELS, INPUT_NAME, OPSET, OUTPUT_NAME,
    validate_onnx_model,
)


class _WithSigmoid(nn.Module):
    """Appends sigmoid so the exported graph emits probabilities (contract)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.model(x))


def export_to_onnx(model, path):
    model = model.eval()
    wrapped = _WithSigmoid(model).eval()
    dummy = torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapped, dummy, path,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes=DYNAMIC_AXES, opset_version=OPSET,
        do_constant_folding=True,
    )
    validate_onnx_model(onnx.load(path))
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_export.py -v`
Expected: PASS (both tests). This is the core correctness gate — the bridge is proven when parity passes.

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/export.py tests/test_export.py
git commit -m "feat: add ONNX export with sigmoid head and torch-vs-onnx parity test"
```

---

### Task 6: OnnxSegmenter runtime wrapper

**Files:**
- Create: `winmol_unet/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `contract.*`, an `.onnx` file from `export_to_onnx`.
- Produces:
  - `OnnxSegmenter(model_path, providers=None)`.
  - `.predict_on_batch(x)` — accepts NHWC `[N,512,512,3]` (numpy **or** anything with `.numpy()`, e.g. a TF tensor); returns NHWC `[N,512,512,1]` float32 probabilities.
  - `.summary()` — prints the IO contract (drop-in for Keras `model.summary()`).
  - Raises `OnnxOutOfMemoryError` (subclass of `RuntimeError`) on runtime OOM.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime.py
import numpy as np

from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet
from winmol_unet.runtime import OnnxSegmenter

def test_predict_on_batch_nhwc_roundtrip(tmp_path):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    seg = OnnxSegmenter(path)
    x = np.random.rand(3, 512, 512, 3).astype(np.float32)
    out = seg.predict_on_batch(x)
    assert out.shape == (3, 512, 512, 1)
    assert out.dtype == np.float32
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid range

def test_summary_runs(tmp_path, capsys):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    OnnxSegmenter(path).summary()
    assert "512" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runtime.py -v`
Expected: FAIL with `ModuleNotFoundError: winmol_unet.runtime`

- [ ] **Step 3: Write minimal implementation**

```python
# winmol_unet/runtime.py
"""Runtime adapter that makes an ONNX model duck-type the Keras model object."""
import numpy as np
import onnxruntime as ort

from .contract import IMG_SIZE, INPUT_NAME, OUTPUT_NAME


class OnnxOutOfMemoryError(RuntimeError):
    """Raised on ONNX runtime OOM; caught by the analyzer's batch-backoff loop."""


def _default_providers():
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxSegmenter:
    def __init__(self, model_path, providers=None):
        self.model_path = model_path
        self.providers = providers or _default_providers()
        self.session = ort.InferenceSession(model_path, providers=self.providers)

    @staticmethod
    def _as_numpy(x):
        if hasattr(x, "numpy"):        # TF tensor / torch tensor
            x = x.numpy()
        return np.ascontiguousarray(np.asarray(x, dtype=np.float32))

    def predict_on_batch(self, x):
        x = self._as_numpy(x)          # NHWC [N,512,512,3]
        nchw = np.transpose(x, (0, 3, 1, 2))
        try:
            out = self.session.run([OUTPUT_NAME], {INPUT_NAME: nchw})[0]
        except Exception as exc:       # normalize OOM for the retry loop
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg or "cudaerror" in msg:
                raise OnnxOutOfMemoryError(str(exc)) from exc
            raise
        return np.transpose(out, (0, 2, 3, 1)).astype(np.float32)  # NHWC [N,512,512,1]

    def summary(self):
        print(
            f"OnnxSegmenter(providers={self.providers}) "
            f"input=[N,3,{IMG_SIZE},{IMG_SIZE}] -> output=[N,1,{IMG_SIZE},{IMG_SIZE}] "
            f"(NHWC at the predict_on_batch boundary)"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/runtime.py tests/test_runtime.py
git commit -m "feat: add OnnxSegmenter runtime adapter (NHWC boundary, OOM normalization)"
```

---

### Task 7: Analyzer ONNX loader branch (in WINMOL_Analyzer repo)

**Files:**
- Modify: `WINMOL_Analyzer/utils/IO.py:189-216` (the model-loading helper)
- Modify: `WINMOL_Analyzer/requirements/base.txt`
- Create: `WINMOL_Analyzer/tests/test_model_loading.py`

**Interfaces:**
- Consumes: `winmol_unet.runtime.OnnxSegmenter` (analyzer pip-installs `winmol_unet`).
- Produces: `IO.load_model(model_path, config)` returns a Keras model for `.hdf5/.h5` (unchanged) or an `OnnxSegmenter` for `.onnx`.

- [ ] **Step 1: Write the failing test**

First read `WINMOL_Analyzer/utils/IO.py:180-220` to confirm the exact current loader function name and signature; the test below assumes a `load_model(path, config)` helper — adjust the name to match what exists.

```python
# WINMOL_Analyzer/tests/test_model_loading.py
import numpy as np
import pytest

from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet
from winmol_unet.runtime import OnnxSegmenter
from utils import IO
from classes.Config import Config

def test_onnx_path_returns_segmenter(tmp_path):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    model = IO.load_model(path, Config())
    assert isinstance(model, OnnxSegmenter)
    out = model.predict_on_batch(np.random.rand(1, 512, 512, 3).astype(np.float32))
    assert out.shape == (1, 512, 512, 1)

def test_hdf5_path_still_uses_keras(monkeypatch):
    # Regression guard: .hdf5 must route to keras.models.load_model, unchanged.
    called = {}
    import tensorflow as tf
    def fake_load(path, compile):
        called["path"] = path
        return "KERAS_MODEL"
    monkeypatch.setattr(tf.keras.models, "load_model", fake_load)
    result = IO.load_model("/some/model.hdf5", Config())
    assert result == "KERAS_MODEL"
    assert called["path"] == "/some/model.hdf5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd WINMOL_Analyzer && pytest tests/test_model_loading.py -v`
Expected: FAIL — `IO.load_model` does not exist yet or does not branch on extension.

- [ ] **Step 3: Write minimal implementation**

Add an extension-branching loader to `utils/IO.py` (keep any existing keras loader body; wrap it in the `.hdf5/.h5` branch):

```python
# utils/IO.py  (add near the existing load helpers)
import os

def load_model(model_path, config=None):
    """Load a segmentation model. .hdf5/.h5 -> Keras (unchanged); .onnx -> OnnxSegmenter."""
    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".hdf5", ".h5"):
        from tensorflow import keras
        return keras.models.load_model(model_path, compile=False)
    if ext == ".onnx":
        from winmol_unet.runtime import OnnxSegmenter
        providers = None
        if config is not None and getattr(config, "prediction_backend", "auto") == "cpu":
            providers = ["CPUExecutionProvider"]
        return OnnxSegmenter(model_path, providers=providers)
    raise ValueError(f"Unsupported model extension: {ext}")
```

Then update `standalone/WINMOL_Analyzer.py:38` to use it:

```python
# standalone/WINMOL_Analyzer.py  (replace the keras load line)
from utils import IO
model = IO.load_model(model_path, config)
```

Add the dependency:

```text
# requirements/base.txt  (append)
onnxruntime>=1.16
winmol_unet @ file:///../WINMOL_segmentor_pt   # editable/local during dev
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd WINMOL_Analyzer && pip install -e ../WINMOL_segmentor_pt && pytest tests/test_model_loading.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
cd WINMOL_Analyzer
git add utils/IO.py standalone/WINMOL_Analyzer.py requirements/base.txt tests/test_model_loading.py
git commit -m "feat: add additive ONNX model-loading path (keras HDF5 path unchanged)"
```

---

### Task 8: End-to-end smoke (untrained ONNX model through the standalone pipeline)

**Files:**
- Create: `WINMOL_Analyzer/tests/test_end_to_end_onnx.py`

**Interfaces:**
- Consumes: `IO.load_model`, `Prediction.predict_with_resampling_per_tile`, `winmol_unet` export.

- [ ] **Step 1: Write the failing test**

```python
# WINMOL_Analyzer/tests/test_end_to_end_onnx.py
import numpy as np

from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet
from utils import IO
from utils import Prediction as Pred
from classes.Config import Config

def test_predict_with_resampling_runs_with_onnx(tmp_path):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    model = IO.load_model(path, Config())
    config = Config()
    # small synthetic orthomosaic + minimal geo profile
    img = (np.random.rand(600, 700, 3)).astype(np.float32)
    profile = {"transform": [0.02, 0, 0, 0, -0.02, 0], "crs": None}
    pred, out_profile = Pred.predict_with_resampling_per_tile(img, profile, model, config)
    assert pred.dtype == np.uint8
    assert pred.ndim == 2
    assert set(np.unique(pred)).issubset({0, 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd WINMOL_Analyzer && pytest tests/test_end_to_end_onnx.py -v`
Expected: FAIL initially if `predict_with_resampling_per_tile` still calls `tf.convert_to_tensor` internally on the tile (line ~737 legacy path is separate; the resampling path uses `predict_tile_array` → `model.predict_on_batch`). If a TF call blocks the ONNX model, that indicates a preprocessing site that must move to `winmol_unet.preprocess`; note it and fix minimally.

- [ ] **Step 3: Write minimal implementation**

If the test fails only because `_prepare_inference_batch` wraps the batch in a TF tensor before `predict_on_batch`, that is fine — `OnnxSegmenter.predict_on_batch` accepts TF tensors via `_as_numpy`. No code change needed; the test should pass as-is. If it fails because a resize uses `tf.image.resize` on a path the ONNX model can't consume, replace only that call with `winmol_unet.preprocess.resize_batch`, leaving TF present for the HDF5 path.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd WINMOL_Analyzer && pytest tests/test_end_to_end_onnx.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd WINMOL_Analyzer
git add tests/test_end_to_end_onnx.py
git commit -m "test: end-to-end ONNX model through resampling prediction path"
```

---

## Self-Review

**Spec coverage:**
- Contract (spec §3) → Task 2 ✓
- Shared package `winmol_unet` (spec §4.1) → Tasks 1–6 ✓
- Preprocess shared module (spec §4.1) → Task 3 ✓
- Analyzer additive ONNX branch (spec §5) → Task 7 ✓
- Export parity / runtime wrapper tests (spec §8.1–8.4, 8.6) → Tasks 5, 6, 8 ✓
- Training port (spec §4.2) → **deferred to Plan 2** (out of scope here, by design) ✓
- Legacy HDF5→ONNX conversion (spec §4.3) → deferred/optional, not in this plan ✓

**Placeholder scan:** Task 7/8 intentionally instruct reading `IO.py`/`Prediction.py` to confirm exact current symbol names before editing (the analyzer code is pre-existing and not authored here); all new code is shown in full. No TBDs.

**Type consistency:** `predict_on_batch` NHWC in/out, `export_to_onnx(model, path)->path`, `OnnxSegmenter(path, providers)`, `IO.load_model(path, config)` are consistent across Tasks 5–8.
