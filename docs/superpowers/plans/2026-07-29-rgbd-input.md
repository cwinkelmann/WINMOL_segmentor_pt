# RGBD (4-Channel) Input Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in RGBD (4-channel) input: datasets carry a per-tile depth channel, models accept 4-channel input, and the ONNX export/serve path handles 4-channel models — with every existing RGB path unchanged.

**Architecture:** Parameterize channels instead of mutating the frozen contract: `IN_CHANNELS = 3` stays the default; a new `RGBD_IN_CHANNELS = 4` constant plus an `in_channels` argument flows through `build_model` → `export_to_onnx` → `validate_onnx_model`. The dataset gains an optional `depth_dir` (`depth/depth{N}.png|.tif` paired by integer N, like `train{N}.jpeg`/`mask{N}.gif`); depth is normalized to [0,1], carried through albumentations as an `additional_targets={"depth": "mask"}` target (geometric transforms apply, photometric skip), and concatenated as the 4th channel after augmentation. `OnnxSegmenter` infers the channel count from the loaded graph. A `--rgbd` CLI flag wires it all.

**Tech Stack:** PyTorch, segmentation-models-pytorch (lazy), albumentations, onnx/onnxruntime, PIL, scikit-image, pytest.

## Global Constraints

- `winmol_unet/` analyzer-facing modules (`contract`, `runtime`) import only `onnxruntime`/`numpy`; `preprocess` additionally `skimage`. Never import torch/tf/albumentations there.
- RGB stays the default everywhere: `IN_CHANNELS = 3` unchanged; every existing test must pass without modification (Task 1 only *adds* tests).
- ONNX contract otherwise unchanged: NCHW, dynamic batch, spatial 512-or-symbolic, opset 17, sigmoid baked in at export.
- Keras HDF5/.keras export remains UNet + RGB only — fail loud (`ValueError`) for RGBD, never silently skip.
- Tests hermetic: synthetic data in `tmp_path`, `encoder_weights=None` for smp archs (no downloads).
- ONNX serve tests pin the CPU EP via `WINMOL_ONNX_FORCE_CPU=1` (monkeypatch env, as existing runtime tests do).
- **Cross-repo note:** a 4-channel ONNX model is only consumable once `WINMOL_Analyzer` feeds 4-channel NHWC batches. This plan keeps 3-channel the default precisely so nothing breaks for the analyzer until it opts in. Do not change analyzer-facing defaults.

**Out of scope (explicit follow-ups, do NOT implement here):**
- `scripts/build_dataset.py` depth conversion (raw depth sources — GeoTIFF DSM/CHM, sensor exports — vary too much; users place `depth{N}` files themselves for now).
- Keras mirror for 4 channels.

**Base note (2026-07-29 rebase):** this plan now targets `main` (branch `feat/depth-simulator`), which has NO `multiscale`, `eval_tiling`, `TilingStemDataset`, `split_ids`, or `StemDataset(resize=...)` — those exist only on `feat/bamforests-benchmark`. The tasks below are written against main's code.

---

### Task 1: Channels-parameterized contract validation

**Files:**
- Modify: `winmol_unet/contract.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Consumes: nothing (root task).
- Produces: `RGBD_IN_CHANNELS = 4` (module constant) and `validate_onnx_model(onnx_model, in_channels=IN_CHANNELS)` — later tasks pass `in_channels=4` when validating RGBD exports.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_contract.py`)

```python
import pytest
from onnx import TensorProto, helper


def _model_with_in_channels(in_ch):
    # validate_onnx_model only inspects graph input/output value_info, so an
    # Identity node is enough to build a well-formed ModelProto for shape tests.
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", in_ch, 512, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1, 512, 512])
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])],
                              "g", [inp], [out])
    return helper.make_model(graph)


def test_rgbd_channel_constant():
    assert contract.RGBD_IN_CHANNELS == 4


def test_validate_accepts_4ch_when_requested():
    contract.validate_onnx_model(_model_with_in_channels(4),
                                 in_channels=contract.RGBD_IN_CHANNELS)


def test_validate_rejects_4ch_by_default():
    with pytest.raises(ValueError):
        contract.validate_onnx_model(_model_with_in_channels(4))


def test_validate_rejects_3ch_when_rgbd_expected():
    with pytest.raises(ValueError):
        contract.validate_onnx_model(_model_with_in_channels(3), in_channels=4)
```

Match the existing import style at the top of `tests/test_contract.py` (it already imports `contract`; add the `pytest`/`onnx.helper` imports if missing).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contract.py -v`
Expected: FAIL — `AttributeError: ... no attribute 'RGBD_IN_CHANNELS'` / `TypeError: validate_onnx_model() got an unexpected keyword argument`.

- [ ] **Step 3: Implement in `winmol_unet/contract.py`**

Add below `IN_CHANNELS = 3`:

```python
RGBD_IN_CHANNELS = 4   # opt-in RGBD variant; IN_CHANNELS stays the frozen default
```

Change the signature and last lines of `validate_onnx_model`:

```python
def validate_onnx_model(onnx_model, in_channels=IN_CHANNELS):
    """Raise ValueError if the model graph violates the contract.

    Batch axis must be dynamic; channels fixed (`in_channels` in — default 3,
    pass RGBD_IN_CHANNELS for RGBD models — 1 out); spatial dims either fixed
    512 or dynamic (see _spatial_ok).
    """
    graph = onnx_model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("Contract requires exactly one input and one output")

    _check_shape(_dim_values(graph.input[0].type.tensor_type), in_channels, "Input")
    _check_shape(_dim_values(graph.output[0].type.tensor_type), OUT_CHANNELS, "Output")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contract.py -v`
Expected: PASS (all, including the pre-existing `IN_CHANNELS == 3` assertion).

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/contract.py tests/test_contract.py
git commit -m "feat(contract): channels-parameterized ONNX validation + RGBD_IN_CHANNELS"
```

---

### Task 2: Depth normalization in preprocess

**Files:**
- Modify: `winmol_unet/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `normalize_depth(arr, vmin=None, vmax=None) -> np.float32 array in [0,1]`, same HW shape as input. Task 5 (dataset) calls it with a raw PIL-decoded depth array.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_preprocess.py`)

```python
from winmol_unet.preprocess import normalize_depth


def test_normalize_depth_minmax_uint16():
    arr = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    out = normalize_depth(arr)
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    np.testing.assert_allclose(out, np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32))


def test_normalize_depth_constant_maps_to_zeros():
    out = normalize_depth(np.full((4, 4), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros((4, 4), dtype=np.float32))


def test_normalize_depth_fixed_range_clips():
    arr = np.array([-5.0, 0.0, 5.0, 15.0], dtype=np.float32)
    out = normalize_depth(arr, vmin=0.0, vmax=10.0)
    np.testing.assert_allclose(out, np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocess.py -v -k depth`
Expected: FAIL with `ImportError: cannot import name 'normalize_depth'`.

- [ ] **Step 3: Implement in `winmol_unet/preprocess.py`** (append)

```python
def normalize_depth(arr, vmin=None, vmax=None):
    """Depth array (any numeric dtype, any shape) -> float32 in [0, 1].

    Defaults to per-image min-max (robust to unknown sensor units); pass
    vmin/vmax for a fixed physical range shared across a dataset. A constant
    image (vmax <= vmin) maps to zeros rather than dividing by zero.
    """
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(arr.min()) if vmin is None else float(vmin)
    hi = float(arr.max()) if vmax is None else float(vmax)
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add winmol_unet/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): normalize_depth for RGBD input"
```

---

### Task 3: Model factory accepts in_channels

**Files:**
- Modify: `training/model_factory.py`
- Test: `tests/test_model_factory.py`

**Interfaces:**
- Consumes: `IN_CHANNELS` from `winmol_unet.contract` (already imported there). `winmol_unet.model.UNet` already has an `in_channels=IN_CHANNELS` constructor parameter — no change to `model.py` needed.
- Produces: `build_model(arch="unet", dropout=0.1, encoder="resnet34", encoder_weights=None, in_channels=IN_CHANNELS)`. Task 7 calls it with `in_channels=cfg.in_channels`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_model_factory.py`)

```python
@pytest.mark.parametrize("arch", ["unet", "deeplabv3plus", "hrnet"])
def test_build_model_rgbd_forward(arch):
    model = build_model(arch=arch, encoder_weights=None, in_channels=4)
    x = torch.randn(1, 4, 64, 64)   # 64: divisible by 32 (smp) and by 16 (UNet pools)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64)
```

Match the file's existing imports (`build_model`, `torch`, `pytest`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_factory.py -v -k rgbd`
Expected: FAIL with `TypeError: build_model() got an unexpected keyword argument 'in_channels'`.

- [ ] **Step 3: Implement in `training/model_factory.py`**

```python
def build_model(arch="unet", dropout=0.1, encoder="resnet34", encoder_weights=None,
                in_channels=IN_CHANNELS):
    if arch == "unet":
        from winmol_unet.model import UNet
        return UNet(in_channels=in_channels, dropout=dropout)
    import segmentation_models_pytorch as smp
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights,
                                 in_channels=in_channels, classes=OUT_CHANNELS)
    if arch == "hrnet":
        return smp.Unet(encoder_name="tu-hrnet_w18", encoder_weights=encoder_weights,
                        in_channels=in_channels, classes=OUT_CHANNELS)
    raise ValueError(f"unknown arch {arch!r}; choose 'unet', 'deeplabv3plus', or 'hrnet'")
```

Update the module docstring's `forward(x:[N,3,512,512])` line to `forward(x:[N,in_channels,512,512]) (default 3; 4 for RGBD)`. Note in the docstring that smp adapts pretrained first-conv weights automatically when `in_channels != 3` (`encoder_weights="imagenet"` still works).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/model_factory.py tests/test_model_factory.py
git commit -m "feat(factory): build_model in_channels pass-through for RGBD"
```

---

### Task 4: ONNX export + runtime for 4-channel models

**Files:**
- Modify: `winmol_unet/export.py`, `winmol_unet/runtime.py`
- Test: `tests/test_export.py`, `tests/test_runtime.py`

**Interfaces:**
- Consumes: `validate_onnx_model(model, in_channels=...)` and `RGBD_IN_CHANNELS` from Task 1; `UNet(in_channels=4)` (constructor param already exists).
- Produces: `export_to_onnx(model, path, in_channels=IN_CHANNELS)`; `OnnxSegmenter.in_channels` (int, inferred from graph) with `predict_on_batch` raising `ValueError` on channel mismatch. Task 7 calls `export_to_onnx(..., in_channels=cfg.in_channels)`.

- [ ] **Step 1: Write the failing export test** (append to `tests/test_export.py`)

```python
def test_export_rgbd_unet(tmp_path):
    from winmol_unet.model import UNet
    path = str(tmp_path / "rgbd.onnx")
    export_to_onnx(UNet(in_channels=4), path, in_channels=4)
    model = onnx.load(path)
    dims = [d.dim_param or d.dim_value
            for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims[1] == 4
```

Match existing imports in the file (`onnx`, `export_to_onnx`).

- [ ] **Step 2: Write the failing runtime test** (append to `tests/test_runtime.py`, mirroring how existing tests there set `WINMOL_ONNX_FORCE_CPU` and export a model)

```python
def test_onnx_segmenter_serves_rgbd(tmp_path, monkeypatch):
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    from winmol_unet.export import export_to_onnx
    from winmol_unet.model import UNet
    path = str(tmp_path / "rgbd.onnx")
    export_to_onnx(UNet(in_channels=4), path, in_channels=4)

    seg = OnnxSegmenter(path)
    assert seg.in_channels == 4
    out = seg.predict_on_batch(np.zeros((1, 512, 512, 4), dtype=np.float32))
    assert out.shape == (1, 512, 512, 1)
    with pytest.raises(ValueError):
        seg.predict_on_batch(np.zeros((1, 512, 512, 3), dtype=np.float32))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_export.py::test_export_rgbd_unet tests/test_runtime.py::test_onnx_segmenter_serves_rgbd -v`
Expected: FAIL — `TypeError: export_to_onnx() got an unexpected keyword argument 'in_channels'`.

- [ ] **Step 4: Implement `winmol_unet/export.py`**

```python
def export_to_onnx(model, path, in_channels=IN_CHANNELS):
    model = model.eval()
    wrapped = _WithSigmoid(model).eval()
    dummy = torch.zeros(1, in_channels, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapped, dummy, path,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes=DYNAMIC_AXES, opset_version=OPSET,
        do_constant_folding=True,
    )
    validate_onnx_model(onnx.load(path), in_channels=in_channels)
    return path
```

(Only the signature, `dummy`, and the `validate_onnx_model` call change.)

- [ ] **Step 5: Implement `winmol_unet/runtime.py`**

In `OnnxSegmenter.__init__`, after creating the session:

```python
        # Channel count comes from the loaded graph (3 = RGB, 4 = RGBD), so one
        # adapter serves both model families without a mode switch.
        self.in_channels = int(self.session.get_inputs()[0].shape[1])
```

In `predict_on_batch`, after `x = self._as_numpy(x)`:

```python
        if x.shape[3] != self.in_channels:
            raise ValueError(
                f"model expects NHWC with C={self.in_channels}, got {x.shape}")
```

In `summary()`, replace the hardcoded `3` with `{self.in_channels}`:

```python
        print(
            f"OnnxSegmenter(providers={self.providers}) "
            f"input=[N,{IMG_SIZE},{IMG_SIZE},{self.in_channels}] "
            f"-> output=[N,{IMG_SIZE},{IMG_SIZE},1] (NHWC)"
        )
```

- [ ] **Step 6: Run tests to verify they pass** (existing RGB runtime/export tests must also stay green)

Run: `pytest tests/test_export.py tests/test_runtime.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add winmol_unet/export.py winmol_unet/runtime.py tests/test_export.py tests/test_runtime.py
git commit -m "feat(onnx): 4-channel export + channel-inferring OnnxSegmenter"
```

---

### Task 5: StemDataset loads a depth channel

**Files:**
- Modify: `training/dataset.py`
- Test: `tests/test_train_dataset.py`

**Interfaces:**
- Consumes: `normalize_depth` from Task 2; existing `resize_batch`, `to_float01`, `_index_by_n`.
- Produces: `StemDataset(image_dir, mask_dir, ..., depth_dir=None)` yielding image tensors `[4, S, S]` when `depth_dir` is set; `_paired_ids(image_dir, mask_dir, depth_dir=None)`; `train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512, transform=None, cache=True, depth_dir=None)`. Depth files: `depth{N}.png` (8/16-bit) or `depth{N}.tif`/`.tiff` (float), paired by integer N. Task 6 adds the augmentation path; Task 7 wires the CLI.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_train_dataset.py`; reuse the file's existing helpers for writing `train{N}.jpeg`/`mask{N}.gif` fixtures)

```python
def _write_depth(depth_dir, n, size=(32, 32)):
    os.makedirs(depth_dir, exist_ok=True)
    h, w = size
    arr = (np.arange(h * w, dtype=np.uint16).reshape(h, w) * 17) % 5000
    Image.fromarray(arr, mode="I;16").save(os.path.join(depth_dir, f"depth{n}.png"))


def test_stem_dataset_rgbd_shapes(tmp_path):
    image_dir, mask_dir = _make_pairs(tmp_path, ids=[1, 2])   # existing fixture helper
    depth_dir = str(tmp_path / "depth")
    for n in (1, 2):
        _write_depth(depth_dir, n)
    ds = StemDataset(image_dir, mask_dir, img_size=64, depth_dir=depth_dir)
    img, mask = ds[0]
    assert img.shape == (4, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert 0.0 <= float(img[3].min()) and float(img[3].max()) <= 1.0


def test_paired_ids_require_depth_when_depth_dir_given(tmp_path):
    image_dir, mask_dir = _make_pairs(tmp_path, ids=[1, 2, 3])
    depth_dir = str(tmp_path / "depth")
    _write_depth(depth_dir, 1)
    _write_depth(depth_dir, 3)
    ds = StemDataset(image_dir, mask_dir, img_size=64, depth_dir=depth_dir)
    assert ds.ids == [1, 3]


def test_stem_dataset_without_depth_unchanged(tmp_path):
    image_dir, mask_dir = _make_pairs(tmp_path, ids=[1])
    ds = StemDataset(image_dir, mask_dir, img_size=64)
    img, _ = ds[0]
    assert img.shape == (3, 64, 64)
```

If `tests/test_train_dataset.py` has no reusable pair-writing helper named `_make_pairs`, adapt to whatever helper it does use (read the file first) — do not duplicate fixture code.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_dataset.py -v -k "rgbd or depth"`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'depth_dir'`.

- [ ] **Step 3: Implement in `training/dataset.py`**

Import: `from winmol_unet.preprocess import normalize_depth, resize_batch, to_float01`.

Module-level, next to `_paired_ids`:

```python
DEPTH_EXTS = (".png", ".tif", ".tiff")


def _depth_index(depth_dir):
    """n -> filename for depth{N}.png|.tif|.tiff (16-bit PNG or float TIFF)."""
    out = {}
    for name in os.listdir(depth_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in DEPTH_EXTS and stem.startswith("depth") and stem[len("depth"):].isdigit():
            out[int(stem[len("depth"):])] = name
    return out


def _paired_ids(image_dir, mask_dir, depth_dir=None):
    imgs = _index_by_n(os.listdir(image_dir), "train", ".jpeg")
    masks = _index_by_n(os.listdir(mask_dir), "mask", ".gif")
    ids = set(imgs) & set(masks)
    if depth_dir is not None:
        ids &= set(_depth_index(depth_dir))
    return sorted(ids)
```

`StemDataset.__init__` gains `depth_dir=None` (appended after `cache` to keep positional args stable):

```python
    def __init__(self, image_dir, mask_dir, img_size=512, transform=None, ids=None, cache=True,
                 depth_dir=None):
        ...existing body unchanged, plus:
        self.depth_dir = depth_dir
        self._depth_names = _depth_index(depth_dir) if depth_dir is not None else None
        self.ids = ids if ids is not None else _paired_ids(image_dir, mask_dir, depth_dir)
```

(The `self.ids` line replaces the existing one; the cache comment becomes `# n -> (image HWC, mask HW[, depth HW1]) float32`.)

New loader, next to `_load_mask`:

```python
    def _load_depth(self, n):
        with Image.open(os.path.join(self.depth_dir, self._depth_names[n])) as im:
            raw = np.asarray(im)
        # per-image min-max to [0,1]; nearest resize to match image/mask handling
        arr = normalize_depth(raw)[..., None]                               # HW1
        arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]
        return np.ascontiguousarray(arr, dtype=np.float32)                  # HW1
```

`__getitem__`: load/cache depth alongside image+mask and concatenate it as the 4th channel **before** the existing HWC→CHW tensor conversion (keep those existing conversion lines untouched):

```python
    def __getitem__(self, i):
        n = self.ids[i]
        if self.cache:
            if n not in self._cache:
                item = (self._load_image(n), self._load_mask(n))
                if self.depth_dir is not None:
                    item = item + (self._load_depth(n),)
                self._cache[n] = item
            item = self._cache[n]
        else:
            item = (self._load_image(n), self._load_mask(n))
            if self.depth_dir is not None:
                item = item + (self._load_depth(n),)
        img, mask = item[0], item[1]
        depth = item[2] if self.depth_dir is not None else None
        if self.transform is not None:
            if depth is not None:
                out = self.transform(image=img, mask=mask, depth=depth)
                img, mask, depth = out["image"], out["mask"], out["depth"]
            else:
                out = self.transform(image=img, mask=mask)
                img, mask = out["image"], out["mask"]
        if depth is not None:
            img = np.concatenate([img, depth], axis=-1)          # HWC -> HW4
        ...existing tensor-conversion + return lines, unchanged...
```

`train_val_split` pass-through (main has no `split_ids` — the split logic lives inline; only the first line and the two `StemDataset(...)` calls change):

```python
def train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512, transform=None,
                    cache=True, depth_dir=None):
    ids = _paired_ids(image_dir, mask_dir, depth_dir)
    ...existing shuffle/split logic unchanged...
    train_ds = StemDataset(image_dir, mask_dir, img_size, transform=transform, ids=train_ids,
                           cache=cache, depth_dir=depth_dir)
    val_ds = StemDataset(image_dir, mask_dir, img_size, transform=None, ids=val_ids, cache=cache,
                         depth_dir=depth_dir)
    return train_ds, val_ds
```

Update the `StemDataset` docstring: mention optional `depth_dir` (`depth{N}.png|.tif`, per-image min-max normalized, appended as 4th channel).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_dataset.py tests/test_dataset_cache.py -v`
Expected: PASS (including the pre-existing RGB tests and cache tests).

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py tests/test_train_dataset.py
git commit -m "feat(dataset): optional depth channel (depth{N}.png|.tif) in StemDataset"
```

---

### Task 6: Depth-aware augmentation

**Files:**
- Modify: `training/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Consumes: `build_augmentation(cfg)` (existing), `TrainConfig` (existing fields only), the `transform(image=..., mask=..., depth=...)` call shape from Task 5.
- Produces: the Compose from `build_augmentation` accepts an optional `depth` target (HW1 float32). Geometric transforms (flip/rotate/crop) move depth with the image using nearest interpolation; photometric ones (brightness/contrast, HSV) leave it untouched — HSV also *requires* a 3-channel image, which is why depth must ride as a separate target rather than a 4th image channel.

- [ ] **Step 1: Write the failing test** (append to `tests/test_augment.py`, reusing however that file constructs a `TrainConfig` — it already builds cfgs for the existing transform tests)

```python
def test_depth_follows_geometric_but_not_photometric(tmp_path):
    cfg = _make_cfg(tmp_path, aug_hflip_p=1.0, aug_vflip_p=0.0, aug_rotate_p=0.0,
                    aug_bc_p=1.0, aug_hsv_p=1.0)   # adapt to the file's cfg helper
    tf = build_augmentation(cfg)
    img = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    mask = np.zeros((16, 16), dtype=np.float32)
    depth = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16, 1)
    out = tf(image=img, mask=mask, depth=depth)
    # hflip (p=1) applies to depth; brightness/HSV (p=1) must NOT touch it
    np.testing.assert_allclose(out["depth"], depth[:, ::-1, :])


def test_compose_without_depth_still_works(tmp_path):
    cfg = _make_cfg(tmp_path, aug_hflip_p=1.0)
    tf = build_augmentation(cfg)
    out = tf(image=np.zeros((16, 16, 3), np.float32), mask=np.zeros((16, 16), np.float32))
    assert out["image"].shape == (16, 16, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_augment.py -v -k depth`
Expected: FAIL — albumentations raises about the unknown `depth` argument (Compose has no such target).

- [ ] **Step 3: Implement in `training/augment.py`**

Change only the final `Compose` construction in `build_augmentation`:

```python
    # depth rides as a mask-type target: geometric transforms carry it (nearest
    # interpolation), photometric ones skip it. Declared unconditionally — a
    # Compose ignores additional targets that are not passed at call time.
    return A.Compose(transforms, seed=cfg.seed,
                     additional_targets={"depth": "mask"})
```

Extend the module docstring line "Geometric transforms carry image+mask" to "image+mask(+depth)".

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_augment.py tests/test_aug_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add training/augment.py tests/test_augment.py
git commit -m "feat(augment): depth as geometric-only additional target"
```

---

### Task 7: End-to-end wiring — TrainConfig, CLI --rgbd, training, export

**Files:**
- Modify: `training/config.py`, `training/run_train.py`
- Test: `tests/test_rgbd_e2e.py` (create)
- Docs: `CLAUDE.md`, `docs/FEATURES.md`

**Interfaces:**
- Consumes: everything above — `build_model(..., in_channels=...)`, `export_to_onnx(..., in_channels=...)`, `StemDataset(..., depth_dir=...)`, `split_ids(..., depth_dir=...)`, `RGBD_IN_CHANNELS`.
- Produces: `TrainConfig.rgbd: bool = False`, `TrainConfig.in_channels` (property: 4 if rgbd else 3), `TrainConfig.depth_dir` / `val_depth_dir` / `gen_depth_dir` / `spec_depth_dir` properties; CLI flag `--rgbd`.

- [ ] **Step 1: Write the failing e2e test** (create `tests/test_rgbd_e2e.py`; model it on `tests/test_train_e2e.py` — read that file first and reuse its synthetic-dataset helper style; write depth files like Task 5's `_write_depth`)

```python
"""RGBD end-to-end: tiny dataset with depth/ -> 1-epoch train -> 4ch ONNX."""
import os

import numpy as np
import onnx
import pytest
from PIL import Image

from training.config import TrainConfig
from training.run_train import run_training
from winmol_unet.contract import RGBD_IN_CHANNELS, validate_onnx_model


def _make_rgbd_dataset(root, n_pairs=4, size=64):
    for sub in ("train", "mask", "depth"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    rng = np.random.default_rng(0)
    for n in range(1, n_pairs + 1):
        img = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(root, "train", f"train{n}.jpeg"))
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        Image.fromarray(mask).convert("P").save(os.path.join(root, "mask", f"mask{n}.gif"))
        depth = ((np.arange(size * size).reshape(size, size) * 13) % 4096).astype(np.uint16)
        Image.fromarray(depth, mode="I;16").save(os.path.join(root, "depth", f"depth{n}.png"))


def test_rgbd_training_exports_4ch_onnx(tmp_path):
    data = str(tmp_path / "ds")
    _make_rgbd_dataset(data)
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=data,
        checkpoint_dir=str(out / "ckpt"), log_dir=str(out / "logs"),
        hdf5_out=str(out / "model.hdf5"), onnx_out=str(out / "model.onnx"),
        pt_out=str(out / "model.pt"),
        rgbd=True, epochs=1, batch_size=2, device="cpu", seed=1,
    )
    run_training(cfg)
    model = onnx.load(cfg.onnx_out)
    validate_onnx_model(model, in_channels=RGBD_IN_CHANNELS)


def test_rgbd_rejects_export_keras(tmp_path):
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path), log_dir=str(tmp_path),
        hdf5_out="x.hdf5", onnx_out="x.onnx", rgbd=True, export_keras=True,
    )
    with pytest.raises(ValueError, match="rgbd"):
        run_training(cfg)


```

Adjust the reject-test to whatever `run_training` actually validates first (it must fail *before* touching the filesystem — that is the point of fail-fast). If `run_training`'s signature differs (e.g. it takes no return), match the real call shape from `tests/test_train_e2e.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rgbd_e2e.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'rgbd'`.

- [ ] **Step 3: Implement `training/config.py`**

Add field (with the other bools, near `multiscale`):

```python
    rgbd: bool = False                       # 4-channel RGBD input; needs depth/ in each dataset dir
```

Add properties (next to the existing `*_dir` properties):

```python
    def _depth_for(self, data_dir):
        return os.path.join(data_dir, "depth") if (self.rgbd and data_dir) else None

    @property
    def in_channels(self) -> int:
        from winmol_unet.contract import IN_CHANNELS, RGBD_IN_CHANNELS
        return RGBD_IN_CHANNELS if self.rgbd else IN_CHANNELS

    @property
    def depth_dir(self) -> Optional[str]:
        return self._depth_for(self.data_dir)

    @property
    def val_depth_dir(self) -> Optional[str]:
        return self._depth_for(self.val_data_dir)

    @property
    def gen_depth_dir(self) -> Optional[str]:
        return self._depth_for(self.gen_data_dir)

    @property
    def spec_depth_dir(self) -> Optional[str]:
        return self._depth_for(self.spec_data_dir)

    @property
    def test_depth_dir(self) -> Optional[str]:
        return self._depth_for(self.test_data_dir)
```

- [ ] **Step 4: Implement `training/run_train.py`**

1. Fail fast — extend the existing `_validate_export(cfg)` (which already rejects `export_keras` for non-unet archs):

```python
    if cfg.rgbd and cfg.export_keras:
        raise ValueError("--export-keras is RGB-only (rgbd=True); drop one of the two")
```

2. Thread `depth_dir` through `_build_loaders` — signature becomes:

```python
def _build_loaders(image_dir, mask_dir, cfg, transform, val_image_dir=None, val_mask_dir=None,
                   depth_dir=None, val_depth_dir=None):
```

Inside `_build_loaders`: the fixed-split branch passes `depth_dir=depth_dir` to the train `StemDataset(...)` and `depth_dir=val_depth_dir` to the val `StemDataset(...)`; the else-branch passes `depth_dir=depth_dir` to `train_val_split(...)`.

3. Call sites:

```python
    # single-stage (run_training):
    ..._build_loaders(cfg.image_dir, cfg.mask_dir, cfg, transform,
                      val_image_dir=cfg.val_image_dir, val_mask_dir=cfg.val_mask_dir,
                      depth_dir=cfg.depth_dir, val_depth_dir=cfg.val_depth_dir)
    # two-stage (run_two_stage):
    ..._build_loaders(cfg.gen_image_dir, cfg.gen_mask_dir, cfg, transform,
                      depth_dir=cfg.gen_depth_dir)
    ..._build_loaders(cfg.spec_image_dir, cfg.spec_mask_dir, cfg, transform,
                      depth_dir=cfg.spec_depth_dir)
    # test stage (_run_test builds its StemDataset directly — add the kwarg):
    test_ds = StemDataset(os.path.join(cfg.test_data_dir, "train"),
                          os.path.join(cfg.test_data_dir, "mask"), cfg.img_size,
                          transform=None, cache=cfg.cache_dataset,
                          depth_dir=cfg.test_depth_dir)
```

4. Model build + export — wherever `build_model(...)` and `export_to_onnx(...)` are called, add `in_channels=cfg.in_channels`.

5. CLI (in the argparse block, near `--arch`):

```python
    p.add_argument("--rgbd", action="store_true",
                   help="4-channel RGBD input; each dataset dir needs depth/depth{N}.png|.tif")
```

and `rgbd=a.rgbd` where the `TrainConfig(...)` is constructed.

- [ ] **Step 5: Run the new tests, then the neighboring suites**

Run: `pytest tests/test_rgbd_e2e.py -v` → PASS.
Run: `pytest tests/test_train_e2e.py tests/test_two_stage.py tests/test_val_data_dir.py tests/test_test_stage.py tests/test_train_config.py -v` → PASS (RGB paths untouched). These train small models — expect minutes.

- [ ] **Step 6: Update docs**

- `CLAUDE.md` → "Dataset convention + scale" paragraph: add one sentence — `--rgbd` adds an optional `depth/depth{N}.png|.tif` channel (per-image min-max normalized, geometric-aug only), producing 4-channel models; Keras export is RGB-only.
- `docs/FEATURES.md` → move "### 4 D input data" out of Backlog (or annotate it "implemented — see `docs/superpowers/plans/2026-07-29-rgbd-input.md`"), noting the analyzer must feed 4-channel NHWC to consume RGBD models.

- [ ] **Step 7: Run the full suite**

Run: `pytest`
Expected: PASS (slow — several architectures train).

- [ ] **Step 8: Commit**

```bash
git add training/config.py training/run_train.py tests/test_rgbd_e2e.py CLAUDE.md docs/FEATURES.md
git commit -m "feat: --rgbd end-to-end (config, loaders, training, 4ch ONNX export)"
```
