# WINMOL Segmentor → PyTorch + ONNX — Design Spec

**Date:** 2026-07-02
**Status:** Approved (brainstorming complete; implementation plan to follow)
**Repo:** `WINMOL_segmentor_pt` (this repo)

## 1. Problem & Goal

The WINMOL tree-stem segmentation U-Net is currently trained in **R + Keras/TensorFlow**
(`WINMOL_segmentor`, script-based) and consumed for inference by the Python
**`WINMOL_Analyzer`**, which loads `.hdf5` models via `keras.models.load_model(...)` and
runs `model.predict_on_batch(tile)`.

**Goal:** re-implement the segmentor (training + model) in **PyTorch**, and make trained
models consumable by the analyzer via **ONNX** — while requiring **as few changes as
possible** in the analyzer and **keeping TensorFlow working** there (existing `.hdf5`
models must continue to load unchanged).

## 2. Decisions (locked during brainstorming)

1. **Bridge = ONNX.** Training moves to PyTorch; models are exported to ONNX. The analyzer
   gains an `onnxruntime` inference path.
2. **TensorFlow stays in the analyzer.** ONNX support is *additive* (extension-based
   branch). The 4 existing HDF5 models keep running via Keras. Legacy HDF5→ONNX conversion
   is **optional / deferred**.
3. **Full training-pipeline parity** with the R segmentor (two-stage GenDS→SpecDS training,
   jpeg/gif data pipeline, augmentation, F1 loss + precision/recall/F1 metrics,
   checkpoint / early-stop / TensorBoard logging).
4. **Shared installable package** (`winmol_unet`) lives in *this* repo and is the single
   source of truth for the model definition + ONNX contract; the analyzer pip-installs it.

## 3. The ONNX Contract (`winmol_unet/contract.py`)

The interface both repos are pinned to. Frozen as constants + validators.

| Property | Value |
|---|---|
| ONNX graph layout | **NCHW**: input `[batch, 3, 512, 512]` → output `[batch, 1, 512, 512]` |
| Batch axis | **dynamic** (analyzer auto-tunes micro-batch size + OOM backoff) |
| Spatial size | **fixed 512×512** (matches analyzer `img_width/height=512`) |
| dtype / range | `float32`, RGB, normalized `[0,1]` (analyzer already divides by 255) |
| Final activation | **sigmoid baked into the ONNX graph** (matches current Keras models) |
| opset | **17** (pinned) |

**NHWC⇄NCHW adaptation happens inside `OnnxSegmenter.predict_on_batch`, not in the graph**,
so the analyzer keeps thinking in NHWC and its `pred[idx,:,:,0]` indexing is unchanged.

**Wrapper contract — `OnnxSegmenter.predict_on_batch(x)`:**
- **in:** NHWC `float32 [N, 512, 512, 3]` in `[0,1]` — exactly what the analyzer's
  `_prepare_inference_batch` already produces (a TF tensor is accepted and `.numpy()`-ed
  internally, since TF stays present).
- **out:** NHWC `float32 [N, 512, 512, 1]` sigmoid probabilities — exactly what the
  downstream crop/threshold loop expects.

## 4. Architecture — Components

### 4.1 Shared package `winmol_unet/` (analyzer depends on this)
- `model.py` — PyTorch U-Net; single source of truth for the architecture.
- `contract.py` — ONNX I/O contract constants + validation helpers (Section 3).
- `preprocess.py` — resize/normalize used by **both** training and inference (eliminates
  train/inference skew for new models).
- `export.py` — `export_to_onnx(torch_model, path)`; writes a contract-conformant ONNX file
  (sigmoid baked in, dynamic batch, fixed 512, opset 17).
- `runtime.py` — `OnnxSegmenter`: wraps `onnxruntime.InferenceSession`; exposes
  `.predict_on_batch()`, `.summary()`, and normalizes ONNX OOM to the exception type the
  analyzer's retry loop catches. Selects execution provider from `config.prediction_backend`.

### 4.2 Training-only `training/` (analyzer never imports)
- `config.py` — paths + hyperparameters (mirrors R `controlling.R` / `environment.R`).
- `datasets.py` — `Dataset` reading jpeg images + gif masks (first frame); **paired**
  geometric augmentation (flips, shared seed for img+mask) + photometric augmentation
  (brightness/contrast/saturation/hue on img only); normalize `/255`; resize via shared
  `preprocess`.
- `losses.py` — `BCE + (1 − soft_F1)` (see §6 note).
- `metrics.py` — precision / recall / F1 (hard-rounded, for reporting).
- `callbacks.py` — checkpoint-best-`val_loss`, `ReduceLROnPlateau`, early-stopping
  (patience 3 for stage 1, 5 for stage 2), TensorBoard via `torch.utils.tensorboard`.
- `train.py` — two-stage GenDS→SpecDS CLI; ends by calling `export_to_onnx`.
- `evaluate.py` — evaluation + side-by-side mask/image/prediction visualization.

### 4.3 Optional / deferred
- `scripts/convert_hdf5_to_onnx.py` — one-time legacy HDF5→ONNX via `tf2onnx`, parity-checked.
  Only needed if/when TF is retired from the analyzer.

### 4.4 Repo layout
```
WINMOL_segmentor_pt/
  pyproject.toml               # package "winmol_unet"; [train] extra for training deps
  winmol_unet/                 # SHARED — analyzer installs this subset
    model.py  contract.py  preprocess.py  export.py  runtime.py
  training/
    config.py  datasets.py  losses.py  metrics.py  callbacks.py  train.py  evaluate.py
  scripts/convert_hdf5_to_onnx.py
  tests/
  docs/
  README.md
```

## 5. Analyzer changes (`WINMOL_Analyzer`) — additive, minimal

TF stays; existing HDF5 path untouched. We only *add* an ONNX branch.

| Concern | Change |
|---|---|
| Model loading (`utils/IO.py:189/208/216`, `standalone/WINMOL_Analyzer.py:38`) | Branch on extension: `.hdf5/.h5` → existing `keras.models.load_model` (unchanged); `.onnx` → `OnnxSegmenter`. Centralize in `IO.py` loader helper. |
| Preprocessing (`Prediction.py:134-150`, `PredictWorkers.py:51-68`) | **Unchanged.** `OnnxSegmenter.predict_on_batch` accepts what `_prepare_inference_batch` returns. |
| OOM retry (`Prediction.py:295`) | Unchanged; wrapper raises a caught exception type on ONNX OOM. |
| GPU/device setup (`winmol_run.py`, `PredictWorkers.py`) | Unchanged for TF; `OnnxSegmenter` selects its own execution provider internally. |
| Deps (`requirements/*`) | **Add** `onnxruntime(-gpu)`; keep `tensorflow`. |

Net analyzer diff ≈ one extension branch in `IO.py` + a dependency line. `OnnxSegmenter`
duck-types the Keras model object, so the swap hides behind one adapter and is opt-in per
model file.

## 6. Faithful-port note — the F1 loss gradient

The R loss is `BCE + (1 − F1Score)`, but `F1Score` applies `k_round(y_pred)` — rounding
zeroes the gradient of the `(1 − F1)` term, so in the R code BCE alone effectively trains
the network. In PyTorch we use a **soft (un-thresholded) F1** inside the *loss* so that term
is genuinely differentiable and useful, while keeping the **hard-rounded** F1 for *reported
metrics*. This is a deliberate, documented improvement over a bug-for-bug port. If exact
original behavior is required instead, replace with BCE-only + detached hard-F1 metric.

## 7. Model geometry note

The R `model_UNet.R` used `IMG_width=256`, but the production models and the analyzer config
use **512**. The PyTorch U-Net targets **512×512×3** input / **512×512×1** sigmoid output to
match deployed models and the analyzer. (Also note the R model contains a probable typo:
block E7 uses `filters = 265` where `256` was intended — the PyTorch port uses 256.)

The port is a **modernized adaptation, not a layer-exact reproduction** of `model_UNet.R`.
Same skeleton (19 Conv, 4 ConvTranspose, 64→128→256→512→1024 ladder, dropout 0.1, ~31M
params, skip concats, sigmoid head), with four deviations — two intentional/geometric (input
512, E7 256-not-265 above) and two structural:

1. **Normalization order.** Port uses **Conv→BN→ReLU**; R uses **Conv→ReLU→BN** (ReLU fused
   into `layer_conv_2d`, then `layer_batch_normalization`). Both are common.
2. **No BN after ConvTranspose.** R inserts a BatchNorm after each of the 4 up-convolutions
   (before the skip concat); the port omits them (**18 BN vs R's 22**).

These do not change the model family or capacity; on real data (beech `TestDS`, 117 pairs,
20 epochs) the port reaches **val F1 ≈ 0.72**. `winmol_unet.keras_model` mirrors *this*
architecture (not R's), keeping the PyTorch↔HDF5/ONNX exports numerically equivalent.

## 8. Verification & Tests (TDD — tests written first)

1. **Contract test** — exported ONNX has input `[N,3,512,512]`, output `[N,1,512,512]`,
   dynamic batch, opset 17, sigmoid terminal.
2. **Export parity (core gate)** — same random input: PyTorch vs. exported ONNX
   (onnxruntime) agree within `atol ≈ 1e-4`.
3. **Runtime wrapper test** — `OnnxSegmenter.predict_on_batch(NHWC)` returns NHWC
   `[N,512,512,1]`, matches raw onnxruntime NCHW run after transpose; `.summary()` works;
   simulated OOM raises the analyzer-caught exception.
4. **Analyzer integration** (populates `WINMOL_Analyzer/tests/`) — `IO.py` extension branch
   loads a tiny `.onnx` and runs a synthetic tile through `predict_with_resampling_per_tile`;
   smoke test confirms the `.hdf5` path still loads (no regression).
5. **Training smoke test** — overfit a 2-tile batch, assert loss decreases; fixed seed.
6. **End-to-end** — run `standalone/WINMOL_Analyzer.py` on a small orthophoto with a new
   ONNX model → produces stem map + GeoJSON without error.
7. **Legacy parity (deferred)** — when converting the 4 HDF5 models: Keras vs converted-ONNX
   output within tolerance on sample tiles.

## 9. Out of scope (YAGNI)

- Retiring TensorFlow from the analyzer (kept indefinitely; conversion script is optional).
- Non-512 tile sizes / dynamic spatial dimensions.
- Rewriting the analyzer's tiling / vectorization / quantification logic (untouched).
- New model architectures beyond the ported U-Net.

## 10. Cross-repo contract dependency

`WINMOL_Analyzer` pip-installs `winmol_unet` from this repo (editable during dev). Any change
to `contract.py` is a breaking change requiring coordinated updates + re-running the parity
suite in both repos.