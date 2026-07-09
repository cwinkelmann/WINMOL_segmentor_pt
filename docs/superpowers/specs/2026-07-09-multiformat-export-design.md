# Multi-format Export (pt + hdf5 + keras + onnx) — Design Spec

**Date:** 2026-07-09
**Status:** Approved (FEATURES.md #4)
**Repo:** `WINMOL_segmentor_pt`

## 1. Goal

Make the trainer export a trained model in **four** formats: PyTorch `.pt`, Keras `.hdf5`
(legacy), Keras native `.keras`, and `.onnx`. `.hdf5` and `.onnx` already exist; add `.pt`
and native `.keras`.

## 2. Scope

- Add native **`.keras`** export (the modern replacement for legacy `.hdf5`; hdf5 stays).
- Add **`.pt`** export (trained `state_dict`, reloadable into `winmol_unet.model.UNet`).
- Wire all four into the training CLI; each artifact path derives from `--out-dir`.
- **Out of scope:** the analyzer's `.keras`/`.pt` *loader* (deferred — the analyzer's
  `keras.models.load_model` already supports `.keras`, but its extension branch is a
  separate change). No new architectures.

## 3. Components

- **`winmol_unet/export_keras.py`** — extract the shared build+weight-transfer into
  `_build_and_transfer(torch_model, dropout=0.1) -> keras.Model`. Keep
  `export_to_keras_hdf5(torch_model, path, dropout=0.1)` (saves legacy `.hdf5`). Add
  `export_to_keras(torch_model, path, dropout=0.1)` → builds via the shared helper and
  `keras_model.save(path)` in **native Keras format** (path ends in `.keras`). Both remain
  numerically equivalent to the PyTorch model.
- **`winmol_unet/export.py`** — add `export_to_pt(model, path) -> path`:
  `torch.save(model.state_dict(), path)`. Reloadable via `UNet(); m.load_state_dict(...)`.
- **`training/config.py`** — add `keras_out: str` and `pt_out: str` (alongside the existing
  `hdf5_out`, `onnx_out`).
- **`training/run_train.py`** — after training (model on CPU), export `.pt`, `.hdf5`,
  `.keras`, `.onnx`; CLI derives `model.pt/.hdf5/.keras/.onnx` under `--out-dir`.

## 4. Data flow

trained `UNet` (CPU) → `export_to_pt` (state_dict) / `export_to_keras_hdf5` /
`export_to_keras` / `export_to_onnx` → four files under `out-dir`.

## 5. Testing

1. **`.keras` parity** — `export_to_keras(UNet, path)`; `keras.models.load_model(path,
   compile=False)` predicts NHWC `[N,512,512,1]` in [0,1], matching
   `torch.sigmoid(unet(x))` within **atol=1e-4** (mirrors the hdf5 parity test).
2. **`.pt` round-trip** — `export_to_pt(model, path)`; a fresh `UNet().load_state_dict(
   torch.load(path))` reproduces the original model's output exactly.
3. **shared helper** — `export_to_keras_hdf5` and `export_to_keras` still each load and match
   torch (no regression from the refactor).
4. **e2e** — `run_training` writes all four files (`.pt`, `.hdf5`, `.keras`, `.onnx`).

## 6. Constraints

- Python 3.9; native `.keras` requires TF/Keras ≥ 3 (present: TF 2.20). Legacy `.hdf5` still
  works in TF 2.20/2.21.
- Export runs on CPU (exporters read weights via numpy); model is moved to CPU first.
- Keras mirror stays standard-layers-only so both `.hdf5` and `.keras` load without
  `custom_objects`.
