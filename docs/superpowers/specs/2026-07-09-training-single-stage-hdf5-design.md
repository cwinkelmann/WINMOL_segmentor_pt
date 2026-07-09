# Single-Stage PyTorch Training + HDF5 Export — Design Spec

**Date:** 2026-07-09
**Status:** Approved (brainstorming complete)
**Repo:** `WINMOL_segmentor_pt`
**Supersedes for now:** the two-stage/ONNX training-port plan
(`docs/superpowers/plans/2026-07-02-training-port.md`) — this is a scoped-down,
HDF5-first first cut. Two-stage GenDS→SpecDS and ortho prediction are deferred.

## 1. Goal

Give this repo the ability to **train** the PyTorch `winmol_unet.UNet` on a real
image/mask dataset and export the trained model as a **Keras `.hdf5`** that drops into
`WINMOL_Analyzer` unmodified (`keras.models.load_model(path, compile=False)` →
`predict_on_batch`). This is the capability the repo is meant to provide; the export +
drop-in half is already built and verified (`winmol_unet.export_keras`).

## 2. Scope (locked)

- **Single-stage** training on one dataset (no two-stage GenDS→SpecDS yet).
- **HDF5** is the primary export artifact; ONNX exported alongside for free.
- Validation is **held-out-tile evaluation** (F1/precision/recall) + HDF5 drop-in parity,
  standing in for `main_prediction.R` (we have tiles, not a georeferenced orthomosaic).
- `training/` is **never imported by the analyzer**; it depends on `winmol_unet`.

## 3. Dataset (verified from `/Users/christian/data/Winmol/data/TestDS`)

- Layout: `train/trainN.jpeg` (RGB uint8) and `mask/maskN.gif` (mode L, single frame,
  binary {0,255}). 117 complete pairs, no missing masks.
- Pairing: image `trainN.jpeg` ↔ mask `maskN.gif` by the same integer `N` (prefix swap).
- Images are **not 512** (sample 313×313, may vary) — must resize to 512.
- Preprocessing (matches R + `winmol_unet.preprocess`): resize to 512 — **bicubic** for
  image, **nearest** for mask — then image `/255` → `[0,1]`; mask `/255` → binarize to
  {0,1}. Loader reads gif **frame 0**, converts to L.

## 4. Components (`training/`, each independently testable)

- `config.py` — `TrainConfig` dataclass: `data_dir` (contains `train/` + `mask/`),
  `checkpoint_dir`, `log_dir`, `hdf5_out`, `onnx_out`, `batch_size=4`, `epochs=100`,
  `lr=1e-3`, `dropout=0.1`, `img_size=512`, `val_fraction=0.2`, `patience=5`, `seed=1`.
- `dataset.py` — `StemDataset(image_dir, mask_dir)` yielding CHW float32 tensors in
  `[0,1]` and 1×H×W binary mask. Pairs by integer N. Paired **geometric** augmentation
  (random horizontal/vertical flip, shared seed for image+mask); **photometric**
  augmentation (brightness/contrast/saturation/hue) on the **image only**. A
  `train_val_split(seed, val_fraction)` helper gives a deterministic 80/20 split with
  augmentation on train only.
- `losses.py` — `bce_soft_f1_loss(logits, target)` = `BCEWithLogitsLoss` +
  `(1 − soft_F1)`, soft-F1 computed on `sigmoid(logits)` (un-thresholded, differentiable).
- `metrics.py` — `precision/recall/f1(logits, target)` hard-rounded (sigmoid + 0.5).
- `train.py` — `train_one_run(model, train_loader, val_loader, cfg)`: Adam, per-epoch
  train/val loss + metrics, checkpoint-best-`val_loss` (torch state_dict), early stop on
  `patience`, TensorBoard logging via `torch.utils.tensorboard`. Returns best model.
- `evaluate.py` — `evaluate(model, val_loader)` → dict of mean precision/recall/F1.
- `run_train.py` — CLI: build dataset from `TrainConfig`, split, `train_one_run`, reload
  best checkpoint, `export_to_keras_hdf5(model, cfg.hdf5_out)` and
  `export_to_onnx(model, cfg.onnx_out)`, print final val metrics.

## 5. Data flow

jpeg+gif → `StemDataset` (resize 512, normalize, augment) → `DataLoader` →
`UNet` (logits) → `bce_soft_f1_loss` ← mask; best checkpoint → reload → `export_to_keras_hdf5`
→ `.hdf5`. Validation: best model → `evaluate` on val tiles → F1/P/R; and
`keras.models.load_model(hdf5, compile=False).predict_on_batch(val_tiles_NHWC)` compared to
`torch.sigmoid(model(x))` within **atol=1e-4**.

## 6. Testing

1. **Dataset test** — pairs resolve by N; output shapes CHW/1HW at 512, image∈[0,1], mask∈{0,1};
   paired flip keeps image/mask aligned; photometric aug leaves mask untouched.
2. **Loss/metric tests** — perfect prediction → loss≈BCE-floor and F1≈1; soft-F1 term is
   differentiable (grad flows); metrics match hand-computed values on a toy tensor.
3. **Overfit smoke test** — 2–4 real tiles, few dozen steps, assert training loss strictly
   decreases (fixed seed).
4. **End-to-end (the `main_prediction.R` stand-in)** — short real run on TestDS (few epochs,
   subset), export HDF5, load `compile=False`, `predict_on_batch` on held-out tiles →
   assert output shape `(N,512,512,1)`, range [0,1], and torch-vs-keras parity <1e-4.
   Marked slow; runs on a small subset to stay CI-friendly.

## 7. Out of scope (YAGNI)

- Two-stage GenDS→SpecDS training (documented follow-up).
- Georeferenced orthomosaic tiling / GeoTIFF output (the analyzer already does this;
  literal `main_prediction.R` port deferred until an orthophoto is available here).
- Hyperparameter search; multi-GPU; resuming R/Keras training.

## 8. Faithful-port note

The R loss `BCE + (1 − F1)` rounds `y_pred` inside F1, zeroing that term's gradient (BCE
effectively trains the net). We use a **soft** F1 in the loss so the term is genuinely
differentiable, while reporting **hard-rounded** F1 as a metric — a deliberate, documented
improvement (see the 2026-07-02 design spec §6).
