# Reder's method: gaps closed from source, and where the docs disagree with the code

The methodological report on Reder et al. 2022 / 2025 lists items "NOT recoverable from
openly accessible sources" and points at `WINMOL_segmentor` (R) and `WINMOL_Analyzer`
(Python) as the authoritative places to read them. Both were read directly. This closes
what is closeable and flags three places where the published description and the code
disagree.

Every value below is quoted from source or measured by running it, with the file and line.

---

## 1. Hyperparameters — closed

| item | value | source |
|---|---|---|
| optimizer | Adam, `lr = 1e-3`, β₁ 0.9, β₂ 0.999, `decay = 0`, `amsgrad = FALSE` | `controlling.R` |
| batch size | **4** | `main_training.R` |
| epochs | 100 cap per stage | `training.R` |
| early stopping | `val_loss`, patience **3** (stage 1) / **5** (stage 2) | `callbacks.R:15-17` |
| LR schedule | ReduceLROnPlateau, factor 0.1, patience 2, `min_delta = 1e-4` | `callbacks.R:7-12` |
| checkpoint | `save_best_only = TRUE` on `val_loss` | `callbacks.R:6` |
| augmentation | flips + brightness/contrast/saturation/hue, train split only | `input_pipeline.R` |

## 2. Split construction — closed, and it is not site-level

```r
train_data <- initial_split(train_data, prop = 0.8)   # training.R, both stages
```

A **random 80/20 split over tiles**, drawn independently per stage. Not spatial, not
site-level. The report infers a site-aware evaluation from the 15-site design; the training
code does not implement one. With the generator's 100× oversampling, tiles overlap, so this
validation signal is optimistic — which matters because early stopping, the LR schedule and
checkpoint selection all read it.

## 3. The composite loss does not train — measured

Both papers present the F1/BCE composite as the key choice for foreground/background
imbalance. In code:

```r
F1Score <- custom_metric("F1Score", function(y_true, y_pred) {
  y_pred <- k_round(y_pred)          # <- hard threshold
  ... })
F1Score_loss <- function(y_true, y_pred) {
  loss_binary_crossentropy(y_true, y_pred) + (1 - F1Score(y_true, y_pred)) }
```

`F1Score_loss` calls the **metric**, and the metric rounds. `k_round` has zero derivative
almost everywhere, so the F1 term contributes no gradient: **the model trains on plain
binary cross-entropy.** The term is real in the reported loss value and absent from the
optimisation.

Verified three independent ways:

- **In torch** — R's expression transcribed gives a gradient `torch.equal` to plain BCE's
  (grad-norm 0.012018 for both), while the reported value is ~2× larger.
- **In R itself** — `LOSS=f1` (Stefan's untouched `controlling.R`) against
  `loss_binary_crossentropy` alone, same seed, 8 epochs: precision, recall and F1 agree to
  **eight decimal places** at every epoch. Only `loss` differs, by exactly `1 − F1`.
- **Ablated** — replacing it with a *differentiable* soft-F1 term, paired over seeds,
  shifts recall **+3.8** and precision **−2.5** (4/4 seeds, paired t 4.86 / −5.47). So the
  term does something once it has a gradient; in the published form it does nothing.

**One important qualification** (re-verified 2026-08-19). "The term does nothing" is too
strong. It contributes no *gradient*, so the optimisation is pure BCE — confirmed again in
torch, where the composite and plain-BCE gradients are bit-identical (max difference exactly
0.0), while a *differentiable* soft-F1 term does change them (2.6e-4), so the test is
sensitive enough to detect a real difference.

But the composite **value** is what Keras monitors: `save_best_only` on `val_loss`,
`ReduceLROnPlateau` on `val_loss`, and early stopping all read `BCE + (1 − F1)`. That ranks
epochs differently from BCE alone, so **which checkpoint is kept and when the learning rate
drops can differ**, even though every training step is identical. The term is inert in the
optimiser and active in model selection.

This does not invalidate the results — BCE is a reasonable loss — but the stated rationale
for the loss choice is not what the model experienced.

## 4. Architecture — a transposed digit

`model_UNet.R:87,91`, decoder stage E7:

```r
layer_conv_2d(filters = 265, ...)    # every other stage is symmetric: 512, 256, 128, 64
```

The transpose conv above it correctly emits 256. Both convs of that block use **265**. The
published `.hdf5` loads at **31,140,642** parameters; the symmetric ladder gives
31,036,673. Harmless numerically, but "31M U-Net" is not reproducible from the paper's
description without the typo.

## 5. Instance-extraction defaults — closed

`classes/Config.py`:

| parameter | default | GUI label in the report |
|---|---:|---|
| `min_length` | **2.0** m | Min Length |
| `max_distance` | **8** | Max Distance |
| `tolerance_angle` | **7** | Maximum Angle |
| `max_tree_height` | **32** m | Maximum Tree Height |
| `tile_size` | **15** m | Tile side length |
| `img_width` | **512** px | Tile pixel extent |
| `overlap_pred` | 8 | — |
| `stem_binary_threshold` | 0.5 | — |

**Tile side length is 15 m, not ≈15.4 m.** The report derives 15.36 m from 3 cm × 512 px;
the code fixes 15 m, giving an effective GSD of **15/512 = 2.93 cm/px**. That number is
load-bearing: matching training GSD to it was worth **+14.6 F1** in our own measurements.

**Skeletonisation** (`utils/Skeletonization.py`): `skimage.morphology.skeletonize` on the
binary mask, after padding by `int(max_tree_height / px_size) + 1`. Endpoints and
branchpoints are found by crossing-number (transitions == 1 endpoint, ≥ 3 branchpoint);
branchpoints are removed, dense nodes eroded, then segments refined to the measuring-point
spacing with a minimum length of `floor((min_length/4) / px_size)`.

**Diameter**: two methods, `diameter_method = "contour"` (default) or `"edt"`, measured on a
local normal (`_local_normal`, `_measurement_vector`, `diameter_vector_half_length_m = 1.0`)
— so "perpendicular to the centerline" is confirmed.

**Volume**: `calc_l_v(p1, p2, d1, d2)`, truncated cone between consecutive nodes. Confirmed.

## 6. Node spacing is 50 cm, not 25 cm

The WINMOL documentation says the diameter is determined "every 25 cm". The code says:

```python
measuring_point_spacing_m = 0.5                          # Config.py:104
measuring_point_spacing = math.floor(
    min(config.min_length, config.measuring_point_spacing_m) / px_size)   # Skeletonization.py
```

`min(2.0, 0.5) = 0.5` → **50 cm**. Measured on real output across four orthomosaics, the
mean gap between consecutive centreline vertices is **0.47–0.50 m**. Either the docs
describe a different version or the figure is wrong; the shipped default is 50 cm.

## 7. Two coordinate traps in the vectoriser

**Re-checked 2026-08-19 against the current `winmol-oom-fix` version: both are still
present.** Both produce plausible geometry rather than an error, so they are invisible
without checking:

- **The skeletonisation pad is never removed.** `find_segments` pads by
  `max_tree_height / px_size` and returns coordinates still carrying that offset — 1530 px
  at 2.09 cm/px. The comment at `Skeletonization.py:175` claims the offset is removed;
  `np.argwhere` returns raw padded indices.
- **Paths are pixel `(row, col)`, not world coordinates**, and `path.length` is a pixel
  count. World coordinates are applied only at write time. `calc_v_d_edt` passes
  `stem.path.coords` to `_xy_to_rowcol`, which reads them as easting/northing — so the
  **EDT diameter branch appears broken**; the default contour path is self-consistent in
  pixel space. The current version adds a bounds check returning `radius = 0.0` when the
  bogus index falls outside the map, so the bug now fails **silently as zero diameters**
  rather than raising — harder to notice, not easier.

## 8. CLI: `Stems` does not vectorise

`run_stem_pipeline` calls the prediction phase and stops; only `run_tree_pipeline`
(`Trees`) runs `run_vector_phase`. A `Stems` run exits cleanly with a stem raster and no
vectors — which reads as a failure and is not one.

---

## What this changes in the report's recommendations

**Stage 1 (data).** Target the code's actual scale — 15 m / 512 px = 2.93 cm/px — rather
than "3 cm". Our measurement: serving a model at a GSD it was not trained on cost 14.6 F1,
and the same published beech model scores **0.8003** on native-512 tiles versus **0.6812**
on Reder's own 313 px TestDS (upsampled ×1.64 to reach the input). Resolution handling, not
model quality, explains a 12-point swing.

**Stage 2 (splits).** The report's advice to use site-level holdout is right and the code
does not do it. Worth knowing what it costs: in our corpus a whole-site holdout costs 8–11
F1, and one site (November beech canopy, 100% amber pixels) returns F1 **exactly 0.0000** —
so a leave-one-site-out mean can be carried entirely by one dead fold. Report per-fold.

**Stage 2 (copy-paste + two-stage).** GenDS tiles are 2000×2000 but carry ~500–650 px of
real information — round-trip loss through 512 is 1.45 grey levels at 2× against SpecDS's
3.73, and edge energy is 7.97/px against 15.52. Effective GSD ≈ 4.6 cm/px, which matches
SpecDS's 4.8 cm/px, so the two stages are well aligned with each other but not with the
Analyzer default. On our own corpus, synthetic pretraining measured **−0.2 to −1.4** once a
validation-split bug was fixed (it had read +1.1 to +3.6 before).

**Stage 4 (evaluation).** Pixel F1 is not comparable to DR25 — different quantity. Also fix
a noise floor before comparing anything: two runs with **bit-identical gradients** landed
2.0 F1 apart without deterministic kernels, 0.5–1.0 with. Single-run differences below that
are unreadable, which is why the ablations above are paired over 5 seeds.

## A reproduction worth noting

The 2022 paper reports pixel F1 **72.6%** for SpecDS-only training and **75.6%** for its
best two-stage pretrained model, on TestDS.

Trained single-stage on the same published SpecDS, scored on the same published TestDS with
our own metric code: **0.7581**. That matches the paper's *best pretrained* result without
any GenDS pretraining. The R U-Net retrained under the same conditions gives 0.7424.

Two caveats. The published **beech** model (`SpecDS_Beech_512`, 2023-02-28) scores only
**0.6812** on that same TestDS — below the 2022 paper's figure, unexplained, and worth
raising with the author. And TestDS as distributed on Zenodo contains **117** tiles, not the
106 stems the paper describes (verified by md5 against the record).

---

*Sources read directly: `WINMOL_segmentor` — `controlling.R`, `training.R`, `callbacks.R`,
`main_training.R`, `model_UNet.R`, `input_pipeline.R`. `WINMOL_Analyzer` — `Config.py`,
`Skeletonization.py`, `Vectorization.py`, `Quantification.py`, `winmol_run.py`. Datasets
verified byte-identical against Zenodo record 5682580.*
