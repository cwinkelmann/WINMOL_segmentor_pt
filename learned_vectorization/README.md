# Learned stem vectorization — a distillation study

**Self-contained mini-project** (branch `feat/learned-vectorization`). It does not touch the
`winmol_unet` package or the training pipeline; it only *reuses* the UNet backbone as an option.

## The point being proved

The WINMOL Analyzer turns a UNet stem-**mask** into vector **stems** (centerline polylines +
per-node **diameter** profile → length, volume) with a hand-tuned heuristic (skeletonize → graph
→ angle-vote connect → measure diameter off the mask). We ask: **can a neural network learn that
mask→graph step, trained only on the heuristic's own output?**

**Thesis:** it can *reproduce* the heuristic, but it **cannot exceed** it — because the training
labels *are* the heuristic. This is a self-distillation ceiling. The value of the learned form is
therefore not accuracy but: no tuned thresholds, differentiability, single forward pass, and
graceful behaviour where the heuristic is brittle (crossings, tile seams). The study's job is to
**demonstrate the ceiling honestly**, not to beat the teacher.

## Approach A — dense fields + light trace

Predict, per pixel, from the mask (optionally + RGB):
- **centerline heatmap** — Gaussian ridge along each stem centerline,
- **orientation field** — tangent as (sin 2θ, cos 2θ) (undirected),
- **diameter map** — stem width in metres at the centerline.

Then a minimal decoder traces polylines along the ridge following the orientation field, sampling
the diameter map. Orientation disambiguates crossings; the learned ridge bridges skeleton gaps.

## Build order (each stage gates the next)

0. **Representation round-trip (NO training, do first):** `gpkg → fields → decode → polylines`,
   compared to the original `gpkg` (stem count, endpoint distance, length/diameter/volume error).
   This bounds the *whole approach* independent of the network. If the GT fields don't round-trip,
   stop and fix the representation/decoder.
1. **Target generation** (`targets.py`): gpkg + raster grid → the three field rasters. Pure
   rasterization math is numpy-testable; geo-IO (gpkg read + geotransform) is integration-tested.
2. **Model** (`model.py`): shared backbone + 3 heads (heatmap / orientation / diameter).
3. **Decoder** (`decode.py`): fields → polylines + diameters.
4. **Train + eval** (`train.py`, `eval.py`): distill on (mask→fields) pairs; eval = model-vs-heuristic
   (expected: approaches, does not exceed) and, if any field data exists, both-vs-field-truth.

## Results

**Step 0 — representation round-trip (DONE, no training).** `round_trip.py` on the real 215-stem
Barnekow gpkg (grid 0.1 m/px, sigma 1.0): decode 0.6 s.

| metric | heuristic (in) | round-trip (out) | ratio |
|--------|---------------:|-----------------:|------:|
| stems | 215 | 218 | 1.01 |
| total length (m) | 1715 | 1806 | 1.05 |
| mean diameter (m) | 0.228 | 0.229 | 1.00 |
| total volume (m³) | 75.9 | 80.0 | 1.05 |

The dense-field representation recovers stem count + diameter to ~1% and length/volume to ~5%
(the +5% is decoder staircasing — tightenable with polyline simplification). **So the
representation is not the ceiling** — a model that predicts these fields well would reproduce the
heuristic to a few percent. That *is* the point: the ceiling is the heuristic labels, not the
architecture. (The momentum tracer also passes crossings straight through — `test_decode_crossing`
— the heuristic's worst case.)

**Step 1 — distillation PoC (DONE).** `train.py` on the same plot, mask→fields, **spatial**
split (left 75% train / right 25% val, 256 px gap → no shared pixels): 104 train / 26 val tiles,
40 epochs, FieldNet base=32. Best val loss 0.239 (heat 0.125, orient 0.019, diam 0.095) — the
orientation head is essentially solved; diameter carries the residual.

A second model was trained identically but with `--corrupt` (input mask degraded with random
erasures, spurious blobs and dilation/erosion, to mimic a real UNet mask rather than a mask
rendered from the labels). `eval.py` on the held-out strip (cols 1536–1907), decoding *predicted*
fields with the same decoder, `heat_thresh=0.5`:

| trained on | eval input | stems | length (m) | mean diam (m) | volume (m³) |
|---|---|--:|--:|--:|--:|
| — | *heuristic (teacher)* | **28** | **156.3** | **0.224** | **6.81** |
| — | *GT-fields round-trip* | 28 | 152.6 | 0.228 | 6.73 |
| clean masks | clean | 31 | 159.6 | 0.226 | 6.88 |
| clean masks | corrupted | 39 | 165.1 | 0.232 | 7.57 |
| corrupted masks | clean | 29 | 165.9 | 0.229 | 7.15 |
| corrupted masks | corrupted | 41 | 174.1 | 0.232 | 7.69 |

**Reading it.** On a clean mask the net lands on the teacher: 29–31 stems vs 28, volume within
1–5%, mean diameter within 1%. Every row is *at or just off* the teacher and **none exceeds it** —
the deviations are over-segmentation (a predicted ridge dipping below threshold mid-stem makes the
tracer emit two stems) plus decoder staircasing in length. Both are error against the label, not
improvement: by construction any departure from 28 is a mistake.

**Corruption training did not buy robustness.** Training on corrupted masks helps slightly on clean
input (29 vs 31 stems) but the corrupted-input rows are no better than the clean-trained model's
(41 vs 39). Degrading the mask costs ~10 spurious stems and ~10% volume regardless. Caveat: the
corrupted eval is a *single* noise draw (seed 7) on one strip, so the 39-vs-41 gap is within noise —
this says corruption-robustness is *unproven*, not that it is impossible.

**Decoder threshold is a real confound, worth recording.** At `heat_thresh=0.3` the corrupt-trained
model decodes into **164** stems while its heat loss is *better* than the clean model's. Component
counts of the thresholded skeleton explain it — training on noise makes the net less confident, so
a low threshold turns its low-probability halo into speckle:

| heat source | t=0.3 | t=0.5 | t=0.7 |
|---|--:|--:|--:|
| ground truth | 28 | 28 | 28 |
| clean-trained | 31 | 27 | 28 |
| corrupt-trained | **491** | 28 | 28 |

Its topology is exactly right at 0.5/0.7. So pixel-wise loss does **not** track the metric anyone
cares about, and a decode threshold tuned on one model silently misreports another. Anything built
on this needs the threshold calibrated per model, or a decoder that doesn't have one.

**The point.** Both the representation (row 2) and the trained model (row 3) sit *at or just off*
the teacher and never above it. There is no signal in the training data that could push the model
past the heuristic, because the heuristic **is** the label. Distillation buys speed, differentiability
and end-to-end composability — not accuracy. To exceed the heuristic you need labels the heuristic
did not produce: field-surveyed stems (DBH tape / TLS), or human-corrected vectorizations. Until
those exist, "learned vectorization" can only be a faster reimplementation of what we already have.

## Step 2 — real teacher labels at scale

The step-1 study had two weaknesses: one plot, and an input mask *rendered from the labels* then
hand-corrupted, so the net could partly read its targets out of its own input. Both are fixed by
running the **real analyzer heuristic** over the segmentor's own training tiles — no orthophoto
required, because `WINMOL_Analyzer/utils/VectorTilePipeline.process_prediction_array_to_gpkg()`
drives skeletonization + vectorization + quantification straight from a mask array.

`build_teacher_dataset.py` does this in two resumable phases: batched UNet prediction → mask PNG,
then a CPU pool → one `.gpkg` per tile. `teacher_tiles.py` turns those into `(mask → fields)`
training pairs; `train_teacher.py` / `eval_teacher.py` train and score on them.

**The GSD is load-bearing.** Tiles are 512 px at the analyzer's own
`Config.tile_size / img_width = 15/512 = 0.0293 m/px`, and every threshold in the heuristic
(`min_length` 2.0 m, `max_distance` 8 m, `measuring_point_spacing` 0.5 m) is in **metres**. A wrong
GSD yields plausible-looking garbage rather than an error, so it is pinned by a test.

| sweep | tiles | vectorized | teacher stems | stems/tile | notes |
|---|--:|--:|--:|--:|---|
| SpecDS_ready, predicted masks | 3230 | 3142 | **12069** | 3.84 (max 12) | the training set |
| GenDS10, predicted masks | 4540 | 3108 | 3584 | 1.15 (max 5) | sparse — see below |

~5 s/tile. The SpecDS sweep alone is ~56× the labels of the single uploaded plot.

**Mask quality dominates the teacher's output — not the vectorizer.** On the same 24 SpecDS tiles,
ground-truth masks yield **155** stems but UNet-predicted masks only **86**. The GenDS10 row says
the same thing at scale: the model used (`twostage_lrfix`) is fine-tuned to beech, so on the
general dataset it predicts sparsely, 867 tiles come out empty and the teacher finds 1.15 stems per
tile instead of 3.84. Whatever a learned vectorizer is trained on, it inherits the segmentation's
errors first and the heuristic's second.

### Result — distilled from 12k real teacher stems

FieldNet (base 32), 20 epochs on 2514 tiles, scored on the **628 held-out tiles** recorded in the
checkpoint. Input is the UNet's own mask; labels are the analyzer's output on that same mask; both
sides get the analyzer's 2 m minimum stem length.

| source | stems | length (m) | mean diam (m) | volume (m³) |
|---|--:|--:|--:|--:|
| heuristic (teacher) | **2356** | **12615.7** | **0.230** | **566.86** |
| GT-fields round-trip | 2161 | 12916.4 | 0.233 | 582.08 |
| model prediction | 2553 | 14530.8 | 0.230 | 638.64 |

**per-tile stem count: exact match 50.6% | MAE 0.64 stems/tile | teacher 3.75 vs model 4.07/tile**

Pooled, the model looks close: mean diameter is *identical* to the teacher's (0.230 m), stem count
+8%, volume +13%. That is the same "approaches but does not exceed" picture as the single-plot
study, now on 56× the data with a mask the net never saw the labels for.

**But the per-tile number is the one to quote.** The model reproduces the teacher's stem count
exactly on only **half** the tiles, and is off by 0.64 stems per tile on average. Pooled totals
hide this: over- and under-counts on different tiles partially cancel, which is exactly why
`eval_teacher.py` reports both. A distilled vectorizer that agrees with its teacher on half the
tiles is not a replacement for it — and it *cannot* become better than it, because the only signal
it has is the teacher's own output.

Note the middle row: the GT-fields round-trip recovers 2161 of 2356 stems (−8%) at this 0.029 m/px
GSD with `sigma=2.0`. The representation itself is now lossy — worse than the +1% it achieved on
the 0.1 m/px plot — so part of the model's error is inherited from the encoding, not learned. That
is a tuning knob (sigma/GSD), not a refutation, but it means these numbers are a *loose* upper
bound on how well Approach A could do here.

## Data status / what's needed

- **Have:** one heuristic output — `…/uploads/…_Barnekow_4_…_detected_stems.gpkg` (3 layers:
  `stems` 215 lines + start/stop/length/volume/d_json/l_json/v_json, `nodes` pts w/ diameter `d`,
  `vectors` segments). CRS ETRS89/UTM-33N, extent ~189×169 m.
- **Needed to train:** the paired input raster per plot — the **orthophoto tiff** (+ regenerate the
  UNet **mask**), and ideally **many plots** ("heuristic output at scale"). Not on this host yet
  (`/data/mnt/storage/hnee` has only models). The Barnekow tiff was mentioned but hasn't landed.
- **Minimal PoC without the fleet:** one plot, train on part / test on a held-out region — proves
  the mechanism + shows the ceiling, though not statistically strong.

## Heuristic reference (what we distil)

`WINMOL_Analyzer/utils/Vectorization.py` (skeleton→graph, `connect_stems` angle-voting,
`tolerance_angle`, dedup) + `Quantification.py` (diameter via `contour` or EDT, outlier-clean,
length+volume). Per-stem outputs match the gpkg schema above.
