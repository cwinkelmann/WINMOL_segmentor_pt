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
