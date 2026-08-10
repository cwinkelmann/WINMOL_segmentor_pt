---
name: winmol-analyzer
description: Use when running the WINMOL Analyzer on an orthomosaic — testing a newly trained model end to end, producing stem maps or vectorised stems, or evaluating a model against a rasterised stem-id ground truth. Covers the CLI contract, model handoff from the segmentor, and full-orthomosaic evaluation.
---

# Running the WINMOL Analyzer

Repo: `/Users/christian/hnee/WINMOL_Analyzer`. This is the deployment path — a model that
scores well on tiles but poorly here has not actually improved anything.

## The CLI contract

`winmol_run.py` takes exactly five positional arguments and no flags:

```bash
cd /Users/christian/hnee/WINMOL_Analyzer
python3 -u winmol_run.py <model_path> <input_tiff> <stem_map_tiff> <output_prefix> <Stems|Trees|Nodes>
```

| argument | meaning |
|---|---|
| `model_path` | `.onnx` (or the legacy Keras `.hdf5`) |
| `input_tiff` | the orthomosaic |
| `stem_map_tiff` | **output** path for the predicted binary stem raster |
| `output_prefix` | prefix for the vector outputs |
| process type | `Stems` (segment + vectorise), `Trees`, or `Nodes` |

It re-execs itself with `PYTHONHASHSEED=0` before anything hashes into a set, because
`connect_stems` joins stems in set-iteration order. **Do not defeat that by setting a
different hash seed** — results become non-reproducible run to run.

For a folder of orthomosaics use `winmol_batch.py`, which adds `--model` (family or id
from `config.json`), `--model-dir`, `--input`, `--output`, `--jobs`, `--merge` and
`--edge-buffer-m`. `--list-models` prints what `config.json` knows about.

## Handing a segmentor model over

A model trained in `WINMOL_segmentor_pt` reaches the Analyzer as ONNX:

```bash
# in the segmentor repo — every architecture exports a contract-conformant .onnx
python -m training.run_train --arch hrnet ... --out-dir output/run
# then point the Analyzer at output/run/model.onnx
```

The ONNX contract is NCHW `[batch, 3, 512, 512] -> [batch, 1, 512, 512]`, dynamic batch,
opset 17, sigmoid baked in. `winmol_unet/contract.py` accepts a spatial dim that is 512
**or symbolic** — a *fixed* non-512 size is rejected, which is why DPT cannot be served
(its ViT encoder wants 384, and the dynamic-size path uses an op with no ONNX lowering).

Only UNet has a Keras `.hdf5` mirror; everything else is ONNX-only.

## The scale knob, and why results move

`ExecutionPlan.py:95-97` cuts `ceil(tile_size / pixel_size)` source pixels and resizes to
`img_width=512`. So:

> **effective ground resolution = `tile_size / 512`, independent of the orthomosaic's own
> resolution.**

At the default `tile_size=15` that is **2.93 cm/px** — exactly what the original R
generator produced. A user changing the tile-size spinbox silently rescales the model's
input: 10 m gives 1.95 cm/px, 25 m gives 4.9 cm/px.

Measured on a held-out site, a fixed-scale model loses **5.2 F1 points across ±30% zoom**,
asymmetrically: −5.2 zooming out (stems become thin), −1.9 zooming in. If Analyzer results
look inconsistent between runs or between users, check `tile_size` before suspecting the
model.

Other config knobs worth knowing (`classes/Config.py`): `stem_binary_threshold` (0.5),
`overlap_pred` (8), `min_length` (2.0 m), `max_distance` (8), `tolerance_angle` (7).

## Evaluating a model on a full orthomosaic

This is the strongest yardstick available, and the one the preprocessing comparison uses:
it is preprocessing-agnostic, so no training variant's tiling can flatter it.

1. **Ground truth** — rasterise the site's annotations onto the ortho grid:
   ```bash
   python scripts/rasterize_annotations.py --shapefile <site>.shp \
       --ortho <site>_ortho.tif --out <site>_gt.tif \
       --instances <site>_trees.tif --instance-level tree
   ```
   `--instance-level tree` groups the fragments of one occluded stem under a shared id,
   which is what stem-id evaluation needs; `segment` gives each polygon its own.

2. **Predict** — run the Analyzer as above, producing `<stem_map_tiff>`.

3. **Score inside the AOI only.**

> **Mask every metric to the windthrow polygon (`*_AOE.shp`).** Outside it the stems are
> real but were never digitised. Scoring the whole raster counts correct detections as
> false positives — and penalises the *better* model hardest, inverting the result. This
> is the single easiest way to get a confidently wrong answer here.

Report precision and recall separately, not only F1: every domain-shift failure measured
in this project has been a recall collapse with precision intact, which F1 hides.

### With stem ids

An instance raster supports questions a binary mask cannot:

- **detection rate** — what fraction of ground-truth trees have any predicted pixel;
- **fragmentation** — how many connected prediction components fall on one true tree,
  which is exactly the work the vectoriser has to undo downstream;
- **merges** — one predicted component spanning several true trees.

Fragmentation is the number to watch when comparing preprocessing or amodal variants,
because it is computed from the prediction alone and stays comparable when the training
target changes — F1 does not.

## Pitfalls

- **Do not evaluate over the whole raster.** See above; this inverts results.
- **Do not compare Analyzer F1 against tile-level F1.** Different tiling, different
  denominator, not the same number.
- **Check `tile_size` matches what the model was trained for** before concluding anything
  about a model.
- **`PYTHONHASHSEED=0` is load-bearing** for reproducible stem joining.
- **The `.venv` in the segmentor repo cannot import onnx and TensorFlow in one process**
  (protobuf ABI clash below onnx 1.18); if the Analyzer environment has both, keep onnx at
  1.18 or newer.
