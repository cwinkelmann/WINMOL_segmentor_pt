# Leak-free RGB vs RGBD experiment from orthomosaics + DEM

A runbook. Whole orthomosaics are assigned to train / val / test, so no tile from
a training site can appear in validation or test.

## Why site-level splits

Splitting tiles randomly leaks. Neighbouring tiles from one flight share
illumination, stand structure, sensor, season and often the same individual
stems crossing a tile boundary. A model can then score well by recognising the
site rather than by finding stems, and the number tells you nothing about the
next storm. Assigning whole sites removes that by construction.

## 1 — Describe the sites

`sites.json`, one entry per orthomosaic:

```json
[
  {"name": "barnekow",    "split": "train",
   "ortho":    "/data/Winmol/orthos/20220212_Barnekow_4.tiff",
   "stem_map": "/data/Winmol/labels/barnekow_stems.tif",
   "dem":      "/data/Winmol/dem/barnekow_dsm.tif"},

  {"name": "bremerhagen", "split": "val",
   "ortho": "...", "stem_map": "...", "dem": "..."},

  {"name": "kraking",     "split": "test",
   "ortho": "...", "stem_map": "...", "dem": "..."}
]
```

- `stem_map` is a raster where >0 marks stem. Analyzer output works; so does a
  rasterized hand annotation.
- `dem` may be a DSM or a CHM. It does **not** need to match the ortho's grid,
  size or resolution — every raster is read by world coordinates. It must share
  the ortho's CRS; the tool refuses rather than guessing if it does not.

## 2 — Cut the tiles

```bash
python scripts/build_site_splits.py \
  --config sites.json --out /data/experiments/rgbd-sites --rgbd \
  --tile 512 --stride 512 \
  --depth-vmin 100 --depth-vmax 140 --dem-nodata -9999
```

Produces `train/ val/ test/`, each with `train/ mask/ depth/`, plus
`splits.json` recording which site produced which tile ids — so any tile's
provenance is recoverable and the leak-free claim is auditable.

**Set `--depth-vmin/--depth-vmax` to a real height range in metres covering all
sites.** Without them each tile is scaled to its own min/max, so a 0.4 m log and
a 2 m root plate both become 1.0 and the absolute height that makes depth
informative is discarded. Pick the range from your DEMs, not per site.

## 3 — Preflight (do not skip)

```bash
for s in train val test; do
  python scripts/validate_dataset.py --data-dir /data/experiments/rgbd-sites/$s --rgbd
done
```

Catches the failures that are invisible once training starts: a partial `depth/`
(which silently shrinks the training set rather than erroring, because the loader
pairs on the id intersection), size disagreements, constant or all-nodata depth.

It also prints **depth→mask AUC**. Read it before trusting any RGBD result:

| AUC | meaning |
|---|---|
| ~0.5 | depth is independent of the label — a clean comparison |
| 0.6–0.8 | depth is informative, as real depth over stems should be |
| >0.9 | depth largely *is* the label; RGBD will "win" by reading channel 4 |

For reference: DEM-derived depth measured 0.486 in testing, while the synthetic
height field measures 0.848 — the latter cannot answer whether depth helps.

## 4 — Train both arms

Identical settings; the only difference is the flag.

```bash
BASE=/data/experiments/rgbd-sites
COMMON="--val-data-dir $BASE/val --test-data-dir $BASE/test \
        --arch deeplabv3plus --encoder resnet34 --encoder-weights imagenet \
        --epochs 40 --batch-size 8 --device cuda --no-cache-dataset --num-workers 8"

python -m training.run_train --data-dir $BASE/train $COMMON --out-dir output/site-rgb
python -m training.run_train --data-dir $BASE/train $COMMON --rgbd \
    --depth-vmin 100 --depth-vmax 140 --out-dir output/site-rgbd
```

Pass the same `--depth-vmin/--depth-vmax` used at tiling time.

Each run writes `test_results.md` in its `--out-dir` with precision / recall / F1
on the held-out **test site** — a site the model has never seen.

## 5 — The control that makes it credible

RGBD beating RGB is not by itself evidence that depth helps: it may be reading
the label off the depth channel. Add a mismatched-depth arm, where each tile is
paired with another tile's depth:

```bash
cp -R $BASE/train $BASE/train-mismatch
python scripts/simulate_depth.py --dataset $BASE/train-mismatch --mismatch --overwrite
python -m training.run_train --data-dir $BASE/train-mismatch $COMMON --rgbd \
    --out-dir output/site-rgbd-mismatch
```

| outcome | reading |
|---|---|
| rgbd > rgb, mismatch ≈ rgb | depth genuinely fused — the result stands |
| rgbd > rgb, mismatch > rgb | leakage; the gain is not about depth |
| rgbd ≈ rgb | the model ignores depth; check the 4th channel reaches it |

## 6 — Full-site inference at the end

The test site is held out end to end, so a whole-ortho prediction is an honest
final check:

```bash
python scripts/predict_stem_map.py \
  --ortho /data/Winmol/orthos/kraking.tiff \
  --model output/site-rgbd/model.onnx \
  --out output/kraking_pred.tif
```

The result is georeferenced, so it can be opened next to the ortho in QGIS.

> Note: `predict_stem_map.py` currently feeds 3-channel tiles. Running an RGBD
> model over a whole ortho additionally needs the DEM read in the same window —
> the tiling logic is already in `build_ortho_pairs.py`, but the two are not yet
> joined. RGB full-site inference works today.
