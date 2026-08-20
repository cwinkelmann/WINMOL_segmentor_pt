# WINMOL Segmentor (PyTorch)

PyTorch re-implementation of the WINMOL tree-stem segmentation U-Net. It turns a drone
orthomosaic of a windthrow area into a map of fallen stems, and exports models as ONNX
(and, for UNet, Keras `.hdf5`) for the WINMOL Analyzer.

## Install

```bash
pip install -e ".[train,geo]"       # the usual one: training + reading orthomosaics
pip install -e ".[train]"           # training only: torch, smp, albumentations, tensorboard
pip install -e ".[geo]"             # rasterio, fiona, shapely — prepare / infer / evaluate
pip install -e ".[optimize]"        # onnxconverter-common — fp16/int8 release artifacts
pip install -e ".[train,keras]"     # + TensorFlow, for the legacy-analyzer HDF5 export
pip install -e ".[train,wandb]"     # + Weights & Biases logging
pip install -e "."                  # serve ONNX only: onnxruntime + numpy, no torch/TF
```

The last line is what the WINMOL Analyzer installs: `winmol_unet` serves exported ONNX
through `OnnxSegmenter` **without** torch, TensorFlow or GDAL.

---

# The pipeline

Four scripts at the repo root, in the order you use them. Each also installs as a console
script — `winmol-prepare`, `winmol-train`, `winmol-infer`, `winmol-evaluate` — running the
same code, so `python train.py …` and `winmol-train …` are interchangeable.

```
  orthomosaic + stems + AOI
            │
            ▼
   prepare.py ───────────► training tiles + tiles.jsonl
            │
            ▼
   train.py  ───────────► model.onnx  (+ model.pt)
            │
            ├──────────► infer.py    ──► georeferenced stem map (GeoTIFF)
            │
            └──────────► evaluate.py ──► precision / recall / F1
```

Three rules decide whether the numbers at the end mean anything. They recur below, but
they are worth reading once up front, because each was learned by getting it wrong:

- **Scale is a knob you set, not a property of your imagery.** A tile covers `--extent-m`
  of ground and is resized to 512 px, so effective ground resolution is `extent_m / 512`
  — 2.93 cm/px at the 15 m default, *regardless of your orthomosaic's own resolution*. It
  must match across `prepare.py`, `infer.py` and the Analyzer's `tile_size`. Getting this
  wrong, rather than the choice of model, is the largest single effect measured here
  (**+14.6 F1**).
- **Splits partition the ground, not the tile list.** Sampling oversamples heavily and cuts
  tiles at random rotations, so tiles overlap; a random split of the tile *list* leaks
  between train and test. Every run writes `tiles.jsonl`, so leak-freedom can be checked
  numerically rather than assumed.
- **Score inside the AOI.** Outside the digitised windthrow polygon the stems are real but
  were never labelled, so scoring the whole raster counts correct detections as false
  positives — and penalises the *better* model hardest.

---

## 1. `prepare.py` — orthomosaic → training tiles

Turns georeferenced source data (an orthomosaic, a stem-polygon layer, and the windthrow
AOI polygon) into the `train/`+`mask/` tile pairs the loader consumes.

### How it works

A regular grid over one of these sites is useless: stems cover only about **1% of the
ground**, so most grid cells are empty and a model trained on them learns to predict
background. `prepare.py` is a port of the original R generator and instead:

1. draws a random point **inside the windthrow polygon** — never the whole ortho, because
   outside that polygon the stems are real but were never digitised;
2. cuts a square `--extent-m` metres across at a **uniformly random rotation**;
3. resizes it to `--tile-px` (512), which fixes the ground resolution at `extent_m / tile_px`;
4. **oversamples heavily** — many more windows are cut than kept;
5. **rejects** any tile whose stem coverage is below `--min-stem-frac` (default 0.5%), or
   which contains too much nodata collar.

Stem polygons are repaired before anything else: 50 of the reference corpus's 3,842
hand-traced outlines self-intersect, and GEOS raises on the first set operation that
touches one — tens of thousands of tiles into a run. Pass `--skip-fix` only if your input
is already clean.

Every run writes **`tiles.jsonl`**: one record per tile with its source ortho, world
centre, rotation, GSD and stem fraction. This is what makes a dataset auditable after the
fact — you can verify a split is leak-free instead of trusting that it is.

### Modes

```bash
# one site
python prepare.py --ortho site.tif --stems site.shp --aoi site_AOE.shp --out data/site

# several sites with leak-free splits — the usual one
python prepare.py --config configs/sites.example.json --out data/beech \
                  --strategy blocks [--jobs 4]

# leave-one-site-out folds, composed from per-site tile sets by symlink
python prepare.py --compose-folds --site-all ALL --site-tv TV \
                  --fold-sites A B C --splittable A B --out folds/

# just the label raster, no tiling
python prepare.py --rasterize --stems site.shp --ortho site.tif --out stem_map.tif
```

Starting from data that is **already raster pairs** rather than from an orthomosaic? These
three modes need no GDAL, so they work on a plain `[train]` install:

```bash
# an existing image/mask folder -> the loader convention (pairs by shared filename key,
# renumbers, binarises palette/instance masks). Reads jpeg/png/tif/bmp.
python prepare.py --from-folder --src raw/ --out data/ready

# COCO polygon annotations -> rasterised masks
python prepare.py --from-coco --coco-json ann.json --images-dir img/ --out data/ready

# a fixed, shareable train/val split, matching the loader's own seeded split
python prepare.py --split --src data/ready --out data/split --val-fraction 0.2 --seed 1
```

### The site config

`--config` takes a JSON file listing the sites to sample and the settings shared across
them. A copy with comments is at `configs/sites.example.json`.

```json
{
  "extent_m": 15.0,
  "tile_px": 512,
  "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
  "split_seed": 1,
  "caps": {"train": 2000, "val": 400, "test": 400},

  "sites": [
    {
      "name": "Campus",
      "ortho": "/path/to/20171016_EW_WW_Campus_ortho.tif",
      "stems": "/path/to/20171016_EW_WW_Campus.shp",
      "aoi":   "/path/to/20171016_EW_WW_Campus_AOE.shp",
      "block_size_m": 140
    }
  ]
}
```

**Top-level keys** — all optional except `sites`:

| key | default | what it does |
|---|---|---|
| `sites` | *required* | the list described below |
| `extent_m` | `10.24` | tile footprint in metres. Effective GSD is `extent_m / tile_px`. |
| `native_px` | *unset* | cut this many pixels at the ortho's **own** resolution instead. Overrides `extent_m`. |
| `tile_px` | `512` | output tile side. Larger than 512 only makes sense with `--multiscale` training, which crops them. |
| `caps` | `{train:2000, val:400, test:400}` | maximum tiles per split, **across all sites** |
| `split_fractions` | `{train:0.7, val:0.15, test:0.15}` | how blocks are apportioned (`blocks` strategy) |
| `split_seed` | `1` | seed for the block→split assignment |
| `block_size_m` | *unset* | fallback block size for sites that don't set their own |
| `block_splits` | all three | which splits a site may contribute to |
| `antialias` | `"auto"` | resampler. `auto` = PIL bicubic; `on`/`off` = skimage order-3. This selects the *resampler*, not just anti-aliasing — measured 3.16 grey levels between `auto` and skimage, vs 0.006 between skimage on and off. |

**Per-site keys:**

| key | required | what it does |
|---|---|---|
| `name` | yes | used for output paths and in `tiles.jsonl` provenance |
| `ortho` | yes | orthomosaic GeoTIFF |
| `stems` | yes | digitised stem polygons |
| `aoi` | yes | the windthrow polygon. Sampling never leaves it. |
| `split` | for `--strategy sites` | which split this whole site becomes |
| `block_size_m` | for `--strategy blocks` | overrides the top-level value |
| `block_splits` | no | restrict this site to certain splits |
| `whole_split` | no | under `blocks`, contribute this site wholesale to one split instead of block-splitting it — for sites too small to block |
| `halves` | no | under `halve`, which two splits the halves become (default `["train", "test"]`) |

**Choosing `block_size_m`.** It must comfortably exceed `extent_m × √2`, the rotated
tile's half-diagonal on either side of a shared block edge. Below that, tiles drawn on
opposite sides of a boundary can overlap and the split leaks. At the 15 m default that
means blocks well above ~21 m; the reference corpus uses 100–140 m. A site whose blocks all
vanish under the inward buffer contributes nothing, and `prepare.py` raises rather than
letting the split quietly shrink.

Which keys matter depends on the strategy: `sites` reads each site's `split`, while
`blocks` and `halve` derive their own and ignore it.

### Split strategies

`--strategy` decides how the *ground* is partitioned before any tile is cut:

| strategy | what it does | use when |
|---|---|---|
| `blocks` | cuts each site into blocks; whole blocks go to train/val/test | the default — every site contributes to every split |
| `halve` | cuts each site in two | few, large sites |
| `sites` | holds entire sites out | you want to measure transfer to unseen ground |

For `blocks`, `block_size_m` must comfortably exceed `extent_m × √2` — the rotated tile's
half-diagonal either side of a shared edge — or tiles across a boundary can overlap and
leak. Small sites may yield no usable block at all; `prepare.py` says so rather than
silently contributing nothing.

`--jobs N` extracts sites concurrently. Decoding the orthomosaic dominates the cost, and
sites are independent, so this scales nearly linearly; the merge reassigns tile indices so
two sites cannot overwrite each other.

### Scale

`--extent-m` (default 15 m) fixes the footprint, giving 2.93 cm/px at 512 px. Use
`--native-px 1024` instead to cut at the orthomosaic's **own** resolution — this is what
the multi-scale training recipes want, because they supply the scale variation themselves
by cropping.

---

## 2. `train.py` — tiles → model

Trains a segmentation model and exports it. Writes `model.pt` and `model.onnx` to
`--out-dir` (plus `model.hdf5`/`model.keras` with `--export-keras`).

The data directory must contain `train/` (`trainN.jpeg`) and `mask/` (`maskN.gif`), paired
by the integer `N` — which is what `prepare.py` produces.

```bash
python train.py --data-dir data/beech --out-dir runs/beech \
                --arch hrnet --recipe robust \
                --epochs 40 --batch-size 16 --seed 1 --deterministic --device cuda
```

### Recipes

`--recipe` applies a named set of augmentation defaults taken from experiments in this
repo, so reproducing a result does not depend on remembering ten flags. **Explicit flags
always beat the recipe**, so `--recipe robust --aug-hsv-p 0.5` is a deliberate weakening
rather than a silent conflict.

| recipe | what it is |
|---|---|
| `baseline` | repo defaults: flips, mild brightness/contrast, mild hue. No rotation, no multi-scale crop. |
| `fixed` | the **control** arm of the scale experiment: multi-scale pipeline pinned to one scale (crop 512–512). |
| `jitter` | the **treatment** arm: crop 394–666 px, i.e. ±30% footprint. Flattens the accuracy-vs-scale curve at no cost at the serving scale. |
| `robust` | `jitter` + strong hue (±90 at p=0.9). Took a held-out site from **F1 0.042 to 0.688**. |
| `mosaic` | `robust` + 2×2 mosaic. **UNEVALUATED** — the knob exists, nothing measures it. |

`python train.py --print-recipe robust` resolves a recipe and exits without training.

Two traps worth knowing before you quote a reproduction, both documented at length in
`winmol_unet/training/recipes.py`:

- **`multiscale=True` is not the jitter switch.** The `fixed` control sets it too — the
  crop *range* is the entire independent variable.
- **Rotation is off by default** (`--aug-rotate-p 0.0`, limit 15°), but every measured run
  used `--aug-rotate-p 0.5 --aug-rotate-limit 180`. The recipes set this explicitly.

Recipes cover **augmentation only**. The schedule the measured runs used —
`--epochs 40 --batch-size 16 --deterministic` — must be passed yourself. Without
`--deterministic` a seed is a label rather than a guarantee: two runs of identical code
landed 2.0 F1 apart.

### Architecture

`--arch {unet,deeplabv3plus,hrnet}` (default `unet`). The latter two use
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch);
tune `--encoder` (default `resnet34`) and `--encoder-weights` (`None`, or `imagenet` —
needs network).

**Every** architecture exports a contract-conformant `.onnx`, so all models load the same
way through `OnnxSegmenter` and the Analyzer never inspects the architecture.
`--width-mult 0.5` halves the channel width for ~4× fewer FLOPs, roughly flat accuracy on
this task.

### Two-stage training

As in the R original: pass `--gen-data-dir` (general, stage 1) and `--spec-data-dir`
(species, stage 2) instead of `--data-dir`. One model is built, trained on the general set
(early-stop patience `--patience-stage1`, default 3), then fine-tuned on the species set
(`--patience-stage2`, default 5). Final metrics and export come from the species model.

To fine-tune from an existing checkpoint instead, use `--init-weights path/to/model.pt`.

### Validation and test

By default training takes a deterministic 80/20 split of `--data-dir`. To pin an explicit,
shareable held-out set, materialise it once and train against it:

```bash
python prepare.py --split --src data/ready --out data/split   # -> split/{train,val}
python train.py --data-dir data/split/train --val-data-dir data/split/val ...
```

`--test-data-dir` adds a held-out test stage (the R `cost_eval`): after training, the final
model is scored with no augmentation, and precision/recall/F1 are printed, logged to
TensorBoard and written to `test_results.md` in `--out-dir`.

### Multi-scale training

`--multiscale` keeps tiles at native resolution and augments with **rotate → random-resized
crop**: the full tile is rotated by an arbitrary angle, then a window of side
`[--crop-min-px, --crop-max-px]` is cropped and resized to 512. Rotating *before* cropping
keeps the crop on valid interior pixels, so black padding only appears near the tile edge.

The sampled crop side spans the effective GSD range (`crop_max` → coarse, native → 1:1,
`crop_min` → zoom-in), which is what makes the model robust to the Analyzer's user-set
`tile_size`. Needs a native-resolution dataset (`prepare.py --native-px 1024`) and
`--no-cache-dataset`.

Pair it with **`--eval-tiling`**, which evaluates val/test by cutting each native tile into
a full-coverage grid of non-overlapping 512 tiles at native resolution, instead of one
downscaled 512 — reproducible, and matching the scale `--multiscale` trains at.

Sampling *larger* tiles and letting `--multiscale` crop them is the better division of
labour: the crop range supplies the scale and rotation variety, so the generator need not
bake one fixed resampling into every tile. If you do that, drop the generator's own
rotation rather than rotating twice — each rotation resamples, and two of them blur the
stem edges the model is looking for.

### Augmentation, in detail

`--aug-*` applies to the **training split only**. Geometric transforms (`hflip`, `vflip`,
`rotate`) carry image and mask together; photometric ones (`bc` = brightness/contrast,
`hsv` = hue/saturation) touch the image only. Each `*-p` is a probability.

`--mosaic-p` stitches a 2×2 grid of tiles into one training image. It is **off by default
and unevaluated in this repo** — no paired run, no leave-one-site-out fold, no results
document. Partner tiles are drawn only from the dataset's own split, so a mosaic cannot
leak across a split boundary.

### Practicalities

- **Device** — `--device auto` (default) prefers Apple MPS, then CUDA, then CPU.
- **Large datasets** — the resize cache is ~4 MB/pair; for thousands of pairs pass
  `--no-cache-dataset --num-workers 4`.
- **Logging** — metrics always go to TensorBoard (`<out-dir>/logs/`). Add `--wandb` (with
  `--wandb-project` / `--wandb-run-name`) for Weights & Biases; put `WANDB_API_KEY` in a
  `.env` at the repo root.
- **Converting an existing dataset** — `python prepare.py --from-folder --src raw --out ready`
  pairs arbitrary image/mask folders by shared key, renumbers them, and binarises masks.

---

## 3. `infer.py` — model → stem map

Runs a model over an orthomosaic and writes a georeferenced probability raster.

```bash
python infer.py --model runs/beech/model.onnx --ortho site.tif --aoi site_AOE.shp \
                --out pred.tif --extent-m 15 --overlap 0.5 [--threshold 0.5]
```

### How it works

The ortho is covered with tiles of `--extent-m` ground each, resized to the model's 512 px
input. Every tile's probability map is resampled back onto a **common world-space
reference grid** and averaged where tiles overlap. The overlap matters: a single
non-overlapping pass leaves seams exactly where a stem crosses a tile edge.

Output is a single-band float32 GeoTIFF of stem probability. With `--threshold`, a
thresholded `_mask.tif` is written alongside it.

**`--extent-m` must match what the model was trained at.** It is a scale knob, not a
speed knob — a fixed-scale model loses **5.2 F1 across a ±30% zoom**. If results look
inconsistent between surveys, check this before blaming the model.

`--aoi` restricts inference to the windthrow polygon. Strongly recommended for anything
you intend to score.

**Vectorising the raster into stem polylines is the WINMOL Analyzer's job**, not this
repo's — hand the Analyzer the same `.onnx` and let it trace.

---

## 4. `evaluate.py` — model → precision / recall / F1

Three modes, one metric implementation. Everything routes through the same code that
produced every `test_results.md` here, so numbers drop straight into those tables.

```bash
# a) tile mode — anything prepare.py produced. Works on .onnx or .pt
python evaluate.py --model model.onnx --data-dir data/beech/test

# b) geospatial mode — predict over the AOI in world space, score there
python evaluate.py --model model.onnx --ortho site.tif --aoi site_AOE.shp --stems site.shp

# c) score an existing stem map (e.g. an Analyzer output) on the same basis
python evaluate.py --stem-map out.tif --aoi site_AOE.shp --stems site.shp
```

### Which mode to use

**Tile F1 measures the model *and the exam*.** A tile cut at 1.2 cm/px and one at
2.93 cm/px are different exams, because a stem is ~2.4× thicker in pixels on the first. Two
models trained at different ground resolutions therefore **cannot** be compared on their
own tile sets — use geospatial mode, which predicts over the same ground and scores both on
one reference grid.

Tile mode is the right choice for comparing arms trained on the *same* tiles.

### The AOI is not optional

In geospatial mode `--aoi` is required. Outside the windthrow polygon stems are real but
were never digitised, so scoring the whole raster counts correct detections as false
positives — inverting the result and penalising the better model hardest.
`--edge-buffer-m` (default 2 m) shrinks the AOI further so tiles straddling its boundary do
not score against absent labels.

`--json-out` writes the metrics as JSON; `--label` names the run in the output.

---

# Results and reports

**The assembled PDF report is not published.** It lives in a private repository along with
its front matter and build script. The documents it is built from are below and are the
authoritative source for every number in it — the report adds assembly and narrative, not
measurements.

**[`docs/README.md`](docs/README.md)** — index of every document, each marked *current* or
*superseded*. Several early results were later refuted; that index says which, so nothing
gets re-cited by mistake.

The four findings that most change how you use this repo:

| finding | where |
|---|---|
| **Scale dominates.** Effective GSD is `tile_size / 512`, a user-set knob — matching it to the training scale was worth **+14.6 F1**. | [`process.md`](docs/process.md) |
| **±30% footprint jitter** flattens the accuracy-vs-scale curve at no cost at the serving scale. | [`scale-augmentation-results.md`](docs/scale-augmentation-results.md) |
| **Strong hue augmentation** takes a held-out site from **F1 0.042 to 0.688**; site holdouts are viable now. | [`scale-augmentation-loso.md`](docs/scale-augmentation-loso.md) |
| **The published method's composite loss does not train** — it calls the rounded metric, so the model trains on plain BCE. Measured directly in R: the gradient sums to exactly `0.000e+00`. | *(detail in the private helper repo)* |

Experiment designs are pre-registered *before* the runs, and deviations are recorded in the
matching results document. Those pre-registrations, together with the full results archive
(including superseded documents and the one-off experiment scripts), live in a private
companion repository — this one keeps the code and the current findings.

# Pretrained models

**Upstream WINMOL Analyzer models — the correct ones.** The four published UNet flavours live on
Zenodo, **DOI [10.5281/zenodo.15907576](https://doi.org/10.5281/zenodo.15907576)**
("WINMOL Analyzer Models", CC-BY-4.0). Use these, not the older mis-referenced weights:

| flavour | Zenodo file | domain |
|---------|-------------|--------|
| generic pretrain | `model_UNet_GenDS_512_*.hdf5` | GenDS (general) |
| beech | `model_UNet_SpecDS_Beech_512_*.hdf5` | beech stems |
| spruce | `model_UNet_SpecDS_Spruce_512_*.hdf5` | spruce stems |
| spruce + deadwood | `model_UNet_SpecDS_Spruce_Deadwood_512_*.hdf5` | spruce + standing deadwood |

They are Keras HDF5 (NHWC); convert any to a contract-conformant ONNX with
`scripts/convert_keras_to_onnx.py`.

**Optimised models + ONNX conversions — GitHub Release [`models-v1`](../../releases/tag/models-v1).**
The release re-hosts *only* the ONNX (the HDF5 stay on Zenodo), each flavour in fp32 / `_fp16`
(GPU) / `_int8` (CPU):

- **PyTorch UNet** — `unet_fp32.onnx` (TestDS F1 0.760) plus the width-0.5 optimised
  `unet_w05_int8_cpu.onnx` (**10× faster CPU, lossless**) and `unet_w05_fp16_gpu.onnx` (GPU, lossless).
- **Zenodo flavours as ONNX** — `model_UNet_<FLAVOUR>_512{.onnx,_fp16.onnx,_int8.onnx}`: converted
  from the Zenodo HDF5 (numerically identical) and quantized **post-training** (no retraining).

**Models trained on the newer data — GitHub Release [`models-v2`](../../releases/tag/models-v2).**
Same ONNX contract, so they drop into the Analyzer unchanged. Two families, each the best of
three seeds, each scored on its own held-out ground (**the two families are not comparable to
each other** — see [`docs/tegel-r12-r13-results.md`](docs/tegel-r12-r13-results.md)):

- **Four-site beech corpus** (Campus, Campus_Oberheide, Bachsee_north, Kaufland) trained with
  ±30% scale jitter — `model_HRNet_Beech4Site_512_jitter` (F1 0.787) and
  `model_UNet_Beech4Site_512_jitter` (F1 0.783).
- **Tegel R12/R13** (July 2025 survey) — `model_UNet_TegelR12R13_512_scratch` (F1 0.777) and
  `..._finetune` (F1 0.789, initialised from the beech UNet). The beech models already reach
  **F1 0.76 zero-shot** on the Tegel test plots, so training on Tegel is worth +3.5 to +6.2 F1.

fp32 (macOS/CoreML) / `_fp16` (GPU) / `_int8` (CPU) as in v1, except HRNet, which has no int8:
the smp decoder's symbolic shapes fail ORT static quantisation. Build and publish with
`scripts/fetch_release_v2.sh` then `scripts/deploy_models_to_release.py --set v2`.

### Reproducing them

The released optimised UNet was trained single-stage, then quantized:

```bash
python train.py --arch unet --width-mult 0.5 \
  --data-dir <SpecDS> --test-data-dir <TestDS> --out-dir output/w05 --device cuda

python scripts/quantize_unet.py output/w05/model.onnx output/w05/model_int8.onnx \
  --mode static --calib-dir <SpecDS> --n-samples 128
```

**A caveat on the v2 beech models.** Their exact training invocations are *not recorded*:
those runs predate the run-config writer, and their directories hold no config, no argv in
the logs and no shell history. The `jitter` recipe is reconstructed from a run one day
later on the same tiles and is consistent with the published numbers — but it is evidence,
not a transcript. A run from `--recipe jitter` reproduces the *recipe*, not the run.

The upstream Zenodo flavours **can't be retrained** (their training imagery isn't public), but
they can still be converted and quantized — which is exactly what the release provides.

# Model optimization — faster CPU/GPU inference

Two independent levers speed up UNet inference while keeping the ONNX contract. Full study +
numbers: `docs/2026-07-21-cpu-inference-speedup-results.md`.

| lever | needs retraining? | backend | typical win | accuracy |
|-------|-------------------|---------|-------------|----------|
| **static int8** quantization | **no** (post-training) | CPU (AVX-VNNI) | ~3× | lossless |
| **fp16** quantization | **no** (post-training) | GPU (Tensor Cores) | ~1.7× | lossless |
| **width scaling** (`width_mult`) | yes (retrain) | CPU + GPU | ~4× FLOPs | ~flat on this task |

```bash
# int8 (CPU) — static, calibrated on real tiles (no labels needed), ~3× lossless:
python scripts/quantize_unet.py model.onnx model_int8.onnx \
  --mode static --calib-dir <StemDataset dir> --n-samples 128

# fp16 (GPU) — no calibration, lossless (needs the [optimize] extra):
python scripts/quantize_unet.py model.onnx model_fp16.onnx --mode fp16

# smaller model (needs retraining) — half the channel width, ~4× fewer FLOPs:
python train.py --arch unet --width-mult 0.5 --data-dir <DS> --out-dir out/w05
```

**Don't use dynamic int8** (`--mode dynamic`) for these conv nets — ORT has no fast dynamic-conv
kernel, so it runs ~3× *slower*. Static int8 or fp16 only.

**Fastest GPU inference — TensorRT EP.** The ONNX we ship *is* what TensorRT consumes: serve any
`.onnx` through the TensorRT execution provider (build `Dockerfile.onnxgpu`, which bundles
onnxruntime-gpu + TensorRT 10) for ~**1.9× over the CUDA EP** (and 2.7× throughput at batch) via
fp16. Set `WINMOL_ONNX_PROVIDERS="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"`;
TensorRT builds+caches an engine on the target GPU (device/TRT-version specific — never shipped).
GPU int8 is *not* worth it (slower than fp16 here) — keep int8 for CPU.

The latency benchmarks that produced these numbers live in the private companion repo.

### Converting a Keras HDF5 model to ONNX

The upstream WINMOL Analyzer models are Keras HDF5 (NHWC). Convert any of them to a
contract-conformant ONNX (NCHW, sigmoid baked in, opset 17) — a pure format conversion, weights
untouched — then quantize as above:

```bash
docker build -f Dockerfile.convert -t winmol-convert .        # TF 2.15 + tf2onnx
docker run --rm -e PYTHONPATH=/app -v "$PWD":/app -w /app -v <models>:/models:ro \
  winmol-convert scripts/convert_keras_to_onnx.py /models/model_UNet_SpecDS_Beech_512_*.hdf5 \
  out/model_UNet_SpecDS_Beech_512.onnx
```

Publish artifacts to a GitHub Release with `scripts/deploy_models_to_release.py`
(`--dry-run` first).

# Legacy analyzer (HDF5 drop-in)

The original WINMOL Analyzer loads a Keras `.hdf5` U-Net. To produce one, install the
`[keras]` extra and pass `--export-keras`:

```bash
pip install -e ".[train,keras]"

python train.py --data-dir /path/to/SpecDS --out-dir output/legacy \
  --arch unet --export-keras --epochs 20 --device mps
```

That writes `model.hdf5` and `model.keras` alongside `model.pt`/`model.onnx`. Drop the
`.hdf5` into the unmodified analyzer in place of its shipped model — the layer topology
mirrors `winmol_unet/model.py` and weights are transferred layer by layer, so no analyzer
change is needed.

Two constraints, both enforced rather than documented-and-hoped:

- **UNet only.** `--export-keras` with `--arch deeplabv3plus` or `hrnet` raises before
  training starts, rather than training for an hour and then failing at export.
- **TensorFlow is imported lazily**, only when `--export-keras` is set, so a TF-less
  environment can still train and export ONNX.

For anything other than the legacy analyzer, prefer ONNX: every architecture exports a
contract-conformant `.onnx`, and `OnnxSegmenter` serves it without torch or TensorFlow.

# The ONNX contract

`winmol_unet/contract.py` is the frozen interface between this repo and the Analyzer:

| | |
|---|---|
| input | NCHW `[batch, 3, 512, 512]`, float32 in `[0,1]` |
| output | `[batch, 1, 512, 512]`, **sigmoid baked in** — probabilities, not logits |
| batch axis | dynamic |
| opset | 17 |

`validate_onnx_model()` enforces it at export. Spatial dims may be fixed 512 *or* symbolic
(some decoders export symbolic shapes) — but a *wrong* fixed size is rejected. Any change
here is breaking and requires a coordinated Analyzer update.

The design document for this contract is not published; `winmol_unet/contract.py` — its
docstring and `validate_onnx_model()` — is the authoritative public description.
