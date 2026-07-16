# WINMOL Segmentor (PyTorch)


PyTorch re-implementation of the WINMOL tree-stem segmentation U-Net, with model
export to ONNX and Keras (`.hdf5` / native `.keras`) for use in the WINMOL Analyzer.

## Install

```bash
pip install -e ".[train,wandb]"   # training deps + optional Weights & Biases
pip install -e ".[keras]"         # + tensorflow (only for the UNet Keras .hdf5/.keras export)
pip install -e ".[dev]"           # + pytest
```

To only **serve** exported ONNX models (e.g. inside the WINMOL Analyzer), install the base
package — it pulls in just `onnxruntime` + `numpy`, no torch/tensorflow:

```bash
pip install .
```

## Datasets

Training expects a **loader-ready** directory: `train/trainN.jpeg` (RGB) + `mask/maskN.gif`
(binary), paired by the integer `N`. Three tools build one:

- **From a COCO instance-segmentation split** (e.g. the BAMFORESTS `coco1024` tree set) — run
  once per split (train / val / test):
  ```bash
  python scripts/coco_to_dataset.py \
    --coco-json .../annotations/instances_tree_train2023.json \
    --images-dir .../train2023 --dst .../dataset/train --limit 1000 --seed 1
  ```
  Samples `--limit` annotated images (deterministic per `--seed`; skips any whose file is
  missing on disk), saves each as RGB `trainN.jpeg` (`--quality`, default 95), and rasterizes
  the union of category polygons to a binary `maskN.gif`. RLE segmentations are not supported.

- **From an arbitrary image+mask folder** (non-integer names, palette/instance masks):
  ```bash
  python scripts/build_dataset.py --src <raw> --dst <loader-ready>
  ```
  Pairs images↔masks by shared key and binarizes the masks.

- **Materialize a fixed train/val split** so multiple runs share an identical split:
  ```bash
  python scripts/split_dataset.py --src <dataset> --dst <split> --val-fraction 0.2 --seed 1
  ```
  Writes `<split>/train` and `<split>/val`; pass the latter to `--val-data-dir`.

## Training

The data directory must contain `train/` (jpeg images named `trainN.jpeg`) and
`mask/` (gif masks named `maskN.gif`), paired by the integer `N`.

```bash
python -m training.run_train \
  --data-dir /path/to/TestDS \
  --out-dir output/run1 \
  --epochs 20 --batch-size 4 --device mps \
  --aug-hflip-p 0.5 --aug-vflip-p 0.5 --aug-rotate-p 0.3 --aug-rotate-limit 20 \
  --aug-bc-p 0.5 --aug-hsv-p 0.5 \
  --wandb --wandb-project winmol --wandb-run-name beech-run1
```

This trains the U-Net (single stage, deterministic 80/20 train/val split) with online
[albumentations](https://albumentations.ai/) augmentation, and writes `model.pt`,
`model.hdf5`, `model.keras`, and `model.onnx` to `--out-dir`.

**Architecture** — `--arch {unet,deeplabv3plus,hrnet}` (default `unet`). `deeplabv3plus`
and `hrnet` use [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch);
tune `--encoder` (default `resnet34`, for `deeplabv3plus`) and `--encoder-weights`
(`None`, or `imagenet` for a pretrained encoder — needs network). **Every** architecture
(incl. UNet) exports a contract-conformant `.onnx` (+ `.pt`) by default, so all models load
the same way via `OnnxSegmenter`. The Keras `.hdf5`/`.keras` mirror is UNet-specific and
opt-in with `--export-keras` (for the unmodified-analyzer drop-in). Example:

```bash
python -m training.run_train --data-dir /path/to/TestDS --out-dir output/deeplab \
  --arch deeplabv3plus --encoder resnet34 --encoder-weights imagenet \
  --epochs 20 --device mps
```

The analyzer consumes any of these transparently through `OnnxSegmenter` once its ONNX
loading branch is in place — it never inspects the architecture.

**Two-stage training** (GenDS → SpecDS fine-tune, as in the R original) — pass
`--gen-data-dir` (general, stage 1) and `--spec-data-dir` (species, stage 2) instead of
`--data-dir`. One model is built, trained on the general set (early-stop patience
`--patience-stage1`, default 3), then fine-tuned on the species set (`--patience-stage2`,
default 5); final metrics + export come from the species model.

```bash
python -m training.run_train --arch deeplabv3plus --encoder resnet34 --encoder-weights imagenet \
  --gen-data-dir /path/to/GenDS --spec-data-dir /path/to/spruce \
  --out-dir output/spruce2stage --epochs 20 --device mps --no-cache-dataset --num-workers 4
```

**Building a dataset** — the loader expects `train/train{N}.jpeg` + `mask/mask{N}.gif`
(integer `N`, binary masks). Convert an arbitrary image/mask folder (non-integer names,
palette/instance masks) with:

```bash
python scripts/build_dataset.py --src /path/to/raw --dst /path/to/ready
```

It pairs by shared key, renames to sequential `train{i}`/`mask{i}`, and binarizes masks
(any pixel > 0 → foreground). Source is never mutated.

**Fixed train/val split** — by default training does a deterministic 80/20 split of
`--data-dir` (`--val-fraction`/`--seed`). To pin an explicit, shareable held-out set (e.g.
so PyTorch and R evaluate on the same tiles), materialize it once and train against it:

```bash
python scripts/split_dataset.py --src /path/to/ready --dst /path/to/split   # -> split/{train,val}
python -m training.run_train --data-dir /path/to/split/train \
  --val-data-dir /path/to/split/val --arch deeplabv3plus ...
```

With `--val-data-dir`, training uses **all** of `--data-dir` for training and the given dir
for validation (no re-splitting). The materialized `val/` matches `train_val_split` for the
same fraction+seed.

**Held-out test stage** (as in the R `cost_eval`) — pass `--test-data-dir /path/to/TestDS`
(a `train/` + `mask/` dataset). After training (single- or two-stage), the final model is
evaluated on it with no augmentation; precision/recall/F1 are printed, logged to TensorBoard
(`logs/test/`), and written to `test_results.md` in `--out-dir`. Training also uses
`ReduceLROnPlateau` (factor 0.1, patience 2) matching the R LR schedule.

**Multi-scale native-fidelity training** (for native-resolution tiles larger than 512,
e.g. the 1024px BAMFORESTS tiles) — pass `--multiscale`. Instead of downscaling each tile to
512 at load time, the loader keeps it at native resolution and augments with a **rotate →
random-resized-crop**: the full tile is rotated by an arbitrary angle (`--aug-rotate-p` /
`--aug-rotate-limit`, e.g. `180` for full 360°; exposed corners pad black), then a window of
side `[--crop-min-px, --crop-max-px]` (default `400`/`1024`) is cropped and resized to 512.
Rotating the full tile *before* cropping keeps the crop on valid interior pixels. The sampled
crop side spans the effective GSD range (`crop_max`→coarse, native→1:1, `crop_min`→zoom-in),
so the model becomes robust to the analyzer's user-set `tile_size`. Requires a native-res
dataset (build one with `scripts/coco_to_dataset.py`) and `--no-cache-dataset` (native tiles
are too large to cache). Pair with `--eval-tiling` (below).

**Deterministic eval tiling** — `--eval-tiling` evaluates val/test by cutting each native tile
into a full-coverage grid of non-overlapping 512 tiles (2×2 for a 1024px image) at native
resolution, instead of one downscaled 512. Reproducible and matches the fine scale that
`--multiscale` trains at. Applies to the val split and the `--test-data-dir` stage.

**Large datasets** — the resize cache is ~4 MB/pair; for thousands of pairs pass
`--no-cache-dataset --num-workers 4` (loads per batch with parallel workers instead of
caching, avoiding out-of-memory).

**Device** — `--device auto` (default) prefers Apple MPS, then CUDA, then CPU; or pass
`mps` / `cuda` / `cpu` explicitly.

**Augmentation** (`--aug-*`, applied on-the-fly to the training split only) — geometric
transforms (`hflip`, `vflip`, `rotate`) are applied to image and mask together; photometric
transforms (`bc` = brightness/contrast, `hsv` = hue/saturation) to the image only. Each
`*-p` is a probability; rotation is off by default (`--aug-rotate-p 0.0`). Omit all `--aug-*`
flags to use the defaults (flips + brightness/contrast + hue/saturation at `p=0.5`).

**Logging** — metrics always go to TensorBoard (`<out-dir>/logs/`). Add `--wandb`
(with `--wandb-project` / `--wandb-run-name`) to also log to Weights & Biases; put your
`WANDB_API_KEY` in a `.env` file at the repo root (loaded automatically).

## End-to-end: BAMFORESTS multi-scale run

Build the three splits from `coco1024`, then train at native fidelity with multi-scale crops
and evaluate with deterministic tiling:

```bash
BASE=/path/to/BAMFORESTS/coco1024
OUT=/path/to/1000_images
for s in "train:instances_tree_train2023.json:train2023" \
         "val:instances_tree_eval2023.json:val2023" \
         "test:instances_tree_TestSet12023.json:test2023/Test-Set-1"; do
  name=${s%%:*}; rest=${s#*:}; json=${rest%%:*}; dir=${rest#*:}
  python scripts/coco_to_dataset.py --coco-json "$BASE/annotations/$json" \
    --images-dir "$BASE/$dir" --dst "$OUT/$name" --limit 1000 --seed 1
done

python -m training.run_train --arch deeplabv3plus --encoder resnet34 --encoder-weights imagenet \
  --data-dir "$OUT/train" --val-data-dir "$OUT/val" --test-data-dir "$OUT/test" \
  --out-dir output/bamforests --device cuda --no-cache-dataset --num-workers 4 \
  --multiscale --crop-min-px 400 --crop-max-px 1024 --eval-tiling \
  --aug-rotate-p 0.5 --aug-rotate-limit 180 --aug-hflip-p 0.5 --aug-vflip-p 0.5 --aug-hsv-p 0.5
```

This writes `model.onnx` / `model.pt`, TensorBoard logs, and `test_results.md`. To compare
architectures on the same data, `scripts/benchmark_architectures.py` trains unet /
deeplabv3plus / hrnet identically and writes a `SUMMARY.md` (`--help` for its flags).

## ONNX inference (WINMOL Analyzer bridge)

Every architecture exports the **same** ONNX graph — NCHW `[N,3,512,512]` → `[N,1,512,512]`,
dynamic batch, sigmoid baked in — served by `OnnxSegmenter`, which duck-types the Keras model
the Analyzer expects (`predict_on_batch(NHWC) → NHWC`):

```python
import numpy as np
from winmol_unet.runtime import OnnxSegmenter

seg = OnnxSegmenter("output/bamforests/model.onnx")     # loads with the best available EP
batch = np.zeros((4, 512, 512, 3), dtype=np.float32)    # NHWC, values in [0,1]
prob = seg.predict_on_batch(batch)                      # NHWC [4, 512, 512, 1] in [0,1]
```

**Execution provider** — `OnnxSegmenter` auto-selects **CUDA → CoreML → CPU** (CPU is always
kept as a fallback). Override via environment variables:

- `WINMOL_ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"` — pin an explicit
  provider list (highest priority).
- `WINMOL_ONNX_FORCE_CPU=1` — force CPU only. Use for **exact fp32 parity** with the
  PyTorch/Keras reference, since CoreML/CUDA compute in fp16 (the ONNX parity tests set this).

## Tests

```bash
pytest                              # full suite (trains small models; a few minutes)
pytest -k onnx                      # by keyword
pytest tests/test_multiscale.py -q  # a single file
```

Tests are hermetic (synthetic data in `tmp_path`, `encoder_weights=None` so smp archs need no
download). Parity tests pin the CPU EP via `WINMOL_ONNX_FORCE_CPU` for bit-exact comparison.
