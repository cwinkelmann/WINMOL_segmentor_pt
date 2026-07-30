# WINMOL Segmentor (PyTorch)


PyTorch re-implementation of the WINMOL tree-stem segmentation U-Net, with model
export to ONNX and Keras (`.hdf5` / native `.keras`) for use in the WINMOL Analyzer.

## Install

```bash
pip install -e ".[train,wandb]"   # training deps + optional Weights & Biases
```

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
`--data-dir` (controlled by `TrainConfig.val_fraction`/`seed`, defaults 0.2/1 — not
exposed as CLI flags). To pin an explicit, shareable held-out set (e.g.
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

## RGBD input: pure RGB model vs RGBD

The trainer optionally consumes a **fourth (depth) channel**: with `--rgbd`, each dataset
dir additionally needs `depth/depth{N}.png|.tif` (paired by the same integer `N`; per-image
min-max normalized, carried through geometric augmentation only), and the model, ONNX
export, and `OnnxSegmenter` all become 4-channel. Until real depth data exists,
`scripts/simulate_depth.py` synthesizes plausible terrain depth from the masks (fractal
terrain + cylindrical stem bulges, degraded with bulge dropout, off-mask distractors, and
sensor noise so depth is a helpful-but-unreliable cue):

```bash
python scripts/simulate_depth.py --dataset /path/to/DS        # writes DS/depth/depth{N}.png
python -m training.run_train --data-dir /path/to/DS --rgbd --arch deeplabv3plus ...
```

Benchmark (deeplabv3plus/resnet34, SpecDS 454 pairs, held-out beech TestDS, synthetic
depth — see [`docs/rgbd-experiment.md`](docs/rgbd-experiment.md) for the full protocol):

![TestDS F1: RGB vs RGBD vs mismatch control](docs/assets/rgbd-vs-rgb-f1.png)

| input | TestDS F1 | precision | recall |
|---|---|---|---|
| RGB (pure model) | 0.7381 | 0.7468 | 0.7296 |
| RGBD, matched depth | **0.8915** | 0.9047 | 0.8787 |
| RGBD, mismatched depth (control) | 0.7546 | 0.7746 | 0.7356 |

The mismatch control (depth generated from the *wrong* image's mask) lands at the RGB
baseline, while matched depth gains **+0.15 F1** — i.e. the network genuinely fuses the
depth channel rather than exploiting a mask-derived shortcut. Because the synthetic test
depth is itself derived from the masks, the absolute RGBD number is an optimistic ceiling;
gains on real photogrammetry depth remain to be measured.
