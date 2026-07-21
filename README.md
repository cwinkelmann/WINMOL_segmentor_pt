# WINMOL Segmentor (PyTorch)
TODO: document where to get some sample data from. 

PyTorch re-implementation of the WINMOL tree-stem segmentation U-Net, with model
export to ONNX and Keras (`.hdf5` / native `.keras`) for use in the WINMOL Analyzer.

## Install

```bash
pip install -e ".[train,wandb]"   # training deps + optional Weights & Biases
```

## Pretrained models

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
`scripts/convert_keras_to_onnx.py` (see [Model optimization](#model-optimization--faster-cpugpu-inference)).

**Optimised models + ONNX conversions — GitHub Release [`models-v1`](../../releases/tag/models-v1).**
The release re-hosts *only* the ONNX (the HDF5 stay on Zenodo), each flavour in fp32 / `_fp16`
(GPU) / `_int8` (CPU):

- **PyTorch UNet** — `unet_fp32.onnx` (TestDS F1 0.760) plus the width-0.5 optimised
  `unet_w05_int8_cpu.onnx` (**10× faster CPU, lossless**) and `unet_w05_fp16_gpu.onnx` (GPU, lossless).
- **Zenodo flavours as ONNX** — `model_UNet_<FLAVOUR>_512{.onnx,_fp16.onnx,_int8.onnx}`: converted
  from the Zenodo HDF5 (numerically identical) and quantized **post-training** (no retraining).

### Retraining / reproducing

The PyTorch UNet is reproducible from the [Training](#training) workflow; the released optimised
model was trained single-stage on SpecDS, then quantized:

```bash
# 1. train (full, or --width-mult 0.5 for the smaller/faster model)
python -m training.run_train --arch unet --width-mult 0.5 \
  --data-dir <SpecDS> --test-data-dir <TestDS> --out-dir output/w05 --device cuda

# 2. quantize the exported ONNX — no retraining (see Model optimization)
python scripts/quantize_unet.py output/w05/model.onnx output/w05/model_int8.onnx \
  --mode static --calib-dir <SpecDS> --n-samples 128
```

The upstream Zenodo flavours **can't be retrained** (their training imagery isn't public), but they
can still be converted and quantized — which is exactly what the release provides.

## Training

The data directory must contain `train/` (jpeg images named `trainN.jpeg`) and
`mask/` (gif masks named `maskN.gif`), paired by the integer `N`.

```bash
##  TODO this looks wrong. --data-dir TestDS is wrong, that would be the testing dataset
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
## Model optimization — faster CPU/GPU inference

Two independent levers speed up UNet inference while keeping the ONNX contract. Full study +
numbers: `docs/2026-07-21-cpu-inference-speedup-results.md`.

| lever | needs retraining? | backend | typical win | accuracy |
|-------|-------------------|---------|-------------|----------|
| **static int8** quantization | **no** (post-training) | CPU (AVX-VNNI) | ~3× | lossless |
| **fp16** quantization | **no** (post-training) | GPU (Tensor Cores) | ~1.7× | lossless |
| **width scaling** (`width_mult`) | yes (retrain) | CPU + GPU | ~4× FLOPs | ~flat on this task |

Quantization is **post-training** — it works on any already-trained ONNX (including the upstream
Keras flavours you can't retrain). Width scaling is the only lever that needs a retrain.

```bash
# int8 (CPU) — static, calibrated on real tiles (no labels needed), ~3× lossless:
python scripts/quantize_unet.py model.onnx model_int8.onnx \
  --mode static --calib-dir <StemDataset dir> --n-samples 128

# fp16 (GPU) — no calibration, lossless:
python scripts/quantize_unet.py model.onnx model_fp16.onnx --mode fp16

# smaller model (needs retraining) — half the channel width, ~4× fewer FLOPs:
python -m training.run_train --arch unet --width-mult 0.5 --data-dir <DS> --out-dir out/w05

# measure it — CPU latency+F1, and GPU latency:
python scripts/benchmark_cpu_latency.py model.onnx model_int8.onnx --test-data-dir <TestDS> \
  --threads 4 --thread-sweep 1,2,4,8
python scripts/benchmark_gpu_latency.py --widths 1.0,0.5,0.25 --dtypes fp32,fp16   # winmol-onnxgpu
```

**Don't use dynamic int8** (`--mode dynamic`) for these conv nets — ORT has no fast dynamic-conv
kernel, so it runs ~3× *slower*. Static int8 or fp16 only.

**Fastest GPU inference — TensorRT EP.** The ONNX we ship *is* what TensorRT consumes: serve any
`.onnx` through the TensorRT execution provider (build `Dockerfile.onnxgpu`, which bundles
onnxruntime-gpu + TensorRT 10) for ~**1.9× over the CUDA EP** (and 2.7× throughput at batch) via
fp16. Set `WINMOL_ONNX_PROVIDERS="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"`;
TensorRT builds+caches an engine on the target GPU (device/TRT-version specific — never shipped).
GPU int8 is *not* worth it (slower than fp16 here) — keep int8 for CPU.

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

`scripts/build_keras_onnx_models.sh` runs the whole flow end-to-end (convert all flavours →
fp16 + domain-calibrated int8 → verify F1). Publish artifacts to a GitHub Release with
`scripts/deploy_models_to_release.py` (`--dry-run` first).

