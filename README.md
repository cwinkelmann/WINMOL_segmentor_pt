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
(`None`, or `imagenet` for a pretrained encoder — needs network). Any architecture exports
a contract-conformant `.onnx` (+ `.pt`); the Keras `.hdf5`/`.keras` mirror is UNet-specific,
so **non-UNet models export ONNX + `.pt` only**. Example:

```bash
python -m training.run_train --data-dir /path/to/TestDS --out-dir output/deeplab \
  --arch deeplabv3plus --encoder resnet34 --encoder-weights imagenet \
  --epochs 20 --device mps
```

The analyzer consumes any of these transparently through `OnnxSegmenter` once its ONNX
loading branch is in place — it never inspects the architecture.

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
