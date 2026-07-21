#!/usr/bin/env bash
# Convert the upstream WINMOL Keras HDF5 flavours to contract-conformant ONNX, then apply
# POST-TRAINING quantization (no retraining): fp16 (GPU, lossless) + static int8 (CPU,
# domain-calibrated). Verifies fp32/fp16/int8 F1 per flavour on a labeled in-domain set.
#
# Naming schema preserved: model_UNet_<FLAVOUR>_512[.onnx|_fp16.onnx|_int8.onnx].
# Two images: winmol-convert (TF2.15 + tf2onnx) for the HDF5 load; winmol-test (torch +
# onnxruntime) for int8 calibration + F1 eval.
set -e
cd "$(dirname "$0")/.."
MODELS=/data/mnt/storage/hnee/WINMOL/models
DATA=/data/mnt/storage/Datasets/Winmol/data
OUT=results/cpu_speedup/keras_onnx
mkdir -p "$OUT"

CONV="docker run --rm -e HOME=/tmp -e PYTHONPATH=/app --user $(id -u):$(id -g) \
  -v $PWD:/app -w /app -v $MODELS:/models:ro winmol-convert"
TEST="docker run --rm -e HOME=/tmp -e PYTHONPATH=/app --user $(id -u):$(id -g) \
  -v $PWD:/app -w /app -v $DATA:/data:ro --entrypoint python winmol-test"

echo "=== [1/2] convert HDF5 -> fp32 ONNX + fp16 (winmol-convert) ==="
$CONV -c '
import sys, os, re, glob; sys.path.insert(0,"scripts")
from convert_keras_to_onnx import convert
from quantize_unet import to_fp16
out="results/cpu_speedup/keras_onnx"
for h in sorted(glob.glob("/models/zenodo_analyzer/*.hdf5")):
    name=re.sub(r"_\d{4}-\d{2}-\d{2}_\d+$","",os.path.splitext(os.path.basename(h))[0])
    convert(h, f"{out}/{name}.onnx"); to_fp16(f"{out}/{name}.onnx", f"{out}/{name}_fp16.onnx")
    print("converted+fp16:", name, flush=True)
'

echo "=== [2/2] static int8 (domain-calibrated) + verify F1 (winmol-test) ==="
$TEST scripts/quantize_verify_keras.py
echo "=== done: results/cpu_speedup/keras_onnx/ ==="
