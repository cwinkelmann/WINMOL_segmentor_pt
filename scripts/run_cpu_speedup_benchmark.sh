#!/usr/bin/env bash
# Final CPU-speedup benchmark: quantize the width-scaled UNets, then run ONE quiet-CPU
# latency+F1 sweep over every candidate. See docs/superpowers/specs/2026-07-21-cpu-inference-speedup-design.md
#
# PREREQUISITE: width retrains finished (results/cpu_speedup/train/{w10,w05,w025}/model.onnx)
# and the fp32 quant variants exist (results/cpu_speedup/models/fp32unet_{dyn,static}.onnx).
# RUN ON A QUIET CPU — no other docker CPU work concurrently, or latency numbers are skewed.
set -e
cd "$(dirname "$0")/.."
DATA=/data/mnt/storage/Datasets/Winmol/data
M=results/cpu_speedup/models
THREADS=${THREADS:-4}

DR="docker run --rm -e HOME=/tmp -e PYTHONPATH=/app --user $(id -u):$(id -g) \
  -v $PWD:/app -w /app -v $DATA:/data:ro --entrypoint python winmol-test"

echo "=== quantize width variants (static int8, SpecDS calib) ==="
$DR -c '
import sys; sys.path.insert(0,"scripts")
from quantize_unet import quantize_static_int8
for tag in ["w10","w05","w025"]:
    src=f"results/cpu_speedup/train/{tag}/model.onnx"
    dst=f"results/cpu_speedup/models/{tag}_static.onnx"
    print("static", tag, flush=True)
    quantize_static_int8(src, dst, "/data/SpecDS", n_samples=128)
print("QUANT_DONE", flush=True)
'

echo "=== ONE clean latency+F1 sweep (threads=$THREADS) ==="
$DR scripts/benchmark_cpu_latency.py \
  "fp32=results/r_vs_pytorch_lrfix/unet/model.onnx" \
  "fp32_dyn=$M/fp32unet_dyn.onnx" \
  "fp32_static=$M/fp32unet_static.onnx" \
  "w10_static=$M/w10_static.onnx" \
  "w05=results/cpu_speedup/train/w05/model.onnx" \
  "w05_static=$M/w05_static.onnx" \
  "w025=results/cpu_speedup/train/w025/model.onnx" \
  "w025_static=$M/w025_static.onnx" \
  --test-data-dir /data/TestDS --threads "$THREADS" \
  --warmup 5 --runs 30 \
  --json-out results/cpu_speedup/sweep.json --md-out results/cpu_speedup/sweep.md

echo "=== thread-scaling on the two extremes (batch-1) ==="
$DR scripts/benchmark_cpu_latency.py \
  "fp32=results/r_vs_pytorch_lrfix/unet/model.onnx" \
  "w025_static=$M/w025_static.onnx" \
  --threads "$THREADS" --thread-sweep 1,2,4,8 --batches 1 --warmup 5 --runs 30 \
  --json-out results/cpu_speedup/thread_sweep.json
echo "=== done: results/cpu_speedup/sweep.md ==="
