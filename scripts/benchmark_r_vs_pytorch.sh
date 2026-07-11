#!/bin/bash
# Orchestrate the R-vs-PyTorch comparison: run the R U-Net container first (per-epoch +
# held-out TestDS), then each PyTorch arch the same way, then assemble one SUMMARY.md.
# Canonical two-stage GenDS10 -> SpecDS -> TestDS. R and PyTorch run SEQUENTIALLY (one GPU).
set -euo pipefail

PT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARE="/data/mnt/storage/Datasets/Winmol/data"
BASE="$PT_REPO/results/r_vs_pytorch"
EPOCHS="${EPOCHS:-100}"
mkdir -p "$BASE/r"

wait_gpu_free() {   # block until GPU memory < 2000 MiB
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 2000 ] && break
    sleep 20
  done
}

echo "=== [1/3] R reference (GenDS10->SpecDS->TestDS) ==="
wait_gpu_free
bash "$PT_REPO/scripts/run_r_reference.sh" "$BASE/r" "$EPOCHS"

echo "=== [2/3] PyTorch archs (unet/deeplabv3plus/hrnet) ==="
wait_gpu_free
docker rm -f winmol-bench 2>/dev/null || true
docker run --rm --gpus all --shm-size=16g -e HOME=/tmp --user "$(id -u):$(id -g)" \
  --name winmol-bench \
  -v "$SHARE:/data" \
  -v "$BASE:/app/output" \
  --entrypoint python winmol-train \
  scripts/benchmark_architectures.py \
    --gen-data-dir /data/beech/GenDS10_ready \
    --spec-data-dir /data/SpecDS \
    --test-data-dir /data/TestDS \
    --out-dir /app/output --epochs "$EPOCHS" --batch-size 4 --device cuda \
    --no-cache-dataset --num-workers 8 --encoder resnet34 --encoder-weights none --seed 1

echo "=== [3/3] combined SUMMARY.md ==="
python3 "$PT_REPO/scripts/write_comparison_summary.py" "$BASE"
echo "=== done -> $BASE/SUMMARY.md ==="
