#!/bin/bash
# Orchestrate the BAMFORESTS single-stage experiments on the SITE-HELD-OUT split (train/val =
# Stadtwald+Tretzendorf, test = Hain — a location the model never saw), SEQUENTIALLY (one GPU).
# Waits for the GPU to be free before each stage so it can be queued behind another run.
#
#   0. build the site-held-out split (if missing) via build_bamforests_sitesplit.py
#   1. R WINMOL_segmentor  — single-stage, classic pipeline, held-out Hain test
#   2. PyTorch headline    — 3 archs, classic (R-matched) aug, single-stage
#   3. PyTorch ablation    — 3 archs x aug arms (RESIZE group + native multiscale arm)
#
# Deliverables land under results/bamforests_experiments/{r_single,headline,ablation}.
#
# Env: COCO_ROOT, SITE_SPLIT (dataset dir), IMG (input size), EPOCHS_HEADLINE, EPOCHS_ABLATION,
#      LIMIT_TRAIN/LIMIT_VAL/LIMIT_TEST (0 = all annotated images per split).
set -uo pipefail

PT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COCO_ROOT="${COCO_ROOT:-/data/mnt/storage/Datasets/BAMFORESTS/coco1024}"
SITE_SPLIT="${SITE_SPLIT:-/data/mnt/storage/Datasets/BAMFORESTS/site_split}"
IMG="${IMG:-512}"
EPOCHS_HEADLINE="${EPOCHS_HEADLINE:-100}"
EPOCHS_ABLATION="${EPOCHS_ABLATION:-40}"
BASE="$PT_REPO/results/bamforests_experiments"
mkdir -p "$BASE"
LOG="$BASE/orchestrator.log"
exec > >(tee -a "$LOG") 2>&1

wait_gpu_free() {   # block until GPU memory < 2000 MiB
  echo "[orch] waiting for GPU to be free..."
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 2000 ] && break
    sleep 30
  done
  echo "[orch] GPU free (${used} MiB used)."
}

pytorch_run() {   # $1=out-subdir  $2..=extra args to augmentation_ablation.py
  local out="$1"; shift
  docker rm -f winmol-bam-exp 2>/dev/null || true
  docker run --rm --gpus all --shm-size=16g -e HOME=/tmp -e PYTHONPATH=/app \
    --user "$(id -u):$(id -g)" --name winmol-bam-exp \
    -v "$SITE_SPLIT:/data:ro" \
    -v "$PT_REPO/scripts:/app/scripts:ro" -v "$PT_REPO/training:/app/training:ro" \
    -v "$PT_REPO/winmol_unet:/app/winmol_unet:ro" \
    -v "$BASE:/app/output" \
    --entrypoint python winmol-train scripts/augmentation_ablation.py \
      --train-dir /data/train --val-dir /data/val --test-dir /data/test \
      --out-dir "/app/output/$out" --device cuda --img-size "$IMG" \
      --no-cache-dataset --num-workers 8 --encoder resnet34 --encoder-weights imagenet "$@"
}

echo "=== [0/3] build site-held-out split (train/val=Stadtwald+Tretzendorf, test=Hain) ==="
if [ ! -d "$SITE_SPLIT/train/train" ]; then
  python3 "$PT_REPO/scripts/build_bamforests_sitesplit.py" --coco-root "$COCO_ROOT" \
    --dst "$SITE_SPLIT" \
    --limit-train "${LIMIT_TRAIN:-0}" --limit-val "${LIMIT_VAL:-0}" \
    --limit-test "${LIMIT_TEST:-0}" --limit-insite "${LIMIT_INSITE:-0}" \
    || { echo "[orch] site-split build failed"; exit 1; }
else
  echo "[orch] site-split already present at $SITE_SPLIT (skipping build)"
fi

echo "=== [1/3] R single-stage ${EPOCHS_HEADLINE}ep @${IMG}px, held-out Hain test ==="
wait_gpu_free
R_BATCH_SIZE=4 bash "$PT_REPO/scripts/run_r_single_stage.sh" "$BASE/r_single" "$SITE_SPLIT" \
  "$EPOCHS_HEADLINE" "$IMG" || echo "[orch] R stage failed (continuing)"

echo "=== [2/3] PyTorch headline: 3 archs, classic aug, ${EPOCHS_HEADLINE}ep @${IMG}px ==="
wait_gpu_free
pytorch_run headline --arms classic --epochs "$EPOCHS_HEADLINE" --patience 12 \
  || echo "[orch] PyTorch headline failed (continuing)"

echo "=== [3/3] PyTorch ablation: 3 archs x arms, ${EPOCHS_ABLATION}ep @${IMG}px ==="
wait_gpu_free
pytorch_run ablation --epochs "$EPOCHS_ABLATION" --patience 8 \
  || echo "[orch] PyTorch ablation failed (continuing)"

echo "=== done -> $BASE (r_single/final.json, headline/SUMMARY.md, ablation/SUMMARY.md) ==="
