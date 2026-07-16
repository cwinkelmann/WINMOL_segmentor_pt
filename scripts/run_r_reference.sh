#!/bin/bash
# Run the R/Keras U-Net reference: canonical two-stage GenDS10 -> SpecDS, then held-out
# TestDS eval. Tees stdout (R container is `--rm`) and parses per-epoch + final metrics.
#
# Usage: scripts/run_r_reference.sh <out_dir> [epochs]
set -euo pipefail

OUT_DIR="${1:?usage: run_r_reference.sh <out_dir> [epochs]}"
EPOCHS="${2:-100}"
R_REPO="/home/christian/hnee/WINMOL_segmentor"
SHARE="/data/mnt/storage/Datasets/Winmol/data"
PT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train_stdout.log"

echo "[r-ref] two-stage GenDS10->SpecDS + TestDS eval, ${EPOCHS} epochs/stage -> $LOG"
# GPU=1 selects Dockerfile.gpu + --gpus all; run.sh mounts the R repo at /work and forwards
# the extra -v/-e flags into `docker run`. Datasets come from the external share at /data
# (DATASET_ROOT), NOT nested under /work (the script warns overlapping mounts race).
GPU=1 bash "$R_REPO/docker/run.sh" train \
  -v "$SHARE:/data" \
  -e DATASET_ROOT=/data \
  -e DATASET_NAME="${R_DATASET_NAME:-beech/GenDS10}" \
  -e SPEC_DATASET="${R_SPEC_DATASET:-SpecDS}" \
  -e TEST_DATASET=TestDS \
  -e EVAL=1 \
  -e EPOCHS="$EPOCHS" \
  -e IMG_WIDTH="${R_IMG_WIDTH:-256}" \
  -e IMG_HEIGHT="${R_IMG_HEIGHT:-256}" \
  -e BATCH_SIZE="${R_BATCH_SIZE:-4}" \
  > "$LOG" 2>&1

echo "[r-ref] parsing $LOG"
python3 "$PT_REPO/scripts/parse_r_run.py" "$LOG" "$OUT_DIR"
echo "[r-ref] done -> $OUT_DIR/metrics.csv + final.json"
