#!/bin/bash
# R/Keras U-Net SINGLE-STAGE reference on an arbitrary 3-way split. DATA_ROOT holds
# <train>/ and <test>/ split subdirs, each with train/ (jpeg) + mask/ (gif). Stage 2 is
# disabled by pointing SPEC_DATASET at a nonexistent name — the R entrypoint's train_stage
# finds no data there and skips it, leaving a clean single-stage run. Held-out eval on TEST.
#
# Usage: scripts/run_r_single_stage.sh <out_dir> <data_root> [epochs] [img_size]
#   env: R_TRAIN_NAME (default train), R_TEST_NAME (default test), R_BATCH_SIZE (default 4)
set -euo pipefail

OUT_DIR="${1:?usage: run_r_single_stage.sh <out_dir> <data_root> [epochs] [img_size]}"
DATA_ROOT="${2:?data_root: dir containing <train>/ and <test>/ split subdirs}"
EPOCHS="${3:-100}"
IMG="${4:-512}"
TRAIN_NAME="${R_TRAIN_NAME:-train}"
TEST_NAME="${R_TEST_NAME:-test}"
R_REPO="/home/christian/hnee/WINMOL_segmentor"
PT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train_stdout.log"

echo "[r-single] single-stage on $DATA_ROOT/$TRAIN_NAME, ${EPOCHS} epochs, ${IMG}px -> $LOG"
# GPU=1 -> Dockerfile.gpu image + --gpus all. Datasets bind-mounted at /data (NOT nested
# under the /work repo mount, which would race). SPEC_DATASET=__none__ => stage 2 skipped.
GPU=1 bash "$R_REPO/docker/run.sh" train \
  -v "$DATA_ROOT:/data" \
  -e DATASET_ROOT=/data \
  -e DATASET_NAME="$TRAIN_NAME" \
  -e SPEC_DATASET=__none__ \
  -e TEST_DATASET="$TEST_NAME" \
  -e EVAL=1 \
  -e EPOCHS="$EPOCHS" \
  -e IMG_WIDTH="$IMG" \
  -e IMG_HEIGHT="$IMG" \
  -e BATCH_SIZE="${R_BATCH_SIZE:-4}" \
  > "$LOG" 2>&1

echo "[r-single] parsing $LOG"
python3 "$PT_REPO/scripts/parse_r_run.py" "$LOG" "$OUT_DIR"
echo "[r-single] done -> $OUT_DIR/metrics.csv + final.json"
