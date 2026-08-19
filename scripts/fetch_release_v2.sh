#!/usr/bin/env bash
# Stage the models-v2 release sources from carrot.
#
# The v2 models are trained on carrot (/raid/cwinkelmann/winmol/runs/...). This pulls the
# already-built release artifacts -- fp32/fp16/int8 ONNX plus the eval_onnx JSON beside each --
# into results/release_v2/, which is what deploy_models_to_release.py --set v2 expects.
#
# Build them there first with build_v2.sh (fp16/int8 conversion + re-scoring), then:
#   scripts/fetch_release_v2.sh
#   python scripts/deploy_models_to_release.py --set v2 --dry-run
set -euo pipefail
HOST="${WINMOL_CARROT:-cwinkelmann@10.188.1.1}"
SRC="${WINMOL_CARROT_RELEASE_DIR:-/raid/cwinkelmann/winmol/release_v2}"
DST="${1:-results/release_v2}"
mkdir -p "$DST"
scp "$HOST:$SRC/*.onnx" "$HOST:$SRC/*.eval.json" "$DST/"
ls -la "$DST"
