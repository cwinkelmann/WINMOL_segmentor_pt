#!/usr/bin/env bash
# Stage the models-v2 release sources from carrot.
#
# The v2 models are trained on the GPU box. This pulls the
# already-built release artifacts -- fp32/fp16/int8 ONNX plus the eval_onnx JSON beside each --
# into results/release_v2/, which is what deploy_models_to_release.py --set v2 expects.
#
# Build them there first with build_v2.sh (fp16/int8 conversion + re-scoring), then:
#   scripts/fetch_release_v2.sh
#   python scripts/deploy_models_to_release.py --set v2 --dry-run
set -euo pipefail
# No baked-in default: this is a public repo, and a hostname plus username in it is
# infrastructure disclosure for no benefit. Set them in your environment, e.g.
#   export WINMOL_CARROT=user@training-box
#   export WINMOL_CARROT_RELEASE_DIR=/path/to/release_v2
HOST="${WINMOL_CARROT:?set WINMOL_CARROT to user@host of the training box}"
SRC="${WINMOL_CARROT_RELEASE_DIR:?set WINMOL_CARROT_RELEASE_DIR to the release dir on that box}"
DST="${1:-results/release_v2}"
mkdir -p "$DST"
scp "$HOST:$SRC/*.onnx" "$HOST:$SRC/*.eval.json" "$DST/"
ls -la "$DST"
