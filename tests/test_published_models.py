"""Published release artifacts must still load and serve through this package.

The models on Zenodo and on the GitHub releases are the deliverable; a refactor that
quietly stops serving them is the worst outcome this repo has. `contract.py` is the
frozen interface that guarantees it, but nothing was checking the guarantee against the
*actual published files* — only against models built in-test, which share whatever
assumption the exporter happens to hold at the time.

Artifacts live in `results/`, which is gitignored (they are 30-120 MB each), so this
skips when they are absent. That makes it a check that runs on the release machine and
in any checkout that has fetched a release, and stays silent elsewhere. It is not a
substitute for the export tests — it is the end-to-end counterpart to them.

Populate the directory with `scripts/fetch_release_v2.sh` before running.
"""
import glob
import os

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from winmol_unet.contract import validate_onnx_model  # noqa: E402
from winmol_unet.runtime import OnnxSegmenter  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_DIRS = ("results/release_v2", "results/release_v1")


def _published_models():
    found = []
    for d in RELEASE_DIRS:
        found.extend(sorted(glob.glob(os.path.join(REPO, d, "*.onnx"))))
    return found


published = _published_models()
requires_artifacts = pytest.mark.skipif(
    not published, reason="no release artifacts present (run scripts/fetch_release_v2.sh)")


@requires_artifacts
@pytest.mark.parametrize("path", published, ids=lambda p: os.path.basename(p))
def test_published_model_satisfies_the_frozen_contract(path):
    validate_onnx_model(onnx.load(path))


@requires_artifacts
@pytest.mark.parametrize("path", published, ids=lambda p: os.path.basename(p))
def test_published_model_serves_through_the_runtime(path, monkeypatch):
    """The analyzer's path: OnnxSegmenter in, NHWC probabilities out.

    CPU execution provider is forced for the same reason the parity tests force it —
    CoreML and CUDA compute in fp16 and are not bit-exact.
    """
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    seg = OnnxSegmenter(path)
    out = seg.predict_on_batch(np.random.rand(1, 512, 512, 3).astype("float32"))
    assert out.shape == (1, 512, 512, 1)
    # sigmoid is baked in at export, so the output is a probability, not a logit
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
