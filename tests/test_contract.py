"""The frozen cross-repo ONNX contract, the preprocessing it assumes, and the
published models that must satisfy it."""
import glob
import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from winmol_unet import contract, preprocess


def test_package_imports():
    import winmol_unet
    assert hasattr(winmol_unet, "__version__")


def test_contract_constants():
    assert contract.IMG_SIZE == 512
    assert contract.IN_CHANNELS == 3
    assert contract.OUT_CHANNELS == 1
    assert contract.OPSET == 17
    assert contract.INPUT_NAME == "input"
    assert contract.OUTPUT_NAME == "output"
    assert contract.DYNAMIC_AXES == {"input": {0: "batch"}, "output": {0: "batch"}}


def _model(in_dims, out_dims, sigmoid=True, opset=contract.OPSET, trailing=None):
    """A contract-conformant graph by default; knobs to violate one term at a time.

    `trailing` appends an op AFTER the sigmoid, which is what quantisation
    (DequantizeLinear), fp16 conversion (Cast) and the Keras converter (Reshape)
    all do to real models — so the sigmoid is present but is not the last node.
    """
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, in_dims)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_dims)
    nodes, cur = [], "input"
    if sigmoid:
        nodes.append(helper.make_node("Sigmoid", [cur], ["sig"]))
        cur = "sig"
    if trailing:
        nodes.append(helper.make_node(trailing, [cur], ["post"]))
        cur = "post"
    nodes.append(helper.make_node("Identity", [cur], ["output"]))
    graph = helper.make_graph(nodes, "g", [inp], [out])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", opset)])


def test_validate_accepts_fixed_512():
    contract.validate_onnx_model(_model(["batch", 3, 512, 512], ["batch", 1, 512, 512]))


def test_validate_accepts_dynamic_spatial_output():
    # smp HRNet exports symbolic spatial dims; at 512 input it still yields 512.
    contract.validate_onnx_model(_model(["batch", 3, 512, 512], ["batch", 1, "h", "w"]))


def test_validate_rejects_wrong_fixed_spatial():
    with pytest.raises(ValueError):
        contract.validate_onnx_model(_model(["batch", 3, 512, 512], ["batch", 1, 256, 256]))


def test_validate_rejects_fixed_batch():
    with pytest.raises(ValueError):
        contract.validate_onnx_model(_model([1, 3, 512, 512], ["batch", 1, 512, 512]))


def test_validate_rejects_wrong_channels():
    with pytest.raises(ValueError):
        contract.validate_onnx_model(_model(["batch", 3, 512, 512], ["batch", 2, 512, 512]))


# --- the sigmoid terminal -----------------------------------------------------
# validate_onnx_model checked shape only, so a logit-output graph passed and then
# produced silently wrong masks in the Analyzer, which thresholds at 0.5 assuming
# probabilities. Measured 2026-08-20.


def test_validate_rejects_logit_output():
    with pytest.raises(ValueError, match="[Ss]igmoid"):
        contract.validate_onnx_model(
            _model(["batch", 3, 512, 512], ["batch", 1, 512, 512], sigmoid=False))


@pytest.mark.parametrize("trailing", ["Cast", "DequantizeLinear", "Reshape"])
def test_validate_accepts_sigmoid_that_is_not_the_last_node(trailing):
    """The predicate is "a sigmoid on the path to the output", not "ends in one".

    Requiring the terminal node to be Sigmoid rejects 11 of the 15 real models in
    this project: fp16 variants end in Cast, int8 in DequantizeLinear, and the
    published Keras-converted UNets in Reshape. All emit probabilities.
    """
    contract.validate_onnx_model(
        _model(["batch", 3, 512, 512], ["batch", 1, 512, 512], trailing=trailing))


# --- the opset ----------------------------------------------------------------


def test_validate_rejects_opset_below_17():
    with pytest.raises(ValueError, match="opset"):
        contract.validate_onnx_model(
            _model(["batch", 3, 512, 512], ["batch", 1, 512, 512], opset=11))


def test_validate_accepts_opset_above_17():
    """`>= 17`, not `== 17` — a future re-export at 18 is not a contract breach."""
    contract.validate_onnx_model(
        _model(["batch", 3, 512, 512], ["batch", 1, 512, 512], opset=18))


# --- preprocessing --------------------------------------------------------
# The pixel-level transforms the contract assumes on both sides of the ONNX
# boundary: uint8 -> float01 on the way in, and resizing to the fixed IMG_SIZE.


def test_to_float01_uint8():
    arr = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = preprocess.to_float01(arr)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)


def test_resize_batch_shape():
    batch = np.zeros((4, 100, 120, 3), dtype=np.float32)
    out = preprocess.resize_batch(batch, size=512)
    assert out.shape == (4, 512, 512, 3)
    assert out.dtype == np.float32


# --- published release artifacts -------------------------------------------
# The models on Zenodo and on the GitHub releases are the deliverable; a refactor
# that quietly stops serving them is the worst outcome this repo has. The tests
# above check the contract against models built in-test, which share whatever
# assumption the exporter happens to hold at the time. These check it against the
# *actual published files*.
#
# Artifacts live in `results/`, which is gitignored (they are 30-120 MB each), so
# this skips when they are absent. That makes it a check that runs on the release
# machine and in any checkout that has fetched a release, and stays silent
# elsewhere. It is not a substitute for the export tests — it is the end-to-end
# counterpart to them.
#
# Populate the directory with `scripts/fetch_release_v2.sh` before running.

from winmol_unet.runtime import OnnxSegmenter  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RELEASE_DIRS = ("results/release_v2", "results/release_v1")


def _published_models():
    found = []
    for d in _RELEASE_DIRS:
        found.extend(sorted(glob.glob(os.path.join(_REPO, d, "*.onnx"))))
    return found


_published = _published_models()
_requires_artifacts = pytest.mark.skipif(
    not _published, reason="no release artifacts present (run scripts/fetch_release_v2.sh)")


@_requires_artifacts
@pytest.mark.parametrize("path", _published, ids=lambda p: os.path.basename(p))
def test_published_model_satisfies_the_frozen_contract(path):
    contract.validate_onnx_model(onnx.load(path))


@_requires_artifacts
@pytest.mark.parametrize("path", _published, ids=lambda p: os.path.basename(p))
def test_published_model_serves_through_the_runtime(path, force_cpu_onnx):
    """The analyzer's path: OnnxSegmenter in, NHWC probabilities out.

    CPU execution provider is forced for the same reason the parity tests force it —
    CoreML and CUDA compute in fp16 and are not bit-exact.
    """
    seg = OnnxSegmenter(path)
    out = seg.predict_on_batch(np.random.rand(1, 512, 512, 3).astype("float32"))
    assert out.shape == (1, 512, 512, 1)
    # sigmoid is baked in at export, so the output is a probability, not a logit
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
