"""Tests for the ONNX contract module."""
import onnx
import pytest
from onnx import TensorProto, helper

from winmol_unet import contract


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
