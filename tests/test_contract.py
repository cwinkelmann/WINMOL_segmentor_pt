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


def _model(in_dims, out_dims):
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, in_dims)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_dims)
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "g", [inp], [out])
    return helper.make_model(graph)


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


def _model_with_in_channels(in_ch):
    # validate_onnx_model only inspects graph input/output value_info, so an
    # Identity node is enough to build a well-formed ModelProto for shape tests.
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", in_ch, 512, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1, 512, 512])
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])],
                              "g", [inp], [out])
    return helper.make_model(graph)


def test_rgbd_channel_constant():
    assert contract.RGBD_IN_CHANNELS == 4


def test_validate_accepts_4ch_when_requested():
    contract.validate_onnx_model(_model_with_in_channels(4),
                                 in_channels=contract.RGBD_IN_CHANNELS)


def test_validate_rejects_4ch_by_default():
    with pytest.raises(ValueError):
        contract.validate_onnx_model(_model_with_in_channels(4))


def test_validate_rejects_3ch_when_rgbd_expected():
    with pytest.raises(ValueError):
        contract.validate_onnx_model(_model_with_in_channels(3), in_channels=4)
