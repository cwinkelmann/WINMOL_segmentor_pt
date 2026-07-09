"""Export a trained (or untrained) UNet to a contract-conformant ONNX file."""
import onnx
import torch
import torch.nn as nn

from .contract import (
    DYNAMIC_AXES, IMG_SIZE, IN_CHANNELS, INPUT_NAME, OPSET, OUTPUT_NAME,
    validate_onnx_model,
)


class _WithSigmoid(nn.Module):
    """Appends sigmoid so the exported graph emits probabilities (contract)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.model(x))


def export_to_onnx(model, path):
    model = model.eval()
    wrapped = _WithSigmoid(model).eval()
    dummy = torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapped, dummy, path,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes=DYNAMIC_AXES, opset_version=OPSET,
        do_constant_folding=True,
    )
    validate_onnx_model(onnx.load(path))
    return path


def export_to_pt(model, path):
    """Save the trained PyTorch weights (state_dict); reload into UNet().load_state_dict()."""
    torch.save(model.eval().state_dict(), path)   # eval() for consistency with other exporters
    return path
