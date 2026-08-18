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
    kwargs = dict(
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes=DYNAMIC_AXES, opset_version=OPSET,
        do_constant_folding=True,
    )
    # torch >= 2.5 defaults to the dynamo exporter, which warns that `dynamic_axes` is
    # "not recommended" and then silently ignores it for some architectures — segformer
    # came out as [1, 1, 512, 512], a FIXED batch, which the contract rejects and which
    # would break batched serving. The contract requires a symbolic batch axis, so pin
    # the TorchScript exporter that honours dynamic_axes. Remove once the dynamo path
    # respects it (validate_onnx_model below is what would catch the regression).
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(wrapped, dummy, path, **kwargs)
    validate_onnx_model(onnx.load(path))
    return path


def export_to_pt(model, path):
    """Save the trained PyTorch weights (state_dict); reload into UNet().load_state_dict()."""
    torch.save(model.eval().state_dict(), path)   # eval() for consistency with other exporters
    return path
