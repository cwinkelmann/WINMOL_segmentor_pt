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


class _WithSoftmax(nn.Module):
    """Appends channel softmax so a multiclass graph emits per-class probabilities."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


class _SpeciesSlice(nn.Module):
    """Softmax, then one class channel — a [N,1,H,W] probability map in [0,1].

    This is how one multiclass training serves every per-species registry slot
    through the FROZEN analyzer contract: the sliced graph is shape- and
    range-identical to a sigmoid binary model, so the Analyzer loads it unchanged.
    """

    def __init__(self, model, class_idx):
        super().__init__()
        self.model = model
        self.class_idx = class_idx

    def forward(self, x):
        probs = torch.softmax(self.model(x), dim=1)
        return probs[:, self.class_idx:self.class_idx + 1]


def _export(wrapped, path):
    dummy = torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    kwargs = dict(
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes=DYNAMIC_AXES, opset_version=OPSET,
        do_constant_folding=True,
    )
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False                 # same regression guard as export_to_onnx
    torch.onnx.export(wrapped.eval(), dummy, path, **kwargs)
    return path


def export_multiclass_to_onnx(model, path):
    """Full multiclass head: [N,C,H,W] softmax. NOT the analyzer contract (that is
    [N,1,H,W]); this artifact is for evaluation/tooling. Per-species serving goes
    through export_species_slice instead."""
    _export(_WithSoftmax(model.eval()), path)
    m = onnx.load(path)                          # relaxed check: symbolic batch only
    batch_dim = m.graph.input[0].type.tensor_type.shape.dim[0]
    assert not batch_dim.dim_value, "multiclass export lost its symbolic batch axis"
    return path


def export_species_slice(model, class_idx, path):
    """One class channel as a contract-conformant [N,1,H,W] probability model."""
    _export(_SpeciesSlice(model.eval(), class_idx), path)
    validate_onnx_model(onnx.load(path))         # the frozen contract, fully enforced
    return path
