"""The analyzer-facing gate: a non-UNet architecture must export to a contract-
conformant ONNX and serve through the exact OnnxSegmenter interface the analyzer uses.
"""
import numpy as np
import pytest

from training.model_factory import build_model
from winmol_unet.export import export_to_onnx
from winmol_unet.runtime import OnnxSegmenter


@pytest.mark.parametrize("arch", ["deeplabv3plus", "hrnet", "segformer"])
def test_arch_exports_and_serves_nhwc(tmp_path, arch, monkeypatch):
    # Pin the CPU EP so the check is EP-independent (CoreML/CUDA compute in fp16 and
    # would nudge sigmoid outputs past a tight epsilon — see WINMOL_ONNX_FORCE_CPU).
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    model = build_model(arch, encoder_weights=None).eval()
    path = export_to_onnx(model, str(tmp_path / f"{arch}.onnx"))   # opset 17 + contract validate
    out = OnnxSegmenter(path).predict_on_batch(
        np.random.rand(2, 512, 512, 3).astype(np.float32))
    assert out.shape == (2, 512, 512, 1)
    # sigmoid probabilities (exact fp32 on CPU, tiny epsilon for onnxruntime sigmoid)
    assert out.min() >= -1e-4 and out.max() <= 1.0 + 1e-4


def test_dpt_cannot_yet_export_onnx():
    """Pin DPT's known limitation so we notice the day it lifts.

    DPT's ViT encoder is built for a fixed 384 (or 224) input and asserts on anything
    else. `dynamic_img_size=True` lets it take our 512 tiles by interpolating the
    position embeddings — but timm does that with antialiased bicubic, and
    `aten::_upsample_bicubic2d_aa` has no ONNX lowering at opset 17 through 20.

    Exporting at the encoder's native 384 does work, but produces fixed 384 spatial dims,
    which the contract rejects (it allows 512 or symbolic, not a different fixed size),
    and a ~485 MB artifact for 122M parameters.

    So DPT is trainable and comparable, but not servable through the current contract.
    If this test starts failing, DPT became exportable and belongs in the parametrization
    above.
    """
    import io

    import torch

    from training.model_factory import build_model

    model = build_model("dpt", encoder_weights=None).eval()
    with pytest.raises(torch.onnx.errors.UnsupportedOperatorError,
                       match="_upsample_bicubic2d_aa"):
        torch.onnx.export(model, torch.zeros(1, 3, 512, 512), io.BytesIO(),
                          opset_version=17, input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
