"""The analyzer-facing gate: a non-UNet architecture must export to a contract-
conformant ONNX and serve through the exact OnnxSegmenter interface the analyzer uses.
"""
import numpy as np
import pytest

from training.model_factory import build_model
from winmol_unet.export import export_to_onnx
from winmol_unet.runtime import OnnxSegmenter


@pytest.mark.parametrize("arch,encoder", [
    ("deeplabv3plus", None),
    ("hrnet", None),
    ("segformer", None),
    # convnext's production default is tu-convnext_large.dinov3_lvd1689m (203.3M), whose
    # ONNX artifact is ~800 MB. Exercise the same code path on the tiny variant: identical
    # stage structure and channel-list plumbing into the Unet decoder, ~28M to export.
    ("convnext", "tu-convnext_tiny.dinov3_lvd1689m"),
])
def test_arch_exports_and_serves_nhwc(tmp_path, arch, encoder, monkeypatch):
    # Pin the CPU EP so the check is EP-independent (CoreML/CUDA compute in fp16 and
    # would nudge sigmoid outputs past a tight epsilon — see WINMOL_ONNX_FORCE_CPU).
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    model = build_model(arch, encoder=encoder, encoder_weights=None).eval()
    path = export_to_onnx(model, str(tmp_path / f"{arch}.onnx"))   # opset 17 + contract validate
    out = OnnxSegmenter(path).predict_on_batch(
        np.random.rand(2, 512, 512, 3).astype(np.float32))
    assert out.shape == (2, 512, 512, 1)
    # sigmoid probabilities (exact fp32 on CPU, tiny epsilon for onnxruntime sigmoid)
    assert out.min() >= -1e-4 and out.max() <= 1.0 + 1e-4


def tmp_onnx_path():
    import tempfile, pathlib
    return pathlib.Path(tempfile.mkdtemp()) / "dpt.onnx"


def test_dpt_cannot_yet_export_onnx():
    """Pin DPT's known limitation so we notice the day it lifts.

    DPT's ViT encoder is built for a fixed 384 (or 224) input; `dynamic_img_size=True`
    lets it take our 512 tiles by interpolating the position embeddings.

    The blocker has MOVED. It used to be that timm's antialiased bicubic interpolation
    (`aten::_upsample_bicubic2d_aa`) had no ONNX lowering at opset 17, so export raised
    outright. With a newer torch the export now succeeds — but the graph it produces has
    a FIXED output batch axis ([1, 1, 512, 512]) while the contract requires a symbolic
    one, and the artifact is ~486 MB for 122M parameters.

    So DPT is still trainable and comparable, but not servable through the contract. This
    asserts the *production* path (export_to_onnx) rejects it, whatever the current
    reason. If this test starts failing, DPT became servable and belongs in the
    parametrization above.
    """
    from training.model_factory import build_model
    from winmol_unet.export import export_to_onnx

    model = build_model("dpt", encoder_weights=None).eval()
    with pytest.raises(Exception) as excinfo:
        export_to_onnx(model, str(tmp_onnx_path()))
    msg = str(excinfo.value)
    assert ("batch axis must be dynamic" in msg          # current: fixed output batch
            or "_upsample_bicubic2d_aa" in msg           # historical: no opset-17 lowering
            or ("Resize" in msg and "17" in msg)), f"DPT failed for a new reason: {msg[:300]}"
