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
