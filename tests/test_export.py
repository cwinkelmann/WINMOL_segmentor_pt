# tests/test_export.py
import numpy as np
import onnx
import onnxruntime as ort
import torch

from winmol_unet import contract
from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet


def test_export_is_contract_conformant(tmp_path):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    contract.validate_onnx_model(onnx.load(path))  # raises on violation


def test_export_parity_torch_vs_onnx(tmp_path):
    model = UNet().eval()
    path = export_to_onnx(model, str(tmp_path / "m.onnx"))
    x = np.random.rand(2, 3, 512, 512).astype(np.float32)

    with torch.no_grad():
        torch_out = torch.sigmoid(model(torch.from_numpy(x))).numpy()

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run([contract.OUTPUT_NAME], {contract.INPUT_NAME: x})[0]

    assert onnx_out.shape == (2, 1, 512, 512)
    assert np.allclose(torch_out, onnx_out, rtol=0.0, atol=1e-4)
