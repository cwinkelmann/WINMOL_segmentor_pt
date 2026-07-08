import numpy as np

from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet
from winmol_unet.runtime import OnnxSegmenter


def test_predict_on_batch_nhwc_roundtrip(tmp_path):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    seg = OnnxSegmenter(path)
    x = np.random.rand(3, 512, 512, 3).astype(np.float32)
    out = seg.predict_on_batch(x)
    assert out.shape == (3, 512, 512, 1)
    assert out.dtype == np.float32
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid range


def test_summary_runs(tmp_path, capsys):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    OnnxSegmenter(path).summary()
    assert "512" in capsys.readouterr().out
