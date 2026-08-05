import numpy as np
import pytest
from unittest.mock import patch

from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet
from winmol_unet.runtime import OnnxSegmenter
import winmol_unet.runtime


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


def test_oom_raises_onnx_oom_error(tmp_path):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    seg = OnnxSegmenter(path)
    x = np.random.rand(1, 512, 512, 3).astype(np.float32)
    with patch.object(seg.session, "run", side_effect=RuntimeError("CUDA error: out of memory")):
        with pytest.raises(winmol_unet.runtime.OnnxOutOfMemoryError):
            seg.predict_on_batch(x)


def test_non_oom_error_propagates_unchanged(tmp_path):
    path = export_to_onnx(UNet().eval(), str(tmp_path / "m.onnx"))
    seg = OnnxSegmenter(path)
    x = np.random.rand(1, 512, 512, 3).astype(np.float32)
    with patch.object(seg.session, "run", side_effect=RuntimeError("some unrelated failure")):
        with pytest.raises(RuntimeError) as exc_info:
            seg.predict_on_batch(x)
        assert not isinstance(exc_info.value, winmol_unet.runtime.OnnxOutOfMemoryError)
        assert "some unrelated failure" in str(exc_info.value)


def test_onnx_segmenter_serves_rgbd(tmp_path, monkeypatch):
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    from winmol_unet.export import export_to_onnx
    from winmol_unet.model import UNet
    path = str(tmp_path / "rgbd.onnx")
    export_to_onnx(UNet(in_channels=4), path, in_channels=4)

    seg = OnnxSegmenter(path)
    assert seg.in_channels == 4
    out = seg.predict_on_batch(np.zeros((1, 512, 512, 4), dtype=np.float32))
    assert out.shape == (1, 512, 512, 1)
    with pytest.raises(ValueError):
        seg.predict_on_batch(np.zeros((1, 512, 512, 3), dtype=np.float32))
