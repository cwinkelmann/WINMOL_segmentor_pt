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


# --- provider selection (CUDA / CoreML / CPU + env overrides) ---------------

from unittest.mock import patch as _patch  # noqa: E402
import os as _os  # noqa: E402
from winmol_unet.runtime import _default_providers  # noqa: E402


def _avail(names):
    return _patch("winmol_unet.runtime.ort.get_available_providers",
                  return_value=list(names))


def _clean_env():
    keys = ("WINMOL_ONNX_PROVIDERS", "WINMOL_ONNX_FORCE_CPU")
    return _patch.dict(_os.environ,
                       {k: "" for k in keys if k in _os.environ},
                       clear=False)


def test_providers_prefers_cuda_when_available():
    with _clean_env(), _avail(["CUDAExecutionProvider",
                               "CoreMLExecutionProvider",
                               "CPUExecutionProvider"]):
        _os.environ.pop("WINMOL_ONNX_FORCE_CPU", None)
        _os.environ.pop("WINMOL_ONNX_PROVIDERS", None)
        assert _default_providers() == ["CUDAExecutionProvider",
                                        "CPUExecutionProvider"]


def test_providers_prefers_coreml_when_no_cuda():
    with _clean_env(), _avail(["CoreMLExecutionProvider",
                               "CPUExecutionProvider"]):
        _os.environ.pop("WINMOL_ONNX_FORCE_CPU", None)
        _os.environ.pop("WINMOL_ONNX_PROVIDERS", None)
        assert _default_providers() == ["CoreMLExecutionProvider",
                                        "CPUExecutionProvider"]


def test_providers_cpu_only_when_nothing_else():
    with _clean_env(), _avail(["CPUExecutionProvider"]):
        _os.environ.pop("WINMOL_ONNX_FORCE_CPU", None)
        _os.environ.pop("WINMOL_ONNX_PROVIDERS", None)
        assert _default_providers() == ["CPUExecutionProvider"]


def test_force_cpu_env_overrides_coreml():
    with _avail(["CoreMLExecutionProvider", "CPUExecutionProvider"]), \
            _patch.dict(_os.environ, {"WINMOL_ONNX_FORCE_CPU": "1"}):
        _os.environ.pop("WINMOL_ONNX_PROVIDERS", None)
        assert _default_providers() == ["CPUExecutionProvider"]


def test_explicit_providers_env_takes_precedence():
    with _avail(["CoreMLExecutionProvider", "CPUExecutionProvider"]), \
            _patch.dict(_os.environ,
                        {"WINMOL_ONNX_PROVIDERS": "CPUExecutionProvider",
                         "WINMOL_ONNX_FORCE_CPU": "1"}):
        assert _default_providers() == ["CPUExecutionProvider"]
