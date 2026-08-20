"""Export any nn.Module to ONNX and serve it: contract shape, torch-vs-ONNX parity across architectures, provider selection, and fp16 quantization."""
import os
import sys
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from winmol_unet import contract
from winmol_unet.contract import validate_onnx_model
from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet
from winmol_unet.runtime import OnnxSegmenter
from winmol_unet.training.model_factory import build_model
import winmol_unet.runtime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from quantize_unet import quantize_dynamic_int8, quantize_static_int8


# --- contract shape + torch-vs-ONNX parity (UNet) ----------------------------


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


# --- cross-architecture export + serve ---------------------------------------


@pytest.mark.parametrize("arch,encoder", [
    ("deeplabv3plus", None),
    ("hrnet", None),
    ("segformer", None),
    # convnext's production default is tu-convnext_large.dinov3_lvd1689m (203.3M), whose
    # ONNX artifact is ~800 MB. Exercise the same code path on the tiny variant: identical
    # stage structure and channel-list plumbing into the Unet decoder, ~28M to export.
    ("convnext", "tu-convnext_tiny.dinov3_lvd1689m"),
])
def test_arch_exports_and_serves_nhwc(tmp_path, arch, encoder, force_cpu_onnx):
    # Pin the CPU EP so the check is EP-independent (CoreML/CUDA compute in fp16 and
    # would nudge sigmoid outputs past a tight epsilon — see WINMOL_ONNX_FORCE_CPU).
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
    from winmol_unet.training.model_factory import build_model

    model = build_model("dpt", encoder_weights=None).eval()
    with pytest.raises(Exception) as excinfo:
        export_to_onnx(model, str(tmp_onnx_path()))
    msg = str(excinfo.value)
    assert ("batch axis must be dynamic" in msg          # current: fixed output batch
            or "_upsample_bicubic2d_aa" in msg           # historical: no opset-17 lowering
            or ("Resize" in msg and "17" in msg)), f"DPT failed for a new reason: {msg[:300]}"


# --- OnnxSegmenter serving: NHWC roundtrip, summary, OOM handling -----------


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
#
# These tests' SUBJECT is provider selection itself — they manipulate
# WINMOL_ONNX_PROVIDERS / WINMOL_ONNX_FORCE_CPU by hand and assert how
# _default_providers() reacts. They must NOT use the force_cpu_onnx fixture:
# that would make them assert a condition their own setup establishes, i.e. a
# test that cannot fail. Keep the manual env handling exactly as it was.

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


# --- fp16 / int8 quantization -------------------------------------------------


def _tiny_unet_onnx(path):
    # small width -> fast to export/quantize, still contract-shaped (512x512x3->512x512x1)
    export_to_onnx(UNet(width_mult=0.25).eval(), path)
    return path


def _weight_bytes(path):
    """Total bytes of a model's initializers, wherever they are stored.

    Comparing os.path.getsize is wrong: from torch 2.9 the ONNX exporter writes weights
    to a sidecar `.onnx.data` file, leaving the `.onnx` a ~13 KB graph stub. The
    quantizer then writes a single self-contained file, so the file-size comparison read
    2.4 MB against 13 KB and failed — while the weights had in fact shrunk. Measuring the
    initializers is storage-agnostic and tests what the assertion actually means.
    """
    from onnx import numpy_helper

    model = onnx.load(path, load_external_data=True)
    return sum(numpy_helper.to_array(t).nbytes for t in model.graph.initializer)


def test_dynamic_int8_roundtrips_and_keeps_contract(tmp_path, force_cpu_onnx):
    src = _tiny_unet_onnx(str(tmp_path / "m.onnx"))
    dst = str(tmp_path / "m.int8.onnx")
    quantize_dynamic_int8(src, dst)

    validate_onnx_model(onnx.load(dst))              # contract still holds
    assert _weight_bytes(dst) < _weight_bytes(src)   # int8 weights are smaller

    seg = OnnxSegmenter(dst)
    y = seg.predict_on_batch(np.zeros((1, 512, 512, 3), np.float32))
    assert y.shape == (1, 512, 512, 1)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0   # sigmoid output


def test_static_int8_calibrates_from_a_tile_dir(tmp_path, force_cpu_onnx):
    from PIL import Image
    # a StemDataset-shaped calibration dir
    d = tmp_path / "calib"
    (d / "train").mkdir(parents=True); (d / "mask").mkdir(parents=True)
    for k in range(1, 5):
        Image.fromarray((np.random.rand(40, 40, 3) * 255).astype(np.uint8), "RGB").save(
            d / "train" / f"train{k}.jpeg")
        Image.fromarray(np.zeros((40, 40), np.uint8), "L").save(d / "mask" / f"mask{k}.gif")

    src = _tiny_unet_onnx(str(tmp_path / "m.onnx"))
    dst = str(tmp_path / "m.static.onnx")
    quantize_static_int8(src, dst, str(d), n_samples=4)

    validate_onnx_model(onnx.load(dst))
    y = OnnxSegmenter(dst).predict_on_batch(np.zeros((1, 512, 512, 3), np.float32))
    assert y.shape == (1, 512, 512, 1)
