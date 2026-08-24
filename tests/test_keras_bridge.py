"""The UNet-only Keras HDF5 drop-in: hand-built mirror, layer-by-layer weight transfer, torch-vs-Keras parity, and coexistence with onnxruntime in one process.

Merges tests/test_export_keras.py, tests/test_export_multiformat.py,
tests/test_keras_model.py, tests/test_onnx_tf_coexist.py and tests/test_deploy_models.py.

TensorFlow is opt-in (`--export-keras`, the `[keras]` extra) and is not installed in CI, so
this file has no module-level `pytest.importorskip("tensorflow")`. Two of its tests are
deliberately designed to keep running when TensorFlow is absent --
`test_onnx_is_new_enough_for_protobuf_6` (reads the onnx version from package metadata,
without importing onnx or tensorflow) and the two `deploy_models` manifest tests (which
never touch TensorFlow at all) -- so a blanket module-level skip would silently drop
coverage exactly where CI relies on it. Every test that does need TensorFlow guards itself
individually instead, either with its own `pytest.importorskip("tensorflow")` or (for the
subprocess-based coexistence check) the pre-existing `HAS_TF`/`skipif` pattern.

tests/test_no_eager_tensorflow.py's single test moved to tests/test_import_boundary.py
rather than living here, for the same reason taken further: it exists specifically to catch
an eager TensorFlow import, so it must be able to run -- and fail -- in a process where
TensorFlow was never imported at all, which a module gated on TensorFlow's presence cannot
guarantee.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from winmol_unet.export import export_to_pt
from winmol_unet.model import UNet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from deploy_models_to_release import build_manifest, sha256_file

HAS_TF = importlib.util.find_spec("tensorflow") is not None


def _torch_probs(model, x_nhwc):
    with torch.no_grad():
        nchw = torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2))
        return torch.sigmoid(model(nchw)).numpy().transpose(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# torch -> Keras HDF5/native export parity (test_export_keras.py, test_export_multiformat.py)
# ---------------------------------------------------------------------------

def test_export_parity_torch_vs_keras(tmp_path):
    pytest.importorskip("tensorflow")
    from winmol_unet.export_keras import export_to_keras_hdf5

    model = UNet().eval()
    path = export_to_keras_hdf5(model, str(tmp_path / "m.hdf5"))
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)

    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)   # unmodified-analyzer load path
    keras_out = loaded.predict(x, verbose=0)

    assert keras_out.shape == (2, 512, 512, 1)
    assert np.allclose(_torch_probs(model, x), keras_out, rtol=0.0, atol=1e-4)


def test_transfer_fails_loudly_on_mismatch(tmp_path):
    pytest.importorskip("tensorflow")
    from winmol_unet.export_keras import export_to_keras_hdf5
    import torch.nn as nn

    # A model with a different architecture must not silently mis-map.
    class Wrong(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 1, 1)

        def forward(self, x):
            return self.c(x)

    with pytest.raises((ValueError, AssertionError)):
        export_to_keras_hdf5(Wrong().eval(), str(tmp_path / "bad.hdf5"))


def test_export_pt_roundtrip(tmp_path):
    pytest.importorskip("tensorflow")
    m = UNet().eval()
    path = export_to_pt(m, str(tmp_path / "m.pt"))
    m2 = UNet()
    m2.load_state_dict(torch.load(path))
    m2.eval()
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-6)


def test_export_keras_native_parity(tmp_path):
    pytest.importorskip("tensorflow")
    from winmol_unet.export_keras import export_to_keras

    model = UNet().eval()
    path = export_to_keras(model, str(tmp_path / "m.keras"))
    assert path.endswith(".keras")
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)
    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)   # native-format load
    out = loaded.predict(x, verbose=0)
    assert out.shape == (2, 512, 512, 1)
    assert np.allclose(_torch_probs(model, x), out, rtol=0.0, atol=1e-4)


def test_export_hdf5_still_works(tmp_path):
    pytest.importorskip("tensorflow")
    from winmol_unet.export_keras import export_to_keras_hdf5

    model = UNet().eval()
    path = export_to_keras_hdf5(model, str(tmp_path / "m.hdf5"))
    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)
    x = np.random.rand(1, 512, 512, 3).astype(np.float32)
    assert np.allclose(_torch_probs(model, x), loaded.predict(x, verbose=0), atol=1e-4)


# ---------------------------------------------------------------------------
# the hand-built Keras UNet mirror (test_keras_model.py)
# ---------------------------------------------------------------------------

def test_keras_unet_output_shape_and_range():
    pytest.importorskip("tensorflow")
    from winmol_unet.keras_model import build_keras_unet

    model = build_keras_unet()
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 512, 512, 1)
    assert (y >= 0).all() and (y <= 1).all()   # sigmoid head


def test_keras_unet_uses_torch_bn_epsilon():
    pytest.importorskip("tensorflow")
    from tensorflow.keras.layers import BatchNormalization

    from winmol_unet.keras_model import build_keras_unet

    model = build_keras_unet()
    bns = [l for l in model.layers if isinstance(l, BatchNormalization)]
    assert bns and all(abs(l.epsilon - 1e-5) < 1e-12 for l in bns)


def test_keras_unet_roundtrips_via_load_model(tmp_path):
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    from winmol_unet.keras_model import build_keras_unet

    path = str(tmp_path / "m.hdf5")
    build_keras_unet().save(path, save_format="h5")
    loaded = keras.models.load_model(path, compile=False)   # analyzer's load path
    assert loaded.predict(np.zeros((1, 512, 512, 3), np.float32), verbose=0).shape == (1, 512, 512, 1)


# ---------------------------------------------------------------------------
# onnx / TensorFlow coexistence in one process (test_onnx_tf_coexist.py)
#
# onnx and TensorFlow ship their own C++ protobuf runtime; before onnx 1.18 those were
# built against incompatible protobuf majors, so whichever loaded second either aborted
# the interpreter ("mutex lock failed: Invalid argument") or deadlocked ("RAW: Lock
# blocking"). Nothing here imports TensorFlow or onnx into the test session -- every actual
# import happens in a subprocess, since on a broken pair doing it here would abort this
# interpreter and take the whole suite down with no report.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF, reason="only reproducible with the [keras] extra installed")
# a fresh interpreter per order -- once a process has loaded either library the outcome is
# already decided, so both orders must start from scratch
@pytest.mark.parametrize("first,second", [("onnx", "tensorflow"), ("tensorflow", "onnx")])
def test_imports_coexist_in_one_process(first, second):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {first}, {second}; print('ok')"],
        # generous for a cold TensorFlow import, short enough that the deadlocking
        # pair reports in ~2 min instead of stalling the suite
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "ok" in proc.stdout, (
        f"importing {first} then {second} failed (exit {proc.returncode}). This is the "
        f"onnx/TensorFlow protobuf clash -- onnx must be >=1.18.\n{proc.stderr[-1500:]}")


def test_onnx_is_new_enough_for_protobuf_6():
    """Runs everywhere: reads the version from metadata, without importing onnx."""
    from importlib.metadata import version

    v = version("onnx")
    major, minor = (int(p) for p in v.split(".")[:2])
    assert (major, minor) >= (1, 18), (
        f"onnx {v} predates protobuf 6 support; with TensorFlow installed it deadlocks or "
        f"aborts on import, breaking --export-keras. pyproject pins onnx>=1.18.")


# ---------------------------------------------------------------------------
# release manifest generation (test_deploy_models.py) -- no TensorFlow involved; the
# consolidation plan groups it into this file rather than leaving it standing alone.
# ---------------------------------------------------------------------------

def test_sha256_file_matches_known_digest(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"winmol")
    # sha256("winmol")
    assert sha256_file(str(p)) == hashlib.sha256(b"winmol").hexdigest()


def test_build_manifest_lists_every_asset_with_hash_and_size():
    entries = [
        {"asset": "unet_fp32.onnx", "title": "Original UNet (fp32)", "backend": "any",
         "sha256": "abc123", "size_mb": 118.4, "notes": "reference, F1 0.760"},
        {"asset": "unet_w05_int8_cpu.onnx", "title": "CPU-optimised", "backend": "CPU",
         "sha256": "def456", "size_mb": 7.5, "notes": "10x, F1 0.760"},
    ]
    md = build_manifest("models-v1", entries)
    assert "models-v1" in md
    for e in entries:
        assert e["asset"] in md
        assert e["sha256"] in md
        assert e["notes"] in md
    # a markdown table header is present
    assert "| asset " in md.lower() or "| asset|" in md.lower().replace(" ", "")
