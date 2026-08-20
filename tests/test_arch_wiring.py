"""Architecture-dependent export behaviour: the CLI-flag half of this file moved to
tests/test_config_wiring.py (task 8 of the 2026-08-20 test-suite consolidation)."""
import os
import numpy as np
import pytest
from PIL import Image

from winmol_unet.training.config import TrainConfig
from winmol_unet.training.run_train import run_training


def _make_ds(tmp_path, n=6):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")


def _cfg(tmp_path, arch, export_keras=False):
    out = tmp_path / "out"
    return TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"), pt_out=str(out / "m.pt"),
        hdf5_out=str(out / "m.hdf5"), keras_out=str(out / "m.keras"),
        onnx_out=str(out / "m.onnx"), epochs=1, batch_size=2, patience=999,
        device="cpu", arch=arch, encoder_weights=None, export_keras=export_keras,
    )


def test_non_unet_exports_onnx_pt_only(tmp_path):
    _make_ds(tmp_path)
    cfg = _cfg(tmp_path, "deeplabv3plus")
    run_training(cfg)
    assert os.path.exists(cfg.onnx_out) and os.path.exists(cfg.pt_out)
    assert not os.path.exists(cfg.hdf5_out)     # UNet-specific mirror skipped
    assert not os.path.exists(cfg.keras_out)


def test_unet_default_is_onnx_pt_only(tmp_path):
    # ONNX is the uniform default for every arch, including UNet.
    _make_ds(tmp_path)
    cfg = _cfg(tmp_path, "unet")
    run_training(cfg)
    assert os.path.exists(cfg.onnx_out) and os.path.exists(cfg.pt_out)
    assert not os.path.exists(cfg.hdf5_out) and not os.path.exists(cfg.keras_out)


def test_unet_export_keras_flag_writes_all_four(tmp_path):
    # writes the Keras HDF5/.keras mirror, so it needs TensorFlow; the sibling
    # non-UNet test does not, because it asserts the fail-fast before export
    pytest.importorskip("tensorflow")
    _make_ds(tmp_path)
    cfg = _cfg(tmp_path, "unet", export_keras=True)
    run_training(cfg)
    for p in (cfg.pt_out, cfg.hdf5_out, cfg.keras_out, cfg.onnx_out):
        assert os.path.exists(p), f"missing {p}"


def test_export_keras_on_non_unet_raises(tmp_path):
    import pytest
    _make_ds(tmp_path)
    cfg = _cfg(tmp_path, "deeplabv3plus", export_keras=True)
    with pytest.raises(ValueError, match="UNet-only"):
        run_training(cfg)
