import os
import numpy as np
from PIL import Image

from training.config import TrainConfig
from training.run_train import config_from_args, run_training


def _make_ds(tmp_path, n=6):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")


def _cfg(tmp_path, arch):
    out = tmp_path / "out"
    return TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"), pt_out=str(out / "m.pt"),
        hdf5_out=str(out / "m.hdf5"), keras_out=str(out / "m.keras"),
        onnx_out=str(out / "m.onnx"), epochs=1, batch_size=2, patience=999,
        device="cpu", arch=arch, encoder_weights=None,
    )


def test_cli_parses_arch_flags():
    cfg = config_from_args(["--data-dir", "d", "--arch", "deeplabv3plus",
                            "--encoder", "resnet18", "--encoder-weights", "imagenet"])
    assert cfg.arch == "deeplabv3plus"
    assert cfg.encoder == "resnet18"
    assert cfg.encoder_weights == "imagenet"
    assert config_from_args(["--data-dir", "d"]).arch == "unet"   # default


def test_non_unet_exports_onnx_pt_only(tmp_path):
    _make_ds(tmp_path)
    cfg = _cfg(tmp_path, "deeplabv3plus")
    run_training(cfg)
    assert os.path.exists(cfg.onnx_out) and os.path.exists(cfg.pt_out)
    assert not os.path.exists(cfg.hdf5_out)     # UNet-specific mirror skipped
    assert not os.path.exists(cfg.keras_out)


def test_unet_still_exports_all_four(tmp_path):
    _make_ds(tmp_path)
    cfg = _cfg(tmp_path, "unet")
    run_training(cfg)
    for p in (cfg.pt_out, cfg.hdf5_out, cfg.keras_out, cfg.onnx_out):
        assert os.path.exists(p), f"missing {p}"
