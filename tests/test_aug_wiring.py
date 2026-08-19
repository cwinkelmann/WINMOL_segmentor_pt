import os
import numpy as np
from PIL import Image

from winmol_unet.training.run_train import config_from_args, run_training
from winmol_unet.training.config import TrainConfig


def test_cli_parses_aug_flags():
    cfg = config_from_args(["--data-dir", "d", "--aug-rotate-p", "0.3",
                            "--aug-rotate-limit", "20", "--aug-hflip-p", "0.25"])
    assert cfg.aug_rotate_p == 0.3
    assert cfg.aug_rotate_limit == 20
    assert cfg.aug_hflip_p == 0.25


def test_aug_flags_default_preserved():
    cfg = config_from_args(["--data-dir", "d"])
    assert cfg.aug_hflip_p == 0.5 and cfg.aug_rotate_p == 0.0


def test_run_training_with_rotation_enabled(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, 7):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"), hdf5_out=str(out / "m.hdf5"),
        onnx_out=str(out / "m.onnx"), epochs=1, batch_size=2, patience=999,
        device="cpu", aug_rotate_p=1.0, aug_rotate_limit=25, aug_hsv_p=1.0,
    )
    metrics = run_training(cfg)          # exercises the full albumentations path
    assert set(metrics) == {"loss", "precision", "recall", "f1"}
    assert os.path.exists(cfg.onnx_out)  # ONNX is the uniform default export
