"""Augmentation wiring end to end: the CLI-flag half of this file moved to
tests/test_config_wiring.py (task 8 of the 2026-08-20 test-suite consolidation)."""
import os
import numpy as np
from PIL import Image

from winmol_unet.training.run_train import run_training
from winmol_unet.training.config import TrainConfig


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
