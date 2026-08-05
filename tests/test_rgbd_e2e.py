"""RGBD end-to-end: tiny dataset with depth/ -> 1-epoch train -> 4ch ONNX."""
import os

import numpy as np
import onnx
import pytest
from PIL import Image

from training.config import TrainConfig
from training.run_train import run_training
from winmol_unet.contract import RGBD_IN_CHANNELS, validate_onnx_model


def _make_rgbd_dataset(root, n_pairs=4, size=64):
    for sub in ("train", "mask", "depth"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    rng = np.random.default_rng(0)
    for n in range(1, n_pairs + 1):
        img = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(root, "train", f"train{n}.jpeg"))
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        Image.fromarray(mask).convert("P").save(os.path.join(root, "mask", f"mask{n}.gif"))
        depth = ((np.arange(size * size).reshape(size, size) * 13) % 4096).astype(np.uint16)
        Image.fromarray(depth, mode="I;16").save(os.path.join(root, "depth", f"depth{n}.png"))


def test_rgbd_training_exports_4ch_onnx(tmp_path):
    data = str(tmp_path / "ds")
    _make_rgbd_dataset(data)
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=data,
        checkpoint_dir=str(out / "ckpt"), log_dir=str(out / "logs"),
        hdf5_out=str(out / "model.hdf5"), onnx_out=str(out / "model.onnx"),
        pt_out=str(out / "model.pt"),
        rgbd=True, epochs=1, batch_size=2, device="cpu", seed=1,
    )
    run_training(cfg)
    model = onnx.load(cfg.onnx_out)
    validate_onnx_model(model, in_channels=RGBD_IN_CHANNELS)


def test_rgbd_rejects_export_keras(tmp_path):
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path), log_dir=str(tmp_path),
        hdf5_out="x.hdf5", onnx_out="x.onnx", rgbd=True, export_keras=True,
    )
    with pytest.raises(ValueError, match="rgbd"):
        run_training(cfg)
