import os

import pytest
import numpy as np
import torch
from PIL import Image

from winmol_unet.training.config import TrainConfig
from winmol_unet.training.run_train import run_training
from winmol_unet.model import UNet


def _make_ds(tmp_path, n=6):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")


def test_end_to_end_train_export_hdf5_dropin(tmp_path):
    # the Keras HDF5 drop-in is opt-in (--export-keras); TensorFlow is not
    # installed in CI
    pytest.importorskip("tensorflow")
    _make_ds(tmp_path)
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"), hdf5_out=str(out / "m.hdf5"),
        onnx_out=str(out / "m.onnx"), epochs=2, batch_size=2, patience=999,
        export_keras=True,     # this test exercises the UNet Keras HDF5 drop-in
    )
    metrics = run_training(cfg)
    assert set(metrics) == {"loss", "precision", "recall", "f1"}
    assert os.path.exists(cfg.hdf5_out) and os.path.exists(cfg.onnx_out)

    # Drop-in: load HDF5 the analyzer's way and predict on NHWC tiles.
    from tensorflow import keras
    km = keras.models.load_model(cfg.hdf5_out, compile=False)
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)
    pred = np.asarray(km.predict_on_batch(x))
    assert pred.shape == (2, 512, 512, 1)
    assert (pred >= 0).all() and (pred <= 1).all()
