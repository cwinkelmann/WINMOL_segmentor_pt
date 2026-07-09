import os
import numpy as np
from PIL import Image

from training.config import TrainConfig
from training.run_train import run_training, config_from_args


def _make_ds(tmp_path, n=6):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")


def test_cli_derives_all_four_paths():
    cfg = config_from_args(["--data-dir", "d", "--out-dir", "o"])
    assert cfg.pt_out == os.path.join("o", "model.pt")
    assert cfg.keras_out == os.path.join("o", "model.keras")
    assert cfg.hdf5_out == os.path.join("o", "model.hdf5")
    assert cfg.onnx_out == os.path.join("o", "model.onnx")


def test_run_training_writes_all_four(tmp_path):
    _make_ds(tmp_path)
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"),
        pt_out=str(out / "m.pt"), hdf5_out=str(out / "m.hdf5"),
        keras_out=str(out / "m.keras"), onnx_out=str(out / "m.onnx"),
        epochs=1, batch_size=2, patience=999, device="cpu",
    )
    run_training(cfg)
    for f in ("m.pt", "m.hdf5", "m.keras", "m.onnx"):
        assert os.path.exists(out / f), f"missing {f}"
