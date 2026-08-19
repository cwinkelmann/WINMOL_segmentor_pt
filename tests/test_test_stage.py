import glob
import os

import numpy as np
from PIL import Image

from winmol_unet.training.config import TrainConfig
from winmol_unet.training.run_train import config_from_args, run_training


def _ds(d, n=6):
    (d / "train").mkdir(parents=True); (d / "mask").mkdir(parents=True)
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(d / "train" / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(d / "mask" / f"mask{k}.gif")


def test_cli_parses_test_data_dir():
    assert config_from_args(["--data-dir", "d", "--test-data-dir", "t"]).test_data_dir == "t"
    assert config_from_args(["--data-dir", "d"]).test_data_dir is None


def test_test_stage_writes_results_and_logs(tmp_path):
    data, test, out = tmp_path / "data", tmp_path / "test", tmp_path / "out"
    _ds(data, 6); _ds(test, 4)
    cfg = TrainConfig(
        data_dir=str(data), test_data_dir=str(test),
        checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "log"),
        hdf5_out="x", onnx_out=str(out / "m.onnx"), epochs=1, batch_size=2,
        patience=999, device="cpu",
    )
    run_training(cfg)

    res = out / "test_results.md"
    assert res.exists()
    txt = res.read_text()
    assert "Test results" in txt and "f1" in txt and "4 tiles" in txt
    assert glob.glob(str(tmp_path / "log" / "test" / "events*"))   # test scalars logged


def test_no_test_stage_without_test_data_dir(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _ds(data, 6)
    cfg = TrainConfig(
        data_dir=str(data), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"), hdf5_out="x", onnx_out=str(out / "m.onnx"),
        epochs=1, batch_size=2, patience=999, device="cpu",
    )
    run_training(cfg)
    assert not (out / "test_results.md").exists()
