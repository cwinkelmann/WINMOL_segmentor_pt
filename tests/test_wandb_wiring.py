import glob
import sys
import types
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from winmol_unet.training.config import TrainConfig
from winmol_unet.training.dataset import StemDataset
from winmol_unet.training.train import train_one_run
from winmol_unet.training.run_train import config_from_args
from winmol_unet.model import UNet


def _tiny_ds(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for n in (1, 2):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{n}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{n}.gif")
    return StemDataset(str(img_dir), str(mask_dir))


def test_cli_parses_wandb_flags():
    cfg = config_from_args(["--data-dir", "d", "--wandb",
                            "--wandb-project", "P", "--wandb-run-name", "R"])
    assert cfg.wandb is True
    assert cfg.wandb_project == "P"
    assert cfg.wandb_run_name == "R"


def test_cli_wandb_off_by_default():
    cfg = config_from_args(["--data-dir", "d"])
    assert cfg.wandb is False


def test_train_logs_train_loss_scalar(tmp_path):
    ds = _tiny_ds(tmp_path)
    loader = DataLoader(ds, batch_size=2)
    cfg = TrainConfig(data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
                      log_dir=str(tmp_path / "log"), hdf5_out="x", onnx_out="x",
                      epochs=1, patience=999)
    train_one_run(UNet(dropout=0.0), loader, loader, cfg)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(sorted(glob.glob(str(tmp_path / "log" / "events*")))[-1]); ea.Reload()
    assert "train/loss" in ea.Tags().get("scalars", [])


def test_train_calls_wandb_log_when_enabled(tmp_path, monkeypatch):
    # End-to-end wiring: cfg.wandb=True -> train_one_run drives wandb.log per epoch.
    calls = []
    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: None
    fake.log = lambda d, step=None: calls.append((dict(d), step))
    fake.finish = lambda: None
    monkeypatch.setitem(sys.modules, "wandb", fake)
    dot = types.ModuleType("dotenv"); dot.load_dotenv = lambda *a, **k: True
    monkeypatch.setitem(sys.modules, "dotenv", dot)

    loader = DataLoader(_tiny_ds(tmp_path), batch_size=2)
    cfg = TrainConfig(data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
                      log_dir=str(tmp_path / "log"), hdf5_out="x", onnx_out="x",
                      epochs=2, patience=999, device="cpu",
                      wandb=True, wandb_project="P", wandb_run_name="R")
    train_one_run(UNet(dropout=0.0), loader, loader, cfg)
    assert len(calls) == 2                                  # one wandb.log per epoch
    scalars, step = calls[0]
    assert set(scalars) == {"train/loss", "val/loss", "val/precision",
                            "val/recall", "val/f1", "lr"}
    assert step == 0
