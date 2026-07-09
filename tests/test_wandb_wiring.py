import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.dataset import StemDataset
from training.train import train_one_run
from training.run_train import config_from_args
from winmol_unet.model import UNet


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
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for n in (1, 2):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{n}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{n}.gif")
    ds = StemDataset(str(img_dir), str(mask_dir))
    loader = DataLoader(ds, batch_size=2)
    cfg = TrainConfig(data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
                      log_dir=str(tmp_path / "log"), hdf5_out="x", onnx_out="x",
                      epochs=1, patience=999)
    train_one_run(UNet(dropout=0.0), loader, loader, cfg)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(sorted(glob.glob(str(tmp_path / "log" / "events*")))[-1]); ea.Reload()
    assert "train/loss" in ea.Tags().get("scalars", [])
