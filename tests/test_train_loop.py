import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.dataset import StemDataset
from training.train import train_one_run
from training.evaluate import evaluate
from winmol_unet.model import UNet


def _tiny_dataset(tmp_path, n_items=2):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for n in range(1, n_items + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{n}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{n}.gif")
    return StemDataset(str(img_dir), str(mask_dir), augment=False)


def test_overfit_loss_decreases(tmp_path):
    torch.manual_seed(0)
    ds = _tiny_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=2)
    cfg = TrainConfig(data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
                      log_dir=str(tmp_path / "log"), hdf5_out="x.hdf5", onnx_out="x.onnx",
                      epochs=8, lr=1e-2, patience=999)
    model = UNet(dropout=0.0)
    before = evaluate(model, loader)["loss"]
    train_one_run(model, loader, loader, cfg)
    after = evaluate(model, loader)["loss"]
    assert after < before


def test_evaluate_returns_metric_keys(tmp_path):
    ds = _tiny_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=2)
    out = evaluate(UNet(dropout=0.0), loader)
    assert set(out) == {"loss", "precision", "recall", "f1"}
