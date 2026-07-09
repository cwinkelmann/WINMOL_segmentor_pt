"""CLI: train on one dataset, export HDF5 (analyzer drop-in) + ONNX."""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from winmol_unet.export import export_to_onnx
from winmol_unet.export_keras import export_to_keras_hdf5
from winmol_unet.model import UNet

from .config import TrainConfig
from .dataset import train_val_split
from .evaluate import evaluate
from .train import train_one_run


def run_training(cfg):
    torch.manual_seed(cfg.seed)
    train_ds, val_ds = train_val_split(
        cfg.image_dir, cfg.mask_dir, cfg.val_fraction, cfg.seed, cfg.img_size)
    # num_workers=0 is required for StemDataset's resize cache to persist across
    # epochs (see StemDataset docstring); do not raise it without persistent_workers.
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=0)

    model = UNet(dropout=cfg.dropout)
    train_one_run(model, train_loader, val_loader, cfg)

    val_metrics = evaluate(model, val_loader)   # on training device

    model.cpu()                                 # exporters read weights via CPU numpy
    os.makedirs(os.path.dirname(cfg.hdf5_out) or ".", exist_ok=True)
    export_to_keras_hdf5(model, cfg.hdf5_out, dropout=cfg.dropout)
    export_to_onnx(model, cfg.onnx_out)

    return val_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    a = p.parse_args()
    cfg = TrainConfig(
        data_dir=a.data_dir,
        checkpoint_dir=os.path.join(a.out_dir, "checkpoints"),
        log_dir=os.path.join(a.out_dir, "logs"),
        hdf5_out=os.path.join(a.out_dir, "model.hdf5"),
        onnx_out=os.path.join(a.out_dir, "model.onnx"),
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
    )
    print(run_training(cfg))


if __name__ == "__main__":
    main()
