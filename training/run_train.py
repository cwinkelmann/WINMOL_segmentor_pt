"""CLI: train on one dataset, export HDF5 (analyzer drop-in) + ONNX."""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from winmol_unet.export import export_to_onnx, export_to_pt
from winmol_unet.export_keras import export_to_keras, export_to_keras_hdf5

from .augment import build_augmentation
from .config import TrainConfig
from .dataset import train_val_split
from .evaluate import evaluate
from .model_factory import build_model
from .train import train_one_run


def run_training(cfg):
    torch.manual_seed(cfg.seed)
    transform = build_augmentation(cfg)       # seeded internally via cfg.seed (reproducible)
    train_ds, val_ds = train_val_split(
        cfg.image_dir, cfg.mask_dir, cfg.val_fraction, cfg.seed, cfg.img_size,
        transform=transform)
    # num_workers=0 is required for StemDataset's resize cache to persist across
    # epochs (see StemDataset docstring); do not raise it without persistent_workers.
    # drop_last avoids a trailing batch of 1, which breaks BatchNorm in architectures
    # whose forward reduces to [N,C,1,1] (e.g. DeepLabV3+ ASPP global pooling) — but only
    # when there is more than one batch's worth, so a tiny train set isn't zeroed out.
    drop_last = len(train_ds) > cfg.batch_size
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=0)

    model = build_model(cfg.arch, dropout=cfg.dropout, encoder=cfg.encoder,
                        encoder_weights=cfg.encoder_weights)
    train_one_run(model, train_loader, val_loader, cfg)

    val_metrics = evaluate(model, val_loader)   # on training device

    model.cpu()                                 # exporters read weights via CPU numpy
    # Create the directory of every configured output (they may differ), so a
    # split path can't FileNotFoundError mid-export and skip the always-on formats.
    for p in (cfg.pt_out, cfg.hdf5_out, cfg.keras_out, cfg.onnx_out):
        if p:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    if cfg.pt_out:
        export_to_pt(model, cfg.pt_out)
    # The Keras HDF5/.keras mirror is UNet-specific; non-UNet models are ONNX-only.
    if cfg.arch == "unet":
        export_to_keras_hdf5(model, cfg.hdf5_out, dropout=cfg.dropout)
        if cfg.keras_out:
            export_to_keras(model, cfg.keras_out, dropout=cfg.dropout)
    export_to_onnx(model, cfg.onnx_out)

    return val_metrics


def config_from_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--aug-hflip-p", type=float, default=0.5)
    p.add_argument("--aug-vflip-p", type=float, default=0.5)
    p.add_argument("--aug-rotate-p", type=float, default=0.0)
    p.add_argument("--aug-rotate-limit", type=float, default=15.0)
    p.add_argument("--aug-bc-p", type=float, default=0.5)
    p.add_argument("--aug-brightness-limit", type=float, default=0.2)
    p.add_argument("--aug-contrast-limit", type=float, default=0.2)
    p.add_argument("--aug-hsv-p", type=float, default=0.5)
    p.add_argument("--arch", default="unet", help="unet|deeplabv3plus|hrnet")
    p.add_argument("--encoder", default="resnet34", help="smp encoder (deeplabv3plus)")
    p.add_argument("--encoder-weights", default=None, help="None or 'imagenet' (needs network)")
    a = p.parse_args(argv)
    return TrainConfig(
        data_dir=a.data_dir,
        checkpoint_dir=os.path.join(a.out_dir, "checkpoints"),
        log_dir=os.path.join(a.out_dir, "logs"),
        pt_out=os.path.join(a.out_dir, "model.pt"),
        hdf5_out=os.path.join(a.out_dir, "model.hdf5"),
        keras_out=os.path.join(a.out_dir, "model.keras"),
        onnx_out=os.path.join(a.out_dir, "model.onnx"),
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
        wandb=a.wandb, wandb_project=a.wandb_project, wandb_run_name=a.wandb_run_name,
        aug_hflip_p=a.aug_hflip_p, aug_vflip_p=a.aug_vflip_p,
        aug_rotate_p=a.aug_rotate_p, aug_rotate_limit=a.aug_rotate_limit,
        aug_bc_p=a.aug_bc_p, aug_brightness_limit=a.aug_brightness_limit,
        aug_contrast_limit=a.aug_contrast_limit, aug_hsv_p=a.aug_hsv_p,
        arch=a.arch, encoder=a.encoder, encoder_weights=a.encoder_weights,
    )


def main():
    print(run_training(config_from_args()))


if __name__ == "__main__":
    main()
