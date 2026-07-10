"""CLI: single-stage or two-stage (GenDS -> SpecDS) training + export.

Single-stage: --data-dir. Two-stage fine-tune: --gen-data-dir + --spec-data-dir
(stage 1 on the general set, stage 2 fine-tunes the SAME model on the species set).
"""
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


def _build_loaders(image_dir, mask_dir, cfg, transform):
    train_ds, val_ds = train_val_split(
        image_dir, mask_dir, cfg.val_fraction, cfg.seed, cfg.img_size,
        transform=transform, cache=cfg.cache_dataset)
    # drop_last avoids a trailing batch of 1 (breaks BatchNorm in DeepLabV3+ ASPP
    # [N,C,1,1]) — only when there is more than one batch's worth, so a tiny set
    # isn't zeroed out. num_workers>0 is only safe with cache_dataset=False.
    drop_last = len(train_ds) > cfg.batch_size
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return train_loader, val_loader


def _export(model, cfg):
    model.cpu()                                 # exporters read weights via CPU numpy
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


def run_training(cfg):
    torch.manual_seed(cfg.seed)
    transform = build_augmentation(cfg)         # seeded internally via cfg.seed
    train_loader, val_loader = _build_loaders(cfg.image_dir, cfg.mask_dir, cfg, transform)
    model = build_model(cfg.arch, dropout=cfg.dropout, encoder=cfg.encoder,
                        encoder_weights=cfg.encoder_weights)
    train_one_run(model, train_loader, val_loader, cfg)
    val_metrics = evaluate(model, val_loader)   # on training device
    _export(model, cfg)
    return val_metrics


def run_two_stage(cfg):
    """Two-stage fine-tune: train on GenDS (stage 1), then fine-tune the same model
    on SpecDS (stage 2). Final metrics + export come from the stage-2 (species) model."""
    torch.manual_seed(cfg.seed)
    transform = build_augmentation(cfg)
    model = build_model(cfg.arch, dropout=cfg.dropout, encoder=cfg.encoder,
                        encoder_weights=cfg.encoder_weights)

    gen_train, gen_val = _build_loaders(cfg.gen_image_dir, cfg.gen_mask_dir, cfg, transform)
    train_one_run(model, gen_train, gen_val, cfg, patience=cfg.patience_stage1,
                  ckpt_name="best_stage1.pt", log_dir=os.path.join(cfg.log_dir, "stage1"))

    spec_train, spec_val = _build_loaders(cfg.spec_image_dir, cfg.spec_mask_dir, cfg, transform)
    train_one_run(model, spec_train, spec_val, cfg, patience=cfg.patience_stage2,
                  ckpt_name="best_stage2.pt", log_dir=os.path.join(cfg.log_dir, "stage2"))

    val_metrics = evaluate(model, spec_val)     # final = species val split
    _export(model, cfg)
    return val_metrics


def config_from_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="single-stage dataset dir")
    p.add_argument("--gen-data-dir", default=None, help="two-stage: general (stage 1) dir")
    p.add_argument("--spec-data-dir", default=None, help="two-stage: species (stage 2) dir")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--patience-stage1", type=int, default=3)
    p.add_argument("--patience-stage2", type=int, default=5)
    p.add_argument("--cache-dataset", dest="cache_dataset", action="store_true", default=True)
    p.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
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
    two_stage = a.gen_data_dir and a.spec_data_dir
    if not a.data_dir and not two_stage:
        p.error("provide --data-dir (single-stage) or both --gen-data-dir and --spec-data-dir")
    return TrainConfig(
        data_dir=a.data_dir or "",
        gen_data_dir=a.gen_data_dir, spec_data_dir=a.spec_data_dir,
        checkpoint_dir=os.path.join(a.out_dir, "checkpoints"),
        log_dir=os.path.join(a.out_dir, "logs"),
        pt_out=os.path.join(a.out_dir, "model.pt"),
        hdf5_out=os.path.join(a.out_dir, "model.hdf5"),
        keras_out=os.path.join(a.out_dir, "model.keras"),
        onnx_out=os.path.join(a.out_dir, "model.onnx"),
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
        patience_stage1=a.patience_stage1, patience_stage2=a.patience_stage2,
        cache_dataset=a.cache_dataset, num_workers=a.num_workers,
        wandb=a.wandb, wandb_project=a.wandb_project, wandb_run_name=a.wandb_run_name,
        aug_hflip_p=a.aug_hflip_p, aug_vflip_p=a.aug_vflip_p,
        aug_rotate_p=a.aug_rotate_p, aug_rotate_limit=a.aug_rotate_limit,
        aug_bc_p=a.aug_bc_p, aug_brightness_limit=a.aug_brightness_limit,
        aug_contrast_limit=a.aug_contrast_limit, aug_hsv_p=a.aug_hsv_p,
        arch=a.arch, encoder=a.encoder, encoder_weights=a.encoder_weights,
    )


def main():
    cfg = config_from_args()
    result = run_two_stage(cfg) if (cfg.gen_data_dir and cfg.spec_data_dir) else run_training(cfg)
    print(result)


if __name__ == "__main__":
    main()
