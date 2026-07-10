"""Single-stage train/validate loop: Adam, best-val-loss checkpoint, early stop,
TensorBoard logging."""
import os

import torch

from .device import resolve_device
from .evaluate import evaluate
from .losses import bce_soft_f1_loss
from .run_logger import RunLogger


def train_one_run(model, train_loader, val_loader, cfg, patience=None,
                  ckpt_name="best.pt", log_dir=None):
    # patience/ckpt_name/log_dir default to the single-stage config; two-stage passes
    # per-stage values so stage 2 doesn't clobber stage 1's checkpoint or TB curves.
    patience = cfg.patience if patience is None else patience
    log_dir = cfg.log_dir if log_dir is None else log_dir
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(cfg.checkpoint_dir, ckpt_name)
    logger = RunLogger(log_dir, cfg.wandb, cfg.wandb_project, cfg.wandb_run_name)
    device = resolve_device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    since_improve = 0
    try:
        for epoch in range(cfg.epochs):
            model.train()
            running, nb = 0.0, 0
            for img, mask in train_loader:
                img, mask = img.to(device), mask.to(device)
                opt.zero_grad()
                loss = bce_soft_f1_loss(model(img), mask)
                loss.backward()
                opt.step()
                running += loss.item()
                nb += 1
            train_loss = running / nb if nb else 0.0

            val = evaluate(model, val_loader)
            logger.log_scalars({
                "train/loss": train_loss,
                "val/loss": val["loss"],
                "val/precision": val["precision"],
                "val/recall": val["recall"],
                "val/f1": val["f1"],
            }, epoch)

            if val["loss"] < best_val:
                best_val = val["loss"]
                since_improve = 0
                torch.save(model.state_dict(), ckpt)
            else:
                since_improve += 1
                if since_improve >= patience:
                    break
    finally:
        logger.close()   # always flush TB + wandb.finish, even on error/early exit

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model
