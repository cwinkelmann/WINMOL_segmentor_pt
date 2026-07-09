"""Single-stage train/validate loop: Adam, best-val-loss checkpoint, early stop,
TensorBoard logging."""
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from .device import resolve_device
from .evaluate import evaluate
from .losses import bce_soft_f1_loss


def train_one_run(model, train_loader, val_loader, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(cfg.checkpoint_dir, "best.pt")
    writer = SummaryWriter(cfg.log_dir)
    device = resolve_device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    since_improve = 0
    for epoch in range(cfg.epochs):
        model.train()
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = bce_soft_f1_loss(logits, mask)
            loss.backward()
            opt.step()

        val = evaluate(model, val_loader)
        writer.add_scalar("val/loss", val["loss"], epoch)
        writer.add_scalar("val/f1", val["f1"], epoch)

        if val["loss"] < best_val:
            best_val = val["loss"]
            since_improve = 0
            torch.save(model.state_dict(), ckpt)
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                break

    writer.close()
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model
