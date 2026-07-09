"""Mean loss + hard-rounded metrics over a loader (no grad).

Metrics are micro-averaged: tp/fp/fn are accumulated globally across the whole
loader and P/R/F1 computed once, so a partial final batch is not over-weighted.
Loss is weighted by batch size (an approximation for the batch-level soft-F1
term, exact for the per-element BCE term) to avoid over-weighting a small last
batch — not a true global recomputation like the metrics get.
"""
import torch

from .losses import bce_soft_f1_loss
from .metrics import counts, prf


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    tp = fp = fn = 0.0
    loss_sum = 0.0
    n_samples = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        bs = img.shape[0]
        loss_sum += bce_soft_f1_loss(logits, mask).item() * bs
        b_tp, b_fp, b_fn = counts(logits, mask)
        tp += b_tp
        fp += b_fp
        fn += b_fn
        n_samples += bs
    p, r, f = prf(tp, fp, fn)
    return {
        "loss": loss_sum / n_samples if n_samples else 0.0,
        "precision": p,
        "recall": r,
        "f1": f,
    }
