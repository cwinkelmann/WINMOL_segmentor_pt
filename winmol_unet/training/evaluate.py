"""Mean loss + hard-rounded metrics over a loader (no grad).

Metrics are micro-averaged: tp/fp/fn are accumulated globally across the whole
loader and P/R/F1 computed once, so a partial final batch is not over-weighted.
Loss is weighted by batch size (an approximation for the batch-level soft-F1
term, exact for the per-element BCE term) to avoid over-weighting a small last
batch — not a true global recomputation like the metrics get.
"""
import torch

from .losses import LOSS_COMPONENTS, bce_soft_f1_loss, ce_soft_f1_loss
from .metrics import counts, multiclass_counts, prf


@torch.no_grad()
def evaluate(model, loader, num_classes=1, ignore_index=255):
    if num_classes > 1:
        return _evaluate_multiclass(model, loader, num_classes, ignore_index)
    model.eval()
    device = next(model.parameters()).device
    tp = fp = fn = 0.0
    loss_sum = 0.0
    n_samples = 0
    # Components of the same composite val loss, weighted like loss_sum. The val loss
    # is bce_soft_f1 by long-standing convention regardless of the training loss, so
    # its parts are always available.
    comp_fn = LOSS_COMPONENTS["bce_soft_f1"]
    comp_sums = {}
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        bs = img.shape[0]
        loss_sum += bce_soft_f1_loss(logits, mask).item() * bs
        for k, v in comp_fn(logits, mask).items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v.item() * bs
        b_tp, b_fp, b_fn = counts(logits, mask)
        tp += b_tp
        fp += b_fp
        fn += b_fn
        n_samples += bs
    p, r, f = prf(tp, fp, fn)
    out = {
        "loss": loss_sum / n_samples if n_samples else 0.0,
        "precision": p,
        "recall": r,
        "f1": f,
    }
    for k, v in comp_sums.items():
        out["loss_" + k] = v / n_samples if n_samples else 0.0
    return out


@torch.no_grad()
def _evaluate_multiclass(model, loader, num_classes, ignore_index=255):
    """Species path: ce_soft_f1 val loss, macro P/R/F1 over foreground classes
    (each species weighs the same regardless of pixel share — the loss's own
    convention), plus per-class F1 under ``f1_c{idx}`` keys. Classes without
    support or predictions are excluded from the macro means, mirroring the
    loss's absent-class rule: a val split that happens to lack a species must
    not drag its macro to zero."""
    model.eval()
    device = next(model.parameters()).device
    per = {c: [0, 0, 0] for c in range(num_classes)}
    loss_sum = 0.0
    n_samples = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        bs = img.shape[0]
        loss_sum += ce_soft_f1_loss(logits, mask, ignore_index=ignore_index).item() * bs
        for c, (tp, fp, fn) in multiclass_counts(
                logits, mask, num_classes, ignore_index).items():
            per[c][0] += tp; per[c][1] += fp; per[c][2] += fn
        n_samples += bs
    # macro over foreground classes WITH support or predictions: a class absent
    # from both GT and prediction (all-zero counts) is neutral, not a zero score
    fg = [prf(*per[c]) for c in range(1, num_classes) if sum(per[c]) > 0]
    if not fg:
        fg = [(0.0, 0.0, 0.0)]
    out = {
        "loss": loss_sum / n_samples if n_samples else 0.0,
        "precision": sum(p for p, _, _ in fg) / len(fg),
        "recall": sum(r for _, r, _ in fg) / len(fg),
        "f1": sum(f for _, _, f in fg) / len(fg),
    }
    for c in range(1, num_classes):
        out[f"f1_c{c}"] = prf(*per[c])[2]
    return out
