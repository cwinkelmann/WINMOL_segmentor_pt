"""Hard-rounded precision/recall/F1 for reporting (sigmoid + 0.5 threshold)."""
import torch


def _counts(logits, target):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return tp, fp, fn


def precision(logits, target):
    tp, fp, _ = _counts(logits, target)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(logits, target):
    tp, _, fn = _counts(logits, target)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(logits, target):
    p, r = precision(logits, target), recall(logits, target)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
