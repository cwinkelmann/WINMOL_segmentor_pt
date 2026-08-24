"""Hard-rounded precision/recall/F1 for reporting (sigmoid + 0.5 threshold)."""
import torch


def counts(logits, target):
    """Return (tp, fp, fn) over the batch, thresholded at sigmoid >= 0.5."""
    pred = (torch.sigmoid(logits) >= 0.5).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return tp, fp, fn


def prf(tp, fp, fn):
    """Precision, recall, F1 from raw counts (0.0 when undefined)."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def precision(logits, target):
    return prf(*counts(logits, target))[0]


def recall(logits, target):
    return prf(*counts(logits, target))[1]


def f1(logits, target):
    return prf(*counts(logits, target))[2]
