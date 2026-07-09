"""BCE + (1 - soft_F1) on logits. Soft (un-thresholded) F1 keeps the term
differentiable, unlike the R F1 which rounds y_pred (see design spec §8)."""
import torch
import torch.nn.functional as F


def bce_soft_f1_loss(logits, target, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum()
    fp = (probs * (1 - target)).sum()
    fn = ((1 - probs) * target).sum()
    soft_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return bce + (1 - soft_f1)
