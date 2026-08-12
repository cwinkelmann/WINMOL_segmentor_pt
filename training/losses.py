"""BCE + (1 - soft_F1) on logits. Soft (un-thresholded) F1 keeps the term
differentiable, unlike the R F1 which rounds y_pred (see design spec §8)."""
import torch
import torch.nn.functional as F  # noqa: F401


def bce_soft_f1_loss(logits, target, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum()
    fp = (probs * (1 - target)).sum()
    fn = ((1 - probs) * target).sum()
    soft_f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return bce + (1 - soft_f1)


def bce_loss(logits, target):
    """Plain BCE — what the R pipeline effectively optimises.

    R's loss is `BCE + (1 - F1)` but its F1 term applies `k_round` to the prediction, and
    a rounded value has zero gradient almost everywhere. The F1 term therefore contributes
    nothing to the update and R trains on BCE alone. Selecting this makes the comparison
    against `bce_soft_f1_loss` an ablation of the one change that actually reaches the
    optimiser.
    """
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(logits, target)


def bce_hard_f1_loss(logits, target, eps=1e-6):
    """R's loss transcribed literally: BCE + (1 - F1(round(sigmoid(x)))).

    Included so the no-gradient claim can be shown in this framework too, not only
    argued. `torch.round` has zero derivative almost everywhere, so this trains
    identically to `bce_loss` while reporting a larger number -- exactly as
    controlling.R's F1Score_loss does. Never select it for real training; it exists
    to reproduce R.
    """
    import torch
    bce = F.binary_cross_entropy_with_logits(logits, target)
    hard = torch.round(torch.sigmoid(logits))
    tp = (hard * target).sum()
    fp = (hard * (1 - target)).sum()
    fn = ((1 - hard) * target).sum()
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return bce + (1 - f1)


LOSSES = {"bce_soft_f1": bce_soft_f1_loss, "bce": bce_loss,
          "bce_hard_f1": bce_hard_f1_loss}
