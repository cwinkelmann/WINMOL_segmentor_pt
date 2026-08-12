"""The loss is selectable, and `bce` is the R-equivalent one.

R's loss is `BCE + (1 - F1)` with `k_round` applied to the prediction. A rounded value has
zero gradient almost everywhere, so R's F1 term never reaches the optimiser and R trains on
BCE alone. `--loss bce` reproduces that, which makes the soft-F1 term ablatable.
"""
import torch

from training.losses import LOSSES, bce_loss, bce_soft_f1_loss
from training.run_train import config_from_args


def test_both_losses_registered_and_distinct():
    assert set(LOSSES) == {"bce_soft_f1", "bce"}
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    assert abs(bce_soft_f1_loss(logits, target) - bce_loss(logits, target)) > 1e-3


def test_soft_f1_term_is_the_only_difference():
    """bce_soft_f1 == bce + (1 - soft_F1), exactly."""
    logits = torch.randn(4, 1, 8, 8)
    target = (torch.rand(4, 1, 8, 8) > 0.5).float()
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum()
    fp = (probs * (1 - target)).sum()
    fn = ((1 - probs) * target).sum()
    soft_f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    assert torch.allclose(bce_soft_f1_loss(logits, target),
                          bce_loss(logits, target) + (1 - soft_f1), atol=1e-6)


def test_a_rounded_f1_term_has_no_gradient():
    """Why `bce` is the honest stand-in for R: the rounded term cannot train anything."""
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    hard = torch.round(torch.sigmoid(logits))          # R's k_round(y_pred)
    tp = (hard * target).sum()
    fp = (hard * (1 - target)).sum()
    fn = ((1 - hard) * target).sum()
    (1 - (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).backward()
    assert logits.grad is None or torch.count_nonzero(logits.grad) == 0


def test_cli_selects_the_loss():
    assert config_from_args(["--data-dir", "x"]).loss == "bce_soft_f1"
    assert config_from_args(["--data-dir", "x", "--loss", "bce"]).loss == "bce"
