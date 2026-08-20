"""Focal loss as a third arm beside `bce` and `bce_soft_f1`.

Focal (Lin et al. 2017) is `-alpha_t * (1 - p_t)^gamma * log(p_t)`: it keeps BCE's shape
but down-weights pixels the model already gets right, so the gradient concentrates on the
hard ones. Motivation here is the same asymmetry the label-smoothing attempt targeted and
failed to fix -- false negatives are 1.59x enriched within 2 px of an annotation edge --
except focal reweights by *difficulty* rather than by distance to an edge.

`alpha` defaults to None (no class re-weighting) so that a focal-vs-bce difference is
attributable to the focusing term alone, matching how every other arm in this repo isolates
exactly one change.
"""
import torch

from winmol_unet.training.losses import LOSSES, bce_loss, focal_loss


def test_gamma_zero_without_alpha_is_exactly_bce():
    """The focusing term is the only thing focal adds; switch it off and BCE remains."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    assert torch.allclose(focal_loss(logits, target, gamma=0.0),
                          bce_loss(logits, target), atol=1e-6)


def test_easy_pixels_are_down_weighted_relative_to_bce():
    """The defining property: on confident-correct pixels focal must fall below BCE."""
    target = torch.ones(1, 1, 8, 8)
    easy = torch.full_like(target, 4.0)      # sigmoid ~ 0.982, already correct
    assert focal_loss(easy, target).item() < 0.25 * bce_loss(easy, target).item()


def test_hard_pixels_keep_almost_all_their_weight():
    """A pixel the model gets wrong is barely modulated -- (1-p_t)^gamma -> 1."""
    target = torch.ones(1, 1, 8, 8)
    hard = torch.full_like(target, -4.0)     # sigmoid ~ 0.018, confidently wrong
    ratio = focal_loss(hard, target).item() / bce_loss(hard, target).item()
    assert ratio > 0.95


def test_focal_is_differentiable():
    target = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.zeros_like(target, requires_grad=True)
    focal_loss(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_alpha_reweights_the_positive_class():
    """alpha is available but off by default, so it never silently confounds an arm."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    positives = torch.ones(2, 1, 16, 16)
    unweighted = focal_loss(logits, positives, alpha=None).item()
    assert abs(focal_loss(logits, positives, alpha=0.25).item()
               - 0.25 * unweighted) < 1e-6


def test_focal_arms_are_registered_and_selectable():
    assert {"focal", "focal_soft_f1"} <= set(LOSSES)
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    # focal_soft_f1 adds the same (1 - soft_F1) term bce_soft_f1 does
    assert LOSSES["focal_soft_f1"](logits, target) > LOSSES["focal"](logits, target)


def test_cli_accepts_the_focal_arms():
    from winmol_unet.training.run_train import config_from_args
    for name in ("focal", "focal_soft_f1"):
        assert config_from_args(["--data-dir", "x", "--loss", name]).loss == name
