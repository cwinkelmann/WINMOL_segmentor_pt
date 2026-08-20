"""The loss is selectable, and `bce` is the R-equivalent one.

R's loss is `BCE + (1 - F1)` with `k_round` applied to the prediction. A rounded value has
zero gradient almost everywhere, so R's F1 term never reaches the optimiser and R trains on
BCE alone. `--loss bce` reproduces that, which makes the soft-F1 term ablatable.
"""
import torch

from winmol_unet.training.losses import LOSSES, bce_loss, bce_soft_f1_loss
from winmol_unet.training.run_train import config_from_args


def test_both_losses_registered_and_distinct():
    assert set(LOSSES) == {"bce_soft_f1", "bce", "bce_hard_f1",
                           "focal", "focal_soft_f1"}
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


def test_r_literal_loss_has_the_same_gradient_as_plain_bce():
    """`bce_hard_f1` is R's F1Score_loss transcribed. It reports a bigger number than
    `bce` and produces a bit-identical gradient, which is the whole finding."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.9).float()

    grads = {}
    values = {}
    for name in ("bce", "bce_hard_f1", "bce_soft_f1"):
        x = logits.clone().requires_grad_(True)
        v = LOSSES[name](x, target)
        v.backward()
        grads[name] = x.grad.clone()
        values[name] = float(v.detach())

    # R's term inflates the reported loss...
    assert values["bce_hard_f1"] > values["bce"] + 0.5
    # ...and changes nothing the optimiser sees
    assert torch.equal(grads["bce_hard_f1"], grads["bce"])
    # while the soft version genuinely does
    assert not torch.allclose(grads["bce_soft_f1"], grads["bce"])


def test_label_smoothing_touches_only_the_edge_band():
    """Boundary-aware smoothing must leave interiors and far background alone.

    That is the whole point: false negatives are 1.59x enriched within 2 px of an
    annotation edge, while interiors are pixels the annotators were sure about.
    """
    from winmol_unet.training.losses import soften_targets

    t = torch.zeros(1, 1, 11, 11)
    t[0, 0, 4:7, 4:7] = 1.0                        # a 3x3 stem

    s = soften_targets(t, eps=0.1, band_px=1)
    assert s[0, 0, 5, 5].item() == 1.0             # interior untouched
    assert abs(s[0, 0, 4, 4].item() - 0.9) < 1e-5  # stem edge pulled down
    assert abs(s[0, 0, 3, 5].item() - 0.1) < 1e-5  # background beside it pulled up
    assert s[0, 0, 0, 0].item() == 0.0             # far background untouched

    # off by default, and exactly the identity when eps=0
    assert torch.equal(soften_targets(t, 0.0, 2), t)

    # band_px=0 is the classic global variant: every pixel moves
    g = soften_targets(t, eps=0.1, band_px=0)
    assert abs(g[0, 0, 5, 5].item() - 0.9) < 1e-5
    assert abs(g[0, 0, 0, 0].item() - 0.1) < 1e-5
