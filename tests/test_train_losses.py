import torch
from winmol_unet.training.losses import bce_soft_f1_loss
from winmol_unet.training.metrics import precision, recall, f1


def _logits_for(target, strong=12.0):
    # logits whose sigmoid ~ target (near 0 or 1)
    return (target * 2 - 1) * strong


def test_perfect_prediction_low_loss_high_f1():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = _logits_for(target)
    loss = bce_soft_f1_loss(logits, target)
    assert loss.item() < 1e-2
    assert f1(logits, target) == 1.0
    assert precision(logits, target) == 1.0
    assert recall(logits, target) == 1.0


def test_loss_is_differentiable_through_soft_f1():
    target = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.zeros_like(target, requires_grad=True)
    bce_soft_f1_loss(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_metrics_match_hand_computed():
    # pred: [1,1,0,0]  target: [1,0,0,0]  -> tp=1 fp=1 fn=0 tn=2
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    logits = _logits_for(torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]))
    assert abs(precision(logits, target) - 0.5) < 1e-6   # tp/(tp+fp)=1/2
    assert abs(recall(logits, target) - 1.0) < 1e-6      # tp/(tp+fn)=1/1
    assert abs(f1(logits, target) - (2 / 3)) < 1e-6      # 2*.5*1/(1.5)
