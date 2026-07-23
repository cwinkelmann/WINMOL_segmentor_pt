"""Multi-task loss for the field heads.

  heat   BCE-with-logits over the whole tile (the ridge is sparse, so `pos_weight` lifts it).
  orient (1 - |cos|) between predicted and target (sin 2t, cos 2t), **masked to the ridge** --
         orientation is undefined away from a centerline. |cos| because the field is undirected
         (theta and theta+pi are the same tangent, and the 2t encoding already folds that in, so
         a plain dot is fine; abs() guards a sign flip from the network).
  diam   L1, **masked to the ridge** -- diameter is only meaningful on a stem.

Masking matters: supervising orientation/diameter on background would swamp the signal with
meaningless targets (that is what test_losses guards).
"""
import torch
import torch.nn.functional as F


def field_loss(pred, target, w_heat=1.0, w_orient=1.0, w_diam=1.0,
               ridge_thresh=0.5, pos_weight=20.0):
    heat_t = target["heat"]
    ridge = (heat_t >= ridge_thresh).float()                    # (N,1,H,W)

    heat = F.binary_cross_entropy_with_logits(
        pred["heat"], heat_t,
        pos_weight=torch.as_tensor(pos_weight, device=heat_t.device, dtype=heat_t.dtype))

    denom = ridge.sum().clamp_min(1.0)

    cos = (pred["orient"] * target["orient"]).sum(dim=1, keepdim=True)   # (N,1,H,W)
    orient = (((1.0 - cos.abs()) * ridge).sum() / denom)

    diam = (((pred["diam"] - target["diam"]).abs() * ridge).sum() / denom)

    total = w_heat * heat + w_orient * orient + w_diam * diam
    return {"total": total, "heat": heat, "orient": orient, "diam": diam}
