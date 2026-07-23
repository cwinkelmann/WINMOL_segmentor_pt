import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from losses import field_loss


def _target(H=8, W=8):
    heat = torch.zeros(1, 1, H, W)
    heat[0, 0, 4, 2:6] = 1.0                      # a short ridge
    orient = torch.zeros(1, 2, H, W)
    orient[0, 0] = 0.0
    orient[0, 1] = -1.0                           # vertical tangent everywhere
    diam = torch.full((1, 1, H, W), 3.0)
    return {"heat": heat, "orient": orient, "diam": diam}


def test_orientation_and_diameter_are_only_supervised_on_the_ridge():
    tgt = _target()
    pred = {
        "heat": torch.zeros(1, 1, 8, 8),
        "orient": tgt["orient"].clone(),
        "diam": tgt["diam"].clone(),
    }
    good = field_loss(pred, tgt)

    # corrupt orientation/diameter ONLY off the ridge -> loss must not change
    bad = {k: v.clone() for k, v in pred.items()}
    off = tgt["heat"][0, 0] < 0.5
    bad["orient"][0, 0][off] = 1.0
    bad["orient"][0, 1][off] = 0.0
    bad["diam"][0, 0][off] = 99.0
    same = field_loss(bad, tgt)

    assert abs(float(good["orient"]) - float(same["orient"])) < 1e-6
    assert abs(float(good["diam"]) - float(same["diam"])) < 1e-6


def test_matching_fields_give_zero_orient_and_diam_loss():
    tgt = _target()
    pred = {"heat": torch.zeros(1, 1, 8, 8), "orient": tgt["orient"].clone(),
            "diam": tgt["diam"].clone()}
    out = field_loss(pred, tgt)
    assert float(out["orient"]) < 1e-6
    assert float(out["diam"]) < 1e-6


def test_wrong_orientation_on_ridge_is_penalised():
    tgt = _target()
    flipped = tgt["orient"].clone()
    flipped[0, 0] = 1.0
    flipped[0, 1] = 0.0                            # 90deg off in (sin2t,cos2t) space
    pred = {"heat": torch.zeros(1, 1, 8, 8), "orient": flipped, "diam": tgt["diam"].clone()}
    assert float(field_loss(pred, tgt)["orient"]) > 0.5


def test_total_is_weighted_sum_of_parts():
    tgt = _target()
    pred = {"heat": torch.zeros(1, 1, 8, 8), "orient": tgt["orient"].clone(),
            "diam": tgt["diam"].clone()}
    out = field_loss(pred, tgt, w_heat=1.0, w_orient=2.0, w_diam=3.0)
    expect = out["heat"] + 2.0 * out["orient"] + 3.0 * out["diam"]
    assert abs(float(out["total"]) - float(expect)) < 1e-6
