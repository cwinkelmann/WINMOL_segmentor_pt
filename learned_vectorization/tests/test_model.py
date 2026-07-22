import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import FieldNet


def test_three_heads_have_field_shapes():
    net = FieldNet(in_channels=1, base=8).eval()
    x = torch.zeros(2, 1, 64, 64)
    with torch.no_grad():
        out = net(x)
    assert out["heat"].shape == (2, 1, 64, 64)
    assert out["orient"].shape == (2, 2, 64, 64)
    assert out["diam"].shape == (2, 1, 64, 64)


def test_orientation_is_unit_norm_and_diameter_non_negative():
    net = FieldNet(in_channels=1, base=8).eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = net(x)
    norm = out["orient"].pow(2).sum(1).sqrt()      # sin^2+cos^2 must be 1
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4)
    assert float(out["diam"].min()) >= 0.0          # diameters can't be negative


def test_accepts_mask_plus_rgb_input():
    net = FieldNet(in_channels=4, base=8).eval()   # mask + RGB
    with torch.no_grad():
        out = net(torch.zeros(1, 4, 32, 32))
    assert out["heat"].shape == (1, 1, 32, 32)
