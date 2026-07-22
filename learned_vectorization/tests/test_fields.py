import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fields import render_fields


def test_vertical_line_gives_ridge_orientation_and_diameter():
    # a single vertical centerline at col=5, rows 2..8, constant diameter 4 px
    line = np.array([[2.0, 5.0], [8.0, 5.0]])          # (row, col) endpoints
    diam = np.array([4.0, 4.0])
    heat, orient, diameter = render_fields([line], [diam], (12, 12), sigma=1.5)

    assert heat.shape == (12, 12)
    assert orient.shape == (2, 12, 12)
    assert diameter.shape == (12, 12)

    # on-line pixel: ridge peaks near 1
    assert heat[5, 5] > 0.9
    # perpendicular decay: 3 px off the line is much weaker
    assert heat[5, 8] < 0.3
    # tangent is vertical -> theta=90deg -> (sin2t, cos2t) = (0, -1)
    assert abs(orient[0, 5, 5] - 0.0) < 0.15
    assert orient[1, 5, 5] < -0.85
    # diameter sampled on the ridge equals the input
    assert abs(diameter[5, 5] - 4.0) < 0.5


def test_diameter_interpolates_along_taper():
    # taper 2 -> 6 px along a vertical line rows 0..10 at col 4
    line = np.array([[0.0, 4.0], [10.0, 4.0]])
    diam = np.array([2.0, 6.0])
    _, _, diameter = render_fields([line], [diam], (11, 9), sigma=1.0)
    # near the top ~2, middle ~4, bottom ~6
    assert abs(diameter[0, 4] - 2.0) < 0.6
    assert abs(diameter[5, 4] - 4.0) < 0.8
    assert abs(diameter[10, 4] - 6.0) < 0.6
