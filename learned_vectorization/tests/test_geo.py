import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geo import Grid, polyline_length, stem_volume


def test_world_pixel_roundtrip_is_identity():
    grid = Grid.from_bounds((100.0, 200.0, 150.0, 260.0), gsd=0.5, pad=4)
    xy = np.array([[110.0, 250.0], [140.0, 205.0], [123.4, 231.7]])
    back = grid.px_to_world(grid.world_to_px(xy))
    assert np.allclose(back, xy, atol=1e-6)


def test_grid_is_north_up_and_metric():
    grid = Grid.from_bounds((0.0, 0.0, 10.0, 10.0), gsd=1.0, pad=0)
    # a point higher in y (north) maps to a smaller row; +x maps to larger col
    rc_top = grid.world_to_px(np.array([0.0, 10.0]))     # top-left
    rc_bot = grid.world_to_px(np.array([0.0, 0.0]))      # bottom-left
    assert rc_top[0] < rc_bot[0]
    rc_right = grid.world_to_px(np.array([10.0, 10.0]))
    assert rc_right[1] > rc_top[1]


def test_volume_of_a_cylinder():
    # straight 10 m stem, constant 0.4 m diameter -> pi r^2 h = pi*0.04*10
    xy = np.array([[0.0, 0.0], [0.0, 10.0]])
    d = np.array([0.4, 0.4])
    assert abs(stem_volume(xy, d) - np.pi * 0.2 ** 2 * 10) < 1e-6
    assert abs(polyline_length(xy) - 10.0) < 1e-9
