import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fields import render_fields
from decode import decode_fields


def test_single_line_decodes_to_one_polyline_with_right_span_and_diameter():
    line = np.array([[2.0, 5.0], [18.0, 5.0]])         # vertical, rows 2..18 col 5
    diam = np.array([4.0, 4.0])
    heat, orient, diameter = render_fields([line], [diam], (21, 11), sigma=1.5)

    stems = decode_fields(heat, orient, diameter, heat_thresh=0.3, min_pixels=4)
    assert len(stems) == 1
    poly = np.asarray(stems[0]["line"])
    rows = poly[:, 0]
    assert rows.min() <= 4 and rows.max() >= 16          # spans most of the line
    assert abs(np.asarray(stems[0]["diam"]).mean() - 4.0) < 0.7
    assert np.allclose(poly[:, 1], 5, atol=1.5)          # stays near col 5


def test_two_separate_lines_decode_to_two_polylines():
    l1 = np.array([[2.0, 3.0], [18.0, 3.0]])
    l2 = np.array([[2.0, 12.0], [18.0, 12.0]])
    d = np.array([3.0, 3.0])
    heat, orient, diameter = render_fields([l1, l2], [d, d], (21, 16), sigma=1.5)
    stems = decode_fields(heat, orient, diameter, heat_thresh=0.3, min_pixels=4)
    assert len(stems) == 2
