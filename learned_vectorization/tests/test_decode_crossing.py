import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fields import render_fields
from decode import decode_fields


def _span(stem):
    p = np.asarray(stem["line"])
    return np.hypot(*(p[0] - p[-1]))


def test_two_crossing_stems_trace_straight_through_the_junction():
    # an X: two diagonals crossing at the centre. Orientation/momentum should keep each
    # stem going straight through, recovering 2 long stems -- not 4 short fragments.
    l1 = np.array([[2.0, 2.0], [18.0, 18.0]])
    l2 = np.array([[2.0, 18.0], [18.0, 2.0]])
    d = np.array([3.0, 3.0])
    heat, orient, diameter = render_fields([l1, l2], [d, d], (21, 21), sigma=1.2)

    stems = decode_fields(heat, orient, diameter, heat_thresh=0.3, min_pixels=4)
    long_stems = [s for s in stems if _span(s) > 18]     # near the full diagonal (~22 px)
    assert len(long_stems) == 2, f"expected 2 through-stems, got {len(stems)} (long={len(long_stems)})"
