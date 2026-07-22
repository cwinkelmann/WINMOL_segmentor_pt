import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fields import render_fields
from decode import decode_fields


def test_dense_crossing_mesh_terminates():
    # 4 vertical + 4 horizontal lines -> 16 crossings, many adjacent junction pixels after
    # skeletonization. Regression guard: the per-walk `seen` set must keep this terminating
    # (pre-fix the momentum tracer ping-ponged among adjacent junctions forever).
    lines, diams = [], []
    for c in (4, 10, 16, 22):
        lines.append(np.array([[2.0, float(c)], [26.0, float(c)]])); diams.append(np.array([2.0, 2.0]))
    for r in (4, 10, 16, 22):
        lines.append(np.array([[float(r), 2.0], [float(r), 26.0]])); diams.append(np.array([2.0, 2.0]))
    heat, orient, diam = render_fields(lines, diams, (29, 29), sigma=1.2)

    stems = decode_fields(heat, orient, diam, heat_thresh=0.3, min_pixels=3)
    # it must return (terminate) and recover the long through-lines
    assert len(stems) >= 6
    spans = [np.hypot(*(np.asarray(s["line"])[0] - np.asarray(s["line"])[-1])) for s in stems]
    assert sum(sp > 18 for sp in spans) >= 6      # most of the 8 lines traced full-length
