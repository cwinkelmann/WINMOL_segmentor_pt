"""Generate synthetic terrain depth maps (depth{N}.png) from stem masks (mask{N}.gif).

Design: docs/superpowers/specs/2026-07-29-depth-simulator-design.md. Synthetic
pretraining/plumbing data for --rgbd, NOT evaluation-grade depth: fractal terrain
+ per-stem cylindrical bulges, degraded by bulge dropout, off-mask distractor
bulges, and sensor noise so depth is a helpful-but-unreliable cue (anti-leakage).

Crossing stems merge into one connected component; the junction gets a larger
inradius and bulges higher, resembling piled logs. Intentional.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def fractal_terrain(shape, rng, amplitude):
    """Fractal heightfield: octaves at 64/16/4 px feature scales, amplitude
    halving per octave, std normalized to `amplitude`."""
    out = np.zeros(shape, np.float32)
    for i, scale in enumerate((64, 16, 4)):
        coarse = rng.standard_normal((shape[0] // scale + 2, shape[1] // scale + 2))
        up = ndi.zoom(coarse, scale, order=3)[: shape[0], : shape[1]]
        out += ndi.gaussian_filter(up, scale / 4) * (0.5 ** i)
    return ((out / out.std()) * amplitude).astype(np.float32)


def bulge_from(region):
    """Half-buried-cylinder bulge for a boolean region: inward distance d gives
    centerline offset x = r - d, so h = r*sqrt(1-(x/r)^2) = r*sqrt(2u-u^2), u=d/r.
    Returns (height HW float32, max inradius r)."""
    d = ndi.distance_transform_edt(region)
    r = float(d.max())
    if r == 0:
        return np.zeros(region.shape, np.float32), 0.0
    u = d / r
    return (r * np.sqrt(np.clip(2 * u - u * u, 0, 1))).astype(np.float32), r


def _distractor_region(shape, rng):
    """Random rotated ellipse (rock / root plate / debris)."""
    cy, cx = rng.integers(15, shape[0] - 15), rng.integers(15, shape[1] - 15)
    ry, rx = rng.uniform(4, 11, 2)
    ang = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0: shape[0], 0: shape[1]]
    ys, xs = yy - cy, xx - cx
    yr = ys * np.cos(ang) - xs * np.sin(ang)
    xr = ys * np.sin(ang) + xs * np.cos(ang)
    return (yr / ry) ** 2 + (xr / rx) ** 2 <= 1


def simulate_depth_map(mask, rng, stem_drop_p=0.2, distractors=3):
    """Full model for one mask -> float32 heightfield (arbitrary relative units)."""
    mask = np.asarray(mask, bool)
    labels, n = ndi.label(mask)
    radii = [ndi.distance_transform_edt(labels == k).max() for k in range(1, n + 1)]
    # terrain amplitude ~ same order as stem height so depth alone can't
    # trivially separate ground from stem
    amp = 0.5 * (float(np.median(radii)) if radii else 6.0)
    depth = fractal_terrain(mask.shape, rng, amp)

    for k in range(1, n + 1):                       # per-stem bulge with dropout
        if rng.random() < stem_drop_p:
            continue
        h, _ = bulge_from(labels == k)
        depth += h

    if distractors > 0:                             # sampled in [d-1, d+2]
        for _ in range(int(rng.integers(distractors - 1, distractors + 3))):
            h, _ = bulge_from(_distractor_region(mask.shape, rng))
            depth += h

    depth += rng.standard_normal(mask.shape).astype(np.float32) * 0.05 * amp
    return ndi.gaussian_filter(depth, 1.0).astype(np.float32)
