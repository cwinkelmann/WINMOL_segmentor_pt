"""Rasterize stem masks straight from sampled SceneSpecs — no renderer.

    python scripts/rasterize_masks.py --out <DIR> --n 1500 [--seed 1]

ControlNet needs only the mask, not a rendered image, so going through Blender
(~8 s/tile) to obtain one is wasteful: the same SceneSpec rasterizes in
milliseconds here. Geometry matches render_bpy's projection — a nadir camera at
a fixed GSD, so a stem of diameter d metres is d/gsd pixels wide — which keeps
these masks interchangeable with rendered ones.

Masks are drawn from fresh seeds, so they are unseen by any model trained on
real masks: generated tiles cannot be mistaken for memorized training pairs.
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synthgen.sampler import GenConfig, sample_scene   # noqa: E402


def _stem_polyline(stem, cfg, size):
    """Backbone points of one stem in pixel coordinates."""
    gsd = cfg.gsd_m_per_px
    cx = size / 2 + stem.center_xy_m[0] / gsd
    cy = size / 2 - stem.center_xy_m[1] / gsd
    L = stem.length_m / gsd
    ang = math.radians(stem.azimuth_deg)
    bend = stem.bend_m / gsd
    pts = []
    for t in np.linspace(-0.5, 0.5, 9):
        # quadratic bow across the stem, same shape the curve bevel produces
        off = bend * (1.0 - 4.0 * t * t)
        x = cx + t * L * math.cos(ang) - off * math.sin(ang)
        y = cy + t * L * math.sin(ang) + off * math.cos(ang)
        pts.append((x, y, t))
    return pts


def rasterize(spec, cfg, size):
    """Binary mask of one scene. Width tapers along the stem, as in the render."""
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    # far stems first so nearer ones overwrite: elevation orders the pile
    for stem in sorted(spec.stems, key=lambda s: s.elevation_m):
        pts = _stem_polyline(stem, cfg, size)
        butt_px = stem.diameter_m / cfg.gsd_m_per_px
        for (x0, y0, t0), (x1, y1, _) in zip(pts, pts[1:]):
            frac = 1.0 + (stem.taper - 1.0) * (t0 + 0.5)      # butt -> tip
            w = max(1.0, butt_px * frac)
            d.line([(x0, y0), (x1, y1)], fill=255, width=int(round(w)))
            r = w / 2.0
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=255)   # round the joint
    return img


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--min-stem-fraction", type=float, default=0.01)
    p.add_argument("--max-stem-fraction", type=float, default=0.35)
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    cfg = GenConfig(tile_px=a.size)
    written, tried = 0, 0
    while written < a.n and tried < a.n * 20:
        spec = sample_scene(cfg, seed=a.seed + tried)
        tried += 1
        m = rasterize(spec, cfg, a.size)
        frac = float((np.asarray(m) >= 128).mean())
        if not a.min_stem_fraction <= frac <= a.max_stem_fraction:
            continue
        written += 1
        m.save(os.path.join(a.out, f"mask{written}.png"))
    print(f"wrote {written} masks to {a.out} (from {tried} sampled scenes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
