"""Rasterize stem masks straight from sampled SceneSpecs — no renderer.

    python scripts/rasterize_masks.py --out <DIR> --n 1500 [--seed 1]

ControlNet needs only the mask, not a rendered image, so going through Blender
(~8 s/tile) to obtain one is wasteful: the same SceneSpec rasterizes in
milliseconds here. Geometry matches render_bpy's projection — a nadir camera at
a fixed GSD, so a stem of diameter d metres is d/gsd pixels wide — which keeps
these masks interchangeable with rendered ones.

Masks are drawn from fresh seeds, so they are unseen by any model trained on
real masks: generated tiles cannot be mistaken for memorized training pairs.

With --depth the same SceneSpec also yields a 16-bit height field: terrain
noise plus each stem's half-cylinder profile, ordered by elevation so crossing
stems stack correctly. Mask and depth therefore describe one geometry and stay
consistent by construction, which is what lets a ControlNet-generated RGB tile
ship as RGBD training data. Caveat: the depth describes the *specified*
geometry, not the generated pixels -- if the generator paints brash over a
stem, the depth will not show that occlusion.
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


def rasterize(spec, cfg, size, instances=False):
    """Mask of one scene. Width tapers along the stem, as in the render.

    `instances=False` paints every stem 255 (the binary label real annotation
    provides, and what the ControlNet expects as conditioning). `instances=True`
    paints stem *k* with value *k+1*, so touching and crossing stems stay
    separable — a label real data cannot supply cheaply, because separating
    overlapping stems by hand is the expensive part of annotation.

    Stems are drawn far-to-near, so where two cross the nearer one owns the
    overlap. That matches what a segmenter can actually see.
    """
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    ordered = sorted(spec.stems, key=lambda s: s.elevation_m)
    if instances and len(ordered) > 255:
        raise ValueError(f"{len(ordered)} stems exceeds the 255 ids an 8-bit mask holds")
    for k, stem in enumerate(ordered):
        value = (k + 1) if instances else 255
        pts = _stem_polyline(stem, cfg, size)
        butt_px = stem.diameter_m / cfg.gsd_m_per_px
        for (x0, y0, t0), (x1, y1, _) in zip(pts, pts[1:]):
            frac = 1.0 + (stem.taper - 1.0) * (t0 + 0.5)      # butt -> tip
            w = max(1.0, butt_px * frac)
            d.line([(x0, y0), (x1, y1)], fill=value, width=int(round(w)))
            r = w / 2.0
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=value)   # round the joint
    return img


def _terrain(size, seed, amp):
    """Low-frequency ground undulation, mirroring the depth simulator's octaves."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size), np.float32)
    for i, scale in enumerate((64, 16)):
        coarse = rng.standard_normal((size // scale + 2, size // scale + 2)).astype(np.float32)
        up = np.asarray(Image.fromarray(coarse).resize((size, size), Image.BICUBIC))
        out += up * (0.5 ** i)
    return (out / (out.std() + 1e-6)) * amp


def rasterize_depth(spec, cfg, size):
    """16-bit height field for the same scene: terrain + half-cylinder stems."""
    gsd = cfg.gsd_m_per_px
    radii = [s.diameter_m / 2.0 for s in spec.stems]
    amp = 0.5 * (float(np.median(radii)) if radii else 0.15)
    height = _terrain(size, spec.seed, amp)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for stem in sorted(spec.stems, key=lambda s: s.elevation_m):
        pts = _stem_polyline(stem, cfg, size)
        r_px = (stem.diameter_m / gsd) / 2.0
        # distance to the backbone -> circular cross-section, as in the renderer
        dist = np.full((size, size), np.inf, np.float32)
        for (x0, y0, t0), (x1, y1, _) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            seg = max(dx * dx + dy * dy, 1e-6)
            t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / seg, 0.0, 1.0)
            dist = np.minimum(dist, np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy)))
        frac = 1.0 + (stem.taper - 1.0) * 0.5
        rr = max(1.0, r_px * frac)
        inside = dist <= rr
        bulge = np.zeros((size, size), np.float32)
        bulge[inside] = np.sqrt(np.clip(rr ** 2 - dist[inside] ** 2, 0, None)) * gsd
        z = stem.elevation_m - stem.burial * (stem.diameter_m / 2.0)
        height[inside] = np.maximum(height[inside], z + bulge[inside])

    lo, hi = float(height.min()), float(height.max())
    norm = np.zeros_like(height) if hi <= lo else (height - lo) / (hi - lo)
    return Image.fromarray((norm * 65535).astype(np.uint16), mode="I;16")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--min-stem-fraction", type=float, default=0.01)
    p.add_argument("--max-stem-fraction", type=float, default=0.35)
    p.add_argument("--depth", action="store_true",
                   help="also write depth{N}.png (16-bit) from the same scene")
    p.add_argument("--instances", action="store_true",
                   help="also write inst{N}.png with one id per stem (0 = background)")
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
        if a.instances:
            # same geometry, per-stem ids: the binary mask above stays the
            # ControlNet conditioning, this is the richer label
            rasterize(spec, cfg, a.size, instances=True).save(
                os.path.join(a.out, f"inst{written}.png"))
        if a.depth:
            rasterize_depth(spec, cfg, a.size).save(
                os.path.join(a.out, f"depth{written}.png"))
    print(f"wrote {written} masks to {a.out} (from {tried} sampled scenes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
