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


# ---- CLI ----


def _mask_ids(mask_dir):
    out = {}
    for name in os.listdir(mask_dir):
        stem, ext = os.path.splitext(name)
        if ext == ".gif" and stem.startswith("mask") and stem[len("mask"):].isdigit():
            out[int(stem[len("mask"):])] = name
    return out


def _load_mask(path):
    im = Image.open(path)
    im.seek(0)
    return np.asarray(im.convert("L")) >= 128       # same binarization as the loader


def _save_uint16(depth, path):
    lo, hi = float(depth.min()), float(depth.max())
    scaled = np.zeros_like(depth) if hi <= lo else (depth - lo) / (hi - lo)
    Image.fromarray((scaled * 65535).astype(np.uint16), mode="I;16").save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="dataset dir containing mask/mask{N}.gif")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stem-drop-p", type=float, default=0.2)
    p.add_argument("--distractors", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--mismatch", action="store_true",
                   help="leakage control: generate depth{N} from a DIFFERENT image's "
                        "mask (ids rotated by one)")
    a = p.parse_args(argv)

    mask_dir = os.path.join(a.dataset, "mask")
    depth_dir = os.path.join(a.dataset, "depth")
    if not os.path.isdir(mask_dir):
        print(f"error: {mask_dir} not found (expected <DS>/mask/mask{{N}}.gif)",
              file=sys.stderr)
        return 2
    if os.path.isdir(depth_dir) and os.listdir(depth_dir) and not a.overwrite:
        print(f"error: {depth_dir} exists; pass --overwrite to regenerate",
              file=sys.stderr)
        return 2
    ids = _mask_ids(mask_dir)
    if not ids:
        print(f"error: no mask{{N}}.gif files in {mask_dir}", file=sys.stderr)
        return 2

    os.makedirs(depth_dir, exist_ok=True)
    ordered = sorted(ids)
    # mismatch: depth{N} uses the NEXT id's mask (rotation = a derangement for >1 id)
    source = {n: ordered[(i + 1) % len(ordered)] if a.mismatch else n
              for i, n in enumerate(ordered)}
    for n in ordered:
        mask = _load_mask(os.path.join(mask_dir, ids[source[n]]))
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, n]))
        depth = simulate_depth_map(mask, rng, a.stem_drop_p, a.distractors)
        _save_uint16(depth, os.path.join(depth_dir, f"depth{n}.png"))
    print(f"wrote {len(ordered)} depth maps to {depth_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
