"""Are a site's stem polygons actually on top of its stems?

A label set can be correctly drawn and still sit metres from the imagery it is scored
against — mixed datums do exactly this, silently, because both CRSs name the same UTM
zone and reprojection between them succeeds. The result looks like a model failure.

**This has to run in the orthomosaic's own grid.** Training tiles are cut at random
rotations, so a constant world offset appears at a *different* pixel offset in every tile
and averaging over tiles smears it away — measuring in tile-pixel space produced a
confident wrong answer here once already.

The test: rasterise the stems onto the native grid, then slide that mask over a grid of
offsets and record where stem pixels separate most strongly, in luminance, from their
surroundings. Well-registered labels peak at (0, 0).

    python scripts/check_registration.py --ortho x.tif --stems x.shp --max-shift-m 1.5
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _windows(geoms, src, n, half_px, rng):
    """Native-grid windows centred on random stem polygons."""
    from rasterio.windows import Window
    idx = rng.permutation(len(geoms))
    out = []
    for i in idx:
        g = geoms[int(i)]
        if g.is_empty:
            continue
        r, c = src.index(g.centroid.x, g.centroid.y)
        if not (half_px < r < src.height - half_px and half_px < c < src.width - half_px):
            continue
        out.append((Window(c - half_px, r - half_px, 2 * half_px, 2 * half_px), g))
        if len(out) >= n:
            break
    return out


def check(ortho, stems, n=60, max_shift_m=1.5, step_px=1, half_px=220, seed=1,
          species=None, quiet=False):
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from shapely.strtree import STRtree

    from scripts.sample_training_tiles import _load_geoms

    src = rasterio.open(ortho)
    gsd = abs(src.transform.a)
    geoms, src_crs = _load_geoms(stems, src.crs, species, "stems", quiet=True)
    if not geoms:
        raise SystemExit(f"no stem polygons read from {stems}")
    tree = STRtree(geoms)
    rng = np.random.default_rng(seed)

    max_px = int(round(max_shift_m / gsd))
    shifts = list(range(-max_px, max_px + 1, step_px))
    acc = np.zeros((len(shifts), len(shifts)))
    used = 0

    for win, _ in _windows(geoms, src, n, half_px, rng):
        arr = src.read(window=win, indexes=[1, 2, 3], boundless=True, fill_value=0)
        if arr.shape[0] < 3:
            continue
        a = np.transpose(arr, (1, 2, 0)).astype(np.float32) / 255.0
        if (a.sum(2) == 0).mean() > 0.2:          # nodata edge
            continue
        lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
        tf = src.window_transform(win)
        from shapely.geometry import box as sbox
        b = rasterio.windows.bounds(win, src.transform)
        hits = [geoms[i] for i in tree.query(sbox(*b))]
        if not hits:
            continue
        m = rio_rasterize([(g, 1) for g in hits], out_shape=lum.shape, transform=tf,
                          fill=0, all_touched=False).astype(bool)
        if m.sum() < 200 or (~m).sum() < 200:
            continue
        sd = lum.std() + 1e-9
        used += 1
        for yi, dy in enumerate(shifts):
            for xi, dx in enumerate(shifts):
                s = np.roll(np.roll(m, dy, 0), dx, 1)
                acc[yi, xi] += (lum[s].mean() - lum[~s].mean()) / sd
    if not used:
        raise SystemExit("no usable windows — check the AOI, nodata, or the stem layer")
    acc /= used

    z = shifts.index(0)
    yi, xi = np.unravel_index(np.abs(acc).argmax(), acc.shape)
    return {"ortho": ortho, "stems": stems, "gsd_m": gsd, "windows": used,
            "stems_crs": str(src_crs), "ortho_crs": str(src.crs),
            "crs_mismatch": str(src_crs) != str(src.crs),
            "contrast_at_zero": float(acc[z, z]),
            "peak_contrast": float(acc[yi, xi]),
            "peak_dy_px": shifts[yi], "peak_dx_px": shifts[xi],
            "peak_dy_m": shifts[yi] * gsd, "peak_dx_m": shifts[xi] * gsd,
            "peak_offset_m": math.hypot(shifts[yi] * gsd, shifts[xi] * gsd)}


def main(argv=None):
    import json
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ortho", required=True)
    p.add_argument("--stems", required=True)
    p.add_argument("--n", type=int, default=60, help="stem windows to average over")
    p.add_argument("--max-shift-m", type=float, default=1.5)
    p.add_argument("--step-px", type=int, default=1)
    p.add_argument("--half-px", type=int, default=220)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--label", default=None)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    r = check(a.ortho, a.stems, a.n, a.max_shift_m, a.step_px, a.half_px, a.seed)
    print(f"{a.label or os.path.basename(a.ortho)}   {r['windows']} windows, "
          f"{100 * r['gsd_m']:.2f} cm/px")
    print(f"  stems CRS {r['stems_crs']}   ortho CRS {r['ortho_crs']}"
          f"   {'MISMATCH' if r['crs_mismatch'] else 'match'}")
    print(f"  contrast at zero shift : {r['contrast_at_zero']:+.3f}")
    print(f"  strongest contrast     : {r['peak_contrast']:+.3f} at "
          f"dx {r['peak_dx_m']:+.2f} m, dy {r['peak_dy_m']:+.2f} m "
          f"(|offset| {r['peak_offset_m']:.2f} m)")
    verdict = ("registered" if r["peak_offset_m"] <= 2 * r["gsd_m"]
               else f"OFFSET {r['peak_offset_m']:.2f} m")
    print(f"  verdict: {verdict}")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
