"""Representation round-trip: gpkg -> dense fields -> decode -> stems, compared to the input gpkg.

Bounds the WHOLE Approach-A ceiling *independent of any network*: if decoding the ground-truth
fields can't recover the heuristic's stems, no trained model could either. No training involved.

  python learned_vectorization/round_trip.py <detected_stems.gpkg> [--gsd 0.05]
"""
import argparse
import sys, os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fields import render_fields
from decode import decode_fields
from geo import (Grid, read_stems, stems_to_pixel, bounds_of, polyline_length, stem_volume)


def _summary(stems, grid=None):
    """(n, total length m, mean diameter m, total volume m3) for world stems (xy,d)."""
    n = len(stems)
    tot_len = sum(polyline_length(s["xy"]) for s in stems)
    diams = np.concatenate([np.asarray(s["d"], float) for s in stems]) if stems else np.array([0.0])
    vol = sum(stem_volume(s["xy"], s["d"]) for s in stems)
    return n, tot_len, float(np.nanmean(diams)), vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpkg")
    ap.add_argument("--gsd", type=float, default=0.05, help="metres/pixel for the synthetic grid")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--heat-thresh", type=float, default=0.3)
    ap.add_argument("--min-len-m", type=float, default=0.5)
    args = ap.parse_args()

    stems = read_stems(args.gpkg)
    verts = np.array([len(s["xy"]) for s in stems])
    print(f"input: {len(stems)} stems | vertices/stem min/med/max = "
          f"{verts.min()}/{int(np.median(verts))}/{verts.max()}")

    grid = Grid.from_bounds(bounds_of(stems), args.gsd)
    print(f"grid: {grid.H}x{grid.W} px @ {args.gsd} m/px")
    polylines, diams = stems_to_pixel(stems, grid)

    import time
    t0 = time.perf_counter()
    heat, orient, diam = render_fields(polylines, diams, (grid.H, grid.W), sigma=args.sigma)
    t1 = time.perf_counter()
    decoded_px = decode_fields(heat, orient, diam, heat_thresh=args.heat_thresh,
                               min_pixels=int(args.min_len_m / args.gsd))
    t2 = time.perf_counter()
    print(f"timing: render {t1-t0:.1f}s | decode {t2-t1:.1f}s")
    # back to world
    decoded = [{"xy": grid.px_to_world(s["line"]), "d": np.asarray(s["diam"]) * args.gsd}
               for s in decoded_px]

    ni, li, di, vi = _summary(stems)
    no, lo, do, vo = _summary(decoded)
    print("\n| metric | heuristic (in) | round-trip (out) | ratio out/in |")
    print("|---|--:|--:|--:|")
    print(f"| stems | {ni} | {no} | {no/ni:.2f} |")
    print(f"| total length (m) | {li:.1f} | {lo:.1f} | {lo/li:.2f} |")
    print(f"| mean diameter (m) | {di:.3f} | {do:.3f} | {do/di:.2f} |")
    print(f"| total volume (m3) | {vi:.2f} | {vo:.2f} | {vo/vi:.2f} |")


if __name__ == "__main__":
    main()
