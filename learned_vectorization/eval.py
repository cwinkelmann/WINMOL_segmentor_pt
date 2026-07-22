"""Evaluate the distilled model against its teacher on the held-out (val) region.

Runs FieldNet on the val strip, decodes its fields into stems, and compares them to the
heuristic's stems **in that same strip** -- and, as the reference point, to the round-trip of the
*ground-truth* fields (the representation ceiling from round_trip.py).

Three numbers matter:
  heuristic (teacher)        -- the labels
  GT-fields round-trip       -- ceiling of the representation+decoder (no network)
  model-predicted fields     -- what the network achieves

The model can approach the round-trip number but not exceed the teacher: that is the ceiling the
study set out to demonstrate.

  python learned_vectorization/eval.py <gpkg> --ckpt runs/poc/best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_plot, split_tiles_spatially, tile_origins
from decode import decode_fields
from geo import Grid, bounds_of, polyline_length, read_stems, stem_volume
from model import FieldNet


def _summary(stems):
    if not stems:
        return 0, 0.0, float("nan"), 0.0
    n = len(stems)
    tot_len = sum(polyline_length(s["xy"]) for s in stems)
    diams = np.concatenate([np.asarray(s["d"], float) for s in stems])
    vol = sum(stem_volume(s["xy"], s["d"]) for s in stems)
    return n, tot_len, float(np.nanmean(diams)), vol


def _decode_to_world(heat, orient, diam, grid, gsd, min_len_m=0.5):
    dec = decode_fields(heat, orient, diam, heat_thresh=0.3, min_pixels=int(min_len_m / gsd))
    return [{"xy": grid.px_to_world(s["line"]), "d": np.asarray(s["diam"]) * gsd} for s in dec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpkg")
    ap.add_argument("--ckpt", default="learned_vectorization/runs/poc/best.pt")
    ap.add_argument("--gsd", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    stems = read_stems(args.gpkg)
    grid = Grid.from_bounds(bounds_of(stems), args.gsd)
    mask, heat, orient, diam = build_plot(stems, grid, sigma=args.sigma)

    # the held-out strip (must match train.py's split)
    origins = tile_origins(grid.H, grid.W, args.tile, args.tile)
    _, va_o = split_tiles_spatially(origins, grid.W, args.tile, val_frac=0.25, gap=args.tile)
    c0 = min(c for _, c in va_o)
    sl = np.s_[:, c0:]
    print(f"val strip: cols {c0}..{grid.W} ({grid.W - c0}px wide)")

    # teacher stems clipped to the strip (by centroid, so a stem is counted once)
    x_cut = grid.px_to_world(np.array([0.0, float(c0)]))[0]
    teacher = [s for s in stems if float(np.asarray(s["xy"])[:, 0].mean()) >= x_cut]

    sub_grid = Grid(grid.minx + c0 * grid.gsd, grid.maxy, grid.gsd, grid.H, grid.W - c0)
    gt = _decode_to_world(heat[sl], orient[:, :, c0:], diam[sl], sub_grid, args.gsd)

    ck = torch.load(args.ckpt, map_location="cpu")
    net = FieldNet(in_channels=1, base=ck["args"].get("base", 32))
    net.load_state_dict(ck["model"]); net.eval().to(args.device)
    m = np.asarray(mask[sl], np.float32)
    # the encoder halves 3x, so pad to a multiple of 8 and crop the predictions back
    mult = 8
    ph, pw = (-m.shape[0]) % mult, (-m.shape[1]) % mult
    x = torch.from_numpy(np.pad(m, ((0, ph), (0, pw))))[None, None].to(args.device)
    with torch.no_grad():
        out = net(x)
    H0, W0 = m.shape
    p_heat = torch.sigmoid(out["heat"])[0, 0].cpu().numpy()[:H0, :W0]
    p_or = out["orient"][0].cpu().numpy()[:, :H0, :W0]
    p_d = out["diam"][0, 0].cpu().numpy()[:H0, :W0]
    pred = _decode_to_world(p_heat, p_or, p_d, sub_grid, args.gsd)

    rows = [("heuristic (teacher)", _summary(teacher)),
            ("GT-fields round-trip", _summary(gt)),
            ("model prediction", _summary(pred))]
    print("\n| source | stems | length (m) | mean diam (m) | volume (m3) |")
    print("|---|--:|--:|--:|--:|")
    for name, (n, l, d, v) in rows:
        print(f"| {name} | {n} | {l:.1f} | {d:.3f} | {v:.2f} |")


if __name__ == "__main__":
    main()
