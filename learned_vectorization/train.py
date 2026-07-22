"""Distil the heuristic vectorizer into FieldNet: mask -> (heat, orient, diam).

Trains on tiles of one plot with a SPATIAL train/val split (held-out geography). The labels are
the heuristic's own gpkg output, so validation loss measures how well the net reproduces the
teacher -- by construction it cannot beat it. That is the study's point.

  python learned_vectorization/train.py <detected_stems.gpkg> --epochs 30 --out runs/poc
Run in the winmol-vec image (needs geopandas for the gpkg).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import PlotTiles, build_plot, split_tiles_spatially, tile_origins
from geo import Grid, bounds_of, read_stems
from losses import field_loss
from model import FieldNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpkg")
    ap.add_argument("--gsd", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--corrupt", action="store_true", help="perturb the mask like a real UNet mask")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="learned_vectorization/runs/poc")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    stems = read_stems(args.gpkg)
    grid = Grid.from_bounds(bounds_of(stems), args.gsd)
    mask, heat, orient, diam = build_plot(stems, grid, sigma=args.sigma)
    print(f"plot {grid.H}x{grid.W}px @ {args.gsd} m/px | {len(stems)} stems | "
          f"mask fg {mask.mean()*100:.2f}%", flush=True)

    origins = tile_origins(grid.H, grid.W, args.tile, args.stride)
    tr_o, va_o = split_tiles_spatially(origins, grid.W, args.tile, val_frac=0.25, gap=args.tile)
    print(f"tiles: {len(tr_o)} train / {len(va_o)} val (spatial split, gap={args.tile}px)", flush=True)

    tr = PlotTiles(mask, heat, orient, diam, tr_o, args.tile, corrupt=args.corrupt, seed=1)
    va = PlotTiles(mask, heat, orient, diam, va_o, args.tile, corrupt=False, seed=2)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    dev = torch.device(args.device)
    net = FieldNet(in_channels=1, base=args.base).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    hist, best = [], float("inf")

    for ep in range(1, args.epochs + 1):
        net.train()
        agg = {"total": 0.0, "heat": 0.0, "orient": 0.0, "diam": 0.0}
        for x, tgt in trl:
            x = x.to(dev); tgt = {k: v.to(dev) for k, v in tgt.items()}
            loss = field_loss(net(x), tgt)
            opt.zero_grad(); loss["total"].backward(); opt.step()
            for k in agg:
                agg[k] += float(loss[k]) * x.size(0)
        for k in agg:
            agg[k] /= max(len(tr), 1)

        net.eval()
        vagg = {"total": 0.0, "heat": 0.0, "orient": 0.0, "diam": 0.0}
        with torch.no_grad():
            for x, tgt in val:
                x = x.to(dev); tgt = {k: v.to(dev) for k, v in tgt.items()}
                loss = field_loss(net(x), tgt)
                for k in vagg:
                    vagg[k] += float(loss[k]) * x.size(0)
        for k in vagg:
            vagg[k] /= max(len(va), 1)

        hist.append({"epoch": ep, "train": agg, "val": vagg})
        print(f"ep {ep:3d} | train {agg['total']:.4f} (h {agg['heat']:.4f} o {agg['orient']:.4f} "
              f"d {agg['diam']:.4f}) | val {vagg['total']:.4f} (h {vagg['heat']:.4f} "
              f"o {vagg['orient']:.4f} d {vagg['diam']:.4f})", flush=True)
        if vagg["total"] < best:
            best = vagg["total"]
            torch.save({"model": net.state_dict(), "args": vars(args)},
                       os.path.join(args.out, "best.pt"))

    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=2)
    print(f"done. best val {best:.4f} -> {args.out}/best.pt", flush=True)


if __name__ == "__main__":
    main()
