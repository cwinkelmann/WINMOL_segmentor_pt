"""Render what the distilled model predicts on held-out tiles, next to its teacher.

Per tile, five panels:
  1. input      -- the UNet mask the model actually sees
  2. teacher    -- the analyzer's stems for that mask (the labels)
  3. pred heat  -- predicted centerline ridge (what the decoder traces)
  4. pred angle -- predicted orientation, hue = tangent direction, masked to the ridge
  5. decoded    -- teacher stems (green) vs the model's decoded stems (red)

  python learned_vectorization/visualize.py work/teacher_specds_pred \
      --ckpt learned_vectorization/runs/teacher_specds/best.pt --out work/viz.png
"""
import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, os.path.dirname(__file__))
from eval_teacher import decode_to_world
from geo import read_stems
from model import FieldNet
from teacher_tiles import list_tiles, load_tile, split_keys, tile_grid


def angle_image(orient, heat, thresh=0.3):
    """Orientation field as hue (undirected: theta and theta+pi share a colour), ridge-masked."""
    theta = 0.5 * np.arctan2(orient[0], orient[1])          # (sin2t, cos2t) -> t in [-pi/2, pi/2]
    hue = (theta / np.pi) % 1.0
    val = np.clip(heat / max(heat.max(), 1e-6), 0, 1) * (heat >= thresh)
    hsv = np.stack([hue, np.ones_like(hue) * 0.9, val], axis=-1)
    return hsv_to_rgb(hsv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ckpt", default="learned_vectorization/runs/teacher_specds/best.pt")
    ap.add_argument("--out", default="work/viz.png")
    ap.add_argument("--n", type=int, default=6, help="number of tiles to show")
    ap.add_argument("--min-stems", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--heat-thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = FieldNet(in_channels=1, base=ck["args"].get("base", 32))
    net.load_state_dict(ck["model"])
    net.eval().to(args.device)

    keys = ck.get("val_keys") or split_keys(list_tiles(args.root), seed=0)[1]

    # prefer tiles with a few stems -- legible, and representative of the median tile (3.75)
    chosen = []
    for k in keys:
        n = len(read_stems(os.path.join(args.root, "gpkg", k + ".gpkg")))
        if n >= args.min_stems:
            chosen.append((k, n))
        if len(chosen) >= args.n:
            break
    print(f"showing {len(chosen)} held-out tiles: {[k for k, _ in chosen]}", flush=True)

    fig, axes = plt.subplots(len(chosen), 5, figsize=(17, 3.4 * len(chosen)))
    axes = np.atleast_2d(axes)
    titles = ["input: UNet mask", "teacher (analyzer)", "predicted ridge",
              "predicted orientation", "decoded: teacher vs model"]

    for r, (key, n_teacher) in enumerate(chosen):
        mask, heat, orient, diam = load_tile(args.root, key, sigma=args.sigma)
        grid = tile_grid(*mask.shape)
        teacher = read_stems(os.path.join(args.root, "gpkg", key + ".gpkg"))

        x = torch.from_numpy(np.asarray(mask, np.float32))[None, None].to(args.device)
        with torch.no_grad():
            out = net(x)
        p_heat = torch.sigmoid(out["heat"])[0, 0].cpu().numpy()
        p_or = out["orient"][0].cpu().numpy()
        p_d = out["diam"][0, 0].cpu().numpy()
        pred = decode_to_world(p_heat, p_or, p_d, grid, args.heat_thresh)

        axes[r, 0].imshow(mask, cmap="gray")
        axes[r, 1].imshow(mask, cmap="gray", alpha=0.35)
        for s in teacher:
            rc = grid.world_to_px(s["xy"])
            axes[r, 1].plot(rc[:, 1], rc[:, 0], "-", color="lime", lw=1.6)
        axes[r, 2].imshow(p_heat, cmap="magma", vmin=0, vmax=1)
        axes[r, 3].imshow(angle_image(p_or, p_heat))
        axes[r, 4].imshow(mask, cmap="gray", alpha=0.3)
        for s in teacher:
            rc = grid.world_to_px(s["xy"])
            axes[r, 4].plot(rc[:, 1], rc[:, 0], "-", color="lime", lw=2.2, alpha=0.85)
        for s in pred:
            rc = grid.world_to_px(s["xy"])
            axes[r, 4].plot(rc[:, 1], rc[:, 0], "-", color="red", lw=1.4)

        axes[r, 0].set_ylabel(f"tile {key}", fontsize=10)
        axes[r, 4].set_title(f"teacher {n_teacher} / model {len(pred)}", fontsize=10, pad=4)
        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0 and c != 4:
                axes[r, c].set_title(titles[c], fontsize=11, pad=6)

    fig.suptitle("Distilled vectorizer on held-out tiles — green = analyzer heuristic (teacher), "
                 "red = model", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
