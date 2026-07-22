"""Compare the distilled model to its teacher over held-out tiles of a teacher sweep.

For each val tile: run FieldNet on the mask, decode the predicted fields into stems, and compare
the aggregate to the teacher's own stems for that tile. Also decodes the teacher's GT fields, which
isolates how much of any gap is the representation+decoder rather than the network.

  python learned_vectorization/eval_teacher.py work/teacher_specds_pred \
      --ckpt learned_vectorization/runs/teacher/best.pt

Reports per-tile stem-count agreement as well as pooled totals: totals alone can hide
compensating errors (a tile that invents two stems cancelling one that drops two).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from build_teacher_dataset import ANALYZER_GSD
from decode import decode_fields
from geo import polyline_length, read_stems, stem_volume
from model import FieldNet
from teacher_tiles import TeacherTiles, list_tiles, load_tile, split_keys, tile_grid


def summarize(stems):
    """(n, total length m, mean diameter m, total volume m3) for world-coord stems."""
    if not stems:
        return 0, 0.0, np.nan, 0.0
    lens = [polyline_length(s["xy"]) for s in stems]
    diams = np.concatenate([np.asarray(s["d"], float) for s in stems])
    vols = [stem_volume(s["xy"], s["d"]) for s in stems]
    return len(stems), float(np.sum(lens)), float(np.nanmean(diams)), float(np.sum(vols))


def decode_to_world(heat, orient, diam, grid, heat_thresh=0.5, min_len_m=2.0):
    """Decode fields to world-coord stems, applying the analyzer's own 2 m minimum length."""
    dec = decode_fields(heat, orient, diam, heat_thresh=heat_thresh,
                        min_pixels=max(int(min_len_m / grid.gsd), 2))
    return [{"xy": grid.px_to_world(s["line"]), "d": np.asarray(s["diam"]) * grid.gsd}
            for s in dec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ckpt", default="learned_vectorization/runs/teacher/best.pt")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--heat-thresh", type=float, default=0.5)
    ap.add_argument("--limit", type=int, help="evaluate only the first N val tiles")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = FieldNet(in_channels=1, base=ck["args"].get("base", 32))
    net.load_state_dict(ck["model"])
    net.eval().to(args.device)

    # reuse the checkpoint's own val split so we never score on tiles it trained on
    keys = ck.get("val_keys")
    if not keys:
        _, keys = split_keys(list_tiles(args.root),
                            val_frac=ck["args"].get("val_frac", 0.2), seed=0)
    if args.limit:
        keys = keys[:args.limit]
    print(f"{len(keys)} held-out tiles from {args.root}", flush=True)

    tot = {"teacher": [0, 0.0, [], 0.0], "gt_rt": [0, 0.0, [], 0.0], "model": [0, 0.0, [], 0.0]}
    per_tile = []
    for i, key in enumerate(keys, 1):
        mask, heat, orient, diam = load_tile(args.root, key, sigma=args.sigma)
        grid = tile_grid(*mask.shape)
        teacher = read_stems(os.path.join(args.root, "gpkg", key + ".gpkg"))

        gt_rt = decode_to_world(heat, orient, diam, grid, args.heat_thresh)

        x = torch.from_numpy(np.asarray(mask, np.float32))[None, None].to(args.device)
        with torch.no_grad():
            out = net(x)
        pred = decode_to_world(torch.sigmoid(out["heat"])[0, 0].cpu().numpy(),
                               out["orient"][0].cpu().numpy(),
                               out["diam"][0, 0].cpu().numpy(), grid, args.heat_thresh)

        for name, stems in (("teacher", teacher), ("gt_rt", gt_rt), ("model", pred)):
            n, l, d, v = summarize(stems)
            tot[name][0] += n
            tot[name][1] += l
            if not np.isnan(d):
                tot[name][2].append(d)
            tot[name][3] += v
        per_tile.append((len(teacher), len(pred)))
        if i % 50 == 0:
            print(f"  {i}/{len(keys)}", flush=True)

    print("\n| source | stems | length (m) | mean diam (m) | volume (m3) |")
    print("|---|--:|--:|--:|--:|")
    for name, label in (("teacher", "heuristic (teacher)"), ("gt_rt", "GT-fields round-trip"),
                        ("model", "model prediction")):
        n, l, ds, v = tot[name]
        print(f"| {label} | {n} | {l:.1f} | {np.mean(ds) if ds else float('nan'):.3f} | {v:.2f} |")

    a = np.array(per_tile, float)
    exact = float(np.mean(a[:, 0] == a[:, 1])) * 100
    mae = float(np.mean(np.abs(a[:, 0] - a[:, 1])))
    print(f"\nper-tile stem count: exact match {exact:.1f}% | MAE {mae:.2f} stems/tile "
          f"| teacher {a[:, 0].mean():.2f} vs model {a[:, 1].mean():.2f} per tile")


if __name__ == "__main__":
    main()
