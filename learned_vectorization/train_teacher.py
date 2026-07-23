"""Distil the heuristic from REAL teacher labels: UNet mask tile -> analyzer fields.

Same net and loss as train.py, but the data comes from a build_teacher_dataset.py sweep instead
of one hand-uploaded plot. Two things change the meaning of the result:

  * the input is a mask the UNet actually produced, not a mask rendered from the labels, so the
    net can no longer partially read its targets out of its input;
  * there are thousands of tiles from many plots, so a held-out split is worth something.

    python learned_vectorization/train_teacher.py work/teacher_specds_pred --epochs 20

Tiles are NOT cached by default: a 512x512 tile carries ~4.5 MB of float32 fields, so caching
thousands of them would need tens of GB. Use --num-workers to hide the rendering cost instead.
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from losses import field_loss
from model import FieldNet
from teacher_tiles import TeacherTiles, list_tiles, split_keys


def run_epoch(net, loader, dev, opt=None):
    train = opt is not None
    net.train() if train else net.eval()
    agg, n = {"total": 0.0, "heat": 0.0, "orient": 0.0, "diam": 0.0}, 0
    for x, tgt in loader:
        x = x.to(dev)
        tgt = {k: v.to(dev) for k, v in tgt.items()}
        with torch.set_grad_enabled(train):
            loss = field_loss(net(x), tgt)
        if train:
            opt.zero_grad()
            loss["total"].backward()
            opt.step()
        for k in agg:
            agg[k] += float(loss[k].detach()) * x.size(0)
        n += x.size(0)
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="a build_teacher_dataset.py output dir (masks/ + gpkg/)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="ridge width px; the tiles are 0.029 m/px, much finer than the plot runs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, help="use only the first N tiles (smoke runs)")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--cache", action="store_true", help="keep decoded tiles in RAM (memory hungry)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="learned_vectorization/runs/teacher")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    keys = list_tiles(args.root)
    if args.limit:
        keys = keys[:args.limit]
    tr_k, va_k = split_keys(keys, val_frac=args.val_frac, seed=0)
    print(f"{len(keys)} teacher tiles in {args.root} -> {len(tr_k)} train / {len(va_k)} val",
          flush=True)

    tr = TeacherTiles(args.root, tr_k, sigma=args.sigma, cache=args.cache)
    va = TeacherTiles(args.root, va_k, sigma=args.sigma, cache=args.cache)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    dev = torch.device(args.device)
    net = FieldNet(in_channels=1, base=args.base).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    hist, best = [], float("inf")

    for ep in range(1, args.epochs + 1):
        a = run_epoch(net, trl, dev, opt)
        v = run_epoch(net, val, dev)
        hist.append({"epoch": ep, "train": a, "val": v})
        print(f"ep {ep:3d} | train {a['total']:.4f} (h {a['heat']:.4f} o {a['orient']:.4f} "
              f"d {a['diam']:.4f}) | val {v['total']:.4f} (h {v['heat']:.4f} "
              f"o {v['orient']:.4f} d {v['diam']:.4f})", flush=True)
        if v["total"] < best:
            best = v["total"]
            torch.save({"model": net.state_dict(), "args": vars(args), "val_keys": va_k},
                       os.path.join(args.out, "best.pt"))

    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=2)
    print(f"done. best val {best:.4f} -> {args.out}/best.pt", flush=True)


if __name__ == "__main__":
    main()
