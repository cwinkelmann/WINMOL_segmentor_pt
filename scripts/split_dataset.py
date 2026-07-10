#!/usr/bin/env python
"""Materialize a fixed train/val split from a loader-ready dataset.

Writes ``<dst>/train/{train,mask}`` and ``<dst>/val/{train,mask}`` using the SAME
deterministic split as ``training.dataset.train_val_split`` (seeded shuffle +
``val_fraction``), so the held-out validation set is explicit, shareable, and
reproducible — e.g. so PyTorch and the R implementation evaluate on the same tiles.
Each split is renumbered ``train{i}.jpeg`` / ``mask{i}.gif`` (i = 1..N) and is itself
a valid loader dataset. The source is never mutated.

Usage:
  python scripts/split_dataset.py --src /Users/christian/data/Winmol/data/SpecDS \
    --dst /Users/christian/data/Winmol/data/SpecDS_split --val-fraction 0.2 --seed 1
"""
import argparse
import os
import random
import shutil
import sys

# make the dev-only `training` package importable when run as `python scripts/...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import _paired_ids


def split_dataset(src_dir, dst_dir, val_fraction=0.2, seed=1):
    image_dir = os.path.join(src_dir, "train")
    mask_dir = os.path.join(src_dir, "mask")
    ids = _paired_ids(image_dir, mask_dir)
    if not ids:
        raise ValueError(f"no paired train*/mask* files under {src_dir}")

    # identical to training.dataset.train_val_split so the split matches at load time
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    for split, split_ids in (("train", train_ids), ("val", val_ids)):
        ti = os.path.join(dst_dir, split, "train")
        mi = os.path.join(dst_dir, split, "mask")
        os.makedirs(ti, exist_ok=True)
        os.makedirs(mi, exist_ok=True)
        for i, n in enumerate(split_ids, start=1):
            shutil.copyfile(os.path.join(image_dir, f"train{n}.jpeg"),
                            os.path.join(ti, f"train{i}.jpeg"))
            shutil.copyfile(os.path.join(mask_dir, f"mask{n}.gif"),
                            os.path.join(mi, f"mask{i}.gif"))
    return len(train_ids), len(val_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="loader-ready dataset (train/ + mask/)")
    p.add_argument("--dst", required=True, help="output dir (gets train/ and val/ subsets)")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    n_train, n_val = split_dataset(a.src, a.dst, a.val_fraction, a.seed)
    print(f"wrote {n_train} train + {n_val} val pairs to {a.dst} "
          f"(val_fraction={a.val_fraction}, seed={a.seed})")


if __name__ == "__main__":
    main()
