"""Build a (stem mask -> real tile) pair dataset for ControlNet training.

    python scripts/build_controlnet_dataset.py --out <DIR> --dataset <DS> [--dataset <DS> ...]

ControlNet learns the inverse of segmentation: given the binary stem mask as
conditioning, produce a photorealistic tile whose stems land exactly on that
mask. The labelled datasets are already in that format — mask{N}.gif is the
conditioning image, train{N}.jpeg is the target — so no new annotation is
needed.

Only well-labelled sources belong here. In datasets where annotation covers
some stems and leaves other woody debris unlabelled, the model learns to draw
stem-like objects *outside* the mask, which is a false-negative source for
anything trained on the output. `--min-stem-fraction` additionally drops
near-empty tiles, which teach nothing about stem appearance.

Writes a HuggingFace dataset via save_to_disk with columns:
  image (target), conditioning_image (mask, RGB), text (caption).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


CAPTION = ("aerial nadir drone photograph of a storm-damaged conifer forest, "
           "fallen tree stems on the forest floor, brash, needles, moss and litter")


def _pairs(ds, min_frac, max_frac):
    train_dir, mask_dir = os.path.join(ds, "train"), os.path.join(ds, "mask")
    if not (os.path.isdir(train_dir) and os.path.isdir(mask_dir)):
        return []
    out = []
    for name in sorted(os.listdir(train_dir)):
        stem, ext = os.path.splitext(name)
        if ext != ".jpeg" or not stem.startswith("train") or not stem[5:].isdigit():
            continue
        mask_path = os.path.join(mask_dir, f"mask{stem[5:]}.gif")
        if not os.path.exists(mask_path):
            continue
        with Image.open(mask_path) as m:
            m.seek(0)
            frac = float((np.asarray(m.convert("L")) >= 128).mean())
        if not min_frac <= frac <= max_frac:
            continue
        out.append((os.path.join(train_dir, name), mask_path, frac))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="save_to_disk target dir")
    p.add_argument("--dataset", action="append", required=True,
                   help="well-labelled loader-ready dataset dir; repeatable")
    p.add_argument("--min-stem-fraction", type=float, default=0.005,
                   help="drop near-empty tiles: they teach nothing about stems")
    p.add_argument("--max-stem-fraction", type=float, default=0.6)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--caption", default=CAPTION)
    a = p.parse_args(argv)

    from datasets import Dataset, Features, Image as HFImage, Value

    rows = []
    for ds in a.dataset:
        got = _pairs(ds, a.min_stem_fraction, a.max_stem_fraction)
        print(f"{len(got):5d} pairs from {ds}")
        rows += got
    if not rows:
        print("error: no pairs found", file=sys.stderr)
        return 2

    fracs = np.array([f for _, _, f in rows])
    print(f"stem fraction: mean {fracs.mean():.3f} median {np.median(fracs):.3f} "
          f"min {fracs.min():.3f} max {fracs.max():.3f}")

    # masks are single-channel; ControlNet conditioning wants 3 channels
    cond_dir = os.path.join(a.out + "_cond")
    os.makedirs(cond_dir, exist_ok=True)
    images, conds = [], []
    for i, (img_path, mask_path, _) in enumerate(rows):
        with Image.open(mask_path) as m:
            m.seek(0)
            arr = (np.asarray(m.convert("L")) >= 128).astype(np.uint8) * 255
        cond_path = os.path.join(cond_dir, f"cond{i:06d}.png")
        Image.fromarray(np.dstack([arr] * 3)).resize(
            (a.size, a.size), Image.NEAREST).save(cond_path)
        images.append(img_path)
        conds.append(cond_path)

    ds = Dataset.from_dict(
        {"image": images, "conditioning_image": conds, "text": [a.caption] * len(rows)},
        features=Features({"image": HFImage(), "conditioning_image": HFImage(),
                           "text": Value("string")}),
    )
    ds.save_to_disk(a.out)
    print(f"saved {len(ds)} pairs to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
