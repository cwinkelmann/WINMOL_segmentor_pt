"""Build a LoRA training set of real forest-floor *background* crops.

    python scripts/build_lora_dataset.py --out <DIR> [--max-stem-fraction 0.02]
        [--per-source 400] [--ortho <tif> ...] [--dataset <DS> ...]

Why background-only: a style LoRA trained on whole tiles learns that these
scenes contain fallen trunks, and then paints trunks into the backgrounds we
inpaint — unlabelled stems, which teach the segmenter that stems are
background.

Whole-tile filtering cannot deliver this: of 8224 labelled tiles only 5 are
stem-free, because every real tile contains stems. So crops are *mined* inside
tiles instead — sliding windows whose mask is empty, plus a dilated margin so
a stem just outside the window does not bleed in. That turns a dataset with no
clean tiles into thousands of provably clean crops.

Orthomosaic crops have no masks and therefore no proof; they are off by
default (--allow-ortho) and screened by a bright-streak heuristic that is a
weak substitute, not an equivalent.

Writes HuggingFace imagefolder layout: images + metadata.jsonl with captions.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None

TRIGGER = "winmolfloor"
CAPTION = (f"{TRIGGER}, aerial nadir drone photograph of a storm-damaged conifer "
           "forest floor, fallen branches, needles, brash, moss and litter")


def _stem_fraction(mask_path):
    with Image.open(mask_path) as m:
        m.seek(0)
        return float((np.asarray(m.convert("L")) >= 128).mean())


def _looks_log_free(rgb):
    """Reject crops holding a long bright streak — an unlabelled trunk.

    Debarked trunks are the brightest thing in these scenes, so a large bright
    connected component elongated along one axis is the signature to avoid.
    """
    g = np.asarray(Image.fromarray(rgb).convert("L"), dtype=float)
    bright = g > np.percentile(g, 96)
    lab, n = ndi.label(bright)
    for sl in ndi.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = (lab[sl] > 0).sum()
        if area > 0.01 * g.size and max(h, w) > 0.55 * g.shape[0] and max(h, w) > 3 * min(h, w):
            return False
    return True


def _mine_windows(mask, window, margin, stride, limit_per_tile):
    """Top-left corners of `window`-sized boxes containing no stem pixel.

    The mask is dilated by `margin` first, so a stem lying just outside the box
    cannot appear in it through the crop boundary.
    """
    blocked = ndi.binary_dilation(mask, iterations=margin) if margin else mask
    # integral image -> O(1) test that a window is empty
    ii = np.cumsum(np.cumsum(blocked.astype(np.int32), 0), 1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    h, w = blocked.shape
    out = []
    for y in range(0, h - window + 1, stride):
        for x in range(0, w - window + 1, stride):
            total = (ii[y + window, x + window] - ii[y, x + window]
                     - ii[y + window, x] + ii[y, x])
            if total == 0:
                out.append((y, x))
                if len(out) >= limit_per_tile:
                    return out
    return out


def _from_dataset(ds, out, limit, max_frac, start_idx, window=None, margin=6,
                  stride=64, per_tile=2):
    train_dir, mask_dir = os.path.join(ds, "train"), os.path.join(ds, "mask")
    if not (os.path.isdir(train_dir) and os.path.isdir(mask_dir)):
        return [], start_idx
    rows, idx = [], start_idx
    names = sorted(os.listdir(train_dir))
    for name in names:
        if len(rows) >= limit:
            break
        stem, ext = os.path.splitext(name)
        if ext != ".jpeg" or not stem.startswith("train") or not stem[5:].isdigit():
            continue
        mask_path = os.path.join(mask_dir, f"mask{stem[5:]}.gif")
        if not os.path.exists(mask_path):
            continue
        with Image.open(mask_path) as mk:
            mk.seek(0)
            mask = np.asarray(mk.convert("L")) >= 128
        img = Image.open(os.path.join(train_dir, name)).convert("RGB")

        if window is None:                       # whole-tile mode
            if mask.mean() > max_frac:
                continue
            boxes = [(0, 0)]
            size = img.size[0]
        else:                                    # mine stem-free windows
            boxes = _mine_windows(mask, window, margin, stride, per_tile)
            size = window

        for (y, x) in boxes:
            if len(rows) >= limit:
                break
            dst = f"floor{idx:05d}.jpeg"
            img.crop((x, y, x + size, y + size)).save(
                os.path.join(out, dst), quality=95)
            rows.append({"file_name": dst, "text": CAPTION})
            idx += 1
    return rows, idx


def _from_ortho(path, out, limit, start_idx, seed, size=512):
    rows, idx = [], start_idx
    rng = np.random.default_rng(seed)
    with Image.open(path) as im:
        tries = 0
        while len(rows) < limit and tries < limit * 40:
            tries += 1
            x = int(rng.integers(0, max(1, im.width - size)))
            y = int(rng.integers(0, max(1, im.height - size)))
            a = np.asarray(im.crop((x, y, x + size, y + size)))
            if a.shape[2] == 4:
                if (a[..., 3] < 250).mean() > 0.01:      # nodata border
                    continue
                a = a[..., :3]
            if a.mean() < 25 or a.std() < 12:            # black or featureless
                continue
            if not _looks_log_free(a):
                continue
            dst = f"floor{idx:05d}.jpeg"
            Image.fromarray(a).save(os.path.join(out, dst), quality=95)
            rows.append({"file_name": dst, "text": CAPTION})
            idx += 1
    return rows, idx


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", action="append", default=[],
                   help="loader-ready dataset dir (train/ + mask/); repeatable")
    p.add_argument("--ortho", action="append", default=[],
                   help="orthomosaic tif to crop from; repeatable")
    p.add_argument("--max-stem-fraction", type=float, default=0.02,
                   help="reject tiles with more labelled stem than this")
    p.add_argument("--per-source", type=int, default=400)
    p.add_argument("--window-px", type=int, default=256,
                   help="mine stem-free windows of this size (0 = whole tiles)")
    p.add_argument("--margin-px", type=int, default=6,
                   help="dilate stems by this before testing a window")
    p.add_argument("--per-tile", type=int, default=2)
    p.add_argument("--allow-ortho", action="store_true",
                   help="include unlabelled orthomosaic crops (no proof they are "
                        "stem-free; heuristic screening only)")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    rows, idx = [], 0
    window = a.window_px or None
    for ds in a.dataset:
        got, idx = _from_dataset(ds, a.out, a.per_source, a.max_stem_fraction, idx,
                                 window=window, margin=a.margin_px, per_tile=a.per_tile)
        print(f"{len(got):5d} background tiles from {ds}")
        rows += got
    for k, ortho in enumerate(a.ortho if a.allow_ortho else []):
        got, idx = _from_ortho(ortho, a.out, a.per_source, idx, a.seed + k)
        print(f"{len(got):5d} crops from {os.path.basename(ortho)}")
        rows += got

    with open(os.path.join(a.out, "metadata.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} images + metadata.jsonl to {a.out} (trigger: {TRIGGER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
