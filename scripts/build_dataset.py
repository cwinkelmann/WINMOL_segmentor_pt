"""Convert a source image/mask dataset into the loader-conformant format.

Reads ``<src>/train`` (images) + ``<src>/mask`` (masks), pairs each by the shared
key (filename minus the ``train``/``mask`` prefix and extension), and writes
``<dst>/train/train{i}.jpeg`` + ``<dst>/mask/mask{i}.gif`` (i = 1..N, sorted by key).

Masks are **binarized** (any pixel > 0 -> 255, saved as mode ``L``), so palette /
instance masks (e.g. spruce's mode-P indices 0-157) become the binary {0,255} the
loader (`training.dataset.StemDataset`) expects. The source is never mutated. Pairs
are numbered by sorted key, so an already-conformant set may be renumbered (pairing
is preserved). Raises if no pairs are found.
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image


def _shared_key(filename, prefix):
    """Filename minus its extension and leading train/mask prefix (the pairing key)."""
    stem = os.path.splitext(filename)[0]
    return stem[len(prefix):] if stem.startswith(prefix) else None


def _index(dirpath, prefix, exts):
    out = {}
    for name in os.listdir(dirpath):
        if name.lower().endswith(exts):
            key = _shared_key(name, prefix)
            if key is not None:
                out[key] = name
    return out


def build_dataset(src_dir, dst_dir):
    img_src = os.path.join(src_dir, "train")
    mask_src = os.path.join(src_dir, "mask")
    imgs = _index(img_src, "train", (".jpeg", ".jpg"))
    masks = _index(mask_src, "mask", (".gif",))
    keys = sorted(set(imgs) & set(masks))
    if not keys:
        raise ValueError(
            f"no paired 'train*'/'mask*' files found under {src_dir} "
            f"({len(imgs)} images, {len(masks)} masks indexed) — check names/extensions")

    img_dst = os.path.join(dst_dir, "train")
    mask_dst = os.path.join(dst_dir, "mask")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    for i, key in enumerate(keys, start=1):
        shutil.copyfile(os.path.join(img_src, imgs[key]),
                        os.path.join(img_dst, f"train{i}.jpeg"))
        mk = Image.open(os.path.join(mask_src, masks[key]))
        mk.seek(0)                                   # first frame
        fg = np.asarray(mk) > 0
        if fg.ndim == 3:                             # RGB/RGBA mask -> any channel set
            fg = fg.any(axis=-1)
        binary = fg.astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(os.path.join(mask_dst, f"mask{i}.gif"))
    return len(keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="dir with train/ (images) + mask/ (masks)")
    p.add_argument("--dst", required=True, help="output dir (train/ + mask/, loader-ready)")
    a = p.parse_args()
    n = build_dataset(a.src, a.dst)
    print(f"wrote {n} pairs to {a.dst}")


if __name__ == "__main__":
    main()
