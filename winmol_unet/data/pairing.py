"""The loader convention: how `train*` and `mask*` files pair up.

This is data-layer knowledge, not training-layer, and it is deliberately torch-free — pure
stdlib. `winmol_unet.data.split` needs it to materialise a split that matches what the
loader would pick, and pulling `training.dataset` in for it dragged torch into a module
whose whole point is that it needs neither torch nor GDAL.

`training.dataset` re-exports these, so `from winmol_unet.training.dataset import
_paired_ids` keeps working.
"""
import os
import re


def _index_by_n(names, prefix, ext):
    """Map the part between `prefix` and `ext` to the filename.

    Our samplers emit `train1.jpeg` / `mask1.gif`, but the published Zenodo sets use
    `train_100_1.jpeg` / `mask_100_1.gif`. Both pair on the shared key, so keying by the
    raw stem reads either layout without renaming anything — important when the data is
    a published artefact that should be used exactly as distributed.

    Keys stay strings; `_sort_key` orders numeric ones numerically so `train2` still
    precedes `train10`. macOS AppleDouble files (`._train1.jpeg`) fail the prefix test.
    """
    out = {}
    for name in names:
        if name.startswith(prefix) and name.endswith(ext):
            out[name[len(prefix):-len(ext)]] = name
    return out


def _sort_key(stem):
    """Numeric where possible, so ordering matches the old int-keyed behaviour."""
    parts = re.split(r"(\d+)", stem)
    return tuple((1, int(p)) if p.isdigit() else (0, p) for p in parts if p != "")


def _paired_ids(image_dir, mask_dir):
    imgs = _index_by_n(os.listdir(image_dir), "train", ".jpeg")
    masks = _index_by_n(os.listdir(mask_dir), "mask", ".gif")
    return sorted(set(imgs) & set(masks), key=_sort_key)
