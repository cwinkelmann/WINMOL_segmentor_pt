"""Training pairs from a build_teacher_dataset.py run: real mask tile -> real teacher fields.

This is the honest version of `PlotTiles`. There, the input mask was *rendered from the labels*
and then hand-corrupted, so the net was partly reading back its own targets. Here the input is a
mask the UNet actually produced (or the dataset's ground truth) and the target is what the real
analyzer heuristic made of that exact mask -- so the (input, label) relation is precisely the
mapping the study claims to distil.

Every tile shares one synthetic georeference (build_teacher_dataset.tile_profile), so world
coordinates in the gpkg map back to pixels with a fixed Grid.
"""
import json
import os

import numpy as np

from build_teacher_dataset import ANALYZER_GSD, load_binary_mask
from fields import render_fields
from geo import Grid, read_stems


def tile_grid(height=512, width=512, gsd=ANALYZER_GSD):
    """The inverse of build_teacher_dataset.tile_profile: world -> pixel for one tile."""
    return Grid(minx=0.0, maxy=0.0, gsd=gsd, H=height, W=width)


def list_tiles(root, require_gpkg=True):
    """Keys with both a mask and (optionally) a teacher gpkg, from a teacher-dataset dir."""
    mask_dir, gpkg_dir = os.path.join(root, "masks"), os.path.join(root, "gpkg")
    if not os.path.isdir(mask_dir):                 # --source gt keeps masks in the dataset dir
        mask_dir = None
    keys = []
    for f in sorted(os.listdir(gpkg_dir)):
        if f.endswith(".gpkg"):
            keys.append(os.path.splitext(f)[0])
    if not require_gpkg:
        return keys
    return [k for k in keys
            if mask_dir is None or os.path.exists(os.path.join(mask_dir, k + ".png"))]


def split_keys(keys, val_frac=0.2, seed=0):
    """Random tile-level split. The tiles come from many plots and are disjoint images, so a
    random split does not leak neighbouring pixels the way it would inside a single plot."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(keys))
    n_val = max(int(round(len(keys) * val_frac)), 1 if keys else 0)
    val = {keys[i] for i in idx[:n_val]}
    return [k for k in keys if k not in val], [k for k in keys if k in val]


def load_tile(root, key, sigma=2.0, size=512):
    """(mask, heat, orient, diam) for one tile, all pixel-space float arrays."""
    mask_path = os.path.join(root, "masks", key + ".png")
    mask = load_binary_mask(mask_path)
    grid = tile_grid(*mask.shape)
    stems = read_stems(os.path.join(root, "gpkg", key + ".gpkg"))
    polylines, diams = [], []
    for s in stems:
        rc = grid.world_to_px(s["xy"])
        d_px = np.asarray(s["d"], float) / grid.gsd
        polylines.append(rc)
        diams.append(d_px)
    heat, orient, diam = render_fields(polylines, diams, mask.shape, sigma=sigma)
    return np.asarray(mask, bool), heat, orient, diam


class TeacherTiles:
    """torch Dataset over teacher-labelled tiles. Caches decoded tiles in memory (they are
    512x512, so a few thousand fit comfortably); pass cache=False for very large sweeps."""

    def __init__(self, root, keys, sigma=2.0, cache=True):
        self.root, self.keys, self.sigma = root, list(keys), sigma
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        import torch
        key = self.keys[i]
        if self._cache is not None and key in self._cache:
            mask, heat, orient, diam = self._cache[key]
        else:
            mask, heat, orient, diam = load_tile(self.root, key, sigma=self.sigma)
            if self._cache is not None:
                self._cache[key] = (mask, heat, orient, diam)
        x = torch.from_numpy(np.asarray(mask, np.float32))[None]
        tgt = {
            "heat": torch.from_numpy(heat.astype(np.float32))[None],
            "orient": torch.from_numpy(orient.astype(np.float32)),
            "diam": torch.from_numpy(diam.astype(np.float32))[None],
        }
        return x, tgt
