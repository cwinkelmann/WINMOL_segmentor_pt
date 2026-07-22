"""Tiles a plot into (mask -> fields) training pairs for the distillation study.

The full plot is rasterized once from the heuristic gpkg:
  input  = synthetic stem **mask** (render_mask, optionally corrupted to mimic a real UNet mask)
  target = the three fields (render_fields)
then cut into tiles. The train/val split is **spatial** (a left/right cut with a gap), so
validation is held-out geography rather than neighbouring pixels -- the honest way to measure
generalization when only one plot is available.
"""
import numpy as np

from fields import render_fields, render_mask


def tile_origins(H, W, tile, stride):
    """Top-left origins covering the image; the last row/col is snapped to the border."""
    rs = list(range(0, max(H - tile, 0) + 1, stride))
    cs = list(range(0, max(W - tile, 0) + 1, stride))
    if rs and rs[-1] != H - tile:
        rs.append(H - tile)
    if cs and cs[-1] != W - tile:
        cs.append(W - tile)
    return [(r, c) for r in rs for c in cs]


def split_tiles_spatially(origins, W, tile, val_frac=0.25, gap=0):
    """Left = train, right = val, separated by `gap` px so no pixel is shared."""
    cut = int(W * (1.0 - val_frac))
    train = [(r, c) for r, c in origins if c + tile <= cut - gap]
    val = [(r, c) for r, c in origins if c >= cut]
    return train, val


def corrupt_mask(mask, rng, drop_p=0.15, blob_p=0.15, dilate_p=0.3):
    """Perturb a synthetic mask toward what a real UNet mask looks like: random erasures
    (gaps/occlusion), spurious blobs (false positives), and slight dilation/erosion."""
    from skimage.morphology import dilation, disk, erosion

    out = np.array(mask, bool, copy=True)
    H, W = out.shape
    # densities are calibrated per 256x256 tile; scale by area so a whole plot/strip is degraded
    # as heavily per-pixel as a tile is (>=1 event each, so small crops still get corrupted)
    area = max(H * W / (256.0 * 256.0), 0.0)
    # erase random patches (simulates missed/occluded stem sections)
    n_drop = max(rng.poisson(drop_p * 12 * area), 1 if drop_p > 0 else 0)
    for _ in range(n_drop):
        h, w = rng.integers(4, 16), rng.integers(4, 16)
        r, c = rng.integers(0, max(H - h, 1)), rng.integers(0, max(W - w, 1))
        out[r:r + h, c:c + w] = False
    # spurious blobs (false positives)
    n_blob = max(rng.poisson(blob_p * 12 * area), 1 if blob_p > 0 else 0)
    for _ in range(n_blob):
        h, w = rng.integers(3, 10), rng.integers(3, 10)
        r, c = rng.integers(0, max(H - h, 1)), rng.integers(0, max(W - w, 1))
        out[r:r + h, c:c + w] = True
    u = rng.random()
    if u < dilate_p:
        out = dilation(out, disk(1))
    elif u > 1.0 - dilate_p:
        out = erosion(out, disk(1))
    return np.asarray(out, bool)


def build_plot(stems, grid, sigma=1.0):
    """Rasterize a whole plot -> (mask, heat, orient, diam) in pixel space."""
    from geo import stems_to_pixel
    polylines, diams = stems_to_pixel(stems, grid)
    shape = (grid.H, grid.W)
    mask = render_mask(polylines, diams, shape)
    heat, orient, diam = render_fields(polylines, diams, shape, sigma=sigma)
    return mask, heat, orient, diam


class PlotTiles:
    """torch Dataset over tiles of one plot. Returns (x, target-dict) float32 tensors."""

    def __init__(self, mask, heat, orient, diam, origins, tile, corrupt=False, seed=0):
        self.mask, self.heat, self.orient, self.diam = mask, heat, orient, diam
        self.origins, self.tile, self.corrupt = origins, tile, corrupt
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        import torch
        r, c = self.origins[i]
        t = self.tile
        m = self.mask[r:r + t, c:c + t]
        if self.corrupt:
            m = corrupt_mask(m, self.rng)
        x = torch.from_numpy(np.asarray(m, np.float32))[None]
        tgt = {
            "heat": torch.from_numpy(self.heat[r:r + t, c:c + t].astype(np.float32))[None],
            "orient": torch.from_numpy(self.orient[:, r:r + t, c:c + t].astype(np.float32)),
            "diam": torch.from_numpy(self.diam[r:r + t, c:c + t].astype(np.float32))[None],
        }
        return x, tgt
