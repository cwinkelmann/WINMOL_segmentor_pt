"""Sample-mixing augmentations for stem segmentation: mosaic, copy-paste, cutout.

These need more than one sample, so they wrap the Dataset rather than living in the
albumentations pipeline (which sees one image at a time). Applied AFTER the base dataset's
own per-sample transform, so photometric/flip augmentation still varies each donor tile.

**Scale is never changed.** The whole pipeline is tied to a fixed ground resolution
(15/512 m/px -- the analyzer's stem-length and diameter thresholds are in metres), so mosaic
composes tiles by *cropping* quadrants rather than downscaling four tiles into one. Resizing
would teach the model stems of the wrong apparent thickness.

  mosaic     4 tiles -> one, split at a jittered centre. More stems and more tile-edge
             context per step; stems get cut at interior seams, which is what the real
             tiled inference has to cope with.
  copy-paste stems from a donor tile are pasted onto an acceptor (mask OR'd). Raises stem
             density and manufactures crossings/overlaps -- the heuristic vectorizer's
             worst case -- without inventing new backgrounds.
  cutout     rectangles erased from image AND mask together, simulating occlusion. Erasing
             the image alone would leave positive labels over blank pixels and teach the
             model to hallucinate stems.
"""
import numpy as np
import torch


def _rand_crop(img, mask, h, w, rng):
    """A random (h, w) window from one (C,H,W)/(1,H,W) sample. No rescaling."""
    H, W = img.shape[-2:]
    h, w = min(h, H), min(w, W)
    r = int(rng.integers(0, H - h + 1))
    c = int(rng.integers(0, W - w + 1))
    return img[:, r:r + h, c:c + w], mask[:, r:r + h, c:c + w]


def mosaic(samples, rng, jitter=0.25):
    """Compose 4 (img, mask) samples into one tile of the same size, by cropping.

    The split point is jittered around the centre so the model does not learn a fixed seam.
    """
    assert len(samples) == 4, "mosaic needs exactly 4 samples"
    img0 = samples[0][0]
    C, H, W = img0.shape
    lo, hi = jitter, 1.0 - jitter
    cy = int(H * float(rng.uniform(lo, hi)))
    cx = int(W * float(rng.uniform(lo, hi)))
    cy = max(1, min(H - 1, cy))
    cx = max(1, min(W - 1, cx))

    out_img = torch.zeros((C, H, W), dtype=img0.dtype)
    out_mask = torch.zeros((1, H, W), dtype=samples[0][1].dtype)
    quads = [(0, 0, cy, cx), (0, cx, cy, W - cx), (cy, 0, H - cy, cx), (cy, cx, H - cy, W - cx)]
    for (r0, c0, qh, qw), (img, msk) in zip(quads, samples):
        pi, pm = _rand_crop(img, msk, qh, qw, rng)
        out_img[:, r0:r0 + pi.shape[-2], c0:c0 + pi.shape[-1]] = pi
        out_mask[:, r0:r0 + pm.shape[-2], c0:c0 + pm.shape[-1]] = pm
    return out_img, out_mask


def copy_paste(dst, src, rng, flip=True):
    """Paste the donor's stem pixels onto the acceptor; masks are OR'd."""
    dst_img, dst_mask = dst
    src_img, src_mask = src
    if flip:
        if rng.random() < 0.5:
            src_img, src_mask = torch.flip(src_img, [-1]), torch.flip(src_mask, [-1])
        if rng.random() < 0.5:
            src_img, src_mask = torch.flip(src_img, [-2]), torch.flip(src_mask, [-2])
        k = int(rng.integers(0, 4))
        if k:
            src_img, src_mask = torch.rot90(src_img, k, [-2, -1]), torch.rot90(src_mask, k, [-2, -1])

    sel = (src_mask[0] >= 0.5)
    if not bool(sel.any()):
        return dst_img, dst_mask
    out_img = dst_img.clone()
    out_mask = dst_mask.clone()
    out_img[:, sel] = src_img[:, sel]
    out_mask[0][sel] = 1.0
    return out_img, out_mask


def cutout(img, mask, rng, n_holes=(1, 4), size_frac=(0.05, 0.20)):
    """Erase rectangles from image AND mask together (occlusion, not label noise)."""
    C, H, W = img.shape
    out_img, out_mask = img.clone(), mask.clone()
    for _ in range(int(rng.integers(n_holes[0], n_holes[1] + 1))):
        fh = float(rng.uniform(*size_frac))
        fw = float(rng.uniform(*size_frac))
        h, w = max(1, int(H * fh)), max(1, int(W * fw))
        r = int(rng.integers(0, H - h + 1))
        c = int(rng.integers(0, W - w + 1))
        out_img[:, r:r + h, c:c + w] = 0.0
        out_mask[:, r:r + h, c:c + w] = 0.0
    return out_img, out_mask


class MixAugmentDataset(torch.utils.data.Dataset):
    """Wraps a sample-level Dataset and applies mosaic / copy-paste / cutout stochastically.

    Order matters: mosaic first (it composes whole tiles), then copy-paste (adds stems to
    whatever composition resulted), then cutout (removes evidence last, so it can also erase
    pasted stems -- otherwise cutout would never touch them).
    """

    def __init__(self, base, mosaic_p=0.0, copypaste_p=0.0, cutout_p=0.0, seed=1):
        self.base = base
        self.mosaic_p, self.copypaste_p, self.cutout_p = mosaic_p, copypaste_p, cutout_p
        self.seed = seed

    def __len__(self):
        return len(self.base)

    @property
    def transform(self):
        """Expose the base dataset's albumentations Compose.

        run_train._worker_init reseeds `dataset.transform` in each DataLoader worker so the
        augmentation streams differ; without this delegation the wrapper would hide it and
        every worker would silently replay the same photometric augmentation.
        """
        return getattr(self.base, "transform", None)

    def _rng(self, i):
        # per-index RNG: deterministic given (seed, index, epoch-free) and safe across
        # DataLoader worker processes, which would otherwise share one RNG state
        return np.random.default_rng((self.seed * 1_000_003 + i) % (2 ** 63))

    def __getitem__(self, i):
        rng = self._rng(i)
        img, mask = self.base[i]

        if self.mosaic_p > 0 and rng.random() < self.mosaic_p and len(self.base) >= 4:
            idx = [i] + [int(rng.integers(0, len(self.base))) for _ in range(3)]
            img, mask = mosaic([self.base[j] for j in idx], rng)

        if self.copypaste_p > 0 and rng.random() < self.copypaste_p and len(self.base) >= 2:
            j = int(rng.integers(0, len(self.base)))
            img, mask = copy_paste((img, mask), self.base[j], rng)

        if self.cutout_p > 0 and rng.random() < self.cutout_p:
            img, mask = cutout(img, mask, rng)

        return img, (mask >= 0.5).float()
