import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from winmol_unet.preprocess import resize_batch, to_float01
# The pairing helpers live in winmol_unet.data, which is torch-free: data.split
# needs them and should not pull torch in for filename arithmetic. Re-exported
# here so existing `from ...training.dataset import _paired_ids` still resolves.
from winmol_unet.data.pairing import _index_by_n, _paired_ids, _sort_key  # noqa: F401


class StemDataset(Dataset):
    """Paired jpeg-image / gif-mask dataset. Resizes to img_size (nearest image +
    mask, mirroring the R input_pipeline.R) via winmol_unet.preprocess; the optional albumentations `transform`
    runs per __getitem__. When cache=True (default) the resized numpy arrays are kept
    in memory (~4 MB/pair) so the skimage resize runs once — this requires
    DataLoader(num_workers=0) to persist across epochs. For large datasets use
    cache=False (loads per __getitem__, bounded memory) with num_workers>0.
    """

    def __init__(self, image_dir, mask_dir, img_size=512, transform=None, ids=None, cache=True,
                 resize=True, mosaic_p=0.0, seed=1, num_classes=1):
        self.image_dir = image_dir
        # num_classes > 1 switches masks from binarized float to palette-index int64
        # (0=background, 1..C-1=species, 255=ignore) for the multiclass losses.
        self.num_classes = num_classes
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform   # albumentations Compose (seeded via cfg.seed) or None
        self.ids = ids if ids is not None else _paired_ids(image_dir, mask_dir)
        # resize=False keeps images/masks at native resolution — the transform is then
        # responsible for producing img_size (e.g. multi-scale RandomSizedCrop crops a
        # native-res window and resizes it), so no fidelity is lost to a pre-downscale.
        self.resize = resize
        # In-memory resize cache (~4 MB/pair) — fast for small sets but unbounded, so
        # disable it for large datasets (e.g. cache=False, num_workers>0). When off,
        # __getitem__ loads+resizes per call (bounded memory).
        self.cache = cache
        self._cache = {}   # n -> (image HWC float32 [0,1], mask HW float32 {0,1})
        # Mosaic needs grid_yx-1 extra tiles per sample. They are drawn from THIS
        # dataset's own `ids` and nowhere else: a val or test dataset is constructed with
        # its own id list, so a mosaic can never reach across a split. That matters more
        # here than in most projects — tiles are oversampled and overlap, so leakage is
        # already the failure mode the split strategy exists to prevent.
        self.mosaic_p = mosaic_p
        # Own RNG rather than the global one, so a mosaic stream is reproducible from
        # cfg.seed alone. NOTE this does not by itself de-correlate DataLoader workers:
        # they fork after __init__, so every worker inherits the SAME Random(seed) and
        # would draw the same partner sequence. run_train._worker_init reseeds it per
        # worker, the way it already does for the albumentations Compose.
        self._mosaic_rng = random.Random(seed)

    def __len__(self):
        return len(self.ids)

    def _mosaic_partners(self, i, k=3):
        """k other tiles from this split, as A.Mosaic's `mosaic_metadata` list.

        Supplied on every call when mosaic is on, because A.Mosaic decides internally
        whether to fire (its own `p`). Gating here instead would mean handing it no
        partners on the other calls, and it replicates the primary image when partners
        are missing — a 2x2 of one tile, which is a zoom-out, not a no-op.

        The cost is k extra tile loads per sample. Free with the default in-memory cache;
        with `--no-cache-dataset` it is k+1 reads per sample, so mosaic and a large
        uncached dataset together are worth benchmarking before use.
        """
        others = [j for j in range(len(self.ids)) if j != i]
        # min(): a split smaller than k+1 tiles yields fewer partners rather than raising.
        picks = self._mosaic_rng.sample(others, min(k, len(others)))
        partners = []
        for j in picks:
            n = self.ids[j]
            if self.cache:
                if n not in self._cache:
                    self._cache[n] = (self._load_image(n), self._load_mask(n))
                pi, pm = self._cache[n]
            else:
                pi, pm = self._load_image(n), self._load_mask(n)
            partners.append({"image": pi, "mask": pm})
        return partners

    def _load_image(self, n):
        im = Image.open(os.path.join(self.image_dir, f"train{n}.jpeg")).convert("RGB")
        arr = to_float01(np.asarray(im))                       # HWC [0,1]
        # nearest to mirror the R input_pipeline.R (tf.image.resize method="nearest"
        # for both image and mask) so the PyTorch benchmark matches the R U-Net's
        # exact preprocessing.
        if self.resize:
            arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]
        return np.ascontiguousarray(arr, dtype=np.float32)     # HWC

    def _load_mask(self, n):
        mk = Image.open(os.path.join(self.mask_dir, f"mask{n}.gif"))
        mk.seek(0)
        if self.num_classes > 1:
            # palette indices ARE the class labels; no [0,1] scaling, nearest only
            arr = np.asarray(mk).astype(np.float32)[..., None]     # HW1 indices
            if self.resize:
                arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]
            return arr[..., 0].astype(np.int64)                    # HW int64
        arr = to_float01(np.asarray(mk.convert("L")))[..., None]   # HW1 [0,1]
        if self.resize:
            arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]  # HW1
        return (arr[..., 0] >= 0.5).astype(np.float32)             # HW binary

    def __getitem__(self, i):
        n = self.ids[i]
        if self.cache:
            if n not in self._cache:
                self._cache[n] = (self._load_image(n), self._load_mask(n))
            img, mask = self._cache[n]
        else:
            img, mask = self._load_image(n), self._load_mask(n)
        if self.transform is not None:
            if self.mosaic_p > 0:
                # The key must be present whenever A.Mosaic is in the pipeline -- it
                # declares mosaic_metadata as a required target and raises without it.
                # An empty or short list is fine (it replicates the primary tile), so
                # this holds for a degenerate one-tile split too.
                out = self.transform(image=img, mask=mask,
                                     mosaic_metadata=self._mosaic_partners(i))
            else:
                out = self.transform(image=img, mask=mask)  # albumentations returns new arrays
            img, mask = out["image"], out["mask"]
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if self.num_classes > 1:
            # index mask, [H,W] int64 — what F.cross_entropy expects; no channel axis
            return img_t.float(), torch.from_numpy(np.ascontiguousarray(mask)).long()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return img_t.float(), (mask_t >= 0.5).float()


def split_ids(image_dir, mask_dir, val_fraction, seed):
    """Deterministic train/val id split (shuffle by seed, first val_fraction -> val)."""
    ids = _paired_ids(image_dir, mask_dir)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    # sort with _sort_key, not plain sorted(): ids are strings now, so the default
    # ordering would put '10' before '2' and disagree with _paired_ids.
    return (sorted(shuffled[n_val:], key=_sort_key),
            sorted(shuffled[:n_val], key=_sort_key))          # train_ids, val_ids


def train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512, transform=None,
                    cache=True):
    train_ids, val_ids = split_ids(image_dir, mask_dir, val_fraction, seed)
    train_ds = StemDataset(image_dir, mask_dir, img_size, transform=transform, ids=train_ids,
                           cache=cache)
    val_ds = StemDataset(image_dir, mask_dir, img_size, transform=None, ids=val_ids, cache=cache)
    return train_ds, val_ds


def _tile_starts(dim, tile):
    """Top-left offsets so `tile`-sized windows cover `dim`; last window clamped to the
    edge (so a non-multiple dim is fully covered, with a small overlap on the last tile)."""
    if dim <= tile:
        return [0]
    starts = list(range(0, dim - tile + 1, tile))
    if starts[-1] != dim - tile:
        starts.append(dim - tile)
    return starts


class TilingStemDataset(Dataset):
    """Deterministic grid tiling of native-resolution pairs into ``tile``x``tile`` windows
    (2x2 for a 1024px image at tile=512) — for reproducible, full-coverage validation/test
    at native fidelity (no downscaling). Never augments; len = total tiles across all images.
    ``cache=True`` keeps every decoded native image in memory; ``cache=False`` keeps only the
    most recent one (bounded), which is enough since the tile index is image-major.
    """

    def __init__(self, image_dir, mask_dir, tile=512, ids=None, cache=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.tile = tile
        self.ids = ids if ids is not None else _paired_ids(image_dir, mask_dir)
        self.cache = cache
        self._cache = {}                 # n -> (img HWC native, mask HW native)
        self._last = None                # (n, img, mask) single-entry cache when cache=False
        self._index = []                 # flat [(n, row_start, col_start), ...]
        for n in self.ids:
            h, w = self._native_hw(n)
            for r in _tile_starts(h, tile):
                for c in _tile_starts(w, tile):
                    self._index.append((n, r, c))

    def _native_hw(self, n):
        with Image.open(os.path.join(self.image_dir, f"train{n}.jpeg")) as im:
            w, h = im.size                # PIL .size is (w, h) and does not decode pixels
        return h, w

    def _load_native(self, n):
        im = Image.open(os.path.join(self.image_dir, f"train{n}.jpeg")).convert("RGB")
        img = np.ascontiguousarray(to_float01(np.asarray(im)), dtype=np.float32)  # HWC
        mk = Image.open(os.path.join(self.mask_dir, f"mask{n}.gif"))
        mk.seek(0)
        mask = (to_float01(np.asarray(mk.convert("L"))) >= 0.5).astype(np.float32)  # HW
        return img, mask

    def _get_native(self, n):
        if self.cache:
            if n not in self._cache:
                self._cache[n] = self._load_native(n)
            return self._cache[n]
        if self._last is None or self._last[0] != n:
            self._last = (n, *self._load_native(n))
        return self._last[1], self._last[2]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        n, r, c = self._index[i]
        img, mask = self._get_native(n)
        t = self.tile
        img_t = img[r:r + t, c:c + t, :]
        mask_t = mask[r:r + t, c:c + t]
        img_out = torch.from_numpy(np.ascontiguousarray(img_t.transpose(2, 0, 1))).float()
        mask_out = torch.from_numpy(np.ascontiguousarray(mask_t))[None]
        return img_out, (mask_out >= 0.5).float()
