import os
import re
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from winmol_unet.preprocess import resize_batch, to_float01


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


class StemDataset(Dataset):
    """Paired jpeg-image / gif-mask dataset. Resizes to img_size (nearest image +
    mask, mirroring the R input_pipeline.R) via winmol_unet.preprocess; the optional albumentations `transform`
    runs per __getitem__. When cache=True (default) the resized numpy arrays are kept
    in memory (~4 MB/pair) so the skimage resize runs once — this requires
    DataLoader(num_workers=0) to persist across epochs. For large datasets use
    cache=False (loads per __getitem__, bounded memory) with num_workers>0.
    """

    def __init__(self, image_dir, mask_dir, img_size=512, transform=None, ids=None, cache=True,
                 resize=True):
        self.image_dir = image_dir
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

    def __len__(self):
        return len(self.ids)

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
            out = self.transform(image=img, mask=mask)   # albumentations returns new arrays
            img, mask = out["image"], out["mask"]
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
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
