import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from winmol_unet.preprocess import normalize_depth, resize_batch, to_float01


def _index_by_n(names, prefix, ext):
    out = {}
    for name in names:
        if name.startswith(prefix) and name.endswith(ext):
            stem = name[len(prefix):-len(ext)]
            if stem.isdigit():
                out[int(stem)] = name
    return out


DEPTH_EXTS = (".png", ".tif", ".tiff")


def _depth_index(depth_dir):
    """n -> filename for depth{N}.png|.tif|.tiff (16-bit PNG or float TIFF)."""
    out = {}
    for name in os.listdir(depth_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in DEPTH_EXTS and stem.startswith("depth") and stem[len("depth"):].isdigit():
            out[int(stem[len("depth"):])] = name
    return out


def _paired_ids(image_dir, mask_dir, depth_dir=None):
    imgs = _index_by_n(os.listdir(image_dir), "train", ".jpeg")
    masks = _index_by_n(os.listdir(mask_dir), "mask", ".gif")
    ids = set(imgs) & set(masks)
    if depth_dir is not None:
        ids &= set(_depth_index(depth_dir))
    return sorted(ids)


class StemDataset(Dataset):
    """Paired jpeg-image / gif-mask dataset. Resizes to img_size (nearest image +
    mask, mirroring the R input_pipeline.R) via winmol_unet.preprocess; the optional albumentations `transform`
    runs per __getitem__. When cache=True (default) the resized numpy arrays are kept
    in memory (~4 MB/pair) so the skimage resize runs once — this requires
    DataLoader(num_workers=0) to persist across epochs. For large datasets use
    cache=False (loads per __getitem__, bounded memory) with num_workers>0.

    Optional `depth_dir` loads a matching `depth{N}.png` (8/16-bit) or
    `depth{N}.tif`/`.tiff` (float) depth map per id, normalizes it per-image to
    [0, 1] (winmol_unet.preprocess.normalize_depth), nearest-resizes it to
    img_size, and appends it as a 4th channel so `__getitem__` yields
    `[4, S, S]` image tensors instead of `[3, S, S]`.
    """

    def __init__(self, image_dir, mask_dir, img_size=512, transform=None, ids=None, cache=True,
                 depth_dir=None, depth_vmin=None, depth_vmax=None, depth_nodata=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform   # albumentations Compose (seeded via cfg.seed) or None
        self.depth_dir = depth_dir
        # a fixed physical range keeps tiles comparable; per-image min-max (the
        # default) rescales each tile alone and discards absolute height
        self.depth_vmin, self.depth_vmax = depth_vmin, depth_vmax
        self.depth_nodata = depth_nodata
        self._depth_names = _depth_index(depth_dir) if depth_dir is not None else None
        self.ids = ids if ids is not None else _paired_ids(image_dir, mask_dir, depth_dir)
        # In-memory resize cache (~4 MB/pair) — fast for small sets but unbounded, so
        # disable it for large datasets (e.g. cache=False, num_workers>0). When off,
        # __getitem__ loads+resizes per call (bounded memory).
        self.cache = cache
        self._cache = {}   # n -> (image HWC, mask HW[, depth HW1]) float32

    def __len__(self):
        return len(self.ids)

    def _load_image(self, n):
        im = Image.open(os.path.join(self.image_dir, f"train{n}.jpeg")).convert("RGB")
        arr = to_float01(np.asarray(im))                       # HWC [0,1]
        # nearest to mirror the R input_pipeline.R (tf.image.resize method="nearest"
        # for both image and mask) so the PyTorch benchmark matches the R U-Net's
        # exact preprocessing.
        arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]
        return np.ascontiguousarray(arr, dtype=np.float32)     # HWC

    def _load_mask(self, n):
        mk = Image.open(os.path.join(self.mask_dir, f"mask{n}.gif"))
        mk.seek(0)
        arr = to_float01(np.asarray(mk.convert("L")))[..., None]   # HW1 [0,1]
        arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]  # HW1
        return (arr[..., 0] >= 0.5).astype(np.float32)             # HW binary

    def _load_depth(self, n):
        with Image.open(os.path.join(self.depth_dir, self._depth_names[n])) as im:
            raw = np.asarray(im)
        # per-image min-max to [0,1]; nearest resize to match image/mask handling
        arr = normalize_depth(raw, vmin=self.depth_vmin, vmax=self.depth_vmax,
                              nodata=self.depth_nodata)[..., None]                               # HW1
        arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]
        return np.ascontiguousarray(arr, dtype=np.float32)                  # HW1

    def __getitem__(self, i):
        n = self.ids[i]
        if self.cache:
            if n not in self._cache:
                item = (self._load_image(n), self._load_mask(n))
                if self.depth_dir is not None:
                    item = item + (self._load_depth(n),)
                self._cache[n] = item
            item = self._cache[n]
        else:
            item = (self._load_image(n), self._load_mask(n))
            if self.depth_dir is not None:
                item = item + (self._load_depth(n),)
        img, mask = item[0], item[1]
        depth = item[2] if self.depth_dir is not None else None
        if self.transform is not None:
            if depth is not None:
                out = self.transform(image=img, mask=mask, depth=depth)
                img, mask, depth = out["image"], out["mask"], out["depth"]
            else:
                out = self.transform(image=img, mask=mask)   # albumentations returns new arrays
                img, mask = out["image"], out["mask"]
        if depth is not None:
            img = np.concatenate([img, depth], axis=-1)          # HWC -> HW4
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return img_t.float(), (mask_t >= 0.5).float()


def train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512, transform=None,
                    cache=True, depth_dir=None):
    ids = _paired_ids(image_dir, mask_dir, depth_dir)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    train_ds = StemDataset(image_dir, mask_dir, img_size, transform=transform, ids=train_ids,
                           cache=cache, depth_dir=depth_dir)
    val_ds = StemDataset(image_dir, mask_dir, img_size, transform=None, ids=val_ids, cache=cache,
                         depth_dir=depth_dir)
    return train_ds, val_ds
