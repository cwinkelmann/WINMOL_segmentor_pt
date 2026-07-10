import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from winmol_unet.preprocess import resize_batch, to_float01


def _index_by_n(names, prefix, ext):
    out = {}
    for name in names:
        if name.startswith(prefix) and name.endswith(ext):
            stem = name[len(prefix):-len(ext)]
            if stem.isdigit():
                out[int(stem)] = name
    return out


def _paired_ids(image_dir, mask_dir):
    imgs = _index_by_n(os.listdir(image_dir), "train", ".jpeg")
    masks = _index_by_n(os.listdir(mask_dir), "mask", ".gif")
    return sorted(set(imgs) & set(masks))


class StemDataset(Dataset):
    """Paired jpeg-image / gif-mask dataset. Resizes to img_size (bicubic image,
    nearest mask) via winmol_unet.preprocess and caches the resized numpy arrays
    (~4 MB/pair) so the skimage resize runs once; the optional albumentations
    `transform` runs per __getitem__ on the cached arrays. Requires
    DataLoader(num_workers=0) for the cache to persist across epochs.
    """

    def __init__(self, image_dir, mask_dir, img_size=512, transform=None, ids=None, cache=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform   # albumentations Compose (seeded via cfg.seed) or None
        self.ids = ids if ids is not None else _paired_ids(image_dir, mask_dir)
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
        arr = resize_batch(arr[None], size=self.img_size, mode="bicubic")[0]
        return np.ascontiguousarray(arr, dtype=np.float32)     # HWC

    def _load_mask(self, n):
        mk = Image.open(os.path.join(self.mask_dir, f"mask{n}.gif"))
        mk.seek(0)
        arr = to_float01(np.asarray(mk.convert("L")))[..., None]   # HW1 [0,1]
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


def train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512, transform=None,
                    cache=True):
    ids = _paired_ids(image_dir, mask_dir)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    train_ds = StemDataset(image_dir, mask_dir, img_size, transform=transform, ids=train_ids,
                           cache=cache)
    val_ds = StemDataset(image_dir, mask_dir, img_size, transform=None, ids=val_ids, cache=cache)
    return train_ds, val_ds
