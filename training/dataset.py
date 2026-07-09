"""Paired jpeg-image / gif-mask dataset with shared-seed geometric augmentation
(flips) on both and photometric augmentation on the image only. Resize to 512
via winmol_unet.preprocess to eliminate train/inference skew."""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as TF

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
    def __init__(self, image_dir, mask_dir, img_size=512, augment=False, seed=1, ids=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.ids = ids if ids is not None else _paired_ids(image_dir, mask_dir)
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.ids)

    def _load_image(self, n):
        im = Image.open(os.path.join(self.image_dir, f"train{n}.jpeg")).convert("RGB")
        arr = to_float01(np.asarray(im))                       # HWC [0,1]
        arr = resize_batch(arr[None], size=self.img_size, mode="bicubic")[0]
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))  # CHW

    def _load_mask(self, n):
        mk = Image.open(os.path.join(self.mask_dir, f"mask{n}.gif"))
        mk.seek(0)
        arr = to_float01(np.asarray(mk.convert("L")))[..., None]   # HW1 [0,1]
        arr = resize_batch(arr[None], size=self.img_size, mode="nearest")[0]  # 512x512x1
        m = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return (m >= 0.5).float()

    def __getitem__(self, i):
        n = self.ids[i]
        img, mask = self._load_image(n), self._load_mask(n)
        if self.augment:
            if self._rng.random() < 0.5:                       # horizontal flip (paired)
                img, mask = TF.hflip(img), TF.hflip(mask)
            if self._rng.random() < 0.5:                       # vertical flip (paired)
                img, mask = TF.vflip(img), TF.vflip(mask)
            # photometric on image only
            img = TF.adjust_brightness(img, 1.0 + self._rng.uniform(-0.2, 0.2))
            img = TF.adjust_contrast(img, 1.0 + self._rng.uniform(-0.2, 0.2))
            img = TF.adjust_saturation(img, 1.0 + self._rng.uniform(-0.2, 0.2))
            img = TF.adjust_hue(img, self._rng.uniform(-0.05, 0.05))
            img = img.clamp(0.0, 1.0)
        return img, mask


def train_val_split(image_dir, mask_dir, val_fraction, seed, img_size=512):
    ids = _paired_ids(image_dir, mask_dir)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    train_ds = StemDataset(image_dir, mask_dir, img_size, augment=True, seed=seed, ids=train_ids)
    val_ds = StemDataset(image_dir, mask_dir, img_size, augment=False, seed=seed, ids=val_ids)
    return train_ds, val_ds
