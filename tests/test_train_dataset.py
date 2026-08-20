import numpy as np
import torch
from PIL import Image
import albumentations as A
from winmol_unet.training.dataset import StemDataset, train_val_split


def test_getitem_shapes_ranges_and_pairing(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, ids=(1, 2, 10))
    img_dir, mask_dir = root / "train", root / "mask"
    ds = StemDataset(str(img_dir), str(mask_dir))       # transform=None
    assert len(ds) == 3
    img, mask = ds[0]
    assert img.shape == (3, 512, 512) and img.dtype == torch.float32
    assert mask.shape == (1, 512, 512)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_transform_paired_flip_keeps_alignment(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=1, size=40, stem_frac=0.5)
    img_dir, mask_dir = root / "train", root / "mask"
    transform = A.Compose([A.HorizontalFlip(p=1.0)])    # always flip
    ds = StemDataset(str(img_dir), str(mask_dir), transform=transform)
    img, mask = ds[0]
    bright = img[:, mask[0] == 1].mean()
    dark = img[:, mask[0] == 0].mean()
    assert bright > dark        # image and mask flipped together

def test_transform_photometric_leaves_mask(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=1)
    img_dir, mask_dir = root / "train", root / "mask"
    base = StemDataset(str(img_dir), str(mask_dir))[0][1]        # mask, no transform
    tf = A.Compose([A.RandomBrightnessContrast(p=1.0)])
    _, mask = StemDataset(str(img_dir), str(mask_dir), transform=tf)[0]
    assert torch.equal(base, mask)                              # mask unchanged by photometric


def test_cache_not_mutated_by_transform(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=1)
    img_dir, mask_dir = root / "train", root / "mask"
    tf = A.Compose([A.HorizontalFlip(p=1.0)])
    ds = StemDataset(str(img_dir), str(mask_dir), transform=tf)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)      # deterministic (p=1 flip) -> cache base intact each call


def test_split_disjoint_and_val_has_no_transform(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=10)
    img_dir, mask_dir = root / "train", root / "mask"
    tf = A.Compose([A.HorizontalFlip(p=1.0)])
    tr, va = train_val_split(str(img_dir), str(mask_dir), 0.2, 1, transform=tf)
    assert len(tr) == 8 and len(va) == 2
    assert set(tr.ids).isdisjoint(set(va.ids))
    assert tr.transform is tf and va.transform is None
