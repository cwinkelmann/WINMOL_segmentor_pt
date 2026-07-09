import numpy as np
import torch
from PIL import Image
import albumentations as A
from training.dataset import StemDataset, train_val_split


def _make_pair(img_dir, mask_dir, n, size=40):
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgb = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(img_dir / f"train{n}.jpeg")
    m = (np.random.rand(size, size) > 0.5).astype(np.uint8) * 255
    Image.fromarray(m, mode="L").save(mask_dir / f"mask{n}.gif")


def test_getitem_shapes_ranges_and_pairing(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    for n in (1, 2, 10):
        _make_pair(img_dir, mask_dir, n)
    ds = StemDataset(str(img_dir), str(mask_dir))       # transform=None
    assert len(ds) == 3
    img, mask = ds[0]
    assert img.shape == (3, 512, 512) and img.dtype == torch.float32
    assert mask.shape == (1, 512, 512)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_transform_paired_flip_keeps_alignment(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    rgb = np.zeros((40, 40, 3), np.uint8); rgb[:, :20, :] = 255
    Image.fromarray(rgb, "RGB").save(img_dir / "train1.jpeg")
    m = np.zeros((40, 40), np.uint8); m[:, :20] = 255
    Image.fromarray(m, "L").save(mask_dir / "mask1.gif")
    transform = A.Compose([A.HorizontalFlip(p=1.0)])    # always flip
    ds = StemDataset(str(img_dir), str(mask_dir), transform=transform)
    img, mask = ds[0]
    bright = img[:, mask[0] == 1].mean()
    dark = img[:, mask[0] == 0].mean()
    assert bright > dark        # image and mask flipped together

def test_transform_photometric_leaves_mask(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    _make_pair(img_dir, mask_dir, 1)
    base = StemDataset(str(img_dir), str(mask_dir))[0][1]        # mask, no transform
    tf = A.Compose([A.RandomBrightnessContrast(p=1.0)])
    _, mask = StemDataset(str(img_dir), str(mask_dir), transform=tf)[0]
    assert torch.equal(base, mask)                              # mask unchanged by photometric


def test_cache_not_mutated_by_transform(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    _make_pair(img_dir, mask_dir, 1)
    tf = A.Compose([A.HorizontalFlip(p=1.0)])
    ds = StemDataset(str(img_dir), str(mask_dir), transform=tf)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)      # deterministic (p=1 flip) -> cache base intact each call


def test_split_disjoint_and_val_has_no_transform(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    for n in range(1, 11):
        _make_pair(img_dir, mask_dir, n)
    tf = A.Compose([A.HorizontalFlip(p=1.0)])
    tr, va = train_val_split(str(img_dir), str(mask_dir), 0.2, 1, transform=tf)
    assert len(tr) == 8 and len(va) == 2
    assert set(tr.ids).isdisjoint(set(va.ids))
    assert tr.transform is tf and va.transform is None
