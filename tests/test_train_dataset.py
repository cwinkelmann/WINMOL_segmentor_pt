import numpy as np
import torch
from PIL import Image
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
    ds = StemDataset(str(img_dir), str(mask_dir))
    assert len(ds) == 3
    img, mask = ds[0]
    assert img.shape == (3, 512, 512) and img.dtype == torch.float32
    assert mask.shape == (1, 512, 512)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_paired_flip_keeps_alignment(tmp_path):
    # A mask that is all-ones on the left half; after any flip, image and mask
    # transform together, so correlation of a constant-structured pair is preserved.
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    rgb = np.zeros((40, 40, 3), np.uint8); rgb[:, :20, :] = 255
    Image.fromarray(rgb, "RGB").save(img_dir / "train1.jpeg")
    m = np.zeros((40, 40), np.uint8); m[:, :20] = 255
    Image.fromarray(m, "L").save(mask_dir / "mask1.gif")
    ds = StemDataset(str(img_dir), str(mask_dir), augment=True, seed=7)
    for _ in range(5):
        img, mask = ds[0]
        # where mask==1, the image (bright side) should be brighter than where mask==0
        bright = img[:, mask[0] == 1].mean()
        dark = img[:, mask[0] == 0].mean()
        assert bright > dark


def test_split_is_deterministic_and_disjoint(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    for n in range(1, 11):
        _make_pair(img_dir, mask_dir, n)
    tr, va = train_val_split(str(img_dir), str(mask_dir), val_fraction=0.2, seed=1)
    tr2, va2 = train_val_split(str(img_dir), str(mask_dir), val_fraction=0.2, seed=1)
    assert len(tr) == 8 and len(va) == 2
    assert tr.ids == tr2.ids and va.ids == va2.ids       # deterministic
    assert set(tr.ids).isdisjoint(set(va.ids))           # disjoint
