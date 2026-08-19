import numpy as np
import torch
from PIL import Image

from winmol_unet.training.dataset import StemDataset


def _pair(img_dir, mask_dir, n):
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgb = (np.random.rand(40, 40, 3) * 255).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(img_dir / f"train{n}.jpeg")
    m = (np.random.rand(40, 40) > 0.5).astype(np.uint8) * 255
    Image.fromarray(m, "L").save(mask_dir / f"mask{n}.gif")


def test_cache_off_matches_cache_on_and_stores_nothing(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    for n in (1, 2):
        _pair(img_dir, mask_dir, n)

    cached = StemDataset(str(img_dir), str(mask_dir), cache=True)
    uncached = StemDataset(str(img_dir), str(mask_dir), cache=False)

    ci, cm = cached[0]
    ui, um = uncached[0]
    assert torch.equal(ci, ui) and torch.equal(cm, um)   # identical output

    _ = uncached[0]; _ = uncached[1]
    assert uncached._cache == {}                          # nothing cached when off
    assert cached._cache != {}                            # cached when on


def test_loader_with_workers_and_transform(tmp_path):
    # num_workers>0 + an albumentations transform must work (Compose picklable under
    # spawn; per-worker reseed via run_train._worker_init). Locks the large-dataset path.
    import albumentations as A
    from torch.utils.data import DataLoader
    from winmol_unet.training.run_train import _worker_init
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    for n in range(1, 5):
        _pair(img_dir, mask_dir, n)
    ds = StemDataset(str(img_dir), str(mask_dir), transform=A.Compose([A.HorizontalFlip(p=0.5)], seed=1),
                     cache=False)
    loader = DataLoader(ds, batch_size=2, num_workers=2, worker_init_fn=_worker_init)
    batches = [b for b in loader]
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 3, 512, 512)
