"""Multi-scale native-fidelity dataloader: native load mode, multi-scale rotate->crop
augmentation, and deterministic grid tiling for eval."""
import glob
import os

import numpy as np
import torch
from PIL import Image

from training.augment import build_augmentation
from training.config import TrainConfig
from training.dataset import StemDataset, TilingStemDataset, _tile_starts, split_ids
from training.run_train import run_training


def _native_ds(d, n=6, size=64):
    """Write n native-resolution RGB jpeg / binary gif pairs of `size`x`size`."""
    (d / "train").mkdir(parents=True)
    (d / "mask").mkdir(parents=True)
    for k in range(1, n + 1):
        rgb = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        Image.fromarray(rgb, "RGB").save(d / "train" / f"train{k}.jpeg", quality=95)
        m = np.zeros((size, size), np.uint8)
        m[: size // 2, : size // 2] = 255                 # a quadrant of foreground
        Image.fromarray(m, "L").save(d / "mask" / f"mask{k}.gif")


class _Cfg:
    """Minimal config stand-in for build_augmentation (only the aug knobs it reads)."""
    def __init__(self, **kw):
        d = dict(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_rotate_limit=15.0,
                 aug_bc_p=0.0, aug_brightness_limit=0.2, aug_contrast_limit=0.2, aug_hsv_p=0.0,
                 aug_hue_shift=20, aug_sat_shift=30, aug_val_shift=20, seed=1, img_size=16,
                 multiscale=False, crop_min_px=8, crop_max_px=40)
        d.update(kw)
        self.__dict__.update(d)


def test_native_mode_skips_resize(tmp_path):
    _native_ds(tmp_path, n=2, size=40)
    native = StemDataset(str(tmp_path / "train"), str(tmp_path / "mask"),
                         img_size=16, resize=False, cache=False)
    img, mask = native[0]
    assert img.shape == (3, 40, 40) and mask.shape == (1, 40, 40)      # untouched native size
    resized = StemDataset(str(tmp_path / "train"), str(tmp_path / "mask"),
                          img_size=16, resize=True, cache=False)
    img2, _ = resized[0]
    assert img2.shape == (3, 16, 16)                                   # forced to img_size


def test_multiscale_transform_outputs_img_size_and_binary_mask(tmp_path):
    _native_ds(tmp_path, n=2, size=40)
    tf = build_augmentation(_Cfg(multiscale=True, img_size=16, crop_min_px=8, crop_max_px=40))
    ds = StemDataset(str(tmp_path / "train"), str(tmp_path / "mask"),
                     img_size=16, transform=tf, resize=False, cache=False)
    img, mask = ds[0]
    assert img.shape == (3, 16, 16) and mask.shape == (1, 16, 16)      # crop resized to img_size
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}             # mask stays binary


def test_multiscale_rotates_full_tile_before_crop():
    # order matters: Rotate (full native tile) must precede RandomSizedCrop so the crop
    # lands on interior pixels; flips/photometric come after.
    import albumentations as A
    tf = build_augmentation(_Cfg(multiscale=True, aug_rotate_p=1.0, aug_rotate_limit=180.0,
                                 aug_hflip_p=0.5))
    names = [type(t).__name__ for t in tf.transforms]
    assert names[0] == "Rotate" and names[1] == "RandomSizedCrop"
    assert names.index("Rotate") < names.index("RandomSizedCrop") < names.index("HorizontalFlip")


def test_non_multiscale_keeps_prior_order():
    tf = build_augmentation(_Cfg(multiscale=False, aug_hflip_p=0.5, aug_rotate_p=1.0))
    names = [type(t).__name__ for t in tf.transforms]
    assert "RandomSizedCrop" not in names                # no crop without multiscale
    assert names.index("HorizontalFlip") < names.index("Rotate")   # flips before rotate


def test_tile_starts():
    assert _tile_starts(1024, 512) == [0, 512]           # clean 2x2
    assert _tile_starts(512, 512) == [0]
    assert _tile_starts(400, 512) == [0]                 # smaller than tile -> single
    assert _tile_starts(1000, 512) == [0, 488]           # last clamped to edge (dim - tile)


def test_tiling_dataset_covers_and_is_deterministic(tmp_path):
    _native_ds(tmp_path, n=3, size=32)
    ds = TilingStemDataset(str(tmp_path / "train"), str(tmp_path / "mask"), tile=16, cache=True)
    assert len(ds) == 3 * 4                               # 2x2 tiles per 32px image
    img, mask = ds[0]
    assert img.shape == (3, 16, 16) and mask.shape == (1, 16, 16)

    # the 4 tiles of image 0 reconstruct the full native image (top-left, top-right, ...)
    native = np.asarray(Image.open(tmp_path / "train" / "train1.jpeg").convert("RGB")) / 255.0
    tiles = [ds[j][0].numpy().transpose(1, 2, 0) for j in range(4)]
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    recon = np.concatenate([top, bot], axis=0)
    assert np.allclose(recon, native, atol=1e-4)

    ds2 = TilingStemDataset(str(tmp_path / "train"), str(tmp_path / "mask"), tile=16, cache=False)
    assert ds._index == ds2._index                       # deterministic index (cache-independent)
    assert torch.equal(ds2[5][0], ds[5][0])              # same tile regardless of cache mode


def test_split_ids_partitions_disjointly(tmp_path):
    _native_ds(tmp_path, n=10, size=16)
    train_ids, val_ids = split_ids(str(tmp_path / "train"), str(tmp_path / "mask"), 0.2, 1)
    assert set(train_ids).isdisjoint(val_ids)
    # ids are the raw filename stems (strings) so published sets like `train_100_1.jpeg`
    # load unrenamed; the partition property is what matters, not the type.
    assert sorted(train_ids + val_ids, key=int) == [str(i) for i in range(1, 11)]
    assert len(val_ids) == 2                              # round(10 * 0.2)


def test_run_training_multiscale_and_eval_tiling(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _native_ds(data, n=6, size=64)
    cfg = TrainConfig(
        data_dir=str(data), checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "log"),
        hdf5_out="x", onnx_out=str(out / "m.onnx"), pt_out=str(out / "m.pt"),
        epochs=1, batch_size=2, patience=999, device="cpu", img_size=32,
        multiscale=True, crop_min_px=32, crop_max_px=64, eval_tiling=True,
        aug_rotate_p=1.0, aug_rotate_limit=180.0, cache_dataset=False,
    )
    metrics = run_training(cfg)
    assert set(metrics) >= {"loss", "precision", "recall", "f1"}
    assert (out / "m.onnx").exists() and (out / "m.pt").exists()
    assert not glob.glob(str(out / "*.hdf5"))            # non-keras path
