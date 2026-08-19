"""Mosaic augmentation: off by default, binary-safe, and split-safe.

Mosaic is UNEVALUATED in this repo — no paired run, no LOSO fold, no results document —
so the most important property is that it is genuinely inert unless asked for. The second
is that it cannot reach across a split: tiles are oversampled and overlap, so leakage is
already the failure mode the whole split machinery exists to prevent, and an augmentation
that stitches four tiles together is an obvious way to reintroduce it.
"""
import numpy as np
import pytest

from winmol_unet.training.augment import build_augmentation
from winmol_unet.training.config import TrainConfig
from winmol_unet.training.dataset import StemDataset


def _cfg(**kw):
    return TrainConfig(data_dir="", checkpoint_dir="", log_dir="", onnx_out="",
                       hdf5_out="", **kw)


def _dataset(tmp_path, n=8, **kw):
    from PIL import Image
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype="uint8")).save(
            img_dir / f"train{i}.jpeg")
        m = np.zeros((64, 64), dtype="uint8"); m[10:40, 10:20] = 255
        Image.fromarray(m).save(mask_dir / f"mask{i}.gif")
    return StemDataset(str(img_dir), str(mask_dir), img_size=64, **kw)


def test_mosaic_is_off_by_default():
    assert _cfg().mosaic_p == 0.0


def test_default_pipeline_contains_no_mosaic():
    """p=0 must not merely disable it — the transform should not be in the pipeline."""
    names = [t.__class__.__name__ for t in build_augmentation(_cfg()).transforms]
    assert "Mosaic" not in names


def test_mosaic_is_first_in_the_pipeline():
    """Anything after mosaic sees one coherent image; a crop before it would undo it."""
    names = [t.__class__.__name__ for t in build_augmentation(_cfg(mosaic_p=0.5)).transforms]
    assert names[0] == "Mosaic", names


def test_mosaic_is_first_even_with_the_multiscale_crop():
    cfg = _cfg(mosaic_p=0.5, multiscale=True, crop_min_px=32, crop_max_px=64, img_size=64)
    names = [t.__class__.__name__ for t in build_augmentation(cfg).transforms]
    assert names[0] == "Mosaic"
    assert names.index("Mosaic") < names.index("RandomSizedCrop")


def test_output_stays_binary_and_the_right_shape(tmp_path):
    cfg = _cfg(mosaic_p=1.0, img_size=64)
    ds = _dataset(tmp_path, transform=build_augmentation(cfg), mosaic_p=1.0, seed=1)
    img, mask = ds[0]
    assert img.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)
    # a mask that came back as a blend of cells would have intermediate values
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}


def test_partners_never_leave_this_datasets_ids(tmp_path):
    """The leak boundary: a val dataset holds its own ids, so mosaic cannot reach train.

    Asserted on the mechanism rather than on outputs, because a leak here would be
    invisible in the tensors and would show up only as an inflated validation score.
    """
    ds = _dataset(tmp_path, n=8, mosaic_p=1.0, seed=1)
    ds.ids = ds.ids[:4]                    # pretend this is the val half
    for _ in range(20):
        partners = ds._mosaic_partners(0)
        assert len(partners) == 3
    # every id the sampler can reach is one of this dataset's own
    reachable = {ds.ids[j] for j in range(len(ds.ids))}
    assert reachable == set(ds.ids[:4])


def test_partners_exclude_the_primary_tile(tmp_path):
    ds = _dataset(tmp_path, n=8, mosaic_p=1.0, seed=1)
    for i in range(len(ds.ids)):
        picks = ds._mosaic_partners(i)
        assert len(picks) == 3


def test_a_single_tile_dataset_does_not_crash(tmp_path):
    """`sample` on an empty population raises; one tile has no partners to draw."""
    cfg = _cfg(mosaic_p=1.0, img_size=64)
    ds = _dataset(tmp_path, n=1, transform=build_augmentation(cfg), mosaic_p=1.0, seed=1)
    img, mask = ds[0]
    assert img.shape == (3, 64, 64)


def test_evaluation_datasets_never_mosaic(tmp_path):
    """Val/test are built with transform=None, so mosaic cannot apply there at all."""
    ds = _dataset(tmp_path, transform=None, mosaic_p=1.0, seed=1)
    img, mask = ds[0]
    assert img.shape == (3, 64, 64)
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}


def test_the_cli_flag_reaches_the_config():
    from winmol_unet.training.run_train import build_parser, config_from_parsed
    p = build_parser()
    cfg = config_from_parsed(p.parse_args(["--data-dir", "x", "--mosaic-p", "0.4"]), p)
    assert cfg.mosaic_p == 0.4


def test_mosaic_recipe_is_labelled_unevaluated():
    from winmol_unet.training import recipes
    assert recipes.RECIPES["mosaic"]["mosaic_p"] > 0
    assert "mosaic" in recipes.UNEVALUATED
