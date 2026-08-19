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
        # A DISTINCT mask per tile: the stripe moves with i. Tiles that are byte-identical
        # make every provenance assertion vacuous -- a leak, a duplicated primary and a
        # correct draw all look the same. This is what lets the tests below check identity
        # rather than just count.
        m = np.zeros((64, 64), dtype="uint8")
        m[10:40, (2 * i) % 50:(2 * i) % 50 + 6] = 255
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
    held_out = set(ds.ids[4:])             # pretend these are the train half
    ds.ids = ds.ids[:4]                    # ...and this dataset is the val half

    # Compare the ACTUAL returned partners against the tiles of the held-out half. Each
    # fixture tile has a distinct mask, so identity is checkable from the pixels.
    import numpy as np
    forbidden = {ds._load_mask(n).tobytes() for n in held_out}
    seen = 0
    for _ in range(30):
        for part in ds._mosaic_partners(0):
            assert part["mask"].tobytes() not in forbidden, \
                "a mosaic partner came from outside this dataset's split"
            seen += 1
    assert seen > 0, "no partners were drawn, so nothing was actually checked"


def test_partners_exclude_the_primary_tile(tmp_path):
    """The primary tile must not also appear as one of its own partners.

    Checked on the returned arrays, not on the count: a 2x2 mosaic of four copies of one
    tile is a zoom-out, not an augmentation, and would be invisible in a length assertion.
    """
    ds = _dataset(tmp_path, n=8, mosaic_p=1.0, seed=1)
    for i in range(len(ds.ids)):
        primary = ds._load_mask(ds.ids[i]).tobytes()
        for _ in range(10):
            picks = ds._mosaic_partners(i)
            assert len(picks) == 3
            assert all(p["mask"].tobytes() != primary for p in picks), \
                f"tile {i} was returned as its own mosaic partner"


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
