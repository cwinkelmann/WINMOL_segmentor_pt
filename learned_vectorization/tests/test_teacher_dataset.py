import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_teacher_dataset import ANALYZER_GSD, load_binary_mask, pair_tiles, tile_key


def test_tile_key_handles_both_dataset_naming_schemes():
    # GenDS10 style (StemDataset's integer pairing cannot do this one)
    assert tile_key("train_100_10.jpeg") == tile_key("mask_100_10.gif") == "100_10"
    # SpecDS_ready style
    assert tile_key("train1.jpeg") == tile_key("mask1.gif") == "1"
    # distinct tiles must not collide
    assert tile_key("train_100_1.jpeg") != tile_key("train_100_10.jpeg")


def test_pair_tiles_matches_images_to_masks_and_drops_unpaired(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "mask").mkdir()
    for n in (1, 2, 3):
        Image.new("RGB", (4, 4)).save(tmp_path / "train" / f"train{n}.jpeg")
    for n in (1, 3):                       # tile 2 has no mask
        Image.new("L", (4, 4)).save(tmp_path / "mask" / f"mask{n}.gif")
    pairs = pair_tiles(str(tmp_path))
    assert [k for k, _, _ in pairs] == ["1", "3"]
    for _, img, msk in pairs:
        assert os.path.exists(img) and os.path.exists(msk)


def test_load_binary_mask_binarizes(tmp_path):
    a = np.zeros((8, 8), np.uint8)
    a[2:5, 2:5] = 255
    p = tmp_path / "m.png"
    Image.fromarray(a).save(p)
    m = load_binary_mask(str(p))
    assert m.dtype == np.uint8 and set(np.unique(m)) <= {0, 1}
    assert m.sum() == 9


def test_profile_uses_the_analyzers_own_resolution(tmp_path):
    # the vectorizer's thresholds are in metres, so a wrong GSD silently ruins every output
    assert abs(ANALYZER_GSD - 15.0 / 512.0) < 1e-12
    from build_teacher_dataset import tile_profile
    prof = tile_profile(512, 512)
    assert prof["height"] == prof["width"] == 512
    assert abs(prof["transform"].a - ANALYZER_GSD) < 1e-12
    assert abs(prof["transform"].e + ANALYZER_GSD) < 1e-12      # north-up: negative y step
