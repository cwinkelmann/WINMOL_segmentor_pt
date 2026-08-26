"""prepare.py --from-folder: converting arbitrary folders (non-integer names, palette and
instance masks, COCO annotations) into the train{N}.jpeg / mask{N}.gif convention
StemDataset expects.

Merges tests/test_build_dataset.py, tests/test_split_dataset.py and
tests/test_coco_to_dataset.py.
"""
import filecmp
import json
import os

import numpy as np
import pytest
from PIL import Image

from winmol_unet.data.build import build_dataset
from winmol_unet.data.coco import coco_to_dataset
from winmol_unet.data.split import split_dataset
from winmol_unet.training.dataset import StemDataset, _paired_ids, train_val_split


# --- build_dataset: arbitrary (non-conventional) folders -> loader convention ------

def _spruce_pair(train_dir, mask_dir, key, fg_index):
    train_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgb = (np.random.rand(20, 20, 3) * 255).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(train_dir / f"train_{key}.jpeg")
    arr = np.zeros((20, 20), np.uint8); arr[:, :10] = fg_index    # mode-P instance index
    Image.fromarray(arr, mode="P").save(mask_dir / f"mask_{key}.gif")


def test_build_dataset_renames_pairs_and_binarizes(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    # spruce-style non-integer names + mode-P masks with different instance indices
    _spruce_pair(src / "train", src / "mask", "1000_19644_SKY162.15", 5)
    _spruce_pair(src / "train", src / "mask", "42_7_SKY99.7", 9)

    n = build_dataset(str(src), str(dst))
    assert n == 2

    # sequential loader-conformant names
    assert set(os.listdir(dst / "train")) == {"train1.jpeg", "train2.jpeg"}
    assert set(os.listdir(dst / "mask")) == {"mask1.gif", "mask2.gif"}

    # masks binarized: the loader opens masks via .convert("L"), which must yield {0,255}
    for i in (1, 2):
        m = np.asarray(Image.open(dst / "mask" / f"mask{i}.gif").convert("L"))
        vals = set(np.unique(m).tolist())
        assert vals <= {0, 255}
        assert 255 in vals                       # nonzero instance -> foreground

    # the converted dataset loads through StemDataset with binary {0,1} masks
    ds = StemDataset(str(dst / "train"), str(dst / "mask"))
    assert len(ds) == 2
    _, mask = ds[0]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}

    # source untouched
    assert any("SKY" in n for n in os.listdir(src / "train"))
    assert Image.open(src / "mask" / "mask_42_7_SKY99.7.gif").mode == "P"


def test_build_dataset_raises_when_no_pairs(tmp_path):
    (tmp_path / "src" / "train").mkdir(parents=True)
    (tmp_path / "src" / "mask").mkdir(parents=True)
    # image with an unrecognized extension -> nothing paired
    Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(
        tmp_path / "src" / "train" / "train_1.png")
    with pytest.raises(ValueError, match="no paired"):
        build_dataset(str(tmp_path / "src"), str(tmp_path / "dst"))


# --- dataset.py's pairing/ordering logic, exercised on well-formed loader folders --

def test_loader_reads_the_published_zenodo_naming(tmp_path, stem_dataset):
    """Zenodo's GenDS uses `train_100_1.jpeg` / `mask_100_1.gif`; ours uses `train1`.

    Both must load without renaming, so a published dataset can be used exactly as
    distributed rather than through a derived copy nobody can verify.
    """
    keys = ["_100_1", "_100_2", "_100_10"]
    root = stem_dataset(tmp_path, ids=keys)
    img, msk = root / "train", root / "mask"
    # an AppleDouble sidecar must not be picked up as a tile
    (img / "._train_100_1.jpeg").write_bytes(b"\x00\x05\x16\x07")

    ids = _paired_ids(str(img), str(msk))
    assert ids == ["_100_1", "_100_2", "_100_10"], ids   # numeric-aware ordering


def test_loader_still_reads_our_own_naming(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, ids=(1, 2, 10))
    assert _paired_ids(str(root / "train"), str(root / "mask")) == ["1", "2", "10"]


# --- split_dataset -------------------------------------------------------------------

def _make_pairs(src, n):
    (src / "train").mkdir(parents=True); (src / "mask").mkdir(parents=True)
    for k in range(1, n + 1):
        rgb = np.full((24, 24, 3), k * 20, np.uint8)   # solid color encodes the id (survives jpeg)
        Image.fromarray(rgb, "RGB").save(src / "train" / f"train{k}.jpeg")
        m = np.zeros((24, 24), np.uint8); m[:, :12] = 255
        Image.fromarray(m, "L").save(src / "mask" / f"mask{k}.gif")


def _id_of(img_path):
    return int(round(float(np.asarray(Image.open(img_path)).mean()) / 20))


def test_split_materializes_and_matches_train_val_split(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_pairs(src, 10)

    n_train, n_val = split_dataset(str(src), str(dst), val_fraction=0.2, seed=1)
    assert (n_train, n_val) == (8, 2)

    # same counts as the runtime split for the same seed/fraction
    tr, va = train_val_split(str(src / "train"), str(src / "mask"), 0.2, 1)
    assert (len(tr), len(va)) == (8, 2)

    # the materialized VAL set holds exactly the same original tiles as train_val_split's val
    mat_val_ids = {_id_of(dst / "val" / "train" / f) for f in os.listdir(dst / "val" / "train")}
    assert mat_val_ids == {int(i) for i in va.ids}   # dataset ids are string stems

    # both splits are loader-ready and disjoint, covering all 10 pairs
    mtr = StemDataset(str(dst / "train" / "train"), str(dst / "train" / "mask"))
    mva = StemDataset(str(dst / "val" / "train"), str(dst / "val" / "mask"))
    assert len(mtr) == 8 and len(mva) == 2
    train_ids = {_id_of(dst / "train" / "train" / f) for f in os.listdir(dst / "train" / "train")}
    assert train_ids.isdisjoint(mat_val_ids)
    assert train_ids | mat_val_ids == set(range(1, 11))

    # source untouched
    assert len(os.listdir(src / "train")) == 10


# --- coco_to_dataset -------------------------------------------------------------------

def _make_coco(tmp_path, n_images=5, n_annotated=4):
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    images, anns = [], []
    for k in range(1, n_images + 1):
        Image.fromarray((np.random.rand(20, 20, 4) * 255).astype(np.uint8), "RGBA").save(
            img_dir / f"im{k}.tif")
        images.append({"id": k, "file_name": f"im{k}.tif", "width": 20, "height": 20})
    for k in range(1, n_annotated + 1):            # a triangle polygon per annotated image
        anns.append({"id": k, "image_id": k, "category_id": 1, "iscrowd": 0,
                     "segmentation": [[2, 2, 15, 2, 8, 15]], "area": 100, "bbox": [2, 2, 13, 13]})
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "tree"}]}
    p = tmp_path / "coco.json"; json.dump(coco, open(p, "w"))
    return str(p), str(img_dir)


def test_converts_binarizes_and_loads(tmp_path):
    coco_json, img_dir = _make_coco(tmp_path, n_images=5, n_annotated=4)
    dst = tmp_path / "out"
    n = coco_to_dataset(coco_json, img_dir, str(dst), limit=3, seed=1)
    assert n == 3
    assert set(os.listdir(dst / "train")) == {"train1.jpeg", "train2.jpeg", "train3.jpeg"}
    assert set(os.listdir(dst / "mask")) == {"mask1.gif", "mask2.gif", "mask3.gif"}
    for i in (1, 2, 3):                            # masks binary + foreground (the triangle)
        m = np.asarray(Image.open(dst / "mask" / f"mask{i}.gif").convert("L"))
        assert set(np.unique(m).tolist()) <= {0, 255} and (m > 0).any()
    ds = StemDataset(str(dst / "train"), str(dst / "mask"))
    assert len(ds) == 3
    _, mask = ds[0]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_only_annotated_and_deterministic(tmp_path):
    coco_json, img_dir = _make_coco(tmp_path, n_images=6, n_annotated=4)
    coco_to_dataset(coco_json, img_dir, str(tmp_path / "a"), limit=4, seed=1)
    coco_to_dataset(coco_json, img_dir, str(tmp_path / "b"), limit=4, seed=1)
    for i in (1, 2, 3, 4):                         # same seed -> byte-identical masks
        assert filecmp.cmp(tmp_path / "a" / "mask" / f"mask{i}.gif",
                           tmp_path / "b" / "mask" / f"mask{i}.gif", shallow=False)


def test_raises_when_too_few_annotated(tmp_path):
    coco_json, img_dir = _make_coco(tmp_path, n_images=5, n_annotated=2)
    with pytest.raises(ValueError, match="eligible"):
        coco_to_dataset(coco_json, img_dir, str(tmp_path / "out"), limit=4, seed=1)


def test_skips_images_missing_on_disk(tmp_path):
    # 5 annotated images referenced by the json, but 2 of their .tif files are absent
    # (some COCO splits reference never-shipped images). Sampling must skip the missing
    # ones and still reach the limit from the 3 present, instead of crashing.
    coco_json, img_dir = _make_coco(tmp_path, n_images=5, n_annotated=5)
    os.remove(os.path.join(img_dir, "im2.tif"))
    os.remove(os.path.join(img_dir, "im4.tif"))
    n = coco_to_dataset(coco_json, img_dir, str(tmp_path / "out"), limit=3, seed=1)
    assert n == 3
    assert len(os.listdir(tmp_path / "out" / "train")) == 3

    # only 3 present -> limit 4 is unreachable
    with pytest.raises(ValueError, match="eligible"):
        coco_to_dataset(coco_json, img_dir, str(tmp_path / "out2"), limit=4, seed=1)
