import filecmp
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from coco_to_dataset import coco_to_dataset


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
    from training.dataset import StemDataset
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
    import pytest
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

    import pytest                                   # only 3 present -> limit 4 is unreachable
    with pytest.raises(ValueError, match="eligible"):
        coco_to_dataset(coco_json, img_dir, str(tmp_path / "out2"), limit=4, seed=1)
