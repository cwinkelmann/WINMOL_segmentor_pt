import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_dataset import build_dataset


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
    from training.dataset import StemDataset
    ds = StemDataset(str(dst / "train"), str(dst / "mask"))
    assert len(ds) == 2
    _, mask = ds[0]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}

    # source untouched
    assert any("SKY" in n for n in os.listdir(src / "train"))
    assert Image.open(src / "mask" / "mask_42_7_SKY99.7.gif").mode == "P"
