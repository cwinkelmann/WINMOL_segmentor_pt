import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from split_dataset import split_dataset
from training.dataset import StemDataset, train_val_split


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
    assert mat_val_ids == set(va.ids)

    # both splits are loader-ready and disjoint, covering all 10 pairs
    mtr = StemDataset(str(dst / "train" / "train"), str(dst / "train" / "mask"))
    mva = StemDataset(str(dst / "val" / "train"), str(dst / "val" / "mask"))
    assert len(mtr) == 8 and len(mva) == 2
    train_ids = {_id_of(dst / "train" / "train" / f) for f in os.listdir(dst / "train" / "train")}
    assert train_ids.isdisjoint(mat_val_ids)
    assert train_ids | mat_val_ids == set(range(1, 11))

    # source untouched
    assert len(os.listdir(src / "train")) == 10
