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
    from winmol_unet.training.dataset import StemDataset
    ds = StemDataset(str(dst / "train"), str(dst / "mask"))
    assert len(ds) == 2
    _, mask = ds[0]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}

    # source untouched
    assert any("SKY" in n for n in os.listdir(src / "train"))
    assert Image.open(src / "mask" / "mask_42_7_SKY99.7.gif").mode == "P"


def test_build_dataset_raises_when_no_pairs(tmp_path):
    import pytest
    (tmp_path / "src" / "train").mkdir(parents=True)
    (tmp_path / "src" / "mask").mkdir(parents=True)
    # image with an unrecognized extension -> nothing paired
    Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(
        tmp_path / "src" / "train" / "train_1.png")
    with pytest.raises(ValueError, match="no paired"):
        build_dataset(str(tmp_path / "src"), str(tmp_path / "dst"))


def test_loader_reads_the_published_zenodo_naming(tmp_path):
    """Zenodo's GenDS uses `train_100_1.jpeg` / `mask_100_1.gif`; ours uses `train1`.

    Both must load without renaming, so a published dataset can be used exactly as
    distributed rather than through a derived copy nobody can verify.
    """
    import numpy as np
    from PIL import Image

    from winmol_unet.training.dataset import _paired_ids, _sort_key

    img = tmp_path / "train"; msk = tmp_path / "mask"
    img.mkdir(); msk.mkdir()
    keys = ["_100_1", "_100_2", "_100_10"]
    for k in keys:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(img / f"train{k}.jpeg")
        Image.fromarray(np.zeros((8, 8), np.uint8), "L").save(msk / f"mask{k}.gif")
    # an AppleDouble sidecar must not be picked up as a tile
    (img / "._train_100_1.jpeg").write_bytes(b"\x00\x05\x16\x07")

    ids = _paired_ids(str(img), str(msk))
    assert ids == ["_100_1", "_100_2", "_100_10"], ids   # numeric-aware ordering


def test_loader_still_reads_our_own_naming(tmp_path):
    import numpy as np
    from PIL import Image

    from winmol_unet.training.dataset import _paired_ids

    img = tmp_path / "train"; msk = tmp_path / "mask"
    img.mkdir(); msk.mkdir()
    for n in (1, 2, 10):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(img / f"train{n}.jpeg")
        Image.fromarray(np.zeros((8, 8), np.uint8), "L").save(msk / f"mask{n}.gif")
    assert _paired_ids(str(img), str(msk)) == ["1", "2", "10"]
