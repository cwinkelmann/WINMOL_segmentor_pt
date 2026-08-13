"""`block_splits` narrows which splits a site is dealt blocks for.

A leave-one-site-out fold needs its *training* sites to yield train and val only: the
test set is the held-out site, and dealing test blocks from a training site would mix a
second site into the yardstick — silently, since the tiles look identical.
"""
import json

import pytest

from scripts import make_splits


@pytest.fixture
def spy(monkeypatch):
    """Record which (site, split) pairs get sampled, without touching a raster."""
    calls = []

    def fake_sample(ortho, stems, aoi, out_dir, **kw):
        calls.append((ortho, kw.get("split")))
        return {"written": 3, "next_index": 1}

    monkeypatch.setattr(make_splits, "sample_tiles", fake_sample)
    monkeypatch.setattr(make_splits, "_clean_stems", lambda site, o, q: (site["stems"], None))
    return calls


def _cfg(tmp_path, sites):
    p = tmp_path / "sites.json"
    p.write_text(json.dumps({"extent_m": 19.512, "tile_px": 666, "block_size_m": 60,
                             "split_fractions": {"train": 0.85, "val": 0.15, "test": 0.0},
                             "sites": sites}))
    return str(p)


def _site(name, **kw):
    return {"name": name, "ortho": f"{name}.tif", "stems": f"{name}.shp",
            "aoi": f"{name}_aoi.shp", **kw}


def test_defaults_to_all_three_splits(tmp_path, spy):
    make_splits.run(_cfg(tmp_path, [_site("A")]), str(tmp_path / "out"), "blocks",
                    quiet=True)
    assert [s for _, s in spy] == ["train", "val", "test"]


def test_restricts_a_training_site_to_train_and_val(tmp_path, spy):
    """The whole point: no test blocks from a site that is not the held-out one."""
    make_splits.run(_cfg(tmp_path, [_site("A", block_splits=["train", "val"])]),
                    str(tmp_path / "out"), "blocks", quiet=True)
    assert [s for _, s in spy] == ["train", "val"]
    assert "test" not in [s for _, s in spy]


def test_top_level_key_applies_to_every_site(tmp_path, spy):
    p = tmp_path / "sites.json"
    p.write_text(json.dumps({"extent_m": 19.512, "tile_px": 666, "block_size_m": 60,
                             "block_splits": ["train", "val"],
                             "split_fractions": {"train": 0.85, "val": 0.15, "test": 0.0},
                             "sites": [_site("A"), _site("B")]}))
    make_splits.run(str(p), str(tmp_path / "out"), "blocks", quiet=True)
    assert [s for _, s in spy] == ["train", "val", "train", "val"]


def test_a_typo_fails_loudly(tmp_path, spy):
    """Silently ignoring 'traon' would build a fold with no training data."""
    with pytest.raises(SystemExit, match="block_splits"):
        make_splits.run(_cfg(tmp_path, [_site("A", block_splits=["traon"])]),
                        str(tmp_path / "out"), "blocks", quiet=True)
