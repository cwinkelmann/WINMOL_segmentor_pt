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


@pytest.mark.parametrize("strategy,extra", [("sites", {"split": "test"}), ("blocks", {})])
def test_tile_px_reaches_every_strategy(tmp_path, monkeypatch, strategy, extra):
    """`sites` and `halve` silently dropped tile_px, so a 666 px config produced 512 px
    tiles at 19.512 m — 3.81 cm/px instead of 2.93. Mixed with block-built tiles it gave
    a training set at two resolutions and a test set at a third, and nothing complained.
    """
    seen = []

    def fake_sample(ortho, stems, aoi, out_dir, **kw):
        seen.append((kw.get("tile_px"), kw.get("antialias")))
        return {"written": 3, "next_index": 1}

    monkeypatch.setattr(make_splits, "sample_tiles", fake_sample)
    monkeypatch.setattr(make_splits, "_clean_stems", lambda site, o, q: (site["stems"], None))
    p = tmp_path / "sites.json"
    p.write_text(json.dumps({"extent_m": 19.512, "tile_px": 666, "block_size_m": 60,
                             "antialias": "on",
                             "split_fractions": {"train": 0.85, "val": 0.15, "test": 0.0},
                             "block_splits": ["train", "val"],
                             "sites": [_site("A", **extra)]}))
    make_splits.run(str(p), str(tmp_path / "out"), strategy, quiet=True)
    assert seen, "no tiles sampled"
    for tile_px, antialias in seen:
        assert tile_px == 666, f"{strategy} dropped tile_px (got {tile_px})"
        assert antialias == "on", f"{strategy} dropped antialias (got {antialias})"
