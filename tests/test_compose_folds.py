"""Fold composition is where a site can silently end up on both sides.

These pin the two properties that make a leave-one-site-out claim mean anything: the
held-out site appears *only* in test, and no tile is dropped or collided during
renumbering.
"""
import json
import os

import pytest

from winmol_unet.geo.folds import _ids, compose, plan

SITES = ["Campus", "Oberheide", "Bachsee", "Kaufland"]
SPLITTABLE = ["Campus", "Oberheide"]


def _make_tiles(d, n, start=1):
    os.makedirs(os.path.join(d, "train"), exist_ok=True)
    os.makedirs(os.path.join(d, "mask"), exist_ok=True)
    for i in range(start, start + n):
        open(os.path.join(d, "train", f"train{i}.jpeg"), "w").write(f"img{i}")
        open(os.path.join(d, "mask", f"mask{i}.gif"), "w").write(f"msk{i}")


@pytest.fixture
def corpus(tmp_path):
    """Two block-splittable sites (train+val) and two whole sites."""
    for s in SPLITTABLE:
        _make_tiles(tmp_path / "tv" / s / "train", 5)
        _make_tiles(tmp_path / "tv" / s / "val", 2)
    for s in SITES:
        _make_tiles(tmp_path / "all" / s, 4)
    return tmp_path


def test_held_out_site_appears_only_in_test(corpus):
    for held in SITES:
        sources = plan(SITES, str(corpus / "all"), str(corpus / "tv"), SPLITTABLE)[held]
        labels = {sp: [l for l, _ in v] for sp, v in sources.items()}
        assert labels["test"] == [held]
        assert held not in labels["train"] and held not in labels["val"]


def test_every_other_site_is_used_for_training(corpus):
    sources = plan(SITES, str(corpus / "all"), str(corpus / "tv"), SPLITTABLE)["Bachsee"]
    assert set(l for l, _ in sources["train"]) == {"Campus", "Oberheide", "Kaufland"}
    assert set(l for l, _ in sources["val"]) == {"Campus", "Oberheide"}


def test_renumbering_is_contiguous_and_loses_nothing(corpus, tmp_path):
    """Two sites both numbered from 1 must not collide when merged into one split."""
    sources = plan(SITES, str(corpus / "all"), str(corpus / "tv"), SPLITTABLE)["Bachsee"]
    out = tmp_path / "fold"
    counts = compose("Bachsee", sources, str(out))
    assert counts["train"] == 5 + 5 + 4        # Campus + Oberheide + Kaufland
    assert counts["val"] == 2 + 2
    assert counts["test"] == 4
    got = sorted(n for n, _ in _ids(str(out / "train")))
    assert got == list(range(1, counts["train"] + 1))


def test_provenance_traces_every_tile_back_to_its_site(corpus, tmp_path):
    sources = plan(SITES, str(corpus / "all"), str(corpus / "tv"), SPLITTABLE)["Kaufland"]
    out = tmp_path / "fold"
    compose("Kaufland", sources, str(out))
    rows = [json.loads(l) for l in open(out / "provenance.jsonl")]
    assert len(rows) == 5 + 5 + 4 + 2 + 2 + 4
    assert not any(r["source"].startswith("Kaufland:train") for r in rows)
    assert sum(r["source"] == "Kaufland:test" for r in rows) == 4


def test_missing_mask_refuses_rather_than_silently_shrinking(corpus, tmp_path):
    os.remove(corpus / "all" / "Bachsee" / "mask" / "mask2.gif")
    sources = plan(SITES, str(corpus / "all"), str(corpus / "tv"), SPLITTABLE)["Bachsee"]
    with pytest.raises(SystemExit, match="no mask2.gif"):
        compose("Bachsee", sources, str(tmp_path / "fold"))


def test_an_empty_split_fails_loudly(corpus, tmp_path):
    """Kaufland's val vanished under the buffer once and the driver moved on silently."""
    sources = {"train": [("A", str(corpus / "all" / "Campus"))], "val": [], "test": []}
    with pytest.raises(SystemExit, match="got no tiles"):
        compose("X", sources, str(tmp_path / "fold"))
