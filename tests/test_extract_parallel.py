"""The merge is where two sites can silently overwrite each other.

Each part numbers its tiles from 1, so a merge that does not renumber loses whole sites;
and `tiles.jsonl` carries the old index, so provenance must be rewritten with it.
"""
import json
import os

import pytest

from winmol_unet.geo.parallel import _part_configs, merge


def _part(work, name, split, n, start=1):
    d = os.path.join(work, name, split)
    os.makedirs(os.path.join(d, "train"), exist_ok=True)
    os.makedirs(os.path.join(d, "mask"), exist_ok=True)
    rows = []
    for i in range(start, start + n):
        open(os.path.join(d, "train", f"train{i}.jpeg"), "w").write(f"{name}-img{i}")
        open(os.path.join(d, "mask", f"mask{i}.gif"), "w").write(f"{name}-msk{i}")
        rows.append({"n": i, "site": name, "centre_x": 100.0 + i})
    with open(os.path.join(d, "tiles.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_two_parts_numbering_from_one_do_not_collide(tmp_path):
    work = tmp_path / "w"
    _part(str(work), "A", "train", 3)
    _part(str(work), "B", "train", 4)
    counts = merge(str(work), ["A", "B"], str(tmp_path / "out"), quiet=True)
    assert counts["train"] == 7, "a colliding merge would lose tiles"
    names = sorted(os.listdir(tmp_path / "out" / "train" / "train"))
    assert len(names) == 7
    got = sorted(int(n[5:-5]) for n in names)
    assert got == list(range(1, 8)), "indices must be contiguous"


def test_provenance_is_renumbered_to_match(tmp_path):
    work = tmp_path / "w"
    _part(str(work), "A", "train", 2)
    _part(str(work), "B", "train", 2)
    merge(str(work), ["A", "B"], str(tmp_path / "out"), quiet=True)
    rows = [json.loads(l) for l in open(tmp_path / "out" / "train" / "tiles.jsonl")]
    assert [r["n"] for r in rows] == [1, 2, 3, 4]
    assert [r["part"] for r in rows] == ["A", "A", "B", "B"]
    # the original centre must travel with the tile, not be reassigned
    assert rows[2]["centre_x"] == pytest.approx(101.0)


def test_missing_mask_refuses_the_merge(tmp_path):
    work = tmp_path / "w"
    _part(str(work), "A", "train", 2)
    os.remove(work / "A" / "train" / "mask" / "mask2.gif")
    with pytest.raises(SystemExit, match="no mask"):
        merge(str(work), ["A"], str(tmp_path / "out"), quiet=True)


def test_each_site_keeps_its_own_split(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"extent_m": 19.512, "tile_px": 666, "sites": [
        {"name": "P1", "ortho": "a.tif", "stems": "a.gpkg", "aoi": "aoi.gpkg", "split": "train"},
        {"name": "P2", "ortho": "b.tif", "stems": "b.gpkg", "aoi": "aoi.gpkg", "split": "test"}]}))
    parts = _part_configs(str(cfg), str(tmp_path / "w"))
    assert [n for n, _ in parts] == ["P1", "P2"]
    for name, p in parts:
        one = json.loads(open(p).read())
        assert len(one["sites"]) == 1 and one["sites"][0]["name"] == name
        assert one["tile_px"] == 666, "shared settings must survive the split"
    assert json.loads(open(parts[1][1]).read())["sites"][0]["split"] == "test"


def test_the_worker_command_is_runnable_as_a_module():
    """The fan-out spawns `python -m winmol_unet.geo.splits`; pin that it resolves.

    This is the one line in parallel.py the rest of the suite cannot reach: the real path
    fans out a process per site over multi-hundred-megapixel orthomosaics, so no hermetic
    test drives it. It used to build a filesystem path as
    `dirname(dirname(__file__))/scripts/make_splits.py`, which silently stopped pointing at
    anything real when the module moved one directory deeper — and would only have failed
    at spawn time, mid-extraction, after the expensive setup.

    Running the module's --help is cheap and catches exactly that class of breakage:
    the module is importable, is a valid -m target, and its parser builds.
    """
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "winmol_unet.geo.splits", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0, f"worker command is not runnable:\n{out.stderr}"
    assert "--config" in out.stdout and "--strategy" in out.stdout


def test_the_spawned_command_names_the_module_not_a_path():
    """A path built from __file__ breaks on a move; `-m` does not. Keep it that way."""
    import inspect

    from winmol_unet.geo import parallel

    src = inspect.getsource(parallel)
    assert '"-m", "winmol_unet.geo.splits"' in src, \
        "parallel.py should spawn the splits module by name, not by filesystem path"
    assert "scripts" not in src.split('"""')[2], \
        "parallel.py still refers to the old scripts/ location outside its docstring"
