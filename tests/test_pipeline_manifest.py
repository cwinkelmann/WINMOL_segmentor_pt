"""Staleness is what makes the stages resumable rather than merely separate."""
import json
import os

from winmol_unet.pipeline.manifest import (
    input_identity, is_stale, read_manifest, write_manifest)


def _src(tmp_path, name="in.txt", text="hello"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_hashes_small_inputs_but_not_large_ones(tmp_path):
    small = _src(tmp_path)
    assert input_identity(small, "stems")["sha256"] is not None

    big = tmp_path / "big.tif"
    big.write_bytes(b"\0" * (65 * 1024 * 1024))
    ident = input_identity(str(big), "ortho")
    # Hashing a 19 GB orthomosaic costs more than re-running the stage it guards.
    assert ident["sha256"] is None
    assert ident["size"] == 65 * 1024 * 1024


def test_a_fresh_directory_is_stale(tmp_path):
    assert is_stale(str(tmp_path), "resample", {"gsd": 0.05}, [])


def test_unchanged_inputs_and_params_are_not_stale(tmp_path):
    inputs = [input_identity(_src(tmp_path), "stems")]
    write_manifest(str(tmp_path), "ingest", {"gsd": 0.05}, inputs, [], {"n": 3})
    assert not is_stale(str(tmp_path), "ingest", {"gsd": 0.05}, inputs)


def test_a_changed_param_is_stale(tmp_path):
    inputs = [input_identity(_src(tmp_path), "stems")]
    write_manifest(str(tmp_path), "ingest", {"gsd": 0.05}, inputs, [], {"n": 3})
    assert is_stale(str(tmp_path), "ingest", {"gsd": 0.10}, inputs)


def test_a_changed_input_is_stale(tmp_path):
    p = _src(tmp_path)
    write_manifest(str(tmp_path), "ingest", {}, [input_identity(p, "stems")], [], {})
    with open(p, "w") as fh:
        fh.write("different")
    assert is_stale(str(tmp_path), "ingest", {}, [input_identity(p, "stems")])


def test_a_different_stage_name_is_stale(tmp_path):
    write_manifest(str(tmp_path), "ingest", {}, [], [], {})
    assert is_stale(str(tmp_path), "resample", {}, [])


def test_the_manifest_records_provenance(tmp_path):
    write_manifest(str(tmp_path), "ingest", {"a": 1}, [], [{"path": "x", "count": 2}], {"n": 2})
    m = read_manifest(str(tmp_path))
    assert m["stage"] == "ingest"
    assert m["params"] == {"a": 1}
    assert m["counts"] == {"n": 2}
    assert m["created_utc"].endswith("Z")
    assert "git_commit" in m and "tool_version" in m
    assert json.loads(open(os.path.join(str(tmp_path), "_manifest.json")).read())
