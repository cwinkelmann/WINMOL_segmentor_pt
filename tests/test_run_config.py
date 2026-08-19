"""A run must record what it was, not just what it scored.

`test_results.md` names the dataset and the arch and nothing else, so a finished run
could not be audited after the fact — the crop range and seed lived only in the shell
that launched it. These tests pin the fields that answer "was this run what it claimed".
"""
import json
import os

from training.config import TrainConfig
from training.run_train import write_run_config


def _cfg(out_dir, **kw):
    return TrainConfig(
        data_dir="/ds/train",
        checkpoint_dir=os.path.join(out_dir, "checkpoints"),
        log_dir=os.path.join(out_dir, "logs"),
        pt_out=os.path.join(out_dir, "model.pt"),
        hdf5_out=os.path.join(out_dir, "model.hdf5"),
        onnx_out=os.path.join(out_dir, "model.onnx"),
        **kw)


def test_records_the_knobs_that_define_an_arm(tmp_path):
    """Crop range and seed are the whole experiment — they must survive the shell."""
    cfg = _cfg(str(tmp_path), multiscale=True, crop_min_px=394, crop_max_px=666,
               seed=1, deterministic=True, arch="hrnet")
    p = write_run_config(cfg, argv=["run_train", "--crop-min-px", "394"])
    d = json.loads(open(p).read())
    assert d["config"]["crop_min_px"] == 394 and d["config"]["crop_max_px"] == 666
    assert d["config"]["seed"] == 1 and d["config"]["deterministic"] is True
    assert d["config"]["arch"] == "hrnet" and d["config"]["multiscale"] is True
    assert d["argv"] == ["run_train", "--crop-min-px", "394"]


def test_lands_beside_the_model_it_describes(tmp_path):
    cfg = _cfg(str(tmp_path))
    assert write_run_config(cfg) == str(tmp_path / "run_config.json")


def test_records_provenance(tmp_path):
    cfg = _cfg(str(tmp_path))
    d = json.loads(open(write_run_config(cfg)).read())
    for k in ("started_utc", "host", "torch", "git_commit", "git_dirty"):
        assert k in d
    assert d["host"] and d["torch"]


def test_survives_a_missing_git_checkout(tmp_path, monkeypatch):
    """A run outside a checkout is still a valid run — provenance degrades, not fails."""
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    d = json.loads(open(write_run_config(_cfg(str(tmp_path)))).read())
    assert d["git_commit"] is None and d["git_dirty"] is None
    assert d["config"]["data_dir"] == "/ds/train"
