import sys
import types
import glob
import pytest


def _fake_wandb():
    calls = {}
    m = types.ModuleType("wandb")
    m.init = lambda **kw: calls.__setitem__("init", kw)
    m.log = lambda d, step=None: calls.__setitem__("log", (dict(d), step))
    m.finish = lambda: calls.__setitem__("finish", True)
    return m, calls


def _fake_dotenv():
    m = types.ModuleType("dotenv")
    m.load_dotenv = lambda *a, **k: True
    return m


def test_tensorboard_only_when_wandb_off(tmp_path, monkeypatch):
    # Ensure wandb is never needed when off.
    monkeypatch.setitem(sys.modules, "wandb", None)   # import wandb -> ImportError if touched
    from winmol_unet.training.run_logger import RunLogger
    lg = RunLogger(str(tmp_path / "log"), use_wandb=False)
    lg.log_scalars({"val/f1": 0.5}, 1)
    lg.close()
    assert glob.glob(str(tmp_path / "log" / "events*"))   # TB wrote something


def test_wandb_init_log_finish(tmp_path, monkeypatch):
    fake, calls = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "dotenv", _fake_dotenv())
    from winmol_unet.training.run_logger import RunLogger
    lg = RunLogger(str(tmp_path / "log"), use_wandb=True, project="P", run_name="R")
    assert calls["init"] == {"project": "P", "name": "R"}
    lg.log_scalars({"val/f1": 0.5}, 3)
    assert calls["log"] == ({"val/f1": 0.5}, 3)
    lg.close()
    assert calls["finish"] is True


def test_missing_wandb_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)   # forces ImportError on `import wandb`
    from winmol_unet.training.run_logger import RunLogger
    with pytest.raises(RuntimeError, match=r"\[wandb\]"):
        RunLogger(str(tmp_path / "log"), use_wandb=True)
