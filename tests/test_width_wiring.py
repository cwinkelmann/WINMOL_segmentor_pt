from winmol_unet.training.model_factory import build_model
from winmol_unet.training.run_train import config_from_args


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


def test_build_model_unet_accepts_width_mult():
    full = build_model("unet")
    half = build_model("unet", width_mult=0.5)
    assert _nparams(half) < 0.32 * _nparams(full)


def test_build_model_default_width_is_full():
    assert _nparams(build_model("unet", width_mult=1.0)) == _nparams(build_model("unet"))


def test_cli_parses_width_mult():
    cfg = config_from_args(["--data-dir", "d", "--width-mult", "0.5"])
    assert cfg.width_mult == 0.5


def test_cli_width_mult_defaults_to_one():
    cfg = config_from_args(["--data-dir", "d"])
    assert cfg.width_mult == 1.0
