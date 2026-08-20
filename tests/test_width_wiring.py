"""CLI wiring for --width-mult. The build_model width tests moved to tests/test_model.py;
this file keeps only the config_from_args tests, pending Task 8's removal of this file."""
from winmol_unet.training.run_train import config_from_args


def test_cli_parses_width_mult():
    cfg = config_from_args(["--data-dir", "d", "--width-mult", "0.5"])
    assert cfg.width_mult == 0.5


def test_cli_width_mult_defaults_to_one():
    cfg = config_from_args(["--data-dir", "d"])
    assert cfg.width_mult == 1.0
