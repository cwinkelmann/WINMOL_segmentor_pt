"""Mosaic augmentation: the CLI wiring and the "unevaluated" recipe label.

Mosaic is UNEVALUATED in this repo — no paired run, no LOSO fold, no results document.
The pipeline-ordering, leak-safety and binary-output tests that used to live here moved
to tests/test_training.py (task 7 of the 2026-08-20 test-suite consolidation); these two
remain because they are CLI/flag-parsing tests, not `run_training()` calls, and are owned
by a later task in that consolidation.
"""


def test_the_cli_flag_reaches_the_config():
    from winmol_unet.training.run_train import build_parser, config_from_parsed
    p = build_parser()
    cfg = config_from_parsed(p.parse_args(["--data-dir", "x", "--mosaic-p", "0.4"]), p)
    assert cfg.mosaic_p == 0.4


def test_mosaic_recipe_is_labelled_unevaluated():
    from winmol_unet.training import recipes
    assert recipes.RECIPES["mosaic"]["mosaic_p"] > 0
    assert "mosaic" in recipes.UNEVALUATED
