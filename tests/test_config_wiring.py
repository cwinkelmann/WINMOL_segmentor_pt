"""Every CLI flag reaches the config field it claims to set.

Collapses the flag-parsing half of eleven former wiring modules — the training half of
each (a full run_training()/train_one_run() call) stayed behind in its own file, owned by
a later consolidation task. Absorbs whole: test_train_config.py, test_train_device.py,
test_recipes.py, test_width_wiring.py, test_val_data_dir.py, test_mosaic.py. Takes only
the flag-parsing tests from: test_arch_wiring.py, test_aug_wiring.py, test_wandb_wiring.py,
test_test_stage.py.

The nine test_recipes.py tests are kept as individual, unparametrized tests: they pin two
traps that have already produced a wrong published number in this repo (multiscale=True is
not the jitter switch — the fixed control sets it too, the crop *range* is the variable;
and rotation is off by default while every measured run passed --aug-rotate-p 0.5
--aug-rotate-limit 180), and their explanatory names are the signal a failing reproduction
needs.
"""
import os

import pytest
import torch

from winmol_unet.training import recipes
from winmol_unet.training.config import TrainConfig
from winmol_unet.training.device import resolve_device
from winmol_unet.training.run_train import build_parser, config_from_args


# --- CLI flag -> TrainConfig field -------------------------------------------------

@pytest.mark.parametrize("flags,expected", [
    # architecture (test_arch_wiring.py::test_cli_parses_arch_flags)
    (["--arch", "deeplabv3plus", "--encoder", "resnet18", "--encoder-weights", "imagenet"],
     {"arch": "deeplabv3plus", "encoder": "resnet18", "encoder_weights": "imagenet"}),
    # augmentation (test_aug_wiring.py::test_cli_parses_aug_flags)
    (["--aug-rotate-p", "0.3", "--aug-rotate-limit", "20", "--aug-hflip-p", "0.25"],
     {"aug_rotate_p": 0.3, "aug_rotate_limit": 20, "aug_hflip_p": 0.25}),
    # width (test_width_wiring.py::test_cli_parses_width_mult)
    (["--width-mult", "0.5"], {"width_mult": 0.5}),
    # fixed validation split (test_val_data_dir.py::test_cli_parses_val_data_dir)
    (["--val-data-dir", "v"],
     {"val_data_dir": "v", "val_image_dir": os.path.join("v", "train")}),
    # wandb (test_wandb_wiring.py::test_cli_parses_wandb_flags)
    (["--wandb", "--wandb-project", "P", "--wandb-run-name", "R"],
     {"wandb": True, "wandb_project": "P", "wandb_run_name": "R"}),
    # held-out test stage (test_test_stage.py::test_cli_parses_test_data_dir)
    (["--test-data-dir", "t"], {"test_data_dir": "t"}),
    # mosaic (test_mosaic.py::test_the_cli_flag_reaches_the_config)
    (["--mosaic-p", "0.4"], {"mosaic_p": 0.4}),
    # static training plots on disk
    (["--plots"], {"plots": True}),
    # defaults -- these rows pin the traps documented in recipes.py: rotation is OFF
    # by default even though every measured run enabled it explicitly.
    ([], {"arch": "unet", "width_mult": 1.0, "aug_hflip_p": 0.5, "aug_rotate_p": 0.0,
          "wandb": False, "val_data_dir": None, "test_data_dir": None,
          "plots": False}),
])
def test_cli_flag_reaches_config(flags, expected):
    cfg = config_from_args(["--data-dir", "d", *flags])
    for field, value in expected.items():
        assert getattr(cfg, field) == value, f"{field} did not reach the config"


# --- test_mosaic.py: recipe labelling (not a flag-parsing test) --------------------

def test_mosaic_recipe_is_labelled_unevaluated():
    assert recipes.RECIPES["mosaic"]["mosaic_p"] > 0
    assert "mosaic" in recipes.UNEVALUATED


# --- test_train_config.py: defaults and derived paths ------------------------------

def test_config_defaults_and_derived_dirs():
    c = TrainConfig(
        data_dir="/data/TestDS", checkpoint_dir="ck", log_dir="log",
        hdf5_out="out/model.hdf5", onnx_out="out/model.onnx",
    )
    assert c.batch_size == 4
    assert c.epochs == 100
    assert c.lr == 1e-3
    assert c.img_size == 512
    assert c.val_fraction == 0.2
    assert c.patience == 5
    assert c.seed == 1
    assert c.image_dir == os.path.join("/data/TestDS", "train")
    assert c.mask_dir == os.path.join("/data/TestDS", "mask")


# --- test_train_device.py: device resolution ----------------------------------------

def test_resolve_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_returns_available_device():
    d = resolve_device("auto")
    assert d.type in {"mps", "cuda", "cpu"}


# --- test_val_data_dir.py: the loader-level behavioural claim -----------------------
# (the trailing run_training() smoke that used to follow this assertion is dropped --
# see the ledger: it is *deduplicated*, covered by the helper repo's full-training
# integration test added in Task 13.)

def test_fixed_val_split_is_used_not_resplit(tmp_path, stem_dataset, train_config):
    """A fixed --val-data-dir is used as given: no 80/20 re-split of either set."""
    from winmol_unet.training.augment import build_augmentation
    from winmol_unet.training.run_train import _build_loaders

    train_d, val_d = tmp_path / "tr", tmp_path / "va"
    stem_dataset(train_d, n=6)
    stem_dataset(val_d, n=4)
    cfg = train_config(data_dir=train_d, val_data_dir=str(val_d))

    tl, vl = _build_loaders(cfg.image_dir, cfg.mask_dir, cfg, build_augmentation(cfg),
                            val_image_dir=cfg.val_image_dir, val_mask_dir=cfg.val_mask_dir)
    assert len(tl.dataset) == 6 and len(vl.dataset) == 4


# --- test_recipes.py, moved in whole -------------------------------------------------
# Recipes must reproduce the arms that were actually measured. Each test here
# corresponds to a way a recipe could silently train the wrong experiment -- no
# exception, no warning, just a number below the published one.

def _resolve(recipe, extra=None):
    argv = ["--data-dir", "x"] + list(extra or [])
    p = build_parser()
    return recipes.apply(recipe, p.parse_args(argv), p, argv)[0]


def test_rotation_is_set_explicitly_by_every_measured_arm():
    """aug_rotate_p defaults to 0.0 and the limit to 15.0; the runs used 0.5 and 180.

    Tiles are cut at random rotations and the block-split separation math assumes the
    full circle, so a recipe that leaves rotation at its default trains a different
    experiment than the one the published numbers came from.
    """
    for name in ("fixed", "jitter", "robust", "mosaic"):
        args = _resolve(name)
        assert args.aug_rotate_p == 0.5, f"{name} left rotation probability at default"
        assert args.aug_rotate_limit == 180.0, f"{name} left rotation limit at default"


def test_the_scale_arms_differ_only_in_crop_range():
    """`multiscale=True` is not the jitter switch — the fixed control sets it too.

    If these two arms ever differ in anything but the crop range, the scale-augmentation
    comparison stops being a controlled one.
    """
    fixed, jitter = recipes.RECIPES["fixed"], recipes.RECIPES["jitter"]
    differing = {k for k in set(fixed) | set(jitter) if fixed.get(k) != jitter.get(k)}
    assert differing == {"crop_min_px", "crop_max_px"}, differing
    assert fixed["crop_min_px"] == fixed["crop_max_px"] == 512
    assert (jitter["crop_min_px"], jitter["crop_max_px"]) == (394, 666)


def test_an_explicit_flag_beats_the_recipe():
    args = _resolve("robust", ["--aug-hsv-p", "0.7"])
    assert args.aug_hsv_p == 0.7           # the user's value survives
    assert args.aug_hue_shift == 90        # the rest of the recipe still applies


def test_an_explicit_flag_wins_even_when_it_equals_the_default():
    """The case that comparing-against-defaults gets wrong.

    --aug-hsv-p 0.5 is also the parser default, so a value comparison cannot tell it
    apart from "not passed" and the recipe would overwrite it with 0.9 — silently
    turning a deliberately mild run into the strong-hue arm.
    """
    args = _resolve("robust", ["--aug-hsv-p", "0.5"])
    assert args.aug_hsv_p == 0.5


def test_overrides_are_reported_not_swallowed():
    p = build_parser()
    argv = ["--data-dir", "x", "--crop-max-px", "700"]
    _, notes = recipes.apply("jitter", p.parse_args(argv), p, argv)
    assert any(n.startswith("OVERRIDDEN crop_max_px=700") for n in notes), notes


def test_baseline_is_not_the_scale_control():
    """`baseline` is repo defaults, not the fixed arm — conflating them was the trap."""
    assert recipes.RECIPES["baseline"] == {}
    assert recipes.RECIPES["fixed"]["multiscale"] is True


def test_mosaic_is_flagged_unevaluated():
    assert "mosaic" in recipes.UNEVALUATED
    assert "UNEVALUATED" in recipes.describe("mosaic")


def test_the_schedule_settings_a_recipe_does_not_cover_are_stated():
    """Recipes are augmentation-only; the note is the only thing stopping that from
    becoming a silent difference against the measured runs."""
    for token in ("--epochs 40", "--batch-size 16", "--deterministic"):
        assert token in recipes.SCHEDULE_NOTE
    # The seed must NOT be quoted as a single value: scale-aug/LOSO used 1, but the
    # released Tegel assets are seed 3, so one number sends half the reproductions to
    # different weights.
    assert "varies by run" in recipes.SCHEDULE_NOTE


def test_unknown_recipe_fails_loudly():
    with pytest.raises(SystemExit, match="unknown recipe"):
        _resolve("nope")
