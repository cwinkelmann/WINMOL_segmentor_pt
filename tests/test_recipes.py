"""Recipes must reproduce the arms that were actually measured.

Each test here corresponds to a way a recipe could silently train the wrong experiment —
no exception, no warning, just a number below the published one.
"""
import pytest

from winmol_unet.training import recipes
from winmol_unet.training.run_train import build_parser


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
