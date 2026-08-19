"""Colour augmentation limits must reach the pipeline, not just the config.

Colour is the axis the beech sites differ on most, and the one autumn site is unlearnable
from the others. A knob that silently keeps its default would make that experiment a
no-op that still produces plausible numbers.
"""
import numpy as np
import pytest

from training.augment import build_augmentation
from training.run_train import config_from_args


def _args(tmp_path, *extra):
    return config_from_args(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path),
                             *extra])


def test_cli_defaults_match_the_config_defaults(tmp_path):
    """Arm A of the colour experiment reuses earlier runs, so defaults must not move."""
    cfg = _args(tmp_path)
    assert (cfg.aug_hue_shift, cfg.aug_sat_shift, cfg.aug_val_shift) == (20, 30, 20)


def test_limits_reach_the_transform(tmp_path):
    cfg = _args(tmp_path, "--aug-hue-shift", "90", "--aug-sat-shift", "60",
                "--aug-val-shift", "30", "--aug-hsv-p", "0.9")
    hsv = [t for t in build_augmentation(cfg).transforms
           if type(t).__name__ == "HueSaturationValue"]
    assert len(hsv) == 1
    h = hsv[0]
    assert tuple(h.hue_shift_limit) == (-90, 90)
    assert tuple(h.sat_shift_limit) == (-60, 60)
    assert h.p == pytest.approx(0.9)


def test_strong_hue_actually_moves_colour(tmp_path):
    """A full-circle hue shift must change hue far more than the default does."""
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = 200                       # a strongly red image
    img[..., 1] = 60
    mask = np.zeros((64, 64), np.uint8)

    def spread(hue_shift):
        cfg = _args(tmp_path, "--aug-hue-shift", str(hue_shift), "--aug-hsv-p", "1.0",
                    "--aug-rotate-p", "0", "--aug-hflip-p", "0", "--aug-vflip-p", "0",
                    "--aug-bc-p", "0")
        aug = build_augmentation(cfg)
        outs = [aug(image=img, mask=mask)["image"].astype(int) for _ in range(12)]
        # channel ordering scatters as hue rotates; measure how far from the original
        return float(np.mean([np.abs(o - img.astype(int)).mean() for o in outs]))

    assert spread(90) > spread(5) * 2, "strong hue shift barely moved the image"
