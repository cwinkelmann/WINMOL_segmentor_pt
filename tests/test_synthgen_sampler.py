"""Sampler + SceneSpec tests — hermetic, no Blender needed."""
import json
import os

import pytest

from synthgen.sampler import GenConfig, sample_scene
from synthgen.scene_spec import SceneSpec

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "golden_scene_seed1.json")


def test_same_seed_is_deterministic():
    a = sample_scene(GenConfig(), seed=1)
    b = sample_scene(GenConfig(), seed=1)
    assert a == b


def test_different_seed_differs():
    assert sample_scene(GenConfig(), seed=1) != sample_scene(GenConfig(), seed=2)


def test_json_round_trip_is_lossless():
    spec = sample_scene(GenConfig(), seed=7)
    assert SceneSpec.from_json(spec.to_json()) == spec


def test_stem_count_and_dimensions_honour_config():
    cfg = GenConfig(n_stems=(2, 5), stem_length_m=(3.0, 12.0), stem_diameter_m=(0.15, 0.5))
    for seed in range(20):
        spec = sample_scene(cfg, seed=seed)
        assert 2 <= len(spec.stems) <= 5
        for s in spec.stems:
            assert 3.0 <= s.length_m <= 12.0
            assert 0.15 <= s.diameter_m <= 0.5


def test_stems_stay_inside_the_tile_footprint():
    cfg = GenConfig()
    half = cfg.tile_px * cfg.gsd_m_per_px / 2.0
    for seed in range(20):
        for s in sample_scene(cfg, seed=seed).stems:
            assert -half <= s.center_xy_m[0] <= half
            assert -half <= s.center_xy_m[1] <= half


def test_sun_elevation_within_range_and_camera_matches_gsd():
    cfg = GenConfig(sun_elevation_deg=(20.0, 70.0), tile_px=512, gsd_m_per_px=0.02)
    spec = sample_scene(cfg, seed=3)
    assert 20.0 <= spec.sun.elevation_deg <= 70.0
    # camera is solved so the rendered tile covers exactly tile_px * gsd metres
    assert spec.camera.footprint_m == pytest.approx(512 * 0.02)
    assert spec.camera.altitude_m > 0


def test_crossing_stems_are_producible():
    # piling/crossing is a modelled axis, not an accident: some scenes must contain
    # a stem resting above ground level on another
    cfg = GenConfig(n_stems=(4, 8), p_crossing=1.0)
    assert any(s.elevation_m > 0 for s in sample_scene(cfg, seed=5).stems)


def test_matches_golden_spec():
    # guards every distribution against silent drift
    spec = sample_scene(GenConfig(), seed=1)
    with open(GOLDEN) as f:
        assert json.loads(spec.to_json()) == json.load(f)
