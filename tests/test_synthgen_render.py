"""Renderer smoke test — skipped where bpy is absent (the Mac dev env, CI).

bpy is a pip module on Python >= 3.11 only, so this runs on the render host and
skips everywhere else, the same way the repo guards other optional-dependency
tests.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("bpy")

from synthgen.render_bpy import render_spec          # noqa: E402
from synthgen.sampler import GenConfig, sample_scene  # noqa: E402


def _find(d, prefix):
    return [f for f in sorted(os.listdir(d)) if f.startswith(prefix) and f.endswith(".png")]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    # small + few samples: this asserts wiring, not image quality
    cfg = GenConfig(tile_px=128, n_stems=(2, 3))
    spec = json.loads(sample_scene(cfg, seed=11).to_json())
    out = str(tmp_path_factory.mktemp("render"))
    render_spec(spec, out, samples=8)
    return out, spec


def test_writes_rgb_mask_and_depth(rendered):
    out, _ = rendered
    for prefix in ("rgb", "mask", "depth"):
        assert _find(out, prefix), f"no {prefix} output"


def test_mask_is_binary_and_coregistered(rendered):
    out, spec = rendered
    rgb = Image.open(os.path.join(out, _find(out, "rgb")[0]))
    mask = np.asarray(Image.open(os.path.join(out, _find(out, "mask")[0])).convert("L"))
    assert rgb.size == (spec["tile_px"], spec["tile_px"])
    assert mask.shape == (spec["tile_px"], spec["tile_px"])
    # object-index pass is discrete: no mid-grey anti-aliased edge to threshold
    assert set(np.unique(mask)) <= {0, 255}


def test_stems_cover_a_plausible_share_of_the_tile(rendered):
    out, _ = rendered
    mask = np.asarray(Image.open(os.path.join(out, _find(out, "mask")[0])).convert("L"))
    frac = float((mask >= 128).mean())
    assert 0.001 < frac < 0.6, f"degenerate stem fraction {frac}"


def test_depth_varies_over_the_scene(rendered):
    out, _ = rendered
    depth = np.asarray(Image.open(os.path.join(out, _find(out, "depth")[0])))
    assert depth.max() > depth.min(), "depth pass is flat"
