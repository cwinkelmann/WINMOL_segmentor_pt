import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_teacher_dataset import ANALYZER_GSD
from teacher_tiles import split_keys, tile_grid


def test_tile_grid_round_trips_pixels_through_world_coords():
    """The gpkg stores world coords; if this inverse is off, every target field is misplaced."""
    grid = tile_grid(512, 512)
    rc = np.array([[0.0, 0.0], [100.0, 250.0], [511.0, 511.0]])
    back = grid.world_to_px(grid.px_to_world(rc))
    assert np.allclose(back, rc, atol=1e-6)


def test_tile_grid_matches_the_profile_used_to_write_the_gpkg():
    pytest.importorskip("rasterio")   # tile_profile needs the geo stack
    from build_teacher_dataset import tile_profile
    prof, grid = tile_profile(512, 512), tile_grid(512, 512)
    assert abs(grid.gsd - prof["transform"].a) < 1e-12
    # origin of the profile transform is the grid's top-left
    assert abs(grid.minx - prof["transform"].c) < 1e-12
    assert abs(grid.maxy - prof["transform"].f) < 1e-12
    assert abs(grid.gsd - ANALYZER_GSD) < 1e-12


def test_split_keys_is_disjoint_and_covers_everything():
    keys = [f"t{i}" for i in range(50)]
    tr, va = split_keys(keys, val_frac=0.2, seed=0)
    assert set(tr) & set(va) == set()
    assert sorted(tr + va) == sorted(keys)
    assert len(va) == 10


def test_split_keys_is_deterministic():
    keys = [f"t{i}" for i in range(50)]
    assert split_keys(keys, seed=3) == split_keys(keys, seed=3)
    assert split_keys(keys, seed=3) != split_keys(keys, seed=4)
