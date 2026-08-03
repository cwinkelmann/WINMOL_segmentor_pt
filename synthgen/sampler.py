"""Seeded sampling of SceneSpecs from a configuration of distributions.

Everything random lives here, drawn from one seeded Generator, so a (config, seed)
pair reproduces a scene exactly — the property the golden-spec test locks and the
refinement loop relies on when it re-renders a scene with one axis changed.
"""
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

from .scene_spec import (
    CameraSpec, ClutterSpec, GroundSpec, SceneSpec, StemSpec, SunSpec,
)


@dataclass
class GenConfig:
    """Ranges are inclusive (lo, hi) pairs sampled uniformly unless noted."""

    tile_px: int = 512
    gsd_m_per_px: float = 0.02              # 2 cm/px — typical WINMOL UAV orthomosaic
    n_stems: Tuple[int, int] = (1, 6)
    stem_length_m: Tuple[float, float] = (2.0, 18.0)
    stem_diameter_m: Tuple[float, float] = (0.10, 0.60)
    bend_frac: Tuple[float, float] = (0.0, 0.08)     # sagitta as a fraction of length
    taper: Tuple[float, float] = (0.45, 0.95)
    burial: Tuple[float, float] = (0.0, 0.45)
    p_crossing: float = 0.35                # chance a stem rests on top of another
    p_root_plate: float = 0.25
    p_broken_end: float = 0.55
    branch_stubs: Tuple[int, int] = (0, 5)
    sun_elevation_deg: Tuple[float, float] = (15.0, 75.0)
    sun_azimuth_deg: Tuple[float, float] = (0.0, 360.0)
    sun_strength: Tuple[float, float] = (2.0, 6.0)
    canopy_shadow: Tuple[float, float] = (0.0, 0.9)
    leaf_density: Tuple[float, float] = (0.0, 1.0)
    branch_density: Tuple[float, float] = (0.0, 0.6)
    stone_density: Tuple[float, float] = (0.0, 0.3)
    over_stem_fraction: Tuple[float, float] = (0.0, 0.35)
    ground_brightness: Tuple[float, float] = (0.7, 1.3)
    ground_textures: Sequence[str] = field(default_factory=lambda: ("procedural",))
    # nadir camera optics; altitude is solved from these to hit the target GSD
    focal_mm: float = 24.0
    sensor_mm: float = 36.0


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def solve_camera(cfg):
    """Nadir camera whose footprint is exactly tile_px * gsd metres.

    Pinhole similar triangles: footprint / altitude = sensor / focal.
    """
    footprint = cfg.tile_px * cfg.gsd_m_per_px
    altitude = footprint * cfg.focal_mm / cfg.sensor_mm
    return CameraSpec(footprint_m=footprint, altitude_m=altitude,
                      focal_mm=cfg.focal_mm, sensor_mm=cfg.sensor_mm)


def _sample_stem(rng, cfg, half, allow_crossing):
    length = _u(rng, cfg.stem_length_m)
    diameter = _u(rng, cfg.stem_diameter_m)
    # centres stay inside the tile: a stem may (and often should) run off the edge,
    # which is what real tiles look like, but its centre must be in frame
    cx = float(rng.uniform(-half, half))
    cy = float(rng.uniform(-half, half))
    crossing = allow_crossing and rng.random() < cfg.p_crossing
    return StemSpec(
        center_xy_m=(cx, cy),
        length_m=length,
        diameter_m=diameter,
        azimuth_deg=float(rng.uniform(0.0, 180.0)),   # a stem is symmetric end-to-end
        bend_m=_u(rng, cfg.bend_frac) * length,
        taper=_u(rng, cfg.taper),
        # a crossing stem rests on the one below, so it sits ~a diameter up and
        # cannot also be buried
        elevation_m=diameter * float(rng.uniform(0.4, 1.0)) if crossing else 0.0,
        burial=0.0 if crossing else _u(rng, cfg.burial),
        root_plate=bool(rng.random() < cfg.p_root_plate),
        broken_end=bool(rng.random() < cfg.p_broken_end),
        branch_stubs=int(rng.integers(cfg.branch_stubs[0], cfg.branch_stubs[1] + 1)),
        bark_hue_shift=float(rng.uniform(-1.0, 1.0)),
        bark_darkness=float(rng.random()),
    )


def sample_scene(cfg, seed):
    """Draw one reproducible SceneSpec. Same (cfg, seed) -> identical spec."""
    rng = np.random.default_rng(seed)
    half = cfg.tile_px * cfg.gsd_m_per_px / 2.0

    ground = GroundSpec(
        texture_key=str(cfg.ground_textures[int(rng.integers(len(cfg.ground_textures)))]),
        rotation_deg=float(rng.uniform(0.0, 360.0)),
        flip=bool(rng.random() < 0.5),
        brightness=_u(rng, cfg.ground_brightness),
    )
    sun = SunSpec(
        elevation_deg=_u(rng, cfg.sun_elevation_deg),
        azimuth_deg=_u(rng, cfg.sun_azimuth_deg),
        strength=_u(rng, cfg.sun_strength),
        canopy_shadow=_u(rng, cfg.canopy_shadow),
    )
    clutter = ClutterSpec(
        leaf_density=_u(rng, cfg.leaf_density),
        branch_density=_u(rng, cfg.branch_density),
        stone_density=_u(rng, cfg.stone_density),
        over_stem_fraction=_u(rng, cfg.over_stem_fraction),
    )
    n = int(rng.integers(cfg.n_stems[0], cfg.n_stems[1] + 1))
    # the first stem has nothing to rest on, so crossing starts from the second
    stems = [_sample_stem(rng, cfg, half, allow_crossing=(i > 0)) for i in range(n)]

    return SceneSpec(seed=int(seed), tile_px=cfg.tile_px, gsd_m_per_px=cfg.gsd_m_per_px,
                     ground=ground, sun=sun, clutter=clutter, camera=solve_camera(cfg),
                     stems=stems)
