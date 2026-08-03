"""Engine-agnostic description of one synthetic scene.

Pure data: no numpy, no bpy, no behaviour beyond (de)serialization. The sampler
produces these, the renderer consumes them, and the (future) refinement loop
mutates them — keeping the description separate from both is what lets the
sampler be tested without a renderer and the renderer be swapped without
touching sampling.

All lengths are metres and all angles degrees; the renderer never guesses a
scale. Coordinates are tile-local: (0, 0) is the tile centre, +x east, +y north.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StemSpec:
    """One lying stem: a tapered, bent cylinder resting on (or above) the ground."""

    center_xy_m: Tuple[float, float]
    length_m: float
    diameter_m: float
    azimuth_deg: float              # heading in the ground plane
    bend_m: float                   # lateral sagitta of the backbone curve; 0 = straight
    taper: float                    # tip diameter as a fraction of butt diameter
    elevation_m: float              # 0 = on the ground; >0 = resting on other debris
    burial: float                   # fraction of the diameter sunk into the ground
    root_plate: bool                # uprooted stems carry a root disc at the butt
    broken_end: bool                # splintered rather than sawn tip
    branch_stubs: int
    bark_hue_shift: float           # [-1, 1] shader jitter
    bark_darkness: float            # [0, 1] shader jitter


@dataclass(frozen=True)
class ClutterSpec:
    """Scattered foliage/branches/stones; some of it deliberately occludes stems."""

    leaf_density: float
    branch_density: float
    stone_density: float
    over_stem_fraction: float       # share of clutter placed on top of stems


@dataclass(frozen=True)
class SunSpec:
    elevation_deg: float
    azimuth_deg: float
    strength: float
    canopy_shadow: float            # 0 = open sky, 1 = heavy dappled shade


@dataclass(frozen=True)
class GroundSpec:
    texture_key: str                # crop identifier, or "procedural"
    rotation_deg: float
    flip: bool
    brightness: float


@dataclass(frozen=True)
class CameraSpec:
    """Nadir camera solved to a target ground sampling distance."""

    footprint_m: float              # ground width covered by the tile
    altitude_m: float
    focal_mm: float
    sensor_mm: float


@dataclass(frozen=True)
class SceneSpec:
    seed: int
    tile_px: int
    gsd_m_per_px: float
    ground: GroundSpec
    sun: SunSpec
    clutter: ClutterSpec
    camera: CameraSpec
    stems: List[StemSpec] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_json(self, indent=1):
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        return cls(
            seed=d["seed"],
            tile_px=d["tile_px"],
            gsd_m_per_px=d["gsd_m_per_px"],
            ground=GroundSpec(**d["ground"]),
            sun=SunSpec(**d["sun"]),
            clutter=ClutterSpec(**d["clutter"]),
            camera=CameraSpec(**d["camera"]),
            # JSON has no tuples; restore them so equality with a freshly
            # sampled spec holds
            stems=[StemSpec(**{**s, "center_xy_m": tuple(s["center_xy_m"])})
                   for s in d["stems"]],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )
