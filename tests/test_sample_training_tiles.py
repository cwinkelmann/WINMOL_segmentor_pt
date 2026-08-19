"""The tiling rules that make a ~1%-stem corpus trainable are the thing to pin.

Everything here builds its own tiny GeoTIFF and shapefiles in tmp_path, so the
tests never touch the real orthomosaics (which are 100-1900 megapixels).
"""
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
shapely = pytest.importorskip("shapely")

from scripts.sample_training_tiles import sample_tiles  # noqa: E402

CRS = "EPSG:25833"
GSD = 0.02          # 2 cm/px, in the range the real orthos sit
ORIGIN = (400000.0, 6000000.0)


def _write_ortho(path, size_m=60.0, gsd=GSD, blank_edge_m=0.0):
    n = int(size_m / gsd)
    transform = rasterio.transform.from_origin(ORIGIN[0], ORIGIN[1], gsd, gsd)
    rng = np.random.default_rng(0)
    # mid-grey noise: never black, so nodata rejection only fires where we make it
    data = rng.integers(60, 200, size=(3, n, n), dtype="uint8")
    if blank_edge_m:
        k = int(blank_edge_m / gsd)
        data[:, :k, :] = 0
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3,
                       dtype="uint8", crs=CRS, transform=transform) as dst:
        dst.write(data)


def _write_polygons(path, polys, props=None):
    schema = {"geometry": "Polygon", "properties": {"id": "int", "Species": "str"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=CRS, schema=schema) as dst:
        for i, poly in enumerate(polys):
            p = (props or [{}] * len(polys))[i]
            dst.write({"geometry": shapely.geometry.mapping(poly),
                       "properties": {"id": p.get("id", i + 1),
                                      "Species": p.get("Species", "GFI")}})


def _stem(x, y, length=3.0, width=0.4, angle=0.0):
    """A stem-shaped polygon: the corpus median is 3.0 m x 0.43 m."""
    from shapely.affinity import rotate
    from shapely.geometry import box
    return rotate(box(x - length / 2, y - width / 2, x + length / 2, y + width / 2), angle)


@pytest.fixture
def site(tmp_path):
    ortho = tmp_path / "site_ortho.tif"
    _write_ortho(str(ortho))
    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0

    from shapely.geometry import box
    # a 40 m AOI leaves room for a 15 m footprint plus its 10.6 m inward buffer
    _write_polygons(str(tmp_path / "aoi.shp"), [box(cx - 20, cy - 20, cx + 20, cy + 20)])
    # a dense mat of stems, so nearly every footprint clears the min-stem test
    stems = [_stem(cx + dx, cy + dy, angle=17 * (dx + dy))
             for dx in range(-18, 19, 3) for dy in range(-18, 19, 3)]
    _write_polygons(str(tmp_path / "stems.shp"), stems)
    return {"ortho": str(ortho), "stems": str(tmp_path / "stems.shp"),
            "aoi": str(tmp_path / "aoi.shp"), "tmp": tmp_path}


def test_writes_loader_convention_pairs(site, tmp_path):
    out = tmp_path / "ds"
    stats = sample_tiles(site["ortho"], site["stems"], site["aoi"], str(out),
                         limit=6, seed=1, quiet=True)

    assert stats["written"] == 6
    imgs = sorted(os.listdir(out / "train"))
    masks = sorted(os.listdir(out / "mask"))
    assert imgs == [f"train{n}.jpeg" for n in range(1, 7)]
    assert masks == [f"mask{n}.gif" for n in range(1, 7)]


def test_tiles_are_512_and_masks_are_binary(site, tmp_path):
    from PIL import Image

    out = tmp_path / "ds"
    sample_tiles(site["ortho"], site["stems"], site["aoi"], str(out),
                 limit=3, seed=1, quiet=True)

    for n in range(1, 4):
        img = Image.open(out / "train" / f"train{n}.jpeg")
        msk = Image.open(out / "mask" / f"mask{n}.gif")
        assert img.size == (512, 512)
        assert msk.size == (512, 512)
        assert set(np.unique(np.asarray(msk.convert("L")))) <= {0, 255}


def test_min_stem_frac_rejects_sparse_tiles(site, tmp_path):
    """The 0.5% floor is what makes the dataset stem-dense; raising it must bite."""
    lax = sample_tiles(site["ortho"], site["stems"], site["aoi"],
                       str(tmp_path / "lax"), min_stem_frac=0.001, seed=3, quiet=True)
    strict = sample_tiles(site["ortho"], site["stems"], site["aoi"],
                          str(tmp_path / "strict"), min_stem_frac=0.30, seed=3, quiet=True)

    assert strict["too_few_stems"] > lax["too_few_stems"]
    assert strict["written"] < lax["written"]
    assert lax["mean_stem_coverage"] > 0


def test_sparse_stems_never_yield_a_tile_below_the_floor(tmp_path):
    """The floor must hold when stems are sparse enough that rotation decides the answer.

    The dense fixture cannot catch a wrong rotation sign: every square contains stems
    either way. With widely spaced stems the checked square and the cropped square
    disagree, and measuring the world footprint instead of the finished mask let 3.2%
    of real Campus tiles through below the floor -- 21 of them completely empty.
    """
    from PIL import Image
    from shapely.geometry import box

    ortho = tmp_path / "sparse_ortho.tif"
    _write_ortho(str(ortho))
    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    _write_polygons(str(tmp_path / "aoi.shp"), [box(cx - 20, cy - 20, cx + 20, cy + 20)])
    # ~8 m apart: most rotated footprints catch one stem, many catch none
    _write_polygons(str(tmp_path / "sparse.shp"),
                    [_stem(cx + dx, cy + dy, angle=31 * (dx - dy))
                     for dx in (-16, -8, 0, 8, 16) for dy in (-16, -8, 0, 8, 16)])

    floor = 0.01
    stats = sample_tiles(str(ortho), str(tmp_path / "sparse.shp"), str(tmp_path / "aoi.shp"),
                         str(tmp_path / "ds"), min_stem_frac=floor, limit=40, seed=11,
                         quiet=True)
    assert stats["written"] > 5, "fixture produced too few tiles to be a real check"

    bad = []
    for n in range(1, stats["written"] + 1):
        cov = (np.asarray(Image.open(tmp_path / "ds" / "mask" / f"mask{n}.gif").convert("L"))
               > 0).mean()
        if cov < floor:
            bad.append((n, cov))
    assert not bad, (f"{len(bad)} of {stats['written']} written tiles fall below the "
                     f"{100*floor:.1f}% floor, e.g. {bad[:3]}")


def test_every_written_tile_clears_the_stem_floor(site, tmp_path):
    from PIL import Image

    out = tmp_path / "ds"
    stats = sample_tiles(site["ortho"], site["stems"], site["aoi"], str(out),
                         min_stem_frac=0.05, limit=8, seed=5, quiet=True)
    assert stats["written"] > 0
    for n in range(1, stats["written"] + 1):
        cov = (np.asarray(Image.open(out / "mask" / f"mask{n}.gif").convert("L")) > 0).mean()
        # measured on the rasterized tile, so allow for resampling at the edges
        assert cov > 0.04, f"tile {n} has {100 * cov:.2f}% stem, below the 5% floor"


def test_same_seed_reproduces_the_same_tiles(site, tmp_path):
    a = sample_tiles(site["ortho"], site["stems"], site["aoi"], str(tmp_path / "a"),
                     limit=4, seed=7, quiet=True)
    b = sample_tiles(site["ortho"], site["stems"], site["aoi"], str(tmp_path / "b"),
                     limit=4, seed=7, quiet=True)

    # `manifest` is an output path and differs between the two directories by
    # construction; everything describing the sampling itself must match
    volatile = {"manifest"}
    assert {k: v for k, v in a.items() if k not in volatile} == \
           {k: v for k, v in b.items() if k not in volatile}
    for n in range(1, 5):
        assert ((tmp_path / "a" / "mask" / f"mask{n}.gif").read_bytes()
                == (tmp_path / "b" / "mask" / f"mask{n}.gif").read_bytes())


def test_sampling_stays_inside_the_annotated_area(site, tmp_path):
    """Tiles from outside the digitized windthrow polygon would be false negatives."""
    import fiona as _fiona
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with _fiona.open(site["aoi"]) as src:
        aoi = unary_union([shape(f["geometry"]) for f in src])

    # a footprint centred anywhere in the buffered area is inside at any rotation;
    # assert the buffer is what the geometry demands rather than the R script's 11
    from scripts.sample_training_tiles import _footprint, _random_points
    import numpy as _np

    inner = aoi.buffer(-(15.0 * _np.sqrt(2) / 2))
    rng = _np.random.default_rng(0)
    for (cx, cy), ang in zip(_random_points(inner, 40, rng), rng.uniform(-179, 180, 40)):
        assert aoi.contains(_footprint(cx, cy, 15.0, ang).buffer(-1e-6))


def test_nodata_collar_is_rejected(site, tmp_path):
    """Orthos have black collars; a tile of black pixels is not forest."""
    ortho = site["tmp"] / "collar_ortho.tif"
    _write_ortho(str(ortho), blank_edge_m=45.0)     # blacks out most of the raster

    stats = sample_tiles(str(ortho), site["stems"], site["aoi"], str(tmp_path / "ds"),
                         limit=5, seed=2, quiet=True)
    assert stats["nodata"] > 0


def test_species_filter_is_case_insensitive(site, tmp_path):
    """The corpus spells beech both 'RBU' and 'rBU'; a case-sensitive match drops 15."""
    from shapely.geometry import box

    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    stems = tmp_path / "mixed.shp"
    _write_polygons(str(stems),
                    [_stem(cx + dx, cy + dy) for dx in range(-9, 10, 3)
                     for dy in range(-9, 10, 3)],
                    [{"Species": "rBU" if (i % 2) else "RBU"} for i in range(49)])

    stats = sample_tiles(site["ortho"], str(stems), site["aoi"], str(tmp_path / "ds"),
                         species={"RBU"}, min_stem_frac=0.001, limit=3, seed=1, quiet=True)
    assert stats["written"] > 0


def test_refuses_an_aoi_too_small_for_the_footprint(site, tmp_path):
    from shapely.geometry import box

    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    tiny = tmp_path / "tiny_aoi.shp"
    _write_polygons(str(tiny), [box(cx - 4, cy - 4, cx + 4, cy + 4)])

    with pytest.raises(SystemExit, match="shrinks to nothing"):
        sample_tiles(site["ortho"], site["stems"], str(tiny), str(tmp_path / "ds"),
                     limit=2, quiet=True)


def test_self_intersecting_polygons_are_repaired_not_fatal(site, tmp_path):
    """Hand-traced stems self-intersect; GEOS aborts on the first intersection test.

    The Campus shapefile contains such rings, and before the loader repaired them a
    sampling run died with `TopologyException: side location conflict` partway through.
    """
    from shapely.geometry import Polygon

    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    # a bowtie: the classic self-intersection a hand-drawn outline produces
    bowties = [Polygon([(cx + dx, cy + dy), (cx + dx + 3, cy + dy + 1),
                        (cx + dx, cy + dy + 1), (cx + dx + 3, cy + dy)])
               for dx in range(-12, 13, 3) for dy in range(-12, 13, 3)]
    assert not bowties[0].is_valid, "the fixture must actually be invalid"

    bad = tmp_path / "bowtie.shp"
    _write_polygons(str(bad), bowties)

    stats = sample_tiles(site["ortho"], str(bad), site["aoi"], str(tmp_path / "ds"),
                         min_stem_frac=0.0005, limit=4, seed=1, quiet=True)
    assert stats["written"] > 0


def test_spatial_blocks_partition_the_site_without_overlap(site, tmp_path):
    """Blocks assigned to different splits must not share a single pixel.

    With three beech sites, holding one out removes a whole acquisition — its
    phenology, colour cast and GSD — so the score measures domain transfer instead of
    segmentation. Block splitting keeps every split spanning the site; it is only
    honest if the blocks are genuinely disjoint after the intra-block buffer.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    from scripts.sample_training_tiles import _spatial_blocks

    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    # 120 m of AOI at 30 m blocks -> 16 cells, enough for all three splits to be
    # non-empty; 30 m clears the 21.2 m the buffer needs on both sides
    area = box(cx - 60, cy - 60, cx + 60, cy + 60)
    buf = 15.0 * np.sqrt(2) / 2
    fr = {"train": 0.6, "val": 0.2, "test": 0.2}

    regions = {s: _spatial_blocks(area, 30.0, s, fr, 7, buf, quiet=True)
               for s in ("train", "val", "test")}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                shared = regions[a].intersection(regions[b]).area
                assert shared == 0, f"{a} and {b} share {shared:.3f} m²"

    # and the same seed must reproduce the same assignment, or the three separate
    # invocations that build train/val/test would disagree about who owns a block
    again = _spatial_blocks(area, 30.0, "train", fr, 7, buf, quiet=True)
    assert again.equals(regions["train"])


def test_block_size_must_exceed_the_buffer(site, tmp_path):
    with pytest.raises(SystemExit, match="leaves nothing after"):
        sample_tiles(site["ortho"], site["stems"], site["aoi"], str(tmp_path / "ds"),
                     block_size_m=10.0, split="train", limit=2, quiet=True)


def test_every_written_tile_is_recorded_in_the_manifest(site, tmp_path):
    """A tile with no provenance cannot be traced back to the ground it came from."""
    import json

    out = tmp_path / "ds"
    stats = sample_tiles(site["ortho"], site["stems"], site["aoi"], str(out),
                         limit=5, seed=1, quiet=True)

    recs = [json.loads(l) for l in (out / "tiles.jsonl").read_text().splitlines() if l.strip()]
    assert len(recs) == stats["written"]
    assert [r["n"] for r in recs] == list(range(1, stats["written"] + 1))
    for r in recs:
        assert r["site"] == "site", "site is derived from the ortho filename"
        assert r["crs"] == CRS
        assert os.path.isabs(r["ortho"]) and os.path.exists(r["ortho"])
        # the recorded centre must be inside the AOI it was drawn from
        assert ORIGIN[0] < r["centre_x"] < ORIGIN[0] + 60
        assert ORIGIN[1] - 60 < r["centre_y"] < ORIGIN[1]
        assert 0.0 <= r["stem_frac"] <= 1.0


def test_manifest_appends_across_sites_rather_than_overwriting(site, tmp_path):
    """Pooling sites into one dataset must not lose the earlier site's provenance."""
    import json

    out = tmp_path / "ds"
    a = sample_tiles(site["ortho"], site["stems"], site["aoi"], str(out),
                     limit=3, seed=1, quiet=True)
    sample_tiles(site["ortho"], site["stems"], site["aoi"], str(out),
                 limit=3, seed=2, start_index=a["next_index"], quiet=True)

    recs = [json.loads(l) for l in (out / "tiles.jsonl").read_text().splitlines() if l.strip()]
    assert len(recs) == 6
    assert [r["n"] for r in recs] == [1, 2, 3, 4, 5, 6]


# test_locate_tile_round_trips_a_tile_to_its_source
# moved to the private helper repo with its subject (scripts/locate_tile.py).

def test_an_empty_split_names_the_split_it_would_silently_drop(site, tmp_path):
    """A split with no usable ground must say so, not leave a hole in the dataset.

    Kaufland's val split hit this: its two 30 m blocks were edge slivers that vanished
    under the buffer, so the site contributed zero val tiles and the driver loop moved on
    without comment.
    """
    from shapely.geometry import box

    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    # a long thin AOI: blocks exist, but nothing survives the inward buffer
    sliver = tmp_path / "sliver.shp"
    _write_polygons(str(sliver), [box(cx - 40, cy - 6, cx + 40, cy + 6)])

    with pytest.raises(SystemExit, match=r"would contribute NO tiles"):
        sample_tiles(site["ortho"], site["stems"], str(sliver), str(tmp_path / "ds"),
                     block_size_m=30.0, split="val", limit=2, quiet=True)


def test_native_px_keeps_the_orthos_own_resolution(site, tmp_path):
    """Native mode must not resample: 1024 source pixels in, 1024 px tile out.

    A fixed-metre footprint normalises every site to one ground resolution, which is
    exactly the scale diversity multi-scale training needs to keep.
    """
    import json

    from PIL import Image

    out = tmp_path / "ds"
    sample_tiles(site["ortho"], site["stems"], site["aoi"], str(out),
                 native_px=256, limit=3, seed=1, quiet=True)

    img = Image.open(out / "train" / "train1.jpeg")
    assert img.size == (256, 256)
    rec = json.loads((out / "tiles.jsonl").read_text().splitlines()[0])
    # the fixture ortho is 2 cm/px, so 256 px is 5.12 m and the tile stays at 2 cm/px
    assert rec["extent_m"] == pytest.approx(256 * GSD, rel=1e-6)
    assert rec["gsd_m_per_px"] == pytest.approx(GSD, rel=1e-3)


@pytest.mark.parametrize("mode", ["on", "off"])
def test_antialias_switch_changes_pixels_only_when_downsampling(site, tmp_path, mode):
    """The flag must actually change the imagery, and must not touch the mask.

    The Analyzer resizes with skimage order=3 and anti_aliasing=False on its CPU path
    while its stream path uses GDAL cubic, which filters. Those disagree only when a tile
    is downsampled, which is exactly when aliasing appears.
    """
    from PIL import Image

    out = tmp_path / f"ds_{mode}"
    # 15 m at the fixture's 2 cm/px reads a 1061 px window and resizes to 512: a
    # downsample, so anti-aliasing is in play
    sample_tiles(site["ortho"], site["stems"], site["aoi"], str(out),
                 extent_m=15.0, limit=2, seed=1, quiet=True, antialias=mode)
    img = np.asarray(Image.open(out / "train" / "train1.jpeg").convert("RGB"), np.int16)
    msk = np.asarray(Image.open(out / "mask" / "mask1.gif").convert("L"))
    assert img.shape == (512, 512, 3)
    assert set(np.unique(msk)) <= {0, 255}, "the mask must stay binary either way"


def test_antialiased_and_aliased_tiles_actually_differ(site, tmp_path):
    from PIL import Image

    for mode in ("on", "off"):
        sample_tiles(site["ortho"], site["stems"], site["aoi"], str(tmp_path / mode),
                     extent_m=15.0, limit=1, seed=1, quiet=True, antialias=mode)
    a = np.asarray(Image.open(tmp_path / "on" / "train" / "train1.jpeg"), np.int16)
    b = np.asarray(Image.open(tmp_path / "off" / "train" / "train1.jpeg"), np.int16)
    diff = np.abs(a - b)
    # The fixture is smooth noise, so the two filters land close together (mean ~0.45
    # levels). What matters is that they differ at all and that some pixels differ
    # visibly; on real imagery with thin bright stems the gap is far larger. An earlier
    # version asserted mean > 0.5, which was a guessed threshold, not a measured one.
    assert diff.max() >= 2, "anti-aliasing must change some pixels materially"
    assert diff.mean() > 0.1, "the switch must change the imagery it produces"
