"""The geo pipeline: tile sampling from an orthomosaic, leak-free block splits, fold
composition, annotation rasterization, parallel extraction, and prediction plotting.

Absorbs tests/test_sample_training_tiles.py, tests/test_block_splits.py,
tests/test_make_splits.py, tests/test_compose_folds.py, tests/test_rasterize_annotations.py,
tests/test_extract_parallel.py and tests/test_plot_inference.py. The synthetic-site builder
(``_write_ortho`` / ``_write_polygons`` / ``_stem``, and the ``CRS`` / ``GSD`` / ``ORIGIN``
constants) moved to tests/geo_helpers.py, which tests/conftest.py wraps as the ``geo_site``
fixture, shared with any future geo test. It is a plain sibling module rather than
conftest itself so that this import works under any pytest import mode. The private helper
repo keeps its own copy of the same builder deliberately -- test suites are not importable
packages across repo boundaries.
"""
import json
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
shapely = pytest.importorskip("shapely")

from tests.geo_helpers import CRS, GSD, ORIGIN, _stem, _write_ortho, _write_polygons  # noqa: E402
from winmol_unet.geo.sample import sample_tiles  # noqa: E402

# =============================================================================
# Tile sampling from an orthomosaic
# (from tests/test_sample_training_tiles.py; synthetic-site builder now in
# tests/conftest.py as the `geo_site` fixture)
# =============================================================================


def test_writes_loader_convention_pairs(geo_site, tmp_path):
    out = tmp_path / "ds"
    stats = sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(out),
                         limit=6, seed=1, quiet=True)

    assert stats["written"] == 6
    imgs = sorted(os.listdir(out / "train"))
    masks = sorted(os.listdir(out / "mask"))
    assert imgs == [f"train{n}.jpeg" for n in range(1, 7)]
    assert masks == [f"mask{n}.gif" for n in range(1, 7)]


def test_tiles_are_512_and_masks_are_binary(geo_site, tmp_path):
    from PIL import Image

    out = tmp_path / "ds"
    sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(out),
                 limit=3, seed=1, quiet=True)

    for n in range(1, 4):
        img = Image.open(out / "train" / f"train{n}.jpeg")
        msk = Image.open(out / "mask" / f"mask{n}.gif")
        assert img.size == (512, 512)
        assert msk.size == (512, 512)
        assert set(np.unique(np.asarray(msk.convert("L")))) <= {0, 255}


def test_min_stem_frac_rejects_sparse_tiles(geo_site, tmp_path):
    """The 0.5% floor is what makes the dataset stem-dense; raising it must bite."""
    lax = sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"],
                       str(tmp_path / "lax"), min_stem_frac=0.001, seed=3, quiet=True)
    strict = sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"],
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


def test_every_written_tile_clears_the_stem_floor(geo_site, tmp_path):
    from PIL import Image

    out = tmp_path / "ds"
    stats = sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(out),
                         min_stem_frac=0.05, limit=8, seed=5, quiet=True)
    assert stats["written"] > 0
    for n in range(1, stats["written"] + 1):
        cov = (np.asarray(Image.open(out / "mask" / f"mask{n}.gif").convert("L")) > 0).mean()
        # measured on the rasterized tile, so allow for resampling at the edges
        assert cov > 0.04, f"tile {n} has {100 * cov:.2f}% stem, below the 5% floor"


def test_same_seed_reproduces_the_same_tiles(geo_site, tmp_path):
    a = sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(tmp_path / "a"),
                     limit=4, seed=7, quiet=True)
    b = sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(tmp_path / "b"),
                     limit=4, seed=7, quiet=True)

    # `manifest` is an output path and differs between the two directories by
    # construction; everything describing the sampling itself must match
    volatile = {"manifest"}
    assert {k: v for k, v in a.items() if k not in volatile} == \
           {k: v for k, v in b.items() if k not in volatile}
    for n in range(1, 5):
        assert ((tmp_path / "a" / "mask" / f"mask{n}.gif").read_bytes()
                == (tmp_path / "b" / "mask" / f"mask{n}.gif").read_bytes())


def test_sampling_stays_inside_the_annotated_area(geo_site, tmp_path):
    """Tiles from outside the digitized windthrow polygon would be false negatives."""
    import fiona as _fiona
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with _fiona.open(geo_site["aoi"]) as src:
        aoi = unary_union([shape(f["geometry"]) for f in src])

    # a footprint centred anywhere in the buffered area is inside at any rotation;
    # assert the buffer is what the geometry demands rather than the R script's 11
    from winmol_unet.geo.sample import _footprint, _random_points
    import numpy as _np

    inner = aoi.buffer(-(15.0 * _np.sqrt(2) / 2))
    rng = _np.random.default_rng(0)
    for (cx, cy), ang in zip(_random_points(inner, 40, rng), rng.uniform(-179, 180, 40)):
        assert aoi.contains(_footprint(cx, cy, 15.0, ang).buffer(-1e-6))


def test_nodata_collar_is_rejected(geo_site, tmp_path):
    """Orthos have black collars; a tile of black pixels is not forest."""
    ortho = geo_site["tmp"] / "collar_ortho.tif"
    _write_ortho(str(ortho), blank_edge_m=45.0)     # blacks out most of the raster

    stats = sample_tiles(str(ortho), geo_site["stems"], geo_site["aoi"], str(tmp_path / "ds"),
                         limit=5, seed=2, quiet=True)
    assert stats["nodata"] > 0


def test_sample_tiles_species_filter_is_case_insensitive(geo_site, tmp_path):
    """The corpus spells beech both 'RBU' and 'rBU'; a case-sensitive match drops 15."""
    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    stems = tmp_path / "mixed.shp"
    _write_polygons(str(stems),
                    [_stem(cx + dx, cy + dy) for dx in range(-9, 10, 3)
                     for dy in range(-9, 10, 3)],
                    [{"Species": "rBU" if (i % 2) else "RBU"} for i in range(49)])

    stats = sample_tiles(geo_site["ortho"], str(stems), geo_site["aoi"], str(tmp_path / "ds"),
                         species={"RBU"}, min_stem_frac=0.001, limit=3, seed=1, quiet=True)
    assert stats["written"] > 0


def test_refuses_an_aoi_too_small_for_the_footprint(geo_site, tmp_path):
    from shapely.geometry import box

    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    tiny = tmp_path / "tiny_aoi.shp"
    _write_polygons(str(tiny), [box(cx - 4, cy - 4, cx + 4, cy + 4)])

    with pytest.raises(SystemExit, match="shrinks to nothing"):
        sample_tiles(geo_site["ortho"], geo_site["stems"], str(tiny), str(tmp_path / "ds"),
                     limit=2, quiet=True)


def test_self_intersecting_polygons_are_repaired_not_fatal(geo_site, tmp_path):
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

    stats = sample_tiles(geo_site["ortho"], str(bad), geo_site["aoi"], str(tmp_path / "ds"),
                         min_stem_frac=0.0005, limit=4, seed=1, quiet=True)
    assert stats["written"] > 0


def test_spatial_blocks_partition_the_site_without_overlap(geo_site, tmp_path):
    """Blocks assigned to different splits must not share a single pixel.

    With three beech sites, holding one out removes a whole acquisition -- its
    phenology, colour cast and GSD -- so the score measures domain transfer instead of
    segmentation. Block splitting keeps every split spanning the site; it is only
    honest if the blocks are genuinely disjoint after the intra-block buffer.
    """
    from shapely.geometry import box

    from winmol_unet.geo.sample import _spatial_blocks

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


def test_block_size_must_exceed_the_buffer(geo_site, tmp_path):
    with pytest.raises(SystemExit, match="leaves nothing after"):
        sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(tmp_path / "ds"),
                     block_size_m=10.0, split="train", limit=2, quiet=True)


def test_every_written_tile_is_recorded_in_the_manifest(geo_site, tmp_path):
    """A tile with no provenance cannot be traced back to the ground it came from."""
    out = tmp_path / "ds"
    stats = sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(out),
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


def test_manifest_appends_across_sites_rather_than_overwriting(geo_site, tmp_path):
    """Pooling sites into one dataset must not lose the earlier site's provenance."""
    out = tmp_path / "ds"
    a = sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(out),
                     limit=3, seed=1, quiet=True)
    sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(out),
                 limit=3, seed=2, start_index=a["next_index"], quiet=True)

    recs = [json.loads(l) for l in (out / "tiles.jsonl").read_text().splitlines() if l.strip()]
    assert len(recs) == 6
    assert [r["n"] for r in recs] == [1, 2, 3, 4, 5, 6]


def test_an_empty_split_names_the_split_it_would_silently_drop(geo_site, tmp_path):
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
        sample_tiles(geo_site["ortho"], geo_site["stems"], str(sliver), str(tmp_path / "ds"),
                     block_size_m=30.0, split="val", limit=2, quiet=True)


def test_native_px_keeps_the_orthos_own_resolution(geo_site, tmp_path):
    """Native mode must not resample: 1024 source pixels in, 1024 px tile out.

    A fixed-metre footprint normalises every site to one ground resolution, which is
    exactly the scale diversity multi-scale training needs to keep.
    """
    from PIL import Image

    out = tmp_path / "ds"
    sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(out),
                 native_px=256, limit=3, seed=1, quiet=True)

    img = Image.open(out / "train" / "train1.jpeg")
    assert img.size == (256, 256)
    rec = json.loads((out / "tiles.jsonl").read_text().splitlines()[0])
    # the fixture ortho is 2 cm/px, so 256 px is 5.12 m and the tile stays at 2 cm/px
    assert rec["extent_m"] == pytest.approx(256 * GSD, rel=1e-6)
    assert rec["gsd_m_per_px"] == pytest.approx(GSD, rel=1e-3)


@pytest.mark.parametrize("mode", ["on", "off"])
def test_antialias_switch_changes_pixels_only_when_downsampling(geo_site, tmp_path, mode):
    """The flag must actually change the imagery, and must not touch the mask.

    The Analyzer resizes with skimage order=3 and anti_aliasing=False on its CPU path
    while its stream path uses GDAL cubic, which filters. Those disagree only when a tile
    is downsampled, which is exactly when aliasing appears.
    """
    from PIL import Image

    out = tmp_path / f"ds_{mode}"
    # 15 m at the fixture's 2 cm/px reads a 1061 px window and resizes to 512: a
    # downsample, so anti-aliasing is in play
    sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(out),
                 extent_m=15.0, limit=2, seed=1, quiet=True, antialias=mode)
    img = np.asarray(Image.open(out / "train" / "train1.jpeg").convert("RGB"), np.int16)
    msk = np.asarray(Image.open(out / "mask" / "mask1.gif").convert("L"))
    assert img.shape == (512, 512, 3)
    assert set(np.unique(msk)) <= {0, 255}, "the mask must stay binary either way"


def test_antialiased_and_aliased_tiles_actually_differ(geo_site, tmp_path):
    from PIL import Image

    for mode in ("on", "off"):
        sample_tiles(geo_site["ortho"], geo_site["stems"], geo_site["aoi"], str(tmp_path / mode),
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


# =============================================================================
# Leak-free block splits
# (from tests/test_block_splits.py and tests/test_make_splits.py)
# =============================================================================

from winmol_unet.geo import splits as make_splits  # noqa: E402
from winmol_unet.geo.splits import _long_axis, halve_aoi  # noqa: E402
from shapely.geometry import box  # noqa: E402


@pytest.fixture
def spy(monkeypatch):
    """Record which (site, split) pairs get sampled, without touching a raster."""
    calls = []

    def fake_sample(ortho, stems, aoi, out_dir, **kw):
        calls.append((ortho, kw.get("split")))
        return {"written": 3, "next_index": 1}

    monkeypatch.setattr(make_splits, "sample_tiles", fake_sample)
    monkeypatch.setattr(make_splits, "_clean_stems", lambda site, o, q: (site["stems"], None))
    return calls


def _splits_cfg(tmp_path, sites):
    p = tmp_path / "sites.json"
    p.write_text(json.dumps({"extent_m": 19.512, "tile_px": 666, "block_size_m": 60,
                             "split_fractions": {"train": 0.85, "val": 0.15, "test": 0.0},
                             "sites": sites}))
    return str(p)


def _splits_site(name, **kw):
    return {"name": name, "ortho": f"{name}.tif", "stems": f"{name}.shp",
            "aoi": f"{name}_aoi.shp", **kw}


def test_defaults_to_all_three_splits(tmp_path, spy):
    make_splits.run(_splits_cfg(tmp_path, [_splits_site("A")]), str(tmp_path / "out"), "blocks",
                    quiet=True)
    assert [s for _, s in spy] == ["train", "val", "test"]


def test_restricts_a_training_site_to_train_and_val(tmp_path, spy):
    """The whole point: no test blocks from a site that is not the held-out one."""
    make_splits.run(_splits_cfg(tmp_path, [_splits_site("A", block_splits=["train", "val"])]),
                    str(tmp_path / "out"), "blocks", quiet=True)
    assert [s for _, s in spy] == ["train", "val"]
    assert "test" not in [s for _, s in spy]


def test_top_level_key_applies_to_every_site(tmp_path, spy):
    p = tmp_path / "sites.json"
    p.write_text(json.dumps({"extent_m": 19.512, "tile_px": 666, "block_size_m": 60,
                             "block_splits": ["train", "val"],
                             "split_fractions": {"train": 0.85, "val": 0.15, "test": 0.0},
                             "sites": [_splits_site("A"), _splits_site("B")]}))
    make_splits.run(str(p), str(tmp_path / "out"), "blocks", quiet=True)
    assert [s for _, s in spy] == ["train", "val", "train", "val"]


def test_a_typo_fails_loudly(tmp_path, spy):
    """Silently ignoring 'traon' would build a fold with no training data."""
    with pytest.raises(SystemExit, match="block_splits"):
        make_splits.run(_splits_cfg(tmp_path, [_splits_site("A", block_splits=["traon"])]),
                        str(tmp_path / "out"), "blocks", quiet=True)


@pytest.mark.parametrize("strategy,extra", [("sites", {"split": "test"}), ("blocks", {})])
def test_tile_px_reaches_every_strategy(tmp_path, monkeypatch, strategy, extra):
    """`sites` and `halve` silently dropped tile_px, so a 666 px config produced 512 px
    tiles at 19.512 m -- 3.81 cm/px instead of 2.93. Mixed with block-built tiles it gave
    a training set at two resolutions and a test set at a third, and nothing complained.
    """
    seen = []

    def fake_sample(ortho, stems, aoi, out_dir, **kw):
        seen.append((kw.get("tile_px"), kw.get("antialias")))
        return {"written": 3, "next_index": 1}

    monkeypatch.setattr(make_splits, "sample_tiles", fake_sample)
    monkeypatch.setattr(make_splits, "_clean_stems", lambda site, o, q: (site["stems"], None))
    p = tmp_path / "sites.json"
    p.write_text(json.dumps({"extent_m": 19.512, "tile_px": 666, "block_size_m": 60,
                             "antialias": "on",
                             "split_fractions": {"train": 0.85, "val": 0.15, "test": 0.0},
                             "block_splits": ["train", "val"],
                             "sites": [_splits_site("A", **extra)]}))
    make_splits.run(str(p), str(tmp_path / "out"), strategy, quiet=True)
    assert seen, "no tiles sampled"
    for tile_px, antialias in seen:
        assert tile_px == 666, f"{strategy} dropped tile_px (got {tile_px})"
        assert antialias == "on", f"{strategy} dropped antialias (got {antialias})"


def test_halves_are_separated_by_the_full_buffer():
    """A tile centred in one half must not reach the other at any rotation."""
    aoi = box(0, 0, 120, 60)
    buf = 10.24 * np.sqrt(2) / 2
    a, b = halve_aoi(aoi, buf)

    assert not a.is_empty and not b.is_empty
    assert a.intersection(b).area == 0
    # the halves must be at least 2*buffer apart: buffer held out on each side of the cut
    assert a.distance(b) == pytest.approx(2 * buf, rel=0.02)


def test_cut_runs_across_the_long_axis_by_default():
    """Cutting a 120x60 AOI along its length halves the area; across it does not."""
    aoi = box(0, 0, 120, 60)
    a, b = halve_aoi(aoi, 5.0)

    # both halves should be wide in y and short in x -- i.e. the cut was vertical
    for half in (a, b):
        minx, miny, maxx, maxy = half.bounds
        assert (maxy - miny) > (maxx - minx), "cut should be across the long axis"
    assert a.area == pytest.approx(b.area, rel=0.02)


def test_forced_compass_axis_overrides_the_fitted_one():
    aoi = box(0, 0, 120, 60)
    a, _ = halve_aoi(aoi, 5.0, cut_axis="ns")
    minx, miny, maxx, maxy = a.bounds
    # cutting north-south splits the y extent, leaving a wide, short half
    assert (maxx - minx) > (maxy - miny)


def test_halving_preserves_almost_all_the_area_minus_the_gap():
    aoi = box(0, 0, 100, 50)
    buf = 4.0
    a, b = halve_aoi(aoi, buf)
    lost = aoi.area - a.area - b.area
    # the gap is 2*buf wide across the 50 m width
    assert lost == pytest.approx(2 * buf * 50, rel=0.05)


def test_long_axis_of_a_wide_rectangle_points_along_its_length():
    _, direction = _long_axis(box(0, 0, 100, 10))
    assert abs(direction[0]) > abs(direction[1])


def test_an_aoi_too_small_to_halve_is_reported_not_silently_emptied():
    """A 10 m AOI cannot survive a cut that holds 7.24 m out on each side."""
    tiny = box(0, 0, 10, 10)
    a, b = halve_aoi(tiny, 7.24)
    assert a.is_empty or b.is_empty, "caller must detect this and refuse"


def _splits_tiny_site(tmp_path):
    """A site whose stem shapefile contains a self-intersecting ring."""
    from shapely.geometry import Polygon, mapping

    crs, gsd = "EPSG:25833", 0.02
    ox, oy = 400000.0, 6000000.0
    n = int(80 / gsd)
    with rasterio.open(tmp_path / "o.tif", "w", driver="GTiff", width=n, height=n,
                       count=3, dtype="uint8", crs=crs,
                       transform=rasterio.transform.from_origin(ox, oy, gsd, gsd)) as dst:
        dst.write(np.full((3, n, n), 120, "uint8"))

    cx, cy = ox + 40, oy - 40
    bowtie = Polygon([(cx, cy), (cx + 3, cy + 1), (cx, cy + 1), (cx + 3, cy)])
    assert not bowtie.is_valid, "the fixture must actually be invalid"
    stems = [bowtie] + [box(cx + dx, cy + dy, cx + dx + 3, cy + dy + 0.4)
                        for dx in range(-24, 25, 4) for dy in range(-24, 25, 4)]
    schema = {"geometry": "Polygon", "properties": {"id": "int", "Species": "str"}}
    with fiona.open(tmp_path / "s.shp", "w", driver="ESRI Shapefile", crs=crs,
                    schema=schema) as dst:
        for i, g in enumerate(stems):
            dst.write({"geometry": mapping(g), "properties": {"id": i, "Species": "RBU"}})
    with fiona.open(tmp_path / "a.shp", "w", driver="ESRI Shapefile", crs=crs,
                    schema={"geometry": "Polygon", "properties": {"n": "int"}}) as dst:
        dst.write({"geometry": mapping(box(cx - 30, cy - 30, cx + 30, cy + 30)),
                   "properties": {"n": 1}})
    return {"name": "S", "ortho": str(tmp_path / "o.tif"),
            "stems": str(tmp_path / "s.shp"), "aoi": str(tmp_path / "a.shp")}


def test_pipeline_repairs_geometry_before_sampling(tmp_path):
    """fix -> sample -> split must be structural, not left to the caller.

    An invalid ring aborts GEOS on the first set operation, killing a sampling run
    partway through, so the repair cannot be optional or ordered by convention.
    """
    from winmol_unet.geo.splits import run

    site = _splits_tiny_site(tmp_path)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"extent_m": 10.24, "caps": {"train": 4, "test": 3},
                               "sites": [dict(site, halves=["train", "test"])]}))

    manifest = run(str(cfg), str(tmp_path / "ds"), "halve", quiet=True)

    rec = manifest["sites"][0]["geometry"]
    assert rec["repaired"] == 1, "the bowtie must have been repaired in step 1"
    assert rec["dropped"] == 0
    # the cleaned shapefile is an artefact, not just an in-memory fix
    assert (tmp_path / "ds" / "_clean" / "S.shp").exists()
    assert (tmp_path / "ds" / "_clean" / "S_geom.json").exists()
    assert manifest["sites"][0]["halves"]["train"]["tiles"] > 0


def test_skip_fix_bypasses_step_one(tmp_path):
    from winmol_unet.geo.splits import run

    site = _splits_tiny_site(tmp_path)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"extent_m": 10.24, "caps": {"train": 3, "test": 2},
                               "sites": [dict(site, halves=["train", "test"])]}))

    manifest = run(str(cfg), str(tmp_path / "ds"), "halve", skip_fix=True, quiet=True)
    assert manifest["sites"][0]["geometry"] is None
    assert not (tmp_path / "ds" / "_clean").exists()


# =============================================================================
# Fold composition
# (from tests/test_compose_folds.py)
# =============================================================================

from winmol_unet.geo.folds import _ids, compose, plan  # noqa: E402

FOLD_SITES = ["Campus", "Oberheide", "Bachsee", "Kaufland"]
FOLD_SPLITTABLE = ["Campus", "Oberheide"]


def _make_fold_tiles(d, n, start=1):
    os.makedirs(os.path.join(d, "train"), exist_ok=True)
    os.makedirs(os.path.join(d, "mask"), exist_ok=True)
    for i in range(start, start + n):
        open(os.path.join(d, "train", f"train{i}.jpeg"), "w").write(f"img{i}")
        open(os.path.join(d, "mask", f"mask{i}.gif"), "w").write(f"msk{i}")


@pytest.fixture
def corpus(tmp_path):
    """Two block-splittable sites (train+val) and two whole sites."""
    for s in FOLD_SPLITTABLE:
        _make_fold_tiles(tmp_path / "tv" / s / "train", 5)
        _make_fold_tiles(tmp_path / "tv" / s / "val", 2)
    for s in FOLD_SITES:
        _make_fold_tiles(tmp_path / "all" / s, 4)
    return tmp_path


def test_held_out_site_appears_only_in_test(corpus):
    for held in FOLD_SITES:
        sources = plan(FOLD_SITES, str(corpus / "all"), str(corpus / "tv"), FOLD_SPLITTABLE)[held]
        labels = {sp: [l for l, _ in v] for sp, v in sources.items()}
        assert labels["test"] == [held]
        assert held not in labels["train"] and held not in labels["val"]


def test_every_other_site_is_used_for_training(corpus):
    sources = plan(FOLD_SITES, str(corpus / "all"), str(corpus / "tv"), FOLD_SPLITTABLE)["Bachsee"]
    assert set(l for l, _ in sources["train"]) == {"Campus", "Oberheide", "Kaufland"}
    assert set(l for l, _ in sources["val"]) == {"Campus", "Oberheide"}


def test_renumbering_is_contiguous_and_loses_nothing(corpus, tmp_path):
    """Two sites both numbered from 1 must not collide when merged into one split."""
    sources = plan(FOLD_SITES, str(corpus / "all"), str(corpus / "tv"), FOLD_SPLITTABLE)["Bachsee"]
    out = tmp_path / "fold"
    counts = compose("Bachsee", sources, str(out))
    assert counts["train"] == 5 + 5 + 4        # Campus + Oberheide + Kaufland
    assert counts["val"] == 2 + 2
    assert counts["test"] == 4
    got = sorted(n for n, _ in _ids(str(out / "train")))
    assert got == list(range(1, counts["train"] + 1))


def test_provenance_traces_every_tile_back_to_its_site(corpus, tmp_path):
    sources = plan(FOLD_SITES, str(corpus / "all"), str(corpus / "tv"), FOLD_SPLITTABLE)["Kaufland"]
    out = tmp_path / "fold"
    compose("Kaufland", sources, str(out))
    rows = [json.loads(l) for l in open(out / "provenance.jsonl")]
    assert len(rows) == 5 + 5 + 4 + 2 + 2 + 4
    assert not any(r["source"].startswith("Kaufland:train") for r in rows)
    assert sum(r["source"] == "Kaufland:test" for r in rows) == 4


def test_missing_mask_refuses_rather_than_silently_shrinking(corpus, tmp_path):
    os.remove(corpus / "all" / "Bachsee" / "mask" / "mask2.gif")
    sources = plan(FOLD_SITES, str(corpus / "all"), str(corpus / "tv"), FOLD_SPLITTABLE)["Bachsee"]
    with pytest.raises(SystemExit, match="no mask2.gif"):
        compose("Bachsee", sources, str(tmp_path / "fold"))


def test_an_empty_split_fails_loudly(corpus, tmp_path):
    """Kaufland's val vanished under the buffer once and the driver moved on silently."""
    sources = {"train": [("A", str(corpus / "all" / "Campus"))], "val": [], "test": []}
    with pytest.raises(SystemExit, match="got no tiles"):
        compose("X", sources, str(tmp_path / "fold"))


# =============================================================================
# Annotation rasterization
# (from tests/test_rasterize_annotations.py)
# =============================================================================

from shapely.geometry import mapping as _rasterize_mapping  # noqa: E402
from winmol_unet.geo.rasterize import rasterize  # noqa: E402


def _rasterize_ortho(path, size_m=40.0, crs=CRS):
    n = int(size_m / GSD)
    transform = rasterio.transform.from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3,
                       dtype="uint8", crs=crs, transform=transform) as dst:
        dst.write(np.full((3, n, n), 120, "uint8"))


def _rasterize_stems(path, polys, props, crs=CRS):
    schema = {"geometry": "Polygon", "properties": {"id": "int", "Species": "str"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as dst:
        for poly, pr in zip(polys, props):
            dst.write({"geometry": _rasterize_mapping(poly), "properties": pr})


def _rasterize_stem(x, y, length=3.0, width=0.4):
    return box(x - length / 2, y - width / 2, x + length / 2, y + width / 2)


@pytest.fixture
def rasterize_site(tmp_path):
    _rasterize_ortho(str(tmp_path / "o.tif"))
    cx, cy = ORIGIN[0] + 20.0, ORIGIN[1] - 20.0
    # three polygons, two of which are fragments of ONE tree (id 7)
    polys = [_rasterize_stem(cx - 6, cy), _rasterize_stem(cx, cy),
             _rasterize_stem(cx + 8, cy + 5)]
    props = [{"id": 7, "Species": "RBU"}, {"id": 7, "Species": "rBU"},
             {"id": 9, "Species": "GFI"}]
    _rasterize_stems(str(tmp_path / "s.shp"), polys, props)
    return {"ortho": str(tmp_path / "o.tif"), "stems": str(tmp_path / "s.shp"), "tmp": tmp_path}


def test_binary_mask_is_strictly_two_valued(rasterize_site, tmp_path):
    out = str(tmp_path / "m.tif")
    rasterize(rasterize_site["stems"], rasterize_site["ortho"], out)
    with rasterio.open(out) as r:
        a = r.read(1)
    assert set(np.unique(a)) <= {0, 255}
    assert (a > 0).any(), "the polygons must land inside the ortho"


def test_tree_level_instances_group_fragments_of_one_stem(rasterize_site, tmp_path):
    """`id` is a tree key: two fragments of tree 7 must share one instance value."""
    inst = str(tmp_path / "i.tif")
    rasterize(rasterize_site["stems"], rasterize_site["ortho"], str(tmp_path / "m.tif"),
              instances_path=inst, instance_level="tree")
    with rasterio.open(inst) as r:
        vals = set(np.unique(r.read(1))) - {0}
    assert vals == {7, 9}, "one value per tree, not per polygon"


def test_segment_level_instances_give_every_polygon_its_own(rasterize_site, tmp_path):
    inst = str(tmp_path / "i.tif")
    rasterize(rasterize_site["stems"], rasterize_site["ortho"], str(tmp_path / "m.tif"),
              instances_path=inst, instance_level="segment")
    with rasterio.open(inst) as r:
        vals = set(np.unique(r.read(1))) - {0}
    assert vals == {1, 2, 3}, "three polygons -> three instances"


def test_rasterize_species_filter_is_case_insensitive(rasterize_site, tmp_path):
    """The corpus spells beech both 'RBU' and 'rBU'; matching case-sensitively drops 15."""
    out = str(tmp_path / "m.tif")
    rasterize(rasterize_site["stems"], rasterize_site["ortho"], out, species={"RBU"})
    with rasterio.open(out) as r:
        got = (r.read(1) > 0).sum()

    both = str(tmp_path / "m2.tif")
    rasterize(rasterize_site["stems"], rasterize_site["ortho"], both, species={"RBU", "GFI"})
    with rasterio.open(both) as r:
        all_three = (r.read(1) > 0).sum()

    assert got > 0
    assert got < all_three, "the spruce polygon must be excluded by the RBU filter"


def test_mismatched_crs_is_reprojected_not_silently_offset(tmp_path):
    """The corpus mixes EPSG:25833 and :32633 -- same zone, different datum.

    Coordinates look plausible either way, so an unhandled mismatch shifts masks by
    metres instead of failing. The mask must still land on the stems.
    """
    _rasterize_ortho(str(tmp_path / "o.tif"), crs="EPSG:32633")
    cx, cy = ORIGIN[0] + 20.0, ORIGIN[1] - 20.0
    _rasterize_stems(str(tmp_path / "s.shp"), [_rasterize_stem(cx, cy)],
                     [{"id": 1, "Species": "RBU"}], crs="EPSG:25833")

    out = str(tmp_path / "m.tif")
    rasterize(str(tmp_path / "s.shp"), str(tmp_path / "o.tif"), out)
    with rasterio.open(out) as r:
        assert (r.read(1) > 0).any(), "reprojection must keep the mask on the ortho"


def test_empty_selection_fails_loudly(rasterize_site, tmp_path):
    with pytest.raises(SystemExit):
        rasterize(rasterize_site["stems"], rasterize_site["ortho"], str(tmp_path / "m.tif"),
                  species={"NOSUCHSPECIES"})


def test_non_overlapping_annotations_are_rejected_not_written_empty(tmp_path):
    """An all-zero mask means the shapefile belongs to a different ortho."""
    _rasterize_ortho(str(tmp_path / "o.tif"))
    far = ORIGIN[0] + 500000.0
    _rasterize_stems(str(tmp_path / "s.shp"), [_rasterize_stem(far, ORIGIN[1])],
                     [{"id": 1, "Species": "RBU"}])

    with pytest.raises(SystemExit, match="do not overlap"):
        rasterize(str(tmp_path / "s.shp"), str(tmp_path / "o.tif"), str(tmp_path / "m.tif"))


# =============================================================================
# Parallel extraction
# (from tests/test_extract_parallel.py)
# =============================================================================

from winmol_unet.geo.parallel import _part_configs, merge  # noqa: E402


def _extract_part(work, name, split, n, start=1):
    d = os.path.join(work, name, split)
    os.makedirs(os.path.join(d, "train"), exist_ok=True)
    os.makedirs(os.path.join(d, "mask"), exist_ok=True)
    rows = []
    for i in range(start, start + n):
        open(os.path.join(d, "train", f"train{i}.jpeg"), "w").write(f"{name}-img{i}")
        open(os.path.join(d, "mask", f"mask{i}.gif"), "w").write(f"{name}-msk{i}")
        rows.append({"n": i, "site": name, "centre_x": 100.0 + i})
    with open(os.path.join(d, "tiles.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_two_parts_numbering_from_one_do_not_collide(tmp_path):
    work = tmp_path / "w"
    _extract_part(str(work), "A", "train", 3)
    _extract_part(str(work), "B", "train", 4)
    counts = merge(str(work), ["A", "B"], str(tmp_path / "out"), quiet=True)
    assert counts["train"] == 7, "a colliding merge would lose tiles"
    names = sorted(os.listdir(tmp_path / "out" / "train" / "train"))
    assert len(names) == 7
    got = sorted(int(n[5:-5]) for n in names)
    assert got == list(range(1, 8)), "indices must be contiguous"


def test_provenance_is_renumbered_to_match(tmp_path):
    work = tmp_path / "w"
    _extract_part(str(work), "A", "train", 2)
    _extract_part(str(work), "B", "train", 2)
    merge(str(work), ["A", "B"], str(tmp_path / "out"), quiet=True)
    rows = [json.loads(l) for l in open(tmp_path / "out" / "train" / "tiles.jsonl")]
    assert [r["n"] for r in rows] == [1, 2, 3, 4]
    assert [r["part"] for r in rows] == ["A", "A", "B", "B"]
    # the original centre must travel with the tile, not be reassigned
    assert rows[2]["centre_x"] == pytest.approx(101.0)


def test_missing_mask_refuses_the_merge(tmp_path):
    work = tmp_path / "w"
    _extract_part(str(work), "A", "train", 2)
    os.remove(work / "A" / "train" / "mask" / "mask2.gif")
    with pytest.raises(SystemExit, match="no mask"):
        merge(str(work), ["A"], str(tmp_path / "out"), quiet=True)


def test_each_site_keeps_its_own_split(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"extent_m": 19.512, "tile_px": 666, "sites": [
        {"name": "P1", "ortho": "a.tif", "stems": "a.gpkg", "aoi": "aoi.gpkg", "split": "train"},
        {"name": "P2", "ortho": "b.tif", "stems": "b.gpkg", "aoi": "aoi.gpkg", "split": "test"}]}))
    parts = _part_configs(str(cfg), str(tmp_path / "w"))
    assert [n for n, _ in parts] == ["P1", "P2"]
    for name, p in parts:
        one = json.loads(open(p).read())
        assert len(one["sites"]) == 1 and one["sites"][0]["name"] == name
        assert one["tile_px"] == 666, "shared settings must survive the split"
    assert json.loads(open(parts[1][1]).read())["sites"][0]["split"] == "test"


def test_the_worker_command_is_runnable_as_a_module():
    """The fan-out spawns `python -m winmol_unet.geo.splits`; pin that it resolves.

    This is the one line in parallel.py the rest of the suite cannot reach: the real path
    fans out a process per site over multi-hundred-megapixel orthomosaics, so no hermetic
    test drives it. It used to build a filesystem path as
    `dirname(dirname(__file__))/scripts/make_splits.py`, which silently stopped pointing at
    anything real when the module moved one directory deeper -- and would only have failed
    at spawn time, mid-extraction, after the expensive setup.

    Running the module's --help is cheap and catches exactly that class of breakage:
    the module is importable, is a valid -m target, and its parser builds.
    """
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "winmol_unet.geo.splits", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0, f"worker command is not runnable:\n{out.stderr}"
    assert "--config" in out.stdout and "--strategy" in out.stdout


def test_the_spawned_command_names_the_module_not_a_path():
    """A path built from __file__ breaks on a move; `-m` does not. Keep it that way."""
    import inspect

    from winmol_unet.geo import parallel

    src = inspect.getsource(parallel)
    assert '"-m", "winmol_unet.geo.splits"' in src, \
        "parallel.py should spawn the splits module by name, not by filesystem path"
    assert "scripts" not in src.split('"""')[2], \
        "parallel.py still refers to the old scripts/ location outside its docstring"


# =============================================================================
# Prediction plotting
# (from tests/test_plot_inference.py)
# =============================================================================

from rasterio.transform import from_origin  # noqa: E402, F401 (kept for parity with source)
from winmol_unet.geo.predict import _grid, score  # noqa: E402


def _aoi_and_stems(tmp_path, gsd=0.03, size=300):
    from shapely.geometry import box as _box, mapping as _mapping
    aoi = _box(0, 0, size * gsd, size * gsd)
    stems = [_box(1.0, 1.0, 1.3, 5.0), _box(3.0, 2.0, 3.3, 6.0)]
    for name, geoms in (("aoi", [aoi]), ("stems", stems)):
        p = tmp_path / f"{name}.gpkg"
        with fiona.open(p, "w", driver="GPKG", crs="EPSG:25833",
                        schema={"geometry": "Polygon", "properties": {}}) as dst:
            for g in geoms:
                dst.write({"geometry": _mapping(g), "properties": {}})
    return aoi, str(tmp_path / "aoi.gpkg"), str(tmp_path / "stems.gpkg")


def test_grid_is_independent_of_any_model_tiling():
    """Same bounds and ref GSD must give the same grid whatever tile size produced it."""
    tf, w, h = _grid((0, 0, 10, 10), 0.029297)
    assert (w, h) == (342, 342)
    assert tf.a == pytest.approx(0.029297) and tf.e == pytest.approx(-0.029297)


def test_perfect_prediction_scores_one(tmp_path):
    aoi, aoi_p, stems_p = _aoi_and_stems(tmp_path)
    ref = 0.03
    tf, W, H = _grid(aoi.bounds, ref)
    from rasterio.features import rasterize as rio_rasterize
    from winmol_unet.geo.sample import _load_geoms
    geoms, _ = _load_geoms(stems_p, rasterio.crs.CRS.from_epsg(25833), None, "s", quiet=True)
    gt = rio_rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=tf, fill=0)
    rows = score(gt.astype(np.float32), tf, W, H, aoi, rasterio.crs.CRS.from_epsg(25833),
                 stems_p, threshold=0.5, edge_buffer_m=0.0)
    assert rows[0]["f1"] == pytest.approx(1.0, abs=1e-6)


def test_edge_buffer_shrinks_the_scored_area(tmp_path):
    aoi, aoi_p, stems_p = _aoi_and_stems(tmp_path)
    tf, W, H = _grid(aoi.bounds, 0.03)
    prob = np.zeros((H, W), np.float32)
    crs = rasterio.crs.CRS.from_epsg(25833)
    wide = score(prob, tf, W, H, aoi, crs, stems_p, edge_buffer_m=0.0)[0]
    tight = score(prob, tf, W, H, aoi, crs, stems_p, edge_buffer_m=1.0)[0]
    assert tight["valid_px"] < wide["valid_px"], "buffer must exclude boundary ground"


def test_threshold_sweep_is_monotone_in_recall(tmp_path):
    """Sanity: raising the threshold cannot increase recall."""
    aoi, aoi_p, stems_p = _aoi_and_stems(tmp_path)
    tf, W, H = _grid(aoi.bounds, 0.03)
    rng = np.random.default_rng(0)
    prob = rng.random((H, W)).astype(np.float32)
    crs = rasterio.crs.CRS.from_epsg(25833)
    rows = score(prob, tf, W, H, aoi, crs, stems_p, thresholds=[0.2, 0.5, 0.8])
    rec = [r["recall"] for r in rows]
    assert rec == sorted(rec, reverse=True)
