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

    assert a == b
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
