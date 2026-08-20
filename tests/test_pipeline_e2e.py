"""Six stages, of which four are new. The seam that matters is stage 5 into
`--from-folder`: `cut` writes `train/` and `mask/`, and `build_dataset` pairs them by
shared key with no changes at all.
"""
import json
import os

import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
pytest.importorskip("shapely")
PIL = pytest.importorskip("PIL")

from tests.pipeline_fixtures import CRS, box_m, write_ortho, write_polygons  # noqa: E402
from winmol_unet.data.build import build_dataset  # noqa: E402
from winmol_unet.geo.rasterize import rasterize  # noqa: E402
from winmol_unet.pipeline.cut import cut  # noqa: E402
from winmol_unet.pipeline.ingest import ingest  # noqa: E402
from winmol_unet.pipeline.layout import layout  # noqa: E402
from winmol_unet.pipeline.resample import resample  # noqa: E402


def test_all_six_stages(tmp_path):
    root = tmp_path / "run"
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=32.0, gsd=0.02)

    src = str(tmp_path / "src.gpkg")
    write_polygons(src, [box_m(0, 0, 20, 20)], layer="AOI")
    write_polygons(src, [box_m(3, 3, 1.0, 1.0), box_m(11, 11, 1.0, 1.0)],
                   layer="hard_negative_AOI")

    # 1 -- ingest
    s1 = ingest(src, ortho, str(root / "00_source"),
                stems_layer="hard_negative_AOI", aoi_layer="AOI", aoi_ids=[1])
    assert s1["n_stems_in_aoi"] == 2

    # 2 -- resample to 8 cm
    s2 = resample(ortho, str(root / "01_gsd"), gsd=0.08)
    gsd_ortho = s2["path"]

    # 3 -- rasterize on that grid (existing mode, untouched)
    stem_map = os.path.join(os.path.dirname(gsd_ortho), "stem_map.tif")
    rasterize(str(root / "00_source" / "stems.gpkg"), gsd_ortho, stem_map)

    # 4 -- layout, no pixels cut
    lay = str(root / "02_layout" / "gsd080_ext10_grid_s100")
    s4 = layout(gsd_ortho, str(root / "00_source" / "aoi.gpkg"),
                str(root / "00_source" / "stems.gpkg"), lay,
                mode="grid", extent_m=10.0, min_stem_frac=0.0)
    assert s4["n_kept"] == 4
    assert s4["tile_px"] == 125
    assert os.path.exists(os.path.join(lay, "footprints.gpkg"))

    # 5 -- cut
    tiles = str(root / "03_tiles" / "gsd080_ext10_grid_s100")
    s5 = cut(lay, gsd_ortho, tiles, stem_map=stem_map)
    assert s5["n_written"] == 4

    # 6 -- from-folder, unmodified
    ds = str(root / "04_dataset" / "gsd080_ext10_grid_s100")
    build_dataset(tiles, ds)
    assert sorted(os.listdir(os.path.join(ds, "train"))) == \
        ["train%d.jpeg" % i for i in range(1, 5)]
    assert sorted(os.listdir(os.path.join(ds, "mask"))) == \
        ["mask%d.gif" % i for i in range(1, 5)]


def test_every_stage_leaves_a_manifest(tmp_path):
    from winmol_unet.pipeline.manifest import read_manifest
    root = tmp_path / "run"
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=32.0, gsd=0.02)
    src = str(tmp_path / "src.gpkg")
    write_polygons(src, [box_m(0, 0, 20, 20)], layer="AOI")
    write_polygons(src, [box_m(3, 3, 1.0, 1.0)], layer="hard_negative_AOI")

    ingest(src, ortho, str(root / "00_source"),
           stems_layer="hard_negative_AOI", aoi_layer="AOI")
    s2 = resample(ortho, str(root / "01_gsd"), gsd=0.08)
    lay = str(root / "02_layout" / "a")
    layout(s2["path"], str(root / "00_source" / "aoi.gpkg"),
           str(root / "00_source" / "stems.gpkg"), lay, extent_m=10.0)
    tiles = str(root / "03_tiles" / "a")
    cut(lay, s2["path"], tiles)

    for d, stage in ((root / "00_source", "ingest"),
                     (os.path.dirname(s2["path"]), "resample"),
                     (lay, "layout"), (tiles, "cut")):
        assert read_manifest(str(d))["stage"] == stage


def test_a_stage_is_not_stale_when_nothing_changed(tmp_path):
    """The property the manifests exist for: re-running is a decision, not a habit."""
    from winmol_unet.pipeline.manifest import input_identity, is_stale
    root = tmp_path / "run"
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=32.0, gsd=0.02)
    s2 = resample(ortho, str(root / "01_gsd"), gsd=0.08)
    d = os.path.dirname(s2["path"])
    params = {"gsd": 0.08, "resampling": "average", "jpeg_quality": 95}
    assert not is_stale(d, "resample", params, [input_identity(ortho, "ortho")])
    assert is_stale(d, "resample", {**params, "gsd": 0.05},
                    [input_identity(ortho, "ortho")])


R13 = "/Volumes/2TB/winmol/training_data/WINDWURF_Tegel/Revier_13"
requires_r13 = pytest.mark.skipif(
    not os.path.exists(os.path.join(R13, "cw_hard_negatives.gpkg")),
    reason="the 2 TB volume is not mounted")


@requires_r13
def test_ingest_survives_the_real_file(tmp_path):
    """Stage 1 against the real GeoPackage -- the one stage whose entire reason for
    existing is a property of this specific file.

    Deliberately stage 1 only. The raster stages here would have to resample the whole
    19 GB / 131.6 Gpx COG, which is not a test, and the AOI-clipped path that makes
    them cheap does not exist yet.

    The assertions are relational, not absolute counts: this file is still being
    labelled, and pinning "166 stems" would turn every labelling session into a red
    build. What must hold is that EVERY stem needs reprojecting (the layer is
    EPSG:4326 while the ortho is EPSG:32633), that selecting one AOI keeps only that
    AOI, and that the selected stems are a strict subset of all of them.
    """
    root = tmp_path / "run"
    ortho = os.path.join(R13, "Ortho", "result_Res1.3_COG.tif")

    s1 = ingest(os.path.join(R13, "cw_hard_negatives.gpkg"), ortho,
                str(root / "00_source"), stems_layer="hard_negative_AOI",
                aoi_layer="AOI", aoi_ids=[3])

    assert s1["n_stems"] > 0
    # Every stem reprojects: the silent-zero-rows defect this stage exists to fix.
    assert s1["reprojected"] == s1["n_stems"]
    assert "32633" in s1["crs"]
    assert s1["n_aoi"] == 1
    assert 0 < s1["n_stems_in_aoi"] <= s1["n_stems"]

    # The stems land on the AOI after reprojection -- reprojecting successfully is not
    # the same as landing in the right place.
    import fiona
    with fiona.open(os.path.join(str(root), "00_source", "stems.gpkg"),
                    layer="stems") as src:
        assert len(src) == s1["n_stems_in_aoi"]
        assert all(f["properties"]["aoi_id"] == 3 for f in src)
