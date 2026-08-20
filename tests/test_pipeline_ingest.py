"""The Revier 13 stems are EPSG:4326 while the AOIs and the ortho are EPSG:32633.
A spatial join across them returns zero rows in silence, which is how the defect was
found. Stage 1 exists to fix that once, on disk, where it can be seen.
"""
import os

import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
shapely = pytest.importorskip("shapely")

from tests.pipeline_fixtures import CRS, box_m, write_ortho, write_polygons  # noqa: E402
from winmol_unet.pipeline.ingest import ingest  # noqa: E402


def _scene(tmp_path, stem_crs=CRS):
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=16.0)
    src = str(tmp_path / "src.gpkg")
    write_polygons(src, [box_m(1, 1, 4, 4), box_m(9, 9, 4, 4)], layer="AOI")
    stems = [box_m(2, 2, 0.4, 0.4), box_m(3, 3, 0.4, 0.4), box_m(10, 10, 0.4, 0.4)]
    if stem_crs != CRS:
        # rasterio.warp, not pyproj: pyproj arrives as a transitive dependency of
        # fiona and is not declared in the [geo] extra, so it must not be imported.
        from rasterio.warp import transform_geom
        from shapely.geometry import shape
        stems = [shape(transform_geom(CRS, stem_crs, s.__geo_interface__))
                 for s in stems]
    write_polygons(src, stems, layer="hard_negative_AOI", crs=stem_crs)
    return ortho, src


def test_reprojects_geographic_stems_onto_the_ortho_crs(tmp_path):
    ortho, src = _scene(tmp_path, stem_crs="EPSG:4326")
    out = str(tmp_path / "00_source")
    stats = ingest(src, ortho, out, stems_layer="hard_negative_AOI", aoi_layer="AOI")

    assert stats["reprojected"] == 3
    with fiona.open(os.path.join(out, "stems.gpkg"), layer="stems") as s:
        assert str(s.crs).endswith("32633")
        # Reprojection must land them back on the AOIs, not merely succeed.
        assert len(s) == 3


def test_tags_each_stem_with_its_aoi(tmp_path):
    ortho, src = _scene(tmp_path)
    out = str(tmp_path / "00_source")
    ingest(src, ortho, out, stems_layer="hard_negative_AOI", aoi_layer="AOI")
    with fiona.open(os.path.join(out, "stems.gpkg"), layer="stems") as s:
        ids = sorted((f["properties"]["aoi_id"] or 0) for f in s)
    assert ids == [1, 1, 2]


def test_aoi_selection_drops_both_aois_and_their_stems(tmp_path):
    ortho, src = _scene(tmp_path)
    out = str(tmp_path / "00_source")
    stats = ingest(src, ortho, out, stems_layer="hard_negative_AOI",
                   aoi_layer="AOI", aoi_ids=[1])
    assert stats["n_aoi"] == 1
    assert stats["n_stems_in_aoi"] == 2
    with fiona.open(os.path.join(out, "aoi.gpkg"), layer="aoi") as s:
        assert [f["properties"]["aoi_id"] for f in s] == [1]


def test_repairs_a_self_intersecting_ring(tmp_path):
    ortho, src = _scene(tmp_path)
    bowtie = shapely.geometry.Polygon([(400002, 5799998), (400003, 5799997),
                                       (400002, 5799997), (400003, 5799998)])
    assert not bowtie.is_valid
    write_polygons(str(tmp_path / "bad.gpkg"), [bowtie], layer="hard_negative_AOI")
    out = str(tmp_path / "00_source")
    # Hand-digitised stems have these; GEOS aborts on the first set operation.
    stats = ingest(str(tmp_path / "bad.gpkg"), ortho, out,
                   stems_layer="hard_negative_AOI", aoi_layer=None)
    assert stats["repaired"] == 1


def test_writes_a_manifest(tmp_path):
    ortho, src = _scene(tmp_path)
    out = str(tmp_path / "00_source")
    ingest(src, ortho, out, stems_layer="hard_negative_AOI", aoi_layer="AOI")
    from winmol_unet.pipeline.manifest import read_manifest
    assert read_manifest(out)["stage"] == "ingest"
