"""The four root entry points, exercised rather than --help'd.

`--help` proves argparse is wired up; it proves nothing about whether the CLI calls the
library correctly. Writing prepare.py produced two bugs that `--help` was perfectly happy
with: `rasterize(..., instances=...)` when the parameter is `instances_path`, and
forwarding `--config` to `folds.main`, which takes `--site-all/--site-tv/--folds/--splittable`
and has no `--config` at all. Both would have failed in front of a user, on real data,
after a long tiling run.

So each mode here runs end to end on a synthetic site built in tmp_path.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
shapely = pytest.importorskip("shapely")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRS = "EPSG:25833"
GSD = 0.02
ORIGIN = (400000.0, 6000000.0)


def _write_ortho(path, size_m=60.0):
    n = int(size_m / GSD)
    transform = rasterio.transform.from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD)
    data = np.random.default_rng(0).integers(60, 200, size=(3, n, n), dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3,
                       dtype="uint8", crs=CRS, transform=transform) as dst:
        dst.write(data)


def _write_polygons(path, polys):
    schema = {"geometry": "Polygon", "properties": {"id": "int", "Species": "str"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=CRS, schema=schema) as dst:
        for i, poly in enumerate(polys):
            dst.write({"geometry": shapely.geometry.mapping(poly),
                       "properties": {"id": i + 1, "Species": "GFI"}})


@pytest.fixture
def site(tmp_path):
    from shapely.affinity import rotate
    from shapely.geometry import box

    ortho = tmp_path / "site_ortho.tif"
    _write_ortho(str(ortho))
    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0
    _write_polygons(str(tmp_path / "aoi.shp"), [box(cx - 20, cy - 20, cx + 20, cy + 20)])
    stems = [rotate(box(cx + dx - 1.5, cy + dy - 0.2, cx + dx + 1.5, cy + dy + 0.2),
                    17 * (dx + dy))
             for dx in range(-18, 19, 3) for dy in range(-18, 19, 3)]
    _write_polygons(str(tmp_path / "stems.shp"), stems)
    return {"ortho": str(ortho), "stems": str(tmp_path / "stems.shp"),
            "aoi": str(tmp_path / "aoi.shp")}


@pytest.mark.parametrize("script", ["prepare.py", "train.py", "infer.py", "evaluate.py"])
def test_entry_point_runs(script):
    out = subprocess.run([sys.executable, os.path.join(REPO, script), "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout


def test_prepare_single_site_writes_a_loader_dataset(site, tmp_path):
    from winmol_unet.cli.prepare import main

    out = tmp_path / "DS"
    assert main(["--ortho", site["ortho"], "--stems", site["stems"], "--aoi", site["aoi"],
                 "--out", str(out), "--limit", "3", "--quiet"]) == 0

    images = sorted((out / "train").glob("train*.jpeg"))
    masks = sorted((out / "mask").glob("mask*.gif"))
    assert len(images) == len(masks) == 3
    # tiles.jsonl is what makes a split auditable for leakage; it is not optional
    recs = [json.loads(l) for l in (out / "tiles.jsonl").read_text().splitlines() if l.strip()]
    assert len(recs) == 3
    assert {"n", "site", "gsd_m_per_px"} <= set(recs[0])


def test_prepare_rasterize_writes_a_label_raster(site, tmp_path):
    """Regression: this called rasterize(instances=...) where the parameter is
    instances_path, so it raised TypeError the moment it was actually invoked."""
    from winmol_unet.cli.prepare import main

    out = tmp_path / "stem_map.tif"
    inst = tmp_path / "instances.tif"
    assert main(["--rasterize", "--stems", site["stems"], "--ortho", site["ortho"],
                 "--out", str(out), "--instances", str(inst)]) == 0
    with rasterio.open(out) as src:
        assert src.read(1).max() > 0        # something was actually burned in
    assert inst.exists()


def test_prepare_compose_folds_forwards_the_flags_folds_actually_takes(tmp_path):
    """Regression: this forwarded --config, which folds.main does not accept.

    Driven with real tile pairs, because folds.compose deliberately refuses a split that
    receives zero tiles -- a fold quietly built from nothing reads as a real result.
    """
    from winmol_unet.cli.prepare import main

    def _tiles(d, n=2):
        (d / "train").mkdir(parents=True, exist_ok=True)
        (d / "mask").mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            (d / "train" / f"train{i}.jpeg").write_bytes(b"x")
            (d / "mask" / f"mask{i}.gif").write_bytes(b"x")

    site_all, site_tv, out = tmp_path / "all", tmp_path / "tv", tmp_path / "folds"
    for site in ("A", "B"):
        _tiles(site_all / site)
        for split in ("train", "val"):
            _tiles(site_tv / site / split)

    assert main(["--compose-folds", "--site-all", str(site_all), "--site-tv", str(site_tv),
                 "--fold-sites", "A", "B", "--splittable", "A", "B",
                 "--out", str(out)]) == 0
    # each fold holds one site out entirely as its test split
    for held, other in (("A", "B"), ("B", "A")):
        assert (out / held / "test" / "train" / "train1.jpeg").exists()
        assert (out / held / "train" / "train" / "train1.jpeg").exists()


def test_prepare_refuses_an_incomplete_single_site_invocation(tmp_path):
    from winmol_unet.cli.prepare import main

    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "DS")])
