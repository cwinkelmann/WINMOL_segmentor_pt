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
import onnx
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


# --- end-to-end: prepare -> train -> infer -> evaluate --------------------------------
# The four entry points are call sites, and --help exercises only argparse. Two Critical
# bugs shipped past a --help-only test: infer.py unpacked predict_plot's 7-tuple as 5, and
# evaluate.py fed a list (geo.predict.score) and a tuple (score_checkpoint.score) into a
# dict splat. Both raise on the first real invocation. These tests drive the real paths.

def _tiny_dataset(dst, n=6):
    """A loader-convention dataset small enough to train on in seconds."""
    import numpy as np
    from PIL import Image
    (dst / "train").mkdir(parents=True); (dst / "mask").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype="uint8")).save(
            dst / "train" / f"train{i}.jpeg")
        m = np.zeros((64, 64), dtype="uint8"); m[20:44, 28:36] = 255
        Image.fromarray(m).save(dst / "mask" / f"mask{i}.gif")
    return dst


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a tiny UNet once and reuse its ONNX across the tests below."""
    torch = pytest.importorskip("torch")
    from winmol_unet.cli.train import main

    root = tmp_path_factory.mktemp("e2e")
    ds = _tiny_dataset(root / "ds")
    out = root / "run"
    assert main(["--data-dir", str(ds), "--out-dir", str(out), "--arch", "unet",
                 "--width-mult", "0.25", "--epochs", "1", "--batch-size", "2",
                 "--device", "cpu", "--seed", "1"]) == 0
    assert (out / "model.onnx").exists(), "training did not export ONNX"
    return {"onnx": str(out / "model.onnx"), "pt": str(out / "model.pt"), "ds": str(ds)}


def test_train_exports_a_contract_conformant_onnx(trained):
    from winmol_unet.contract import validate_onnx_model
    validate_onnx_model(onnx.load(trained["onnx"]))


def test_evaluate_tile_mode_emits_a_flat_metric_dict(trained, capsys, monkeypatch):
    """Regression for the tuple/list splat: --label must not raise, and the payload
    must be a flat mapping rather than a list or a (metrics, n) pair."""
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    from winmol_unet.cli.evaluate import main
    assert main(["--model", trained["onnx"], "--data-dir", trained["ds"],
                 "--label", "e2e"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["label"] == "e2e"
    assert {"f1", "precision", "recall"} <= set(payload)
    assert all(not isinstance(v, list) for v in payload.values())


def test_evaluate_checkpoint_mode_emits_a_flat_metric_dict(trained, capsys):
    """score_checkpoint returns (metrics, n_tiles); the CLI must flatten it."""
    pytest.importorskip("torch")
    from winmol_unet.cli.evaluate import main
    assert main(["--model", trained["pt"], "--data-dir", trained["ds"], "--arch", "unet",
                 "--width-mult", "0.25",   # must match what the checkpoint was trained at
                 "--device", "cpu", "--label", "ckpt"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["label"] == "ckpt"
    assert {"f1", "precision", "recall"} <= set(payload)


def test_infer_writes_a_georeferenced_raster(site, trained, tmp_path, monkeypatch):
    """Regression for the 7-tuple unpack AND the silently-missing CRS: a stem map with
    crs=None is not georeferenced, which is the entire point of the output."""
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    from winmol_unet.cli.infer import main

    out = tmp_path / "pred.tif"
    assert main(["--model", trained["onnx"], "--ortho", site["ortho"],
                 "--aoi", site["aoi"], "--out", str(out),
                 "--extent-m", "15", "--threshold", "0.5", "--quiet"]) == 0
    with rasterio.open(out) as src:
        assert src.crs is not None, "output raster has no CRS"
        assert src.crs.to_string() == CRS
        assert src.count == 1 and src.dtypes[0] == "float32"
    assert (tmp_path / "pred_mask.tif").exists(), "--threshold did not write the mask"


def test_evaluate_geospatial_mode_scores_against_the_ground(site, trained, tmp_path,
                                                            capsys, monkeypatch):
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    from winmol_unet.cli.evaluate import main
    assert main(["--model", trained["onnx"], "--ortho", site["ortho"],
                 "--aoi", site["aoi"], "--stems", site["stems"],
                 "--extent-m", "15", "--edge-buffer-m", "0", "--label", "geo",
                 "--quiet"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["label"] == "geo"
    assert {"f1", "precision", "recall"} <= set(payload)
    assert 0.0 <= payload["f1"] <= 1.0


# --- prepare.py's non-geo modes ------------------------------------------------------
# build/coco/split moved from scripts/ into winmol_unet.data and are now prepare.py modes.
# They need no GDAL, so unlike the tiling modes they work on a base [train] install.

def _pairs(dst, n=4, ext="png"):
    from PIL import Image
    import numpy as np
    (dst / "train").mkdir(parents=True); (dst / "mask").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((16, 16, 3), 100 + i, dtype="uint8")).save(
            dst / "train" / f"train_{i}.{ext}")
        m = np.zeros((16, 16), dtype="uint8"); m[4:9, 3 + i] = 7   # palette-ish, non-binary
        Image.fromarray(m).save(dst / "mask" / f"mask_{i}.png")
    return dst


def test_prepare_from_folder_produces_the_loader_convention(tmp_path):
    from winmol_unet.cli.prepare import main
    src, out = _pairs(tmp_path / "raw"), tmp_path / "ready"
    assert main(["--from-folder", "--src", str(src), "--out", str(out)]) == 0
    imgs = sorted((out / "train").glob("train*.jpeg"))
    masks = sorted((out / "mask").glob("mask*.gif"))
    assert len(imgs) == len(masks) == 4
    # masks must come out binary: the source used palette index 7, not 255
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(masks[0]).convert("L"))
    assert set(np.unique(arr)) <= {0, 255}


def test_prepare_split_matches_the_loaders_own_split(tmp_path):
    """The materialised split must equal what train_val_split would pick for the same
    fraction and seed — otherwise a 'shareable held-out set' is a different set."""
    from winmol_unet.cli.prepare import main
    from winmol_unet.training.dataset import split_ids

    src = tmp_path / "ready"
    (src / "train").mkdir(parents=True); (src / "mask").mkdir(parents=True)
    from PIL import Image
    import numpy as np
    for i in range(1, 11):
        Image.fromarray(np.full((16, 16, 3), i, dtype="uint8")).save(src / "train" / f"train{i}.jpeg")
        Image.fromarray(np.zeros((16, 16), dtype="uint8")).save(src / "mask" / f"mask{i}.gif")

    out = tmp_path / "split"
    assert main(["--split", "--src", str(src), "--out", str(out),
                 "--val-fraction", "0.2", "--seed", "1"]) == 0
    _, val_ids = split_ids(str(src / "train"), str(src / "mask"), 0.2, 1)
    assert len(list((out / "val" / "train").glob("*.jpeg"))) == len(val_ids)
    assert len(list((out / "train" / "train").glob("*.jpeg"))) == 10 - len(val_ids)


def test_prepare_conversion_modes_need_no_gdal():
    """These modes must not import rasterio/fiona/shapely — they are not in [train]."""
    import subprocess, sys as _sys, textwrap
    code = textwrap.dedent("""
        import sys
        import winmol_unet.data.build, winmol_unet.data.coco, winmol_unet.data.split
        heavy = [m for m in ("rasterio", "fiona", "shapely") if m in sys.modules]
        print(",".join(heavy))
    """)
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"conversion modules pulled in GDAL: {out.stdout}"


def test_prepare_conversion_modes_report_missing_arguments(tmp_path):
    from winmol_unet.cli.prepare import main
    for argv in (["--from-folder"], ["--from-coco"], ["--split"]):
        with pytest.raises(SystemExit):
            main(argv + ["--out", str(tmp_path / "x")])
