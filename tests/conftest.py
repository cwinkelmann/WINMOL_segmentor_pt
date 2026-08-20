"""Shared fixtures for the WINMOL test suite.

Replaces the synthetic-dataset builder that was duplicated across 22 test modules
and the TrainConfig block duplicated across 15. Both are factories rather than
plain fixtures because call sites need several datasets (train + val + test) or
several configs (stage 1 + stage 2) inside one test.
"""
import pathlib

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def stem_dataset():
    """Build a StemDataset-shaped folder: train/train{k}.jpeg + mask/mask{k}.gif.

    The image is a hard vertical split -- white where the mask is white -- so a
    model can actually fit it, which is what the overfit test needs.

    ``ids`` overrides the default contiguous ``1..n`` numbering when a caller
    needs specific (e.g. non-contiguous) integer ids to prove pairing logic;
    it defaults to None so every existing call site is unaffected. When ``ids``
    is passed, ``n`` is silently ignored -- the number of files written is
    ``len(ids)``, not ``n``.
    """
    def _make(root, n=6, size=32, stem_frac=0.5, ids=None):
        root = pathlib.Path(root)
        img_dir, mask_dir = root / "train", root / "mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        split = max(1, int(size * stem_frac))
        for k in (ids if ids is not None else range(1, n + 1)):
            rgb = np.zeros((size, size, 3), np.uint8)
            rgb[:, :split, :] = 255
            Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
            m = np.zeros((size, size), np.uint8)
            m[:, :split] = 255
            Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")
        return root
    return _make


@pytest.fixture
def train_config(tmp_path):
    """A hermetic TrainConfig: one epoch, CPU, no encoder download."""
    def _make(data_dir=None, **overrides):
        from winmol_unet.training.config import TrainConfig   # lazy: pulls torch
        out = tmp_path / "out"
        base = dict(
            data_dir=str(data_dir if data_dir is not None else tmp_path),
            checkpoint_dir=str(tmp_path / "ck"),
            log_dir=str(tmp_path / "log"),
            hdf5_out=str(out / "m.hdf5"),
            onnx_out=str(out / "m.onnx"),
            pt_out=str(out / "m.pt"),
            keras_out=str(out / "m.keras"),
            epochs=1,
            batch_size=2,
            patience=999,
            device="cpu",
            encoder_weights=None,
        )
        base.update(overrides)
        return TrainConfig(**base)
    return _make


@pytest.fixture
def force_cpu_onnx(monkeypatch):
    """Pin the CPU EP so ONNX parity assertions are bit-exact.

    CoreML and CUDA compute in fp16 and drift past a tight epsilon.
    """
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    yield
    monkeypatch.delenv("WINMOL_ONNX_FORCE_CPU", raising=False)


# --- geo fixtures -----------------------------------------------------------
# rasterio/fiona/shapely are an optional extra (`pip install -e ".[geo]"`), and
# this module is imported for every test in the suite, so the guard has to sit
# on the fixture, not at module scope -- a module-level pytest.importorskip
# here would make conftest.py itself fail to load when geo is not installed,
# taking every other fixture down with it.

CRS = "EPSG:25833"
GSD = 0.02          # 2 cm/px, in the range the real orthos sit
ORIGIN = (400000.0, 6000000.0)


def _write_ortho(path, size_m=60.0, gsd=GSD, blank_edge_m=0.0):
    import rasterio

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
    import fiona
    from shapely.geometry import mapping

    schema = {"geometry": "Polygon", "properties": {"id": "int", "Species": "str"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=CRS, schema=schema) as dst:
        for i, poly in enumerate(polys):
            p = (props or [{}] * len(polys))[i]
            dst.write({"geometry": mapping(poly),
                       "properties": {"id": p.get("id", i + 1),
                                      "Species": p.get("Species", "GFI")}})


def _stem(x, y, length=3.0, width=0.4, angle=0.0):
    """A stem-shaped polygon: the corpus median is 3.0 m x 0.43 m."""
    from shapely.affinity import rotate
    from shapely.geometry import box
    return rotate(box(x - length / 2, y - width / 2, x + length / 2, y + width / 2), angle)


@pytest.fixture
def geo_site(tmp_path):
    """A tiny synthetic orthomosaic + stem/AOI shapefiles, in a real CRS.

    Lifted verbatim from the `site` fixture in the former test_sample_training_tiles.py
    (now folded into tests/test_geo.py). Everything is built in tmp_path, so nothing
    touches the real orthomosaics (100-1900 megapixels). The CRS, GSD and mid-grey noise
    range are load-bearing: nodata-collar rejection is asserted against exactly these
    values.
    """
    pytest.importorskip("rasterio")
    pytest.importorskip("fiona")
    pytest.importorskip("shapely")
    from shapely.geometry import box

    ortho = tmp_path / "site_ortho.tif"
    _write_ortho(str(ortho))
    cx, cy = ORIGIN[0] + 30.0, ORIGIN[1] - 30.0

    # a 40 m AOI leaves room for a 15 m footprint plus its 10.6 m inward buffer
    _write_polygons(str(tmp_path / "aoi.shp"), [box(cx - 20, cy - 20, cx + 20, cy + 20)])
    # a dense mat of stems, so nearly every footprint clears the min-stem test
    stems = [_stem(cx + dx, cy + dy, angle=17 * (dx + dy))
             for dx in range(-18, 19, 3) for dy in range(-18, 19, 3)]
    _write_polygons(str(tmp_path / "stems.shp"), stems)
    return {"ortho": str(ortho), "stems": str(tmp_path / "stems.shp"),
            "aoi": str(tmp_path / "aoi.shp"), "tmp": tmp_path}
