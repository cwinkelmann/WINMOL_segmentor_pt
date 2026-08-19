"""Score an Analyzer stem-map raster on the same basis as `geo.predict`.

The Analyzer is the deployment path, so its output is the number that matters — but it
must be scored the same way as anything it is compared against: resampled onto one
reference grid, masked to the AOI shrunk by an edge buffer, precision and recall reported
separately.

Outside the windthrow polygon the stems are real but were never digitised, so scoring the
whole raster counts correct detections as false positives and penalises the better model
hardest. That masking is not optional.

    python evaluate.py --stem-map out.tif --aoi aoi.gpkg --stems stems.gpkg
"""
import argparse
import json

import numpy as np



def score_map(stem_map, aoi, stems, ref_gsd=0.029297, edge_buffer_m=2.0, threshold=0.5):
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.warp import Resampling, reproject
    from shapely.ops import unary_union

    from .predict import _grid
    from .sample import _load_geoms

    src = rasterio.open(stem_map)
    area = unary_union(_load_geoms(aoi, src.crs, None, "aoi", quiet=True)[0])
    tf, W, H = _grid(area.bounds, ref_gsd)

    dst = np.zeros((H, W), np.float32)
    reproject(source=rasterio.band(src, 1), destination=dst,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=tf, dst_crs=src.crs, resampling=Resampling.average)
    # Analyzer stem maps are 0/1 or 0/255 depending on version; normalise either way.
    if dst.max() > 1.5:
        dst = dst / 255.0

    geoms, _ = _load_geoms(stems, src.crs, None, "stems", quiet=True)
    gt = rio_rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=tf,
                       fill=0).astype(bool)
    inner = area.buffer(-edge_buffer_m) if edge_buffer_m else area
    valid = rio_rasterize([(inner, 1)], out_shape=(H, W), transform=tf, fill=0).astype(bool)

    pred = (dst > threshold) & valid
    g = gt & valid
    tp = int((pred & g).sum()); fp = int((pred & ~g).sum()); fn = int((~pred & g).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"stem_map": stem_map, "f1": 2 * p * r / max(p + r, 1e-9), "precision": p,
            "recall": r, "tp": tp, "fp": fp, "fn": fn, "gt_px": int(g.sum()),
            "valid_px": int(valid.sum()), "ref_gsd_cm": ref_gsd * 100,
            "edge_buffer_m": edge_buffer_m}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem-map", required=True)
    p.add_argument("--aoi", required=True)
    p.add_argument("--stems", required=True)
    p.add_argument("--ref-gsd-cm", type=float, default=2.9297)
    p.add_argument("--edge-buffer-m", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--label", default=None)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)
    r = score_map(a.stem_map, a.aoi, a.stems, a.ref_gsd_cm / 100.0, a.edge_buffer_m,
                  a.threshold)
    print(f"{a.label or os.path.basename(a.stem_map)}   F1 {r['f1']:.4f}  "
          f"P {r['precision']:.4f}  R {r['recall']:.4f}   "
          f"(gt {r['gt_px']:,} px in {r['valid_px']:,} px of AOI)")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
