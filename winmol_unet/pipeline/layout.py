"""Stage 4 — decide which tiles to cut, and write that decision down.

Nothing is read from the image bands here. The output is a GeoPackage of footprints
and a JSONL of their properties, which means the sampling plan can be opened in QGIS
and judged before an hour is spent cutting it.

Every footprint is snapped to the GSD raster's own pixel grid. That is what lets
stage 5 be a pure window read: no interpolation at cut time, so a tile is
byte-identical to the pixels in its GSD copy, and all resampling stays in stage 2
where it is visible.

`stem_frac` here is computed from the polygons -- exact and cheap for the 69 stems of
Revier 13 -- and confirmed against the label raster in stage 5.

`min_aoi_frac` trades a hard "entirely inside the AOI" test for a fraction, so a tile
can reach the AOI boundary instead of always sitting a full footprint short of it.
Ground inside the tile but outside the AOI is *unlabelled*, not *stem-free* -- the
annotator never looked there. A tile at `min_aoi_frac=0.25` therefore teaches the
model that three-quarters of its area is background on no evidence. That is a real
risk for a positive-bearing dataset, and much less so for a hard-negative set, where
the surrounding ground genuinely is negative. This module does not judge that
trade-off; whoever sets the flag does.
"""
import json
import os

from winmol_unet.pipeline.manifest import input_identity, write_manifest


def tile_geometry(gsd, extent_m=None, tile_px=None):
    """`(extent_m, tile_px)` from whichever one was given.

    Passing both would reintroduce the hidden second resample this pipeline exists to
    remove: the tile would be cut at one scale and resized to another.
    """
    if (extent_m is None) == (tile_px is None):
        raise ValueError("give exactly one of extent_m or tile_px; the GSD fixes the other")
    if extent_m is None:
        extent_m = tile_px * gsd
    return float(extent_m), int(round(extent_m / gsd))


def _snap(value, origin, gsd):
    """Move a world coordinate onto the raster's grid lines."""
    return origin + round((value - origin) / gsd) * gsd


def enumerate_grid(bounds, origin, gsd, extent_m, stride_frac):
    """Interior tile positions tiling `bounds`, plus a one-tile skirt straddling
    each edge so a footprint can reach past it.

    The skirt candidate on a side is centred exactly on that edge -- half inside,
    half out -- rather than sized to whatever gap the interior grid happens to
    leave, so it is always exactly one extra candidate per edge, independent of
    `stride_frac` or how evenly the tile size divides the region. Whether a skirt
    candidate survives is left entirely to `min_aoi_frac` in `layout()`: at the
    default of 1.0 a skirt candidate is at most half full and is always dropped,
    so the interior list below -- and every count that depended on it -- is
    unchanged from before this existed.
    """
    minx, miny, maxx, maxy = bounds
    step = extent_m * stride_frac

    xs = []
    x = _snap(minx, origin[0], gsd)
    while x + extent_m <= maxx + 1e-9:
        xs.append(x)
        x = _snap(x + step, origin[0], gsd)
    for skirt_x in (_snap(minx - extent_m / 2, origin[0], gsd),
                    _snap(maxx - extent_m / 2, origin[0], gsd)):
        if skirt_x not in xs:
            xs.append(skirt_x)

    ys = []
    y = _snap(maxy, origin[1], gsd)
    while y - extent_m >= miny - 1e-9:
        ys.append(y)
        y = _snap(y - step, origin[1], gsd)
    for skirt_y in (_snap(maxy + extent_m / 2, origin[1], gsd),
                    _snap(miny + extent_m / 2, origin[1], gsd)):
        if skirt_y not in ys:
            ys.append(skirt_y)

    return [(x, y) for x in xs for y in ys]


def _read_layer(path, layer):
    import fiona
    from shapely.geometry import shape
    with fiona.open(path, layer=layer) as src:
        return [(shape(f["geometry"]), dict(f["properties"] or {}))
                for f in src if f["geometry"] is not None]


def layout(gsd_ortho, aoi_path, stems_path, out_dir, mode="grid", extent_m=None,
           tile_px=None, stride_frac=1.0, min_valid_frac=0.5, min_stem_frac=0.0,
           min_aoi_frac=1.0, n_tiles=None, seed=1, quiet=False):
    """Stage 4 -- write the sampling plan, without reading any image bands.

    `n_tiles` (mode="random") is applied **per AOI region**, not per run: each
    leakage group (AOI) is sampled to the same size regardless of its own area, so a
    small AOI is not starved relative to a large one just because the run covers
    several sites at once.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    from shapely.geometry import box, mapping
    from shapely.strtree import STRtree
    import fiona

    with rasterio.open(gsd_ortho) as src:
        gsd = src.res[0]
        crs = src.crs
        origin = (src.transform.c, src.transform.f)
        extent_m, tile_px = tile_geometry(gsd, extent_m, tile_px)

        aois = _read_layer(aoi_path, "aoi") if aoi_path else []
        stems = _read_layer(stems_path, "stems") if stems_path else []
        stem_tree = STRtree([g for g, _ in stems]) if stems else None

        if aois:
            # aoi_id is normally written by ingest (stage 1), 1-based by position in
            # the AOI layer. Fall back to that same position when a caller hands in
            # an AOI layer that doesn't carry the field, rather than collapsing every
            # region onto the "no AOI given at all" sentinel of 0 -- two different
            # situations that must not share a value, or every real AOI's tiles get
            # tagged as ungrouped for leakage purposes.
            regions = []
            for i, (poly, props) in enumerate(aois, start=1):
                aoi_id = props.get("aoi_id")
                aoi_id = int(aoi_id) if aoi_id is not None else i
                regions.append((poly, {"aoi_id": aoi_id}))
        else:
            regions = [(box(*src.bounds), {"aoi_id": 0})]

        # The screen reads the mask from an overview, cheap enough for millions of
        # candidates. It can only be approximate, so borderline cases are passed to
        # stage 5 rather than dropped here.
        ov = src.overviews(1)
        decim = ov[min(2, len(ov) - 1)] if ov else 1
        tol = 0.05

        cands = []
        for poly, props in regions:
            aoi_id = int(props.get("aoi_id") or 0)
            if mode == "grid":
                tops = enumerate_grid(poly.bounds, origin, gsd, extent_m, stride_frac)
            elif mode == "random":
                rng = np.random.default_rng(seed + aoi_id)
                minx, miny, maxx, maxy = poly.bounds
                tops = []
                for _ in range(int((n_tiles or 100) * 20)):
                    if len(tops) >= (n_tiles or 100):
                        break
                    x = _snap(rng.uniform(minx, maxx - extent_m), origin[0], gsd)
                    y = _snap(rng.uniform(miny + extent_m, maxy), origin[1], gsd)
                    tops.append((x, y))
            else:
                raise ValueError(f"unknown mode {mode!r}; use grid or random")
            cands += [(x, y, aoi_id, poly) for x, y in tops]

        kept, dropped = [], {"aoi": 0, "valid": 0, "stem": 0}
        for x, y, aoi_id, region in cands:
            fp = box(x, y - extent_m, x + extent_m, y)
            aoi_frac = fp.intersection(region).area / fp.area
            if aoi_frac < min_aoi_frac - 1e-9:
                dropped["aoi"] += 1
                continue

            win = from_bounds(*fp.bounds, transform=src.transform)
            m = src.read_masks(1, window=win,
                               out_shape=(max(1, int(tile_px / decim)),) * 2,
                               resampling=Resampling.average)
            valid_frac = float(m.mean()) / 255.0
            if valid_frac < min_valid_frac - tol:
                dropped["valid"] += 1
                continue

            stem_area = 0.0
            n_stems = 0
            if stem_tree is not None:
                for idx in stem_tree.query(fp):
                    g = stems[int(idx)][0]
                    inter = g.intersection(fp).area
                    if inter > 0:
                        stem_area += inter
                        n_stems += 1
            stem_frac = stem_area / (extent_m * extent_m)
            if stem_frac < min_stem_frac:
                dropped["stem"] += 1
                continue

            kept.append({"id": "%06d" % len(kept), "source_ortho": os.path.abspath(gsd_ortho),
                         "gsd_m": gsd, "crs": crs.to_string(),
                         "minx": x, "miny": y - extent_m, "maxx": x + extent_m, "maxy": y,
                         "width_px": tile_px, "height_px": tile_px, "rotation": 0.0,
                         "aoi_id": aoi_id, "mode": mode, "seed": seed,
                         "stride_frac": stride_frac, "valid_frac": valid_frac,
                         "aoi_frac": aoi_frac,
                         "stem_frac": stem_frac, "n_stems": n_stems, "stage": "layout"})

    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "tiles.jsonl")
    with open(jsonl, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    fp_path = os.path.join(out_dir, "footprints.gpkg")
    schema = {"geometry": "Polygon",
              "properties": {"id": "str", "aoi_id": "int", "valid_frac": "float",
                             "aoi_frac": "float",
                             "stem_frac": "float", "n_stems": "int"}}
    with fiona.open(fp_path, "w", driver="GPKG", layer="footprints",
                    crs=crs.to_string(), schema=schema) as dst:
        for r in kept:
            dst.write({"geometry": mapping(box(r["minx"], r["miny"], r["maxx"], r["maxy"])),
                       "properties": {k: r[k] for k in
                                      ("id", "aoi_id", "valid_frac", "aoi_frac",
                                       "stem_frac", "n_stems")}})

    stats = {"n_candidates": len(cands), "n_kept": len(kept),
             "dropped_aoi": dropped["aoi"], "dropped_valid": dropped["valid"],
             "dropped_stem": dropped["stem"], "extent_m": extent_m, "tile_px": tile_px}
    params = {"mode": mode, "extent_m": extent_m, "tile_px": tile_px,
              "stride_frac": stride_frac, "min_valid_frac": min_valid_frac,
              "min_stem_frac": min_stem_frac, "min_aoi_frac": min_aoi_frac,
              "n_tiles": n_tiles, "seed": seed}
    inputs = [input_identity(gsd_ortho, "ortho")]
    if aoi_path:
        inputs.append(input_identity(aoi_path, "aoi"))
    if stems_path:
        inputs.append(input_identity(stems_path, "stems"))
    write_manifest(out_dir, "layout", params, inputs,
                   [{"path": jsonl, "count": len(kept)},
                    {"path": fp_path, "count": len(kept)}], stats)
    if not quiet:
        print(stats)
    return stats
