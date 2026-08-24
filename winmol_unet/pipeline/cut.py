"""Stage 5 — read the windows stage 4 chose.

Because every footprint was snapped to this raster's own pixel grid, the window is an
integer window and the read is a copy. Nothing is resampled here, which is the point;
`tests/test_pipeline_cut.py` asserts the bytes match a direct read.

The mask comes from the same window of the label raster produced by `--rasterize`
against this GSD's grid, so image and mask are aligned by construction rather than by
a second rasterisation that could drift. With no label raster the mask is all zeros,
which is the right answer for a tile with no annotated stems in it.

Output is `<out>/train/train<id>.tif` + `<out>/mask/mask<id>.tif`. The `train`/`mask`
prefix is on the FILENAME, not merely the directory: `data/build.py:_shared_key` pairs by
stripping that literal prefix off the stem and returns None without it, so bare `<id>.tif`
would leave `--from-folder` with nothing to pair and raise. With the prefix, stage 6 runs
unmodified.
"""
import json
import os

from winmol_unet.pipeline.manifest import input_identity, write_manifest


def cut(layout_dir, gsd_ortho, out_dir, stem_map=None, min_valid_frac=0.5,
        min_stem_frac=0.0, quiet=False):
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    recs = [json.loads(l) for l in open(os.path.join(layout_dir, "tiles.jsonl"))]
    img_dir = os.path.join(out_dir, "train")
    msk_dir = os.path.join(out_dir, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    written, dropped = [], 0
    src = rasterio.open(gsd_ortho)
    lbl = rasterio.open(stem_map) if stem_map else None
    try:
        for r in recs:
            col = int(round((r["minx"] - src.transform.c) / src.res[0]))
            row = int(round((src.transform.f - r["maxy"]) / src.res[1]))
            win = Window(col, row, r["width_px"], r["height_px"])

            # A footprint from a skirt candidate (min_aoi_frac < 1) or from an AOI-less
            # run over the raster's own bounds can overhang the raster edge. A plain
            # (non-boundless) read silently returns a SMALLER array for such a window --
            # not an error -- and `dst.write` below would then stretch it nearest-
            # neighbour to width_px x height_px, georeferenced as if it covered the
            # full footprint: a corrupted, resampled tile that looks byte-identical
            # everywhere the eye would check. boundless=True keeps the read exactly
            # width_px x height_px, with the off-raster part filled at 0 -- real pixels
            # where the raster has them, explicit fill everywhere else, never a stretch.
            data = src.read(window=win, boundless=True, fill_value=0)
            valid = src.read_masks(1, window=win, boundless=True)
            valid_frac = float((valid > 0).mean())

            if lbl is not None:
                # lbl shares gsd_ortho's exact grid (winmol_unet/geo/rasterize.py writes
                # it with the same transform and shape), so the same window can overhang
                # it the same way; boundless=True fill_value=0 says "no stem" for ground
                # the label raster doesn't cover, which is already excluded from training
                # by valid_frac above.
                mask = lbl.read(1, window=win, boundless=True, fill_value=0)
            else:
                mask = np.zeros((r["height_px"], r["width_px"]), dtype="uint8")
            mask = np.where(mask > 0, 255, 0).astype("uint8")
            stem_frac = float((mask > 0).mean())

            if valid_frac < min_valid_frac or stem_frac < min_stem_frac:
                dropped += 1
                continue

            transform = rasterio.Affine(src.res[0], 0.0, r["minx"],
                                        0.0, -src.res[1], r["maxy"])
            common = {"driver": "GTiff", "width": r["width_px"],
                      "height": r["height_px"], "crs": src.crs,
                      "transform": transform, "compress": "DEFLATE"}
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(os.path.join(img_dir, "train" + r["id"] + ".tif"), "w",
                                   count=src.count, dtype="uint8", **common) as dst:
                    dst.write(data)
                    dst.write_mask(valid)
                with rasterio.open(os.path.join(msk_dir, "mask" + r["id"] + ".tif"), "w",
                                   count=1, dtype="uint8", **common) as dst:
                    dst.write(mask, 1)

            out = dict(r)
            out.update({"valid_frac": valid_frac, "stem_frac": stem_frac,
                        "stage": "cut"})
            written.append(out)
    finally:
        src.close()
        if lbl is not None:
            lbl.close()

    jsonl = os.path.join(out_dir, "tiles.jsonl")
    with open(jsonl, "w") as fh:
        for r in written:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    stats = {"n_written": len(written), "n_dropped_exact": dropped}
    inputs = [input_identity(os.path.join(layout_dir, "tiles.jsonl"), "layout"),
              input_identity(gsd_ortho, "ortho")]
    if stem_map:
        inputs.append(input_identity(stem_map, "stem_map"))
    write_manifest(out_dir, "cut",
                   {"min_valid_frac": min_valid_frac, "min_stem_frac": min_stem_frac},
                   inputs, [{"path": img_dir, "count": len(written)},
                            {"path": msk_dir, "count": len(written)}], stats)
    if not quiet:
        print(stats)
    return stats
