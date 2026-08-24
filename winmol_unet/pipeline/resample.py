"""Stage 2 — a full-size copy of the orthomosaic at one ground sample distance.

`average` is a box mean over exactly the source pixels each target pixel covers. It
matches how GDAL builds overviews and loses no energy. `bilinear` would sample rather
than integrate, and at the 15.6x this pipeline asks for at 20 cm that aliases: fine
canopy detail becomes false texture a model can learn.

The mask band is resampled the same way and then thresholded at 50%, so a target
pixel is valid when the majority of its source pixels were. That is deliberately the
same rule `--min-valid-frac` applies to a whole tile, one scale down.

Writing at quality 95 rather than the ~80 default costs disk and buys back most of one
JPEG generation: the source is already lossy YCbCr JPEG, and the export stage will
encode once more.

This stage never reprojects -- the output CRS is always the source's -- so the
downsampling is done with plain windowed, decimated reads (`out_shape` + `resampling`
on `read`/`read_masks`) rather than a `WarpedVRT`. That distinction is not cosmetic:
empirically (GDAL 3.9.2), a `WarpedVRT`'s `average`-resampled alpha/mask output is not
graded at all here -- `dataset_mask()` through the VRT returns 255 for a target pixel
even when zero of its source pixels were valid. A plain decimated `read_masks` call,
by contrast, returns the true coverage fraction (verified against a hand-built 4x4
block: 4 of 16 source pixels valid reads back as 64 = 255 * 4/16), and a plain
decimated `read` on the colour bands is correctly weighted by that same source mask
(a block with one valid row at value 100 and three invalid rows at 0 reads back as
100, not the unweighted 25) -- both are exactly what a manual box mean over only the
valid source pixels would give.
"""
import os

from winmol_unet.pipeline.manifest import input_identity, write_manifest

BLOCK = 2048


def gsd_dir_name(gsd):
    """`0.05 -> 'gsd_050'`. Millimetres, so the names sort the way the scales do."""
    return "gsd_%03d" % int(round(gsd * 1000))


def _clip_window(src, clip_aoi, aoi_ids, ratio):
    """The source-pixel window covering the selected AOIs, snapped outward to whole pixels.

    Snapping outward to whole SOURCE pixels (floor the near edge, ceil the far edge) matters:
    a window with fractional edges would make the clipped raster's origin fall between source
    pixels, and every average-resampled target pixel would then be a different blend of source
    pixels than the tile an unclipped run produces.

    That is necessary but not sufficient once the output is JPEG-compressed. Verified
    empirically (GDAL 3.9.2 / libjpeg, photometric=YCbCr): JPEG's 4:2:0 chroma subsampling
    groups pixels into 16x16 MCU blocks relative to each *file's own* tile grid, which always
    starts fresh at local pixel (0, 0). Two files holding bit-identical pre-compression pixel
    data reconstruct different bytes after a JPEG round trip unless the region lands on the
    same MCU phase in both -- checked against a hand-built pair of rasters holding the same
    150x150 block: written at a source-pixel-exact but MCU-unaligned offset (e.g. 50, not a
    multiple of 16) the two do not match (up to 94/255 off per pixel, spread over the whole
    block, not just the edges); snapped so the offset AND size are both multiples of 16 they
    match exactly (0 pixels differ). So the window is snapped a second time, outward again, to
    whole 16-pixel blocks in the *target* GSD's grid (`16 * ratio` source pixels), on top of the
    whole-source-pixel snap above. This makes the window somewhat larger than the tightest box
    that covers the AOIs -- the extra ground outside the AOI polygons is still marked invalid by
    the mask burn-in, so it costs a few invalid border pixels, not correctness.
    """
    import fiona
    import numpy as np
    from rasterio.windows import Window, from_bounds
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with fiona.open(clip_aoi, layer="aoi") as lyr:
        geoms = []
        for i, f in enumerate(lyr, start=1):
            aid = (f["properties"] or {}).get("aoi_id") or i
            if aoi_ids and int(aid) not in set(aoi_ids):
                continue
            geoms.append(shape(f["geometry"]))
    if not geoms:
        raise ValueError(
            f"no AOI matched {aoi_ids!r} in {clip_aoi}; refusing to clip to nothing")

    union = unary_union(geoms)
    win = from_bounds(*union.bounds, transform=src.transform)
    col_off = int(np.floor(win.col_off))
    row_off = int(np.floor(win.row_off))
    far_col = int(np.ceil(win.col_off + win.width))
    far_row = int(np.ceil(win.row_off + win.height))

    # JPEG MCU alignment (see docstring): 16 target-GSD pixels is `16 * ratio` source pixels.
    mcu = 16 * ratio
    col_off = int(np.floor(col_off / mcu) * mcu)
    row_off = int(np.floor(row_off / mcu) * mcu)
    far_col = int(np.ceil(far_col / mcu) * mcu)
    far_row = int(np.ceil(far_row / mcu) * mcu)

    col_off = max(0, col_off)
    row_off = max(0, row_off)
    width = min(far_col - col_off, src.width - col_off)
    height = min(far_row - row_off, src.height - row_off)
    return Window(col_off, row_off, width, height), geoms


def resample(ortho, out_root, gsd, clip_aoi=None, aoi_ids=None, jpeg_quality=95,
             resampling="average", quiet=False):
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.windows import Window

    out_dir = os.path.join(out_root, gsd_dir_name(gsd))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ortho.tif")

    with rasterio.open(ortho) as src:
        src_gsd = src.res[0]
        ratio = gsd / src_gsd
        if ratio < 0.999:
            raise ValueError(
                f"requested {gsd} m/px is finer than the source {src_gsd} m/px; "
                "this pipeline downsamples only")

        clip_win, clip_geoms = None, None
        if clip_aoi:
            clip_win, clip_geoms = _clip_window(src, clip_aoi, aoi_ids, ratio)
            src_w, src_h = clip_win.width, clip_win.height
            left = src.transform.c + clip_win.col_off * src.res[0]
            top = src.transform.f - clip_win.row_off * src.res[1]
        else:
            src_w, src_h = src.width, src.height
            left, top = src.bounds.left, src.bounds.top
        clip_off_x = clip_win.col_off if clip_win is not None else 0
        clip_off_y = clip_win.row_off if clip_win is not None else 0

        # A clip is a real change to the raster, so it always writes -- the native-GSD
        # symlink passthrough only applies when nothing is being clipped.
        if abs(ratio - 1.0) < 1e-9 and not clip_aoi:
            # Native is the source, not a re-encode of it. A copy would cost a JPEG
            # generation for nothing.
            if os.path.lexists(out_path):
                os.remove(out_path)
            os.symlink(os.path.abspath(ortho), out_path)
            width, height = src.width, src.height
        else:
            width = int(src_w / ratio)
            height = int(src_h / ratio)
            transform = rasterio.Affine(gsd, 0.0, left, 0.0, -gsd, top)
            kernel = getattr(Resampling, resampling)
            profile = {"driver": "GTiff", "width": width, "height": height,
                       "count": src.count, "dtype": "uint8", "crs": src.crs,
                       "transform": transform, "tiled": True,
                       "blockxsize": 512, "blockysize": 512,
                       "compress": "JPEG", "jpeg_quality": jpeg_quality,
                       "photometric": "YCbCr" if src.count == 3 else "MINISBLACK",
                       "BIGTIFF": "IF_SAFER"}
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), \
                    rasterio.open(out_path, "w", **profile) as dst:
                for j in range(0, height, BLOCK):
                    for i in range(0, width, BLOCK):
                        bw = min(BLOCK, width - i)
                        bh = min(BLOCK, height - j)
                        win = Window(i, j, bw, bh)
                        # Same-CRS scaling only, so the source window is a plain
                        # multiply by ratio -- no warp needed.
                        src_win = Window(clip_off_x + i * ratio, clip_off_y + j * ratio,
                                         bw * ratio, bh * ratio)
                        data = src.read(window=src_win, out_shape=(src.count, bh, bw),
                                        resampling=kernel, boundless=True, fill_value=0)
                        dst.write(data, window=win)
                        # A plain decimated read_masks gives the true coverage
                        # fraction (see module docstring); >=128 is the majority vote.
                        m = src.read_masks(1, window=src_win, out_shape=(bh, bw),
                                           resampling=kernel, boundless=True)
                        m = np.where(m >= 128, 255, 0).astype("uint8")
                        if clip_geoms is not None:
                            inside = geometry_mask(
                                clip_geoms, out_shape=(bh, bw), invert=True,
                                transform=rasterio.windows.transform(win, transform))
                            m = np.where(inside, m, 0)
                        dst.write_mask(m, window=win)
                # After the pixels, never before: build_overviews on an empty dataset
                # decimates nothing and is not refreshed by the later writes.
                dst.build_overviews([2, 4, 8, 16, 32], kernel)

    clip_bounds = None
    if clip_win is not None:
        clip_bounds = [left, top - height * gsd, left + width * gsd, top]
    stats = {"path": out_path, "width": width, "height": height,
             "gsd": gsd, "ratio": ratio, "clipped": clip_aoi is not None,
             "clip_bounds": clip_bounds}
    params = {"gsd": gsd, "resampling": resampling, "jpeg_quality": jpeg_quality}
    if clip_aoi:
        # Only added when clipping is actually used, so a manifest written by an
        # unclipped call is byte-for-byte the same shape it always was -- is_stale()
        # compares this dict verbatim, and an old-style params dict without these keys
        # must still read back as "not stale" for a plain, unclipped resample() call.
        params["clip_aoi"] = clip_aoi
        params["aoi_ids"] = sorted(aoi_ids) if aoi_ids else None
    write_manifest(out_dir, "resample", params,
                   [input_identity(ortho, "ortho")],
                   [{"path": out_path, "count": 1}], stats)
    if not quiet:
        print(stats)
    return stats
