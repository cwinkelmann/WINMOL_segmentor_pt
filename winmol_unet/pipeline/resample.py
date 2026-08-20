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


def resample(ortho, out_root, gsd, jpeg_quality=95, resampling="average", quiet=False):
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
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

        if abs(ratio - 1.0) < 1e-9:
            # Native is the source, not a re-encode of it. A copy would cost a JPEG
            # generation for nothing.
            if os.path.lexists(out_path):
                os.remove(out_path)
            os.symlink(os.path.abspath(ortho), out_path)
            width, height = src.width, src.height
        else:
            width = int(src.width / ratio)
            height = int(src.height / ratio)
            transform = rasterio.Affine(gsd, 0.0, src.bounds.left,
                                        0.0, -gsd, src.bounds.top)
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
                        src_win = Window(i * ratio, j * ratio, bw * ratio, bh * ratio)
                        data = src.read(window=src_win, out_shape=(src.count, bh, bw),
                                        resampling=kernel, boundless=True, fill_value=0)
                        dst.write(data, window=win)
                        # A plain decimated read_masks gives the true coverage
                        # fraction (see module docstring); >=128 is the majority vote.
                        m = src.read_masks(1, window=src_win, out_shape=(bh, bw),
                                           resampling=kernel, boundless=True)
                        dst.write_mask(
                            np.where(m >= 128, 255, 0).astype("uint8"), window=win)
                # After the pixels, never before: build_overviews on an empty dataset
                # decimates nothing and is not refreshed by the later writes.
                dst.build_overviews([2, 4, 8, 16, 32], kernel)

    stats = {"path": out_path, "width": width, "height": height,
             "gsd": gsd, "ratio": ratio}
    write_manifest(out_dir, "resample",
                   {"gsd": gsd, "resampling": resampling,
                    "jpeg_quality": jpeg_quality},
                   [input_identity(ortho, "ortho")],
                   [{"path": out_path, "count": 1}], stats)
    if not quiet:
        print(stats)
    return stats
