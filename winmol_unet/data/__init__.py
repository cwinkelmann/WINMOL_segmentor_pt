"""Dataset format conversion: existing image/mask folders -> the loader convention.

The counterpart to `winmol_unet.geo`, which starts from georeferenced source data. These
modules start from data that is already raster pairs, and only need PIL and numpy — no
GDAL — so they stay out of the `[geo]` extra.

The loader convention is `train/train{N}.jpeg` + `mask/mask{N}.gif`, paired by the integer
N, with binary masks. Everything here produces exactly that, so a dataset from any of
these paths is interchangeable with one `prepare.py` cut from an orthomosaic.

  build   pair an arbitrary image/mask folder by shared filename key, renumber, binarise
  coco    rasterise COCO polygon annotations into masks
  split   materialise a fixed train/val split, matching the loader's own seeded split

Nothing is imported here: `import winmol_unet.data` must stay as cheap as
`import winmol_unet` (see tests/test_import_boundary.py).
"""
