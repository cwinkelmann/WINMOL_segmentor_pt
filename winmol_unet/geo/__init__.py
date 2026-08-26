"""Georeferenced I/O: orthomosaic in, training tiles or a scored stem map out.

Everything here needs the `[geo]` extra (rasterio, fiona, shapely). That is deliberately
NOT part of the base install: the analyzer installs `winmol_unet` to serve ONNX and must
not be made to pull GDAL. Nothing is imported at module level for the same reason —
`import winmol_unet.geo` stays free, and each submodule imports its own dependencies.

The modules, and why each exists rather than being a few lines inline:

  geometry   repair self-intersecting stem polygons. 50 of the corpus's 3,842 outlines
             self-intersect (hand-tracing produces them readily) and GEOS raises on the
             first set operation that touches one — tens of thousands of tiles into a
             sampling run. Has to happen before anything else.
  rasterize  burn polygon annotations onto an ortho's exact grid, as a binary mask and
             optionally an instance raster.
  sample     the rotated-tile sampler ported from the original R generator. A plain grid
             over one of these orthomosaics is ~99% background, because stems cover about
             1% of a site; this samples inside the windthrow polygon at random rotations
             and oversamples heavily. Writes tiles.jsonl.
  splits     leak-free train/val/test splits: by spatial block, by halving a site, or by
             holding whole sites out. Oversampled tiles overlap, so a random tile split
             leaks — this is the module that stops that.
  folds      compose leave-one-site-out folds from per-site tile sets by symlink, so the
             same ground is sampled once rather than once per fold.
  parallel   run `splits` per site in its own process and merge, reassigning tile indices
             (each part numbers from 1, so a naive merge silently overwrites).
  predict    run a model over an AOI in world space and resample onto a common reference
             grid. Two models trained at different ground resolutions cannot be compared
             on their own tile sets — a stem is thicker in pixels on the finer one, so
             tile F1 measures the exam, not the model.
  score      score a stem map against digitised stems on that common grid, masked to the
             AOI. Outside the windthrow polygon stems are real but were never digitised,
             so scoring the whole raster counts correct detections as false positives and
             penalises the better model hardest. The masking is not optional.

`tiles.jsonl` (source ortho, world centre, rotation, GSD, stem fraction) is written by
every sampling path. It is what makes a split auditable for leakage after the fact, and
what `locate_tile` in the helper repo uses to re-cut a tile at native resolution.
"""
