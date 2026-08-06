"""Rasterize polygon stem annotations onto an orthomosaic's grid.

    python scripts/rasterize_annotations.py --shapefile stems.shp --ortho ortho.tif \
        --out stem_map.tif [--instances inst.tif] [--species GFI RBU] [--id-field id]

The WINMOL training shapefiles hold one traced polygon per stem, with `id` and
`Species` attributes (median aspect ratio ~7:1, ~3 m long and ~0.4 m wide — stem
outlines, not crowns). This burns them onto the ortho's exact grid so the result
drops straight into build_ortho_pairs.py / build_site_splits.py, which expect a
raster where >0 means stem.

Two outputs are available from the same polygons:

  * `--out` — binary mask, the label the segmenter trains on.
  * `--instances tree` — one id per **tree**, taken from the `id` attribute. That
    attribute is not a polygon key: in Barnekow_5, 1012 polygons carry only 332
    distinct ids, and the polygons sharing one are collinear (measured centroid
    collinearity 0.006) and spaced ~8.7 m apart. One windthrown stem was traced as
    several segments along its length, broken where branches or other stems occlude
    it, and the `id` records which fragments are the same tree.

    That is stem continuity across occlusion — the thing a downstream vectorizer
    normally has to infer — already labelled by hand.

  * `--instances segment` — one id per polygon instead, i.e. the visible pieces
    treated as separate objects.

**Geometries are reprojected when the shapefile's CRS differs from the ortho's.**
The corpus mixes EPSG:25833 and EPSG:32633 — the same UTM zone on different
datums, so coordinates look plausible either way and a silent mismatch would
shift masks by metres rather than failing.
"""
import argparse
import sys

import numpy as np


def _open_annotations(path, species=None):
    """Yield (geometry, id) for the polygons to burn, in file order."""
    import fiona

    with fiona.open(path) as src:
        crs = src.crs
        keep = []
        for i, f in enumerate(src):
            props = f["properties"] or {}
            sp = props.get("Species")
            if species:
                # the corpus contains both 'RBU' and 'rBU'; matching case-sensitively
                # would silently drop 15 beech polygons
                if sp is None or str(sp).strip().upper() not in species:
                    continue
            fid = props.get("id")
            keep.append((f["geometry"], int(fid) if fid is not None else i + 1))
    return crs, keep


def rasterize(shapefile, ortho_path, out_path, instances_path=None, species=None,
              all_touched=False, instance_level="tree"):
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.warp import transform_geom

    src = rasterio.open(ortho_path)
    ann_crs, feats = _open_annotations(shapefile, species)
    if not feats:
        raise SystemExit(f"no polygons selected from {shapefile}"
                         + (f" for species {sorted(species)}" if species else ""))

    reprojected = 0
    if ann_crs and src.crs and str(ann_crs) != str(src.crs):
        feats = [(transform_geom(ann_crs, src.crs, g), i) for g, i in feats]
        reprojected = len(feats)
        print(f"reprojected {reprojected} geometries {ann_crs} -> {src.crs}")

    shape = (src.height, src.width)
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="deflate", nodata=None)
    for k in ("photometric", "alpha", "interleave"):
        profile.pop(k, None)

    binary = rio_rasterize([(g, 255) for g, _ in feats], out_shape=shape,
                           transform=src.transform, fill=0, dtype="uint8",
                           all_touched=all_touched)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(binary, 1)

    cov = float((binary > 0).mean())
    print(f"{len(feats)} polygons -> {out_path}  ({100 * cov:.2f}% of the ortho is stem)")
    if cov == 0:
        raise SystemExit(
            "the mask is empty: the polygons do not overlap the ortho. Check that this "
            "shapefile belongs to this orthomosaic, and that both carry a CRS.")

    if instances_path:
        if instance_level == "segment":
            feats = [(g, k + 1) for k, (g, _) in enumerate(feats)]
        ids = [i for _, i in feats]
        n_objects = len(set(ids))
        if max(ids) > 65535:
            raise SystemExit(f"instance id {max(ids)} exceeds uint16; use a smaller id field")
        # later polygons win where they overlap, which matches how a modal instance
        # label behaves: the visible stem owns the pixel
        inst = rio_rasterize([(g, i) for g, i in feats], out_shape=shape,
                             transform=src.transform, fill=0, dtype="uint16",
                             all_touched=all_touched)
        iprof = profile.copy()
        iprof.update(dtype="uint16")
        with rasterio.open(instances_path, "w", **iprof) as dst:
            dst.write(inst, 1)
        present = len(np.unique(inst)) - 1
        print(f"{present}/{n_objects} {instance_level} instances rasterized "
              f"(from {len(feats)} polygons) -> {instances_path}")
        if present < n_objects:
            print(f"  note: {n_objects - present} objects are absent — fully overlapped "
                  f"by a later polygon, or smaller than a pixel", file=sys.stderr)
    return binary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shapefile", required=True, help="polygon stem annotations (.shp/.gpkg)")
    p.add_argument("--ortho", required=True, help="orthomosaic defining the output grid")
    p.add_argument("--out", required=True, help="binary stem mask GeoTIFF")
    p.add_argument("--instances", default=None,
                   help="write a uint16 instance raster to this path")
    p.add_argument("--instance-level", choices=("tree", "segment"), default="tree",
                   help="'tree' groups polygons by the id attribute (fragments of one "
                        "stem share a value); 'segment' gives every polygon its own")
    p.add_argument("--species", nargs="*", default=None,
                   help="keep only these species codes (case-insensitive), e.g. GFI RBU")
    p.add_argument("--all-touched", action="store_true",
                   help="burn every pixel a polygon touches; default covers pixel centres "
                        "only, which thins narrow stems")
    a = p.parse_args(argv)
    species = {s.strip().upper() for s in a.species} if a.species else None
    rasterize(a.shapefile, a.ortho, a.out, a.instances, species, a.all_touched,
              a.instance_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
