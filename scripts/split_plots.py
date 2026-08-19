"""Split a plot-structured survey GeoPackage into one stems+AOI pair per sample plot.

Surveys like Revier 12/13 digitise a few widely separated sample plots (Probekreise)
rather than one contiguous area: each plot is its own layer, and the `_AOI` layer holds
one polygon per plot. `make_splits.py` wants a stems path and an aoi path per "site", so
each plot becomes a site — which is also the natural leak-free split, since plots sit
hundreds of metres to kilometres apart.

Geometry is repaired on the way out (`make_valid`): these layers contain self-touching
rings that make `unary_union` raise, and a plot silently dropped for that reason would
shrink the test set without saying so.

    python scripts/split_plots.py --gpkg Rev_12.gpkg --out plots/ --prefix R12
"""
import argparse
import json
import os


def _valid(g):
    """Repair, then coerce to MultiPolygon.

    `make_valid` can turn a Polygon into a MultiPolygon or a GeometryCollection (it splits
    self-touching rings), which a Polygon-typed layer then refuses to accept. Keeping only
    the polygonal parts and always writing MultiPolygon avoids both the refusal and the
    silent loss of a repaired plot.
    """
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.validation import make_valid
    if not g.is_valid:
        g = make_valid(g)
    if isinstance(g, GeometryCollection):
        parts = [x for x in g.geoms if isinstance(x, (Polygon, MultiPolygon))]
        g = MultiPolygon([p for x in parts
                          for p in (x.geoms if isinstance(x, MultiPolygon) else [x])])
    if isinstance(g, Polygon):
        g = MultiPolygon([g])
    return g


def split(gpkg, out_dir, prefix, quiet=False):
    import fiona
    from shapely.geometry import mapping, shape

    layers = fiona.listlayers(gpkg)
    aoi_layers = [l for l in layers if l.endswith("_AOI")]
    if len(aoi_layers) != 1:
        raise SystemExit(f"expected exactly one *_AOI layer in {gpkg}, got {aoi_layers}")
    with fiona.open(gpkg, layer=aoi_layers[0]) as src:
        aoi_crs = src.crs
        aois = [_valid(shape(f["geometry"])) for f in src if f["geometry"]]

    stem_layers = [l for l in layers if l != aoi_layers[0]]
    out = []
    for i, lyr in enumerate(sorted(stem_layers)):
        with fiona.open(gpkg, layer=lyr) as src:
            crs, schema = src.crs, src.schema
            feats = [(f, _valid(shape(f["geometry"]))) for f in src if f["geometry"]]
        if not feats:
            raise SystemExit(f"layer {lyr!r} has no geometry — refusing a silent empty plot")

        # Which AOI polygon is this plot in? Match by containment of the stem centroids,
        # not by layer name: the naming is inconsistent across surveys.
        cx = sum(g.centroid.x for _, g in feats) / len(feats)
        cy = sum(g.centroid.y for _, g in feats) / len(feats)
        from shapely.geometry import Point
        c = Point(cx, cy)
        hits = [a for a in aois if a.buffer(1.0).contains(c)]
        if len(hits) != 1:
            hits = sorted(aois, key=lambda a: a.centroid.distance(c))[:1]
            if not quiet:
                print(f"  {lyr}: no unique AOI by containment, took the nearest")
        aoi = hits[0]

        name = f"{prefix}-P{i + 1}"
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        sp = os.path.join(d, "stems.gpkg")
        ap = os.path.join(d, "aoi.gpkg")
        schema = {**schema, "geometry": "MultiPolygon"}
        with fiona.open(sp, "w", driver="GPKG", crs=crs, schema=schema) as dst:
            for f, g in feats:
                dst.write({"geometry": mapping(g), "properties": f["properties"]})
        with fiona.open(ap, "w", driver="GPKG", crs=aoi_crs,
                        schema={"geometry": "MultiPolygon", "properties": {}}) as dst:
            dst.write({"geometry": mapping(aoi), "properties": {}})

        rec = {"plot": name, "source_layer": lyr, "stems": sp, "aoi": ap,
               "polygons": len(feats), "stem_area_m2": round(sum(g.area for _, g in feats), 1),
               "aoi_area_m2": round(aoi.area, 1),
               "centroid": [round(cx, 1), round(cy, 1)], "crs": str(crs)}
        out.append(rec)
        if not quiet:
            print(f"  {name}: {rec['polygons']:5d} polys  {rec['stem_area_m2']:8.0f} m2 stem  "
                  f"AOI {rec['aoi_area_m2']:9,.0f} m2   <- {lyr}")
    with open(os.path.join(out_dir, f"{prefix}-plots.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpkg", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", required=True, help="plot name prefix, e.g. R12")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    split(a.gpkg, a.out, a.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
