"""Stage 1 — read the source vectors, put them on the orthomosaic's CRS, keep the
AOIs asked for, and write the result where it can be opened in QGIS.

The Revier 13 file is the reason this is a stage rather than a step: its stems are
EPSG:4326 while its AOIs and its orthomosaic are EPSG:32633, so a spatial join
across the two returns zero rows without raising anything. Fixing that once, here,
lets every later stage assume a single CRS and fail loudly if that breaks.

The layer holding the stems there is named `hard_negative_AOI` -- the name describes
the purpose of the collection, not the content of the layer -- so the layer names are
parameters, never guessed.
"""
import os

from winmol_unet.pipeline.manifest import input_identity, write_manifest


def _repair(geoms):
    """Make hand-digitised polygons safe for GEOS set operations.

    Same job as `geo.sample._repair`; kept local because that one prints and takes a
    label, and this stage reports through its return value and manifest instead.
    """
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon

    out, repaired = [], 0
    for g in geoms:
        if g.is_valid:
            out.append(g)
            continue
        fixed = make_valid(g)
        parts = ([fixed] if isinstance(fixed, Polygon)
                 else list(fixed.geoms) if isinstance(fixed, MultiPolygon)
                 else [p for p in getattr(fixed, "geoms", []) if isinstance(p, Polygon)])
        if parts:
            out.append(parts[0] if len(parts) == 1 else MultiPolygon(parts))
            repaired += 1
    return out, repaired


def _read_layer(path, layer, dst_crs):
    """Geometries + properties from one layer, reprojected to `dst_crs`."""
    import fiona
    from rasterio.warp import transform_geom
    from shapely.geometry import shape

    with fiona.open(path, layer=layer) as src:
        crs = src.crs
        recs = [(f["geometry"], dict(f["properties"] or {}), f.get("id"))
                for f in src if f["geometry"] is not None]

    if not crs:
        # This stage exists to catch exactly this class of silent CRS defect --
        # a layer with no CRS at all would otherwise pass the equality check below
        # and be written through unreprojected, on the unverified assumption that
        # it already matches the orthomosaic.
        raise ValueError(f"{path!r} layer {layer!r} has no CRS; refusing to assume "
                          "it already matches the orthomosaic")

    reprojected = 0
    if dst_crs and str(crs) != str(dst_crs):
        recs = [(transform_geom(crs, dst_crs, g), p, i) for g, p, i in recs]
        reprojected = len(recs)
    geoms, repaired = _repair([shape(g) for g, _, _ in recs])
    return geoms, [p for _, p, _ in recs], reprojected, repaired


def _species(p):
    """The species code, verbatim, from whichever spelling the corpus used.

    Revier 13 spells the field lowercase `species` and has typed it both `Integer`
    and absent entirely between reads (it has an open WAL). The older corpus spells
    it `Species`, capitalised, with text codes (`GFI`, `RBU`, `DGL`). Both are read
    here, `species` first, so one code path serves both -- stored as text and
    verbatim, never coerced to int, since a text code cannot survive that coercion.
    """
    v = p.get("species")
    if v is None:
        v = p.get("Species")
    return "" if v is None else str(v)


def ingest(gpkg, ortho, out_dir, stems_layer, aoi_layer=None, aoi_ids=None,
           species=None, quiet=False):
    import fiona
    import rasterio
    from shapely.geometry import mapping

    with rasterio.open(ortho) as src:
        dst_crs = src.crs

    stems, stem_props, reprojected, repaired = _read_layer(gpkg, stems_layer, dst_crs)
    if species:
        # `species` here is expected already upper-cased and stripped by the caller
        # (the CLI does this before passing it in); only the stem's own value is
        # normalised on this side of the comparison.
        keep = [i for i, p in enumerate(stem_props)
                if _species(p).strip().upper() in species]
        stems = [stems[i] for i in keep]
        stem_props = [stem_props[i] for i in keep]

    aois, aoi_props = [], []
    if aoi_layer:
        aois, aoi_props, _, _ = _read_layer(gpkg, aoi_layer, dst_crs)
        # AOIs are keyed by their position in the layer; the Revier 13 AOI layer
        # carries no attributes at all, so fid order is the only identifier there is.
        ids = list(range(1, len(aois) + 1))
        if aoi_ids:
            sel = [i for i, k in enumerate(ids) if k in set(aoi_ids)]
            aois = [aois[i] for i in sel]
            ids = [ids[i] for i in sel]
        aoi_props = [{"aoi_id": k} for k in ids]

    os.makedirs(out_dir, exist_ok=True)

    tagged = []
    for g, p in zip(stems, stem_props):
        aoi_id = 0
        # First-intersection-wins: a stem intersecting two AOIs is tagged with
        # whichever AOI comes first in the layer's iteration order. Undefined for
        # overlapping AOIs -- fine for the current corpus, worth revisiting if that
        # changes.
        for poly, ap in zip(aois, aoi_props):
            if poly.intersects(g):
                aoi_id = ap["aoi_id"]
                break
        if aois and aoi_id == 0:
            continue
        tagged.append((g, {"stem_id": int(p.get("stem_id") or 0),
                           "species": _species(p),
                           "old_tree": int(p.get("old_tree") or 0),
                           "aoi_id": aoi_id}))

    stems_path = os.path.join(out_dir, "stems.gpkg")
    schema = {"geometry": "Unknown",
              "properties": {"stem_id": "int", "species": "str",
                             "old_tree": "int", "aoi_id": "int"}}
    with fiona.open(stems_path, "w", driver="GPKG", layer="stems",
                    crs=dst_crs.to_string(), schema=schema) as dst:
        for g, p in tagged:
            dst.write({"geometry": mapping(g), "properties": p})

    outputs = [{"path": stems_path, "count": len(tagged)}]
    if aoi_layer:
        aoi_path = os.path.join(out_dir, "aoi.gpkg")
        with fiona.open(aoi_path, "w", driver="GPKG", layer="aoi",
                        crs=dst_crs.to_string(),
                        schema={"geometry": "Unknown",
                                "properties": {"aoi_id": "int"}}) as dst:
            for g, p in zip(aois, aoi_props):
                dst.write({"geometry": mapping(g), "properties": p})
        outputs.append({"path": aoi_path, "count": len(aois)})

    stats = {"n_stems": len(stems), "n_stems_in_aoi": len(tagged), "n_aoi": len(aois),
             "reprojected": reprojected, "repaired": repaired,
             "crs": dst_crs.to_string()}
    params = {"stems_layer": stems_layer, "aoi_layer": aoi_layer,
              "aoi_ids": sorted(aoi_ids) if aoi_ids else None,
              "species": sorted(species) if species else None}
    write_manifest(out_dir, "ingest", params,
                   [input_identity(gpkg, "vectors"), input_identity(ortho, "ortho")],
                   outputs, stats)
    if not quiet:
        print(stats)
    return stats
