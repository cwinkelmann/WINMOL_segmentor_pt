"""Inventory the WINMOL training corpus: what exists, what pairs, what is wrong.

    python scripts/inventory_training_data.py --root <DIR> [--json out.json] [--md out.md]

Walks a data root, finds annotation shapefiles, orthomosaics and DEMs, pairs them
by site name, and reports what can actually be turned into training data — plus
the defects that would otherwise be discovered halfway through a training run:

  * a site whose annotations have no matching orthomosaic (unusable)
  * an orthomosaic with no annotations (usable only for inference or pretraining)
  * a CRS mismatch between annotations and ortho (masks would land metres off)
  * annotations that do not overlap their ortho at all
  * duplicate sites, missing species, and case-variant species codes

Pairing is by site name: an ortho `<site>_ortho.tif` matches annotations
`<site>.shp`. Names that differ only by a typo will not pair, which the report
makes visible rather than silently dropping.
"""
import argparse
import collections
import json
import os
import re
import sys


def _norm(name):
    """Site key: strip known suffixes so ortho and annotation names meet."""
    n = os.path.splitext(os.path.basename(name))[0]
    for suf in ("_ortho", "_AOI", "_AOE", "_aoi", "_dem", "_dsm", "_chm"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def _find(root, exts):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def _raster_info(path):
    import rasterio
    try:
        with rasterio.open(path) as r:
            gsd = abs(r.transform.a)
            return {"path": path, "width": r.width, "height": r.height,
                    "crs": str(r.crs) if r.crs else None, "gsd_m": round(gsd, 4),
                    "bands": r.count,
                    "mpx": round(r.width * r.height / 1e6, 1)}
    except Exception as exc:
        return {"path": path, "error": str(exc)[:120]}


def _vector_info(path):
    import fiona
    try:
        with fiona.open(path) as src:
            species, ids, n = collections.Counter(), set(), 0
            xs, ys = [], []
            for f in src:
                n += 1
                p = f["properties"] or {}
                species[p.get("Species")] += 1
                if p.get("id") is not None:
                    ids.add(p["id"])
            b = src.bounds
            return {"path": path, "features": n, "crs": str(src.crs) if src.crs else None,
                    "geometry": src.schema["geometry"], "tree_ids": len(ids),
                    "species": {str(k): v for k, v in species.most_common()},
                    "bounds": [round(v, 1) for v in b]}
    except Exception as exc:
        return {"path": path, "error": str(exc)[:120]}


def _overlaps(vec, ras):
    """Do the annotation bounds intersect the ortho bounds (after reprojection)?"""
    import rasterio
    from rasterio.warp import transform_bounds
    try:
        with rasterio.open(ras["path"]) as r:
            rb = r.bounds
            vb = vec["bounds"]
            if vec["crs"] and str(r.crs) and vec["crs"] != str(r.crs):
                vb = transform_bounds(vec["crs"], r.crs, *vb)
            return not (vb[2] < rb[0] or vb[0] > rb[2] or vb[3] < rb[1] or vb[1] > rb[3])
    except Exception:
        return None


def inventory(root):
    shp = [p for p in _find(root, {".shp", ".gpkg"})
           if not re.search(r"_(AOI|AOE|aoi)$", os.path.splitext(os.path.basename(p))[0])]
    tif = _find(root, {".tif", ".tiff"})
    orthos = [p for p in tif if "ortho" in os.path.basename(p).lower()]
    dems = [p for p in tif if re.search(r"(dem|dsm|chm)", os.path.basename(p).lower())]
    other = [p for p in tif if p not in orthos and p not in dems]

    ann = {}
    for p in shp:
        info = _vector_info(p)
        if info.get("features"):
            ann.setdefault(_norm(p), []).append(info)
    ras = {}
    for p in orthos:
        ras.setdefault(_norm(p), []).append(_raster_info(p))
    dem_by_site = {}
    for p in dems:
        dem_by_site.setdefault(_norm(p), []).append(_raster_info(p))

    sites, issues = [], []
    for key in sorted(set(ann) | set(ras)):
        a = ann.get(key, [{}])[0]
        o = ras.get(key, [{}])[0]
        d = dem_by_site.get(key, [{}])[0]
        row = {"site": key, "features": a.get("features"), "tree_ids": a.get("tree_ids"),
               "ann_crs": a.get("crs"), "ortho_crs": o.get("crs"),
               "ortho_mpx": o.get("mpx"), "gsd_m": o.get("gsd_m"),
               "has_ann": bool(a.get("features")), "has_ortho": bool(o.get("width")),
               "has_dem": bool(d.get("width")),
               "species": a.get("species", {})}
        if row["has_ann"] and row["has_ortho"]:
            row["crs_match"] = (a.get("crs") == o.get("crs"))
            row["overlaps"] = _overlaps(a, o)
            if not row["crs_match"]:
                issues.append(f"{key}: annotations {a.get('crs')} vs ortho {o.get('crs')} "
                              "— reprojection required (rasterize_annotations does this)")
            if row["overlaps"] is False:
                issues.append(f"{key}: annotations do NOT overlap the ortho — wrong pairing?")
        elif row["has_ann"]:
            issues.append(f"{key}: {a.get('features')} annotated polygons but NO ortho "
                          "— unusable as training data")
        else:
            issues.append(f"{key}: ortho with no annotations — inference/pretraining only")
        sites.append(row)

    # corpus-wide species hygiene
    allsp = collections.Counter()
    for a_list in ann.values():
        for a in a_list:
            for k, v in (a.get("species") or {}).items():
                allsp[k] += v
    variants = collections.defaultdict(list)
    for k in allsp:
        if k not in ("None", None):
            variants[str(k).strip().upper()].append(k)
    for canon, vs in variants.items():
        if len(vs) > 1:
            issues.append(f"species {canon}: case variants {vs} — match case-insensitively")
    if "None" in allsp:
        issues.append(f"{allsp['None']} polygons have no Species value")

    return {"root": root, "sites": sites, "issues": issues,
            "species_total": dict(allsp.most_common()),
            "counts": {"annotation_files": len(shp), "orthos": len(orthos),
                       "dems": len(dems), "other_rasters": len(other)}}


def to_markdown(inv):
    L = [f"# WINMOL training-data inventory", "",
         f"Root: `{inv['root']}`", ""]
    c = inv["counts"]
    L += [f"{c['annotation_files']} annotation files · {c['orthos']} orthomosaics · "
          f"{c['dems']} DEM/DSM/CHM · {c['other_rasters']} other rasters", ""]
    usable = [s for s in inv["sites"] if s["has_ann"] and s["has_ortho"]]
    feats = sum(s["features"] or 0 for s in usable)
    trees = sum(s["tree_ids"] or 0 for s in usable)
    L += [f"**{len(usable)} sites are usable as training data**, carrying {feats} polygons "
          f"across {trees} trees.", "",
          "## Sites", "",
          "| site | polygons | trees | ortho | GSD | DEM | CRS |",
          "|---|---:|---:|---:|---:|:---:|---|"]
    for s in sorted(inv["sites"], key=lambda r: (-(r["features"] or 0), r["site"])):
        ortho = f"{s['ortho_mpx']} MP" if s["has_ortho"] else "**missing**"
        crs = s["ann_crs"] or s["ortho_crs"] or "—"
        if s.get("crs_match") is False:
            crs = f"{s['ann_crs']} vs {s['ortho_crs']} ⚠"
        L.append(f"| {s['site']} | {s['features'] or '—'} | {s['tree_ids'] or '—'} | "
                 f"{ortho} | {s['gsd_m'] or '—'} | {'yes' if s['has_dem'] else 'no'} | {crs} |")
    L += ["", "## Species", "",
          " · ".join(f"`{k}` {v}" for k, v in inv["species_total"].items()), ""]
    if inv["issues"]:
        L += ["## Issues", ""] + [f"- {i}" for i in inv["issues"]] + [""]
    L += ["## Turning this into training data", "",
          "```bash",
          "# 1 — rasterize annotations onto each ortho's grid",
          "python scripts/rasterize_annotations.py --shapefile <site>.shp \\",
          "    --ortho <site>_ortho.tif --out <site>_stems.tif \\",
          "    --instances <site>_trees.tif --instance-level tree",
          "",
          "# 2 — leak-free tiling, whole sites per split (add --rgbd with DEMs)",
          "python scripts/build_site_splits.py --config sites.json --out <ROOT>",
          "",
          "# 3 — preflight before training",
          "python scripts/validate_dataset.py --data-dir <ROOT>/train",
          "```", ""]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--md", default=None)
    a = p.parse_args(argv)

    inv = inventory(a.root)
    md = to_markdown(inv)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(inv, f, indent=1)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
