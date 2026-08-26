"""Repair invalid stem polygons and write a cleaned shapefile plus a report.

    python -m winmol_unet.geo.geometry --stems site.shp --out site_fixed.shp \
        [--report site_geom.json] [--check-only]

50 of the corpus's 3,842 stem polygons have self-intersecting rings — hand-tracing an
outline produces them readily. GEOS raises `TopologyException: side location conflict` on
the first set operation that touches one, which kills a sampling run tens of thousands of
tiles in, so this has to happen before anything else in the pipeline.

Repair happens inside the samplers too, but silently and in memory. Doing it as its own
step means the corrected geometry is an artefact you can inspect, and the report says what
changed rather than leaving you to trust that it was harmless:

  * `repaired` — invalid rings rebuilt by `make_valid`, with the area before and after
  * `dropped`  — geometries that repair reduced to nothing (a zero-area sliver)
  * `area_delta_m2` — how much stem area the repair moved, in total

A large `area_delta_m2` is a warning: repair should nudge a ring, not redraw a stem. If
it is large, look at the individual entries before using the output.

`--check-only` reports without writing, for use as a preflight.
"""
import argparse
import json
import os
import sys


def _repair_one(geom):
    """Return (geometry_or_None, entry_or_None) for a single polygon."""
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon

    if geom.is_valid:
        return geom, None
    before = geom.area
    r = make_valid(geom)
    parts = [p for p in getattr(r, "geoms", [r])
             if isinstance(p, (Polygon, MultiPolygon)) and not p.is_empty and p.area > 0]
    if not parts:
        return None, {"dropped": True, "area_m2": round(before, 4)}
    fixed = parts[0] if len(parts) == 1 else MultiPolygon(
        [q for p in parts for q in getattr(p, "geoms", [p])])
    return fixed, {"dropped": False, "area_before_m2": round(before, 4),
                   "area_after_m2": round(fixed.area, 4),
                   "delta_m2": round(fixed.area - before, 4), "parts": len(parts)}


def fix(stems_path, out_path=None, report_path=None, check_only=False, quiet=False):
    import fiona
    from shapely.geometry import mapping, shape

    with fiona.open(stems_path) as src:
        meta = dict(driver=src.driver, crs=src.crs, schema=src.schema)
        records = [f for f in src if f["geometry"] is not None]

    entries, dropped, written = [], [], 0
    writer = None
    if not check_only and out_path:
        writer = fiona.open(out_path, "w", **meta)
    try:
        for i, f in enumerate(records):
            key = (f["properties"] or {}).get("id", i)
            fixed, entry = _repair_one(shape(f["geometry"]))
            if entry is not None:
                entry["id"] = key
                (dropped if entry["dropped"] else entries).append(entry)
            if fixed is None:
                continue
            if writer is not None:
                writer.write({"geometry": mapping(fixed), "properties": f["properties"]})
                written += 1
    finally:
        if writer is not None:
            writer.close()

    report = {
        "source": os.path.abspath(stems_path),
        "polygons": len(records),
        "invalid": len(entries) + len(dropped),
        "repaired": len(entries),
        "dropped": len(dropped),
        "area_delta_m2": round(sum(e["delta_m2"] for e in entries), 4),
        "entries": entries, "dropped_entries": dropped,
    }
    if writer is not None:
        report["written"] = written
        report["output"] = os.path.abspath(out_path)

    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1)
    if not quiet:
        print(f"{report['polygons']} polygons, {report['invalid']} invalid "
              f"({report['repaired']} repaired, {report['dropped']} dropped)")
        if entries:
            print(f"  area moved by repair: {report['area_delta_m2']:+.3f} m² total, "
                  f"largest single change {max(abs(e['delta_m2']) for e in entries):.3f} m²")
        if dropped:
            print(f"  dropped (no area after repair): {[d['id'] for d in dropped][:8]}")
        if writer is not None:
            print(f"  wrote {written} polygons -> {out_path}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stems", required=True)
    p.add_argument("--out", default=None, help="cleaned shapefile (omit with --check-only)")
    p.add_argument("--report", default=None, help="JSON detailing every repair")
    p.add_argument("--check-only", action="store_true",
                   help="report without writing, as a preflight")
    a = p.parse_args(argv)
    if not a.check_only and not a.out:
        p.error("--out is required unless --check-only")
    r = fix(a.stems, a.out, a.report, a.check_only)
    return 0 if r["dropped"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
