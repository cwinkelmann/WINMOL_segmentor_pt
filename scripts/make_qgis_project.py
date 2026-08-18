"""Generate a QGIS project (.qgs) for comparing stem predictions side by side.

Written as a generator rather than a saved project so the comparison can be rebuilt when
models change, and so paths stay relative — the folder can be moved or shared as a unit.

Layer order is deliberate: ground truth sits on top of the predictions, because the
question being asked is "did the model find this stem", and a prediction drawn over the
truth hides exactly the evidence needed to answer it.

    python scripts/make_qgis_project.py --dir data/qgis --out data/qgis/tegel.qgs
"""
import argparse
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

CRS = {"auth": "EPSG:32633", "srid": "32633", "epsg": "32633",
       "desc": "WGS 84 / UTM zone 33N",
       "proj": "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"}


def _crs_block(parent, tag="srs"):
    srs = ET.SubElement(parent, tag)
    sp = ET.SubElement(srs, "spatialrefsys")
    for k, v in (("wkt", ""), ("proj4", CRS["proj"]), ("srsid", "3100"),
                 ("srid", CRS["srid"]), ("authid", CRS["auth"]),
                 ("description", CRS["desc"]), ("projectionacronym", "utm"),
                 ("ellipsoidacronym", "EPSG:7030"), ("geographicflag", "false")):
        ET.SubElement(sp, k).text = v
    return srs


def _line_symbol(color, width="0.6", opacity="1"):
    return {"type": "line", "color": color, "width": width, "opacity": opacity}


def _maplayer(parent, lid, name, gpkg, layername, geom, style):
    ml = ET.SubElement(parent, "maplayer", {
        "geometry": geom, "type": "vector", "hasScaleBasedVisibilityFlag": "0",
        "styleCategories": "AllStyleCategories"})
    ET.SubElement(ml, "id").text = lid
    ET.SubElement(ml, "datasource").text = f"./{gpkg}|layername={layername}"
    ET.SubElement(ml, "layername").text = name
    _crs_block(ml)
    ET.SubElement(ml, "provider", {"encoding": "UTF-8"}).text = "ogr"
    rr = ET.SubElement(ml, "renderer-v2", {"type": "singleSymbol",
                                           "forceraster": "0", "symbollevels": "0"})
    syms = ET.SubElement(rr, "symbols")
    sym = ET.SubElement(syms, "symbol", {"type": style["type"], "name": "0",
                                         "alpha": style["opacity"], "clip_to_extent": "1"})
    if style["type"] == "line":
        props = {"line_color": style["color"], "line_width": style["width"],
                 "line_width_unit": "MM", "line_style": "solid", "capstyle": "round",
                 "joinstyle": "round"}
        layer_class = "SimpleLine"
    else:
        props = {"color": style["color"], "outline_color": style.get("outline", "0,0,0,255"),
                 "outline_width": style["width"], "outline_width_unit": "MM",
                 "style": style.get("fill", "no")}
        layer_class = "SimpleFill"
    sl = ET.SubElement(sym, "layer", {"class": layer_class, "enabled": "1", "pass": "0",
                                      "locked": "0"})
    for k, v in props.items():
        ET.SubElement(sl, "prop", {"k": k, "v": v})
    ET.SubElement(ml, "blendMode").text = "0"
    return ml


def build(d, out, entries):
    root = ET.Element("qgis", {"projectname": "Tegel stem predictions",
                               "version": "3.34.0-Prizren"})
    ET.SubElement(root, "homePath", {"path": ""})
    ET.SubElement(root, "title").text = "Tegel R12 / R13 — new vs published model"
    tree = ET.SubElement(root, "layer-tree-group")
    layers_el = ET.SubElement(root, "projectlayers")

    groups = {}
    for e in entries:
        if not os.path.exists(os.path.join(d, e["gpkg"])):
            continue
        g = groups.setdefault(e["group"], ET.SubElement(
            tree, "layer-tree-group", {"name": e["group"], "checked": "Qt::Checked",
                                       "expanded": "1"}))
        lid = e["id"]
        ET.SubElement(g, "layer-tree-layer", {
            "id": lid, "name": e["name"], "source": f"./{e['gpkg']}|layername={e['layer']}",
            "providerKey": "ogr", "checked": e.get("checked", "Qt::Checked"),
            "expanded": "0"})
        _maplayer(layers_el, lid, e["name"], e["gpkg"], e["layer"], e["geom"], e["style"])

    props = ET.SubElement(root, "properties")
    ET.SubElement(props, "WMSServiceTitle").text = "Tegel stem predictions"
    _crs_block(ET.SubElement(root, "projectCrs"), tag="spatialrefsys") if False else None
    pc = ET.SubElement(root, "projectCrs")
    sp = ET.SubElement(pc, "spatialrefsys")
    for k, v in (("proj4", CRS["proj"]), ("srsid", "3100"), ("srid", CRS["srid"]),
                 ("authid", CRS["auth"]), ("description", CRS["desc"]),
                 ("projectionacronym", "utm"), ("ellipsoidacronym", "EPSG:7030"),
                 ("geographicflag", "false")):
        ET.SubElement(sp, k).text = v

    xml = minidom.parseString(ET.tostring(root, "utf-8")).toprettyxml(indent="  ")
    with open(out, "w") as f:
        f.write(xml)
    return out, len(groups)


def default_entries():
    out = []
    for rev in ("R12", "R13"):
        # ground truth LAST in this list = drawn on top (QGIS draws the tree top-down)
        out.append({"group": rev, "id": f"{rev}_new_stems", "name": f"{rev} stems — NEW model",
                    "gpkg": f"{rev}_new.gpkg", "layer": "stems", "geom": "Line",
                    "style": _line_symbol("0,160,60,255", "0.7")})
        out.append({"group": rev, "id": f"{rev}_old_stems", "name": f"{rev} stems — published model",
                    "gpkg": f"{rev}_old.gpkg", "layer": "stems", "geom": "Line",
                    "style": _line_symbol("220,50,32,255", "0.7")})
        out.append({"group": rev, "id": f"{rev}_new_nodes",
                    "name": f"{rev} nodes (diameter every 50 cm) — NEW",
                    "gpkg": f"{rev}_new.gpkg", "layer": "nodes", "geom": "Point",
                    "checked": "Qt::Unchecked",
                    "style": {"type": "marker", "color": "0,160,60,120", "width": "0.2",
                              "opacity": "1"}})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    entries = default_entries()
    # ground truth per plot, added last so it draws on top of the predictions
    plots = os.path.join(a.dir, "tegel_plots")
    if os.path.isdir(plots):
        for plot in sorted(os.listdir(plots)):
            if not os.path.isdir(os.path.join(plots, plot)):
                continue
            rev = plot.split("-")[0]
            for kind, colour, width in (("stems", "255,215,0,255", "0.9"),
                                        ("aoi", "40,120,255,255", "0.5")):
                src = f"tegel_plots/{plot}/{kind}.gpkg"
                if not os.path.exists(os.path.join(a.dir, src)):
                    continue
                entries.append({
                    "group": rev, "id": f"{plot}_{kind}",
                    "name": f"{plot} {'GROUND TRUTH' if kind == 'stems' else 'plot AOI'}",
                    "gpkg": src, "layer": os.path.splitext(os.path.basename(src))[0],
                    "geom": "Polygon",
                    "style": {"type": "fill", "color": colour, "width": width,
                              "opacity": "1", "outline": colour, "fill": "no"}})
    out, n = build(a.dir, a.out, entries)
    print(f"wrote {out} with {n} groups, {len(entries)} layer entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
