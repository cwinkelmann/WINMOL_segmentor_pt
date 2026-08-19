"""Zoom in on where predictions and human polygons disagree, so the disagreement can be judged.

A plot-wide overlay shows *that* they disagree; it cannot show *who is right*. At plot scale
a 30 cm stem is a few pixels. This crops each disagreement at native resolution and puts it
next to the imagery bare, so the question "is there actually a stem here" is answerable by
looking.

Three categories, and the middle one is the point:

* **matched**    — a prediction with a label near it. Reference for what agreement looks like.
* **unmatched**  — a prediction with NO label near it. Either a false positive or a real stem
  the annotator missed. If these are real, precision is understated and the F1 is wrong.
* **missed**     — a labelled stem with no prediction near it. A genuine miss by the model.

    python scripts/inspect_predictions.py --pred R13_new.gpkg --gt stems.gpkg \\
        --aoi aoi.gpkg --ortho crop.tif --out fig.png
"""
import argparse
import os

import numpy as np


def _load(path, layer=None):
    import fiona
    from shapely.geometry import shape
    with fiona.open(path, **({"layer": layer} if layer else {})) as s:
        return [shape(f["geometry"]) for f in s], str(s.crs)


def _to(crs_from, crs_to, geoms):
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    if str(crs_from) == str(crs_to):
        return geoms
    tr = Transformer.from_crs(crs_from, crs_to, always_xy=True).transform
    return [shp_transform(tr, g) for g in geoms]


def categorise(pred, gt, tol_m=0.5):
    """Split predictions and labels into matched / unmatched / missed."""
    from shapely.ops import unary_union
    gt_buf = unary_union(gt).buffer(tol_m)
    pred_buf = unary_union(pred).buffer(tol_m) if pred else None
    matched = [p for p in pred if p.intersects(gt_buf)]
    unmatched = [p for p in pred if not p.intersects(gt_buf)]
    missed = [g for g in gt if pred_buf is None or not g.intersects(pred_buf)]
    return matched, unmatched, missed


def figure(ortho, groups, out, half_m=6.0, cols=5, seed=1, title=None):
    """A row per category: native-resolution crops, bare above, annotated below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import from_bounds

    src = rasterio.open(ortho)
    rng = np.random.default_rng(seed)
    rows = [(k, v, c) for k, v, c in groups if v]
    fig, axes = plt.subplots(2 * len(rows), cols,
                             figsize=(2.5 * cols, 2.7 * 2 * len(rows)), squeeze=False)
    for r, (name, geoms, colour) in enumerate(rows):
        pick = rng.choice(len(geoms), size=min(cols, len(geoms)), replace=False)
        for c in range(cols):
            ax_bare, ax_ann = axes[2 * r][c], axes[2 * r + 1][c]
            if c >= len(pick):
                ax_bare.axis("off"); ax_ann.axis("off"); continue
            g = geoms[int(pick[c])]
            cx, cy = g.centroid.x, g.centroid.y
            b = (cx - half_m, cy - half_m, cx + half_m, cy + half_m)
            img = src.read([1, 2, 3], window=from_bounds(*b, src.transform),
                           out_shape=(3, 400, 400), boundless=True, fill_value=0)
            img = np.transpose(img, (1, 2, 0))
            ext = [b[0], b[2], b[1], b[3]]
            for ax in (ax_bare, ax_ann):
                ax.imshow(img, extent=ext)
                ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
                ax.set_xticks([]); ax.set_yticks([])
            parts = g.geoms if g.geom_type.startswith("Multi") else [g]
            for p in parts:
                xy = p.exterior.xy if p.geom_type == "Polygon" else p.xy
                ax_ann.plot(*xy, color=colour, lw=2.0)
        axes[2 * r][0].set_ylabel(f"{name}\n({len(geoms)}) — bare", fontsize=8)
        axes[2 * r + 1][0].set_ylabel("annotated", fontsize=8)
    fig.suptitle(title or "Prediction vs human label — 12 m windows, native resolution",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    return out


def overview(ortho, aoi, matched, unmatched, missed, out, title=None, px=1500):
    """Whole-plot overlay: where do predictions and labels agree, and where does neither?

    The zoom panels answer "is this one real"; this answers "how much of the plot is
    affected and is it clustered". A high unmatched count spread evenly reads as
    systematic under-labelling; clustered reads as one missed corner.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.patches import Patch
    from rasterio.windows import from_bounds

    def draw(ax, g, colour, lw):
        for part in (g.geoms if g.geom_type.startswith("Multi") else [g]):
            xy = part.exterior.xy if part.geom_type == "Polygon" else part.xy
            ax.plot(*xy, color=colour, lw=lw)

    src = rasterio.open(ortho)
    b = aoi.bounds
    img = src.read([1, 2, 3], window=from_bounds(*b, src.transform),
                   out_shape=(3, px, px), boundless=True, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(np.transpose(img, (1, 2, 0)), extent=[b[0], b[2], b[1], b[3]])
    draw(ax, aoi, "#2a78ff", 1.5)
    for g in missed:
        draw(ax, g, "#ffd700", 1.6)
    for g in matched:
        draw(ax, g, "#00c020", 1.2)
    for g in unmatched:
        draw(ax, g, "#ff2020", 2.2)
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[
        Patch(color="#ff2020", label=f"prediction, NO label near it ({len(unmatched)})"),
        Patch(color="#ffd700", label=f"label the model missed ({len(missed)})"),
        Patch(color="#00c020", label=f"prediction matching a label ({len(matched)})"),
        Patch(color="#2a78ff", label="plot AOI")],
        loc="lower left", framealpha=0.92, fontsize=9)
    ax.set_title(title or "Label completeness", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    p.add_argument("--pred-layer", default="stems")
    p.add_argument("--gt", required=True)
    p.add_argument("--aoi", required=True)
    p.add_argument("--ortho", required=True)
    p.add_argument("--tol-m", type=float, default=0.5)
    p.add_argument("--half-m", type=float, default=6.0)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--overview-out", default=None,
                   help="also write a whole-plot overlay showing where the disagreements are")
    a = p.parse_args(argv)

    import rasterio
    from shapely.ops import unary_union
    target = str(rasterio.open(a.ortho).crs)
    aoi, aoi_crs = _load(a.aoi); aoi = unary_union(_to(aoi_crs, target, aoi))
    gt, gt_crs = _load(a.gt); gt = [g for g in _to(gt_crs, target, gt) if g.intersects(aoi)]
    pred, pr_crs = _load(a.pred, a.pred_layer)
    pred = [g for g in _to(pr_crs, target, pred) if g.intersects(aoi)]

    matched, unmatched, missed = categorise(pred, gt, a.tol_m)
    print(f"  predictions {len(pred)}, labels {len(gt)}  (tolerance {a.tol_m} m)")
    print(f"    matched   {len(matched):4d}  prediction has a label nearby")
    print(f"    unmatched {len(unmatched):4d}  NO label nearby -> false positive OR missing label")
    print(f"    missed    {len(missed):4d}  labelled stem with no prediction")
    if a.overview_out:
        print(overview(a.ortho, aoi, matched, unmatched, missed, a.overview_out, a.title))
    print(figure(a.ortho, [("unmatched prediction", unmatched, "#ff2020"),
                           ("missed label", missed, "#ffd700"),
                           ("matched", matched, "#00c020")],
                 a.out, a.half_m, a.cols, a.seed, a.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
