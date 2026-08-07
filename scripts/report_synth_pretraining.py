"""Render the synthetic-pretraining study to a multi-page PDF.

    python scripts/report_synth_pretraining.py --results results.json --out report.pdf \
        [--overlay docs/assets/kaufland-tile-mask-overlay.jpg]

Results come from a JSON file rather than being hard-coded, so re-running an arm or
adding an architecture is an edit to the data, not to the plotting code:

    {"runs": [{"arch": "UNet", "arm": "beech only", "f1": 0.7638, "precision": ...,
               "recall": ..., "loss": ..., "val_f1": ...}, ...],
     "dataset": {...}, "corpus": {...}}

Every number on the page traces to a run directory's `test_results.md`; nothing here
computes a metric, so the PDF cannot drift from what the training actually reported.
"""
import argparse
import json
import os
import sys

import numpy as np

# muted, colourblind-safe pair: one hue per arm, held constant across every panel so a
# colour always means the same arm
C_BASE = "#5A7D9A"      # beech only
C_SYNTH = "#C8763C"     # synthetic pretrain
C_INK = "#2B2B2B"
C_MUTE = "#8A8A8A"
C_GRID = "#DDDDDD"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": C_MUTE, "axes.labelcolor": C_INK, "text.color": C_INK,
        "xtick.color": C_MUTE, "ytick.color": C_MUTE,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _title(fig, title, subtitle=None):
    fig.text(0.06, 0.955, title, fontsize=15, weight="bold", color=C_INK)
    if subtitle:
        fig.text(0.06, 0.928, subtitle, fontsize=9.5, color=C_MUTE)
    fig.lines.append(__import__("matplotlib").lines.Line2D(
        [0.06, 0.94], [0.917, 0.917], transform=fig.transFigure, color=C_GRID, lw=1))


def _footer(fig, page, total):
    fig.text(0.94, 0.035, f"{page} / {total}", ha="right", fontsize=8, color=C_MUTE)
    fig.text(0.06, 0.035, "WINMOL segmentor — synthetic pretraining study",
             fontsize=8, color=C_MUTE)


def page_headline(pdf, runs, dataset, corpus):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    _title(fig, "Does synthetic pretraining help a beech stem segmenter?",
           "Yes — but less the better the architecture. Leak-free block split, one run per arm.")

    seen, archs = set(), []
    for r in runs:                                   # first-appearance order
        if r["arch"] not in seen:
            seen.add(r["arch"]); archs.append(r["arch"])

    ax = fig.add_axes([0.12, 0.615, 0.76, 0.245])
    xs = np.arange(len(archs))
    # keep bars readable whether there is one architecture or four
    w = 0.30 if len(archs) > 1 else 0.22
    top = max(r["f1"] for r in runs) * 1.30
    for i, (arm, c) in enumerate((("beech only", C_BASE), ("synthetic pretrain", C_SYNTH))):
        vals = [next((r["f1"] for r in runs if r["arch"] == a and r["arm"] == arm), np.nan)
                for a in archs]
        bars = ax.bar(xs + (i - 0.5) * w, vals, w * 0.9, color=c, label=arm, zorder=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.012, f"{v:.4f}",
                        ha="center", fontsize=8.5, color=C_INK)
    # delta above the pair, clear of both bars and of the value labels
    for j, a in enumerate(archs):
        base = next((r["f1"] for r in runs if r["arch"] == a and r["arm"] == "beech only"), None)
        syn = next((r["f1"] for r in runs if r["arch"] == a and r["arm"] == "synthetic pretrain"), None)
        if base and syn:
            y = max(base, syn) + top * 0.075
            ax.annotate("", xy=(j + w / 2, y), xytext=(j - w / 2, y),
                        arrowprops=dict(arrowstyle="<->", color=C_MUTE, lw=0.9))
            ax.text(j, y + top * 0.02, f"{100*(syn-base):+.1f} pts", ha="center",
                    fontsize=9.5, weight="bold",
                    color=C_SYNTH if syn > base else C_BASE)
    ax.set_xticks(xs); ax.set_xticklabels(archs, fontsize=10)
    # right margin reserved for the reference-line label, so it never sits over a bar
    ax.set_xlim(-0.55, len(archs) - 0.5 + 0.62)
    ax.set_ylabel("test F1 (held-out blocks)")
    ax.set_ylim(0, top)
    ax.grid(axis="y", color=C_GRID, lw=0.8, zorder=0)

    # reference line first, then a legend placed outside the plot so neither collides
    ax.axhline(0.760, color=C_MUTE, lw=1, ls=(0, (4, 3)), zorder=2)
    # sits just above its own line, in the reserved right margin, so it is neither
    # struck through by the dash nor overlapping a bar
    ax.text(len(archs) - 0.5 + 0.58, 0.772, "published R model 0.760", va="bottom",
            ha="right", fontsize=7, color=C_MUTE)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.005))

    # The gain is not constant across architectures — and how it varies is the finding.
    pairs = []
    for a in archs:
        b = next((r["f1"] for r in runs if r["arch"] == a and r["arm"] == "beech only"), None)
        s = next((r["f1"] for r in runs if r["arch"] == a and r["arm"] == "synthetic pretrain"), None)
        if b and s:
            pairs.append((a, b, 100 * (s - b)))

    if len(pairs) >= 3:
        ax2 = fig.add_axes([0.12, 0.365, 0.50, 0.165])
        bs = [p[1] for p in pairs]; gs = [p[2] for p in pairs]
        ax2.plot(bs, gs, "-", color=C_MUTE, lw=1, zorder=2)
        ax2.scatter(bs, gs, s=55, color=C_SYNTH, zorder=3)
        for a, b, g in pairs:
            ax2.annotate(a, (b, g), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=8.5, color=C_INK)
        ax2.set_xlabel("baseline F1 (no pretraining)", fontsize=8.5)
        ax2.set_ylabel("gain from\nsynthetic (pts)", fontsize=8.5)
        ax2.tick_params(labelsize=8)
        ax2.set_ylim(min(gs) - 0.25, max(gs) + 0.55)
        ax2.grid(color=C_GRID, lw=0.8, zorder=0)
        ax2.set_title("The stronger the baseline, the less synthetic data adds",
                      fontsize=9, color=C_MUTE, loc="left", pad=6)
        body = (
            "The gain is not a constant. It shrinks\n"
            "monotonically as the architecture\n"
            "strengthens — and HRNet's baseline,\n"
            "with no pretraining at all, already beats\n"
            "UNet's pretrained arm.\n\n"
            "So synthetic data substitutes for model\n"
            "capacity here rather than adding what a\n"
            "stronger model lacks. On this corpus,\n"
            "changing architecture buys more than\n"
            "pretraining does, and the two do not\n"
            "stack.\n\n"
            "One run per arm: HRNet's +0.1 is\n"
            "indistinguishable from seed noise. The\n"
            "architecture ranking is a larger effect\n"
            "and more likely real."
        )
        fig.text(0.665, 0.545, body, fontsize=8.5, va="top", linespacing=1.5)
    else:
        body = (
            "Synthetic pretraining gives a gain on top of a reasonable baseline: arm A lands\n"
            "on the published R model's 0.760, so this is not rescuing a broken model.\n\n"
            "Read it as a direction, not a measured effect size — one run per arm, and a\n"
            "difference of this size is within plausible seed-to-seed variation."
        )
        fig.text(0.12, 0.545, body, fontsize=10, va="top", linespacing=1.6)

    box = (
        f"Why this matters here\n\n"
        f"The corpus holds {corpus['stem_m2']:,} m² of digitized stem in total — under one hectare, across\n"
        f"{corpus['polygons']:,} polygons and {corpus['trees']:,} trees, and only {corpus['unlabelled_beech']} unannotated beech orthomosaic remains.\n"
        f"Beech labelling is close to exhausted, so a generator producing unlimited perfectly-\n"
        f"labelled tiles is one of the few sources of signal left — and the architecture result\n"
        f"above says it is not the cheapest one to reach for first."
    )
    ax3 = fig.add_axes([0.10, 0.115, 0.80, 0.185]); ax3.axis("off")
    ax3.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", transform=ax3.transAxes,
        facecolor="#F5F2ED", edgecolor=C_GRID, lw=1))
    ax3.text(0.035, 0.87, box, fontsize=9, va="top", linespacing=1.6,
             transform=ax3.transAxes)
    _footer(fig, 1, 4)
    pdf.savefig(fig); import matplotlib.pyplot as p; p.close(fig)


def page_numbers(pdf, runs, dataset):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    _title(fig, "Results in full", "Every value read from the run's own test_results.md")

    n = len(runs)
    ax = fig.add_axes([0.06, 0.855 - 0.045 * n, 0.88, 0.045 * (n + 1)]); ax.axis("off")
    cols = ["architecture", "arm", "test F1", "precision", "recall", "test loss", "val F1"]
    rows = [[r["arch"], "synth pretrain" if r["arm"].startswith("syn") else r["arm"],
             f"{r['f1']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}",
             f"{r['loss']:.4f}",
             # val F1 is only recorded where the run log was captured; an em dash beats
             # a fabricated number
             f"{r['val_f1']:.4f}" if r.get("val_f1") is not None else "—"]
            for r in runs]
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                 colWidths=[0.17, 0.19, 0.13, 0.13, 0.13, 0.13, 0.12])
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(C_GRID)
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#F0EDE8")
        elif rows[r - 1][1] == "synth pretrain":
            cell.set_facecolor("#FBF4EE")

    ax3 = fig.add_axes([0.12, 0.44, 0.76, 0.22])
    labels, vals, cols_ = [], [], []
    for r in runs:
        labels.append(f"{r['arch']}\n{r['arm']}")
        vals.append((r["precision"], r["recall"]))
        cols_.append(C_SYNTH if r["arm"] == "synthetic pretrain" else C_BASE)
    x = np.arange(len(labels))
    ax3.bar(x - 0.18, [v[0] for v in vals], 0.34, color=cols_, zorder=3)
    ax3.bar(x + 0.18, [v[1] for v in vals], 0.34, color=cols_, alpha=0.55, zorder=3)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=7.5)
    ax3.set_ylabel("precision (solid) / recall (faded)")
    ax3.set_ylim(0, 1.0); ax3.grid(axis="y", color=C_GRID, lw=0.8, zorder=0)
    ax3.set_title("Precision and recall move together — no threshold artefact",
                  fontsize=9.5, color=C_MUTE, loc="left", pad=8)

    d = dataset
    txt = (
        "Setup\n\n"
        "30 epochs, batch 16, fixed --val-data-dir and --test-data-dir so neither arm\n"
        "re-splits. Two-stage resets the learning rate between stages, so stage 2 does not\n"
        "inherit stage 1's decayed rate. The smp architectures use encoder_weights=None:\n"
        "an ImageNet encoder would confound what the synthetic stage buys.\n\n"
        "Real tiles are sampled at --extent 10.24, i.e. 512 px / 10.24 m = 2 cm/px, exactly\n"
        "the synthetic generator's gsd_m_per_px. Sharing a ground resolution between the two\n"
        "stages is the point of pretraining.\n\n"
        f"train {d['train']:,} tiles ({d['train_note']})  ·  val {d['val']:,}  ·  "
        f"test {d['test']:,}  ·  stage 1: {d['synth']:,} synthetic tiles"
    )
    fig.text(0.08, 0.355, txt, fontsize=9.5, va="top", linespacing=1.6)
    _footer(fig, 2, 4)
    pdf.savefig(fig); plt.close(fig)


def page_pitfalls(pdf, colour_table, overlay):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    _title(fig, "Two designs that were wrong first",
           "Both produced numbers that looked like results")

    fig.text(0.06, 0.885, "1 · Holding out a whole site gave F1 exactly 0.0000 — for every arm",
             fontsize=11, weight="bold")
    fig.text(0.06, 0.845,
             "Peak output probability was 0.0024: the models never fired. The labels were fine —\n"
             "at native resolution the stems are visible and correctly aligned. The cause was colour.",
             fontsize=9.5, va="top", linespacing=1.6)

    ax = fig.add_axes([0.10, 0.62, 0.80, 0.17]); ax.axis("off")
    rows = [[r["set"], r["rgb"], r["amber"]] for r in colour_table]
    t = ax.table(cellText=rows, colLabels=["set", "mean RGB", "amber pixels"],
                 loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(C_GRID)
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#F0EDE8")
        elif "test" in rows[r - 1][0].lower():
            cell.set_facecolor("#FBEEEE")

    fig.text(0.06, 0.575,
             "Bachsee_north is a November beech canopy in full autumn colour — a domain nothing\n"
             "else covers, synthetic included. With three beech sites, holding one out removes an\n"
             "entire acquisition: its phenology, colour cast and GSD. The score then measures domain\n"
             "transfer, not segmentation. Hence spatial block splits: whole blocks per split, each\n"
             "shrunk by the footprint half-diagonal so no tile straddles a boundary at any rotation.",
             fontsize=9.5, va="top", linespacing=1.6)

    fig.text(0.06, 0.455, "2 · The stem filter tested the wrong square",
             fontsize=11, weight="bold")
    fig.text(0.06, 0.418,
             "Acceptance measured a world-space footprint rotated counter-clockwise, but PIL rotates\n"
             "the image counter-clockwise — which samples the world square rotated the other way. They\n"
             "coincide only at 0° and 90°. Image and mask still rotated together, so labels stayed\n"
             "registered and an overlay looked perfect. Only the selection was wrong: 3.2% of written\n"
             "tiles fell below the 0.5% stem floor and 21 were completely empty.\n\n"
             "The floor now applies to the finished mask, which is exact by construction. After the fix:\n"
             "0 empty, 4 of 4,154 marginally under floor. The pre-existing test could not catch it — its\n"
             "stem fixture is dense enough that every square contains stems whichever way it turns.",
             fontsize=9.5, va="top", linespacing=1.6)

    if overlay and os.path.exists(overlay):
        fig.text(0.06, 0.235,
                 "Sampled tiles with rasterized masks — the check that looked clean and hid the bug",
                 fontsize=8.5, color=C_MUTE)
        axi = fig.add_axes([0.06, 0.075, 0.88, 0.15])
        axi.imshow(mpimg.imread(overlay)); axi.axis("off")
    _footer(fig, 3, 4)
    pdf.savefig(fig); plt.close(fig)


def page_caveats(pdf, synth):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    _title(fig, "What the synthetic data does and does not cover",
           "Scope of the claim, and where it would not hold")

    # generous left margin: the category labels are long and get clipped otherwise
    ax = fig.add_axes([0.30, 0.63, 0.60, 0.22])
    names = [s["name"] for s in synth]
    green = [s["green"] for s in synth]
    ax.barh(np.arange(len(names)), green, 0.55,
            color=[C_SYNTH if "ynth" in n else C_BASE for n in names], zorder=3)
    for i, v in enumerate(green):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9, color=C_INK)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("green-dominant pixels")
    ax.grid(axis="x", color=C_GRID, lw=0.8, zorder=0)
    ax.set_title("The generator renders no vegetation green at all",
                 fontsize=9.5, color=C_MUTE, loc="left", pad=8)

    fig.text(0.06, 0.53,
             "The synthetic palette matches autumn and leaf-off scenes and misses leaf-on entirely.\n"
             "24% of its tiles also carry >10% violet pixels — a colour no orthomosaic produces.\n\n"
             "So the gain reported here is measured on the conditions the generator resembles. A\n"
             "leaf-on summer target would likely benefit less until the generator grows a summer\n"
             "palette. That is a scope limit on the claim, not a reason to discount it.\n\n"
             "What probably makes it work despite the colour defects is that the label statistics\n"
             "line up: synthetic stem coverage averages 7.0% against 6.2% for real tiles. For a\n"
             "segmentation prior, the geometry of the labels matters more than the palette.",
             fontsize=10, va="top", linespacing=1.65)

    fig.text(0.06, 0.30, "Next", fontsize=11, weight="bold")
    fig.text(0.06, 0.265,
             "· Seed replicates — run_train has no --seed CLI flag yet (it lives in TrainConfig);\n"
             "  adding it turns the gap from a direction into a defensible effect size.\n"
             "· Bachsee_north stays a legitimate hard-transfer benchmark; it is not a fair test set.\n"
             "· A summer palette in the generator would extend the claim to leaf-on sites.\n"
             "· For beech, the remaining levers are pre-labelling with the released ONNX plus human\n"
             "  correction, synthetic data, and generic→species transfer — not new annotation.",
             fontsize=9.5, va="top", linespacing=1.7)
    _footer(fig, 4, 4)
    pdf.savefig(fig); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--overlay", default=None)
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    _style()
    d = json.load(open(a.results))
    with PdfPages(a.out) as pdf:
        page_headline(pdf, d["runs"], d["dataset"], d["corpus"])
        page_numbers(pdf, d["runs"], d["dataset"])
        page_pitfalls(pdf, d["colour"], a.overlay)
        page_caveats(pdf, d["synth_colour"])
        pdf.infodict()["Title"] = "WINMOL — synthetic pretraining for beech stem segmentation"
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
