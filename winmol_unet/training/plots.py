"""Static training figures on disk: loss/metric curves and prediction panels.

Everything here is opt-in (`--plots`) and every matplotlib import is inside a
function. The analyzer installs this package with no torch and no matplotlib, and
`tests/test_import_boundary.py` asserts that importing `winmol_unet` pulls in
neither -- so a module-level `import matplotlib` here would break that boundary.

train.py already logs these same numbers and a per-epoch prediction panel to
TensorBoard. This module is what makes them openable without TensorBoard: files
you can attach to a results document or drop into a report.
"""
import os

import numpy as np      # already a core dependency: contract.py and runtime.py use it

# The series drawn on the metric panel, in the order they appear in the legend.
_METRIC_KEYS = ("val/precision", "val/recall", "val/f1")


def _pyplot():
    """matplotlib with a headless backend, or None when it is not installed.

    Agg is selected before pyplot is imported: training runs on machines with no
    display, and the default backend would fail there.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _series(history, key):
    """(steps, values) for one key, skipping entries that do not carry it."""
    pairs = [(e["step"], e[key]) for e in history if key in e]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def plot_history(history, out_dir, prefix=""):
    """Write curve PNGs from a metrics history. Returns the paths written.

    `history` is the list RunLogger writes to metrics_history.json: one dict per
    epoch, each carrying "step" plus whatever scalars that run logged. Series that
    a run did not produce are simply absent -- a run with no per-component losses
    still gets its loss curve.
    """
    plt = _pyplot()
    if plt is None:
        return []
    import pathlib

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    loss_keys = sorted(k for k in {k for e in history for k in e}
                       if k.endswith("loss") or "loss_" in k)
    if loss_keys:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key in loss_keys:
            steps, values = _series(history, key)
            # Component losses are supporting detail; the two headline curves lead.
            headline = key in ("train/loss", "val/loss")
            ax.plot(steps, values, label=key, linewidth=2 if headline else 1,
                    alpha=1.0 if headline else 0.6,
                    linestyle="-" if headline else "--")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title("Training and validation loss")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / f"{prefix}loss_curves.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)

    present = [k for k in _METRIC_KEYS if any(k in e for e in history)]
    if present:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key in present:
            steps, values = _series(history, key)
            ax.plot(steps, values, label=key.split("/")[-1], linewidth=2)
        ax.set_xlabel("epoch"); ax.set_ylabel("score")
        ax.set_ylim(0, 1)                      # these are all rates; pin the axis
        ax.set_title("Validation precision, recall and F1")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / f"{prefix}metric_curves.png"
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path)

    return written


def _tile_f1(prob, mask, threshold=0.5):
    """Micro F1 for one tile, on hard-rounded predictions.

    Matches how the training loop scores: sigmoid, threshold, then compare. A tile
    with no stem pixels and no predicted stem pixels scores 1.0 -- it is perfectly
    predicted, and ranking it "worst" would fill the panel with empty ground.
    """
    pred = (prob >= threshold)
    truth = (mask >= 0.5)
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def _predict(model, dataset, index, device=None):
    """(image HWC, mask HW, probability HW) for one dataset index."""
    import torch

    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else "cpu"
    img, mask = dataset[index]
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu()
    return (img.permute(1, 2, 0).cpu().numpy(), mask.squeeze().cpu().numpy(),
            prob.numpy())


def worst_f1_indices(model, dataset, k, device=None):
    """The k dataset indices the model scores worst on, lowest F1 first.

    Selected, not random -- panels built from this must say so, which is why
    plot_predictions names its file predictions_selected_worst_f1.png.
    """
    scored = []
    for i in range(len(dataset)):
        _, mask, prob = _predict(model, dataset, i, device)
        scored.append((_tile_f1(prob, mask), i))
    scored.sort(key=lambda pair: (pair[0], pair[1]))
    return [i for _, i in scored[:k]]


def _panel(plt, model, dataset, indices, title, out_path, device=None):
    """One row per tile: the tile, with prediction filled and ground truth outlined."""
    fig, axes = plt.subplots(len(indices), 2, figsize=(7, 3.2 * len(indices)),
                             squeeze=False)
    for row, idx in enumerate(indices):
        img, mask, prob = _predict(model, dataset, idx, device)
        f1 = _tile_f1(prob, mask)

        axes[row][0].imshow(img)
        axes[row][0].set_ylabel(f"tile {idx}\nF1 {f1:.2f}", fontsize=8)
        axes[row][0].set_title("tile" if row == 0 else "", fontsize=9)

        axes[row][1].imshow(img)
        # Prediction as a translucent fill, ground truth as an outline on top, so
        # over- and under-prediction are both visible against the actual imagery.
        axes[row][1].imshow(np.ma.masked_where(prob < 0.5, prob), cmap="autumn",
                            alpha=0.55, vmin=0, vmax=1)
        if mask.max() > 0:
            axes[row][1].contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
        axes[row][1].set_title("prediction (fill) vs truth (outline)" if row == 0 else "",
                               fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_predictions(model, dataset, out_dir, n=4, seed=0, device=None):
    """Two panels: a random sample, and the worst-scoring tiles.

    Both are written, always. The repo's figure rule is that a selected figure is
    labelled as selected and shown alongside a random sample -- a worst-F1 panel on
    its own would make any model look broken.
    """
    plt = _pyplot()
    if plt is None or len(dataset) == 0:
        return []
    import pathlib
    import random

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = min(n, len(dataset))

    rnd = random.Random(seed).sample(range(len(dataset)), k)
    random_path = out / "predictions_random.png"
    _panel(plt, model, dataset, sorted(rnd), f"Random validation tiles (n={k})",
           random_path, device)

    worst_path = out / "predictions_selected_worst_f1.png"
    _panel(plt, model, dataset, worst_f1_indices(model, dataset, k, device),
           f"SELECTED: lowest-F1 validation tiles (n={k}) -- not representative",
           worst_path, device)

    return [random_path, worst_path]


def load_history(log_dir):
    """Read back what RunLogger wrote, or [] if the run produced no history."""
    import json

    path = os.path.join(log_dir, "metrics_history.json")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return json.load(fh)
