"""Named augmentation recipes — the configurations this repo actually measured.

Every knob below is individually reachable as a CLI flag; a recipe is just a set of
defaults so a run that reproduces a published result does not depend on remembering
seven numbers. Explicit flags always win over the recipe (see `apply` below), so
`--recipe robust --aug-hsv-p 0.5` is a deliberate weakening, not a silent conflict.

Provenance for each recipe is in `docs/`, and the numbers are quoted here so the
choice can be audited without leaving the file. Do not add a recipe without a
measurement behind it; add it as UNEVALUATED and say so, as `mosaic` does.
"""

RECIPES = {
    # Today's defaults: flips, mild brightness/contrast, mild hue. This is what every
    # pre-scale-augmentation result in the repo was trained with.
    "baseline": {},

    # docs/scale-augmentation-results.md. RandomSizedCrop 394-666 px on a native-res
    # tile, resized to 512 — a +/-30% footprint, i.e. +/-30% effective GSD. Flattens the
    # accuracy-vs-scale curve (spread 0.0334 -> 0.0146 on UNet) at no measurable cost at
    # the serving scale. Requires a native-resolution dataset: `prepare.py --native-px`.
    "jitter": {
        "multiscale": True,
        "crop_min_px": 394,
        "crop_max_px": 666,
        "eval_tiling": True,
    },

    # jitter + docs/scale-augmentation-loso.md's hue recommendation. Bachsee_north as a
    # held-out site goes from F1 0.042 to 0.688; in-domain cost is -1.7 F1. The two
    # ingredients are not separable — a full hue circle at default probability reaches
    # only 0.261, and half the range at full probability only 0.203. The doc's wording is
    # "do not soften it", so these values are not defaults to be nudged.
    "robust": {
        "multiscale": True,
        "crop_min_px": 394,
        "crop_max_px": 666,
        "eval_tiling": True,
        "aug_hue_shift": 90,
        "aug_sat_shift": 60,
        "aug_val_shift": 30,
        "aug_hsv_p": 0.9,
    },

    # UNEVALUATED. Mosaic has no measurement in this repo — no paired run, no LOSO fold,
    # no results document. It is here so the knob is reachable, not because it is
    # recommended. Before any claim rests on it, run it through the winmol-experiment
    # skill: paired seeds against `robust`, leave-one-site-out, reported as a fold count.
    "mosaic": {
        "multiscale": True,
        "crop_min_px": 394,
        "crop_max_px": 666,
        "eval_tiling": True,
        "aug_hue_shift": 90,
        "aug_sat_shift": 60,
        "aug_val_shift": 30,
        "aug_hsv_p": 0.9,
        "mosaic_p": 0.5,
    },
}

UNEVALUATED = {"mosaic"}


def apply(recipe_name, args, parser):
    """Fold a recipe into parsed `args`, letting explicit CLI flags win.

    `parser` is used to tell "the user passed the default value" apart from "the user
    passed nothing": argparse cannot distinguish those after the fact, so we compare
    against the parser's own defaults. A flag whose value differs from its default was
    set deliberately and is left alone.
    """
    if not recipe_name:
        return args, []
    try:
        recipe = RECIPES[recipe_name]
    except KeyError:
        raise SystemExit(
            f"unknown recipe {recipe_name!r}; choose from {', '.join(sorted(RECIPES))}")

    applied, overridden = [], []
    for key, value in recipe.items():
        current = getattr(args, key, None)
        default = parser.get_default(key)
        if current != default:
            overridden.append(f"{key}={current} (recipe wanted {value})")
            continue
        setattr(args, key, value)
        applied.append(f"{key}={value}")
    return args, applied + [f"OVERRIDDEN {o}" for o in overridden]


def describe(recipe_name):
    """One-line summary for --help and for the run log."""
    body = RECIPES.get(recipe_name, {})
    if not body:
        return "repo defaults (flips, mild brightness/contrast, mild hue)"
    parts = ", ".join(f"{k}={v}" for k, v in sorted(body.items()))
    flag = " [UNEVALUATED]" if recipe_name in UNEVALUATED else ""
    return parts + flag
