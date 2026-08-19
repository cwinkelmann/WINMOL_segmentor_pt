"""Named augmentation recipes — the configurations this repo actually measured.

Every knob below is individually reachable as a CLI flag; a recipe is just a set of
defaults so a run that reproduces a published result does not depend on remembering ten
numbers. Explicit flags always win over the recipe (see `apply`), so
`--recipe robust --aug-hsv-p 0.5` is a deliberate weakening, not a silent conflict.

Two traps, both of which can produce a confidently wrong "reproduction":

**`multiscale=True` is not the jitter switch.** The *fixed* control sets it too, with
`crop_min_px == crop_max_px == 512`. The crop RANGE is the entire independent variable in
the scale-augmentation experiment. Reading `multiscale=True` as "this is the jitter arm",
and `multiscale=False` as "this is the baseline", compares against an arm that was never
run — and `multiscale=False` in particular changes the input pipeline (the dataset stops
handing over native-resolution tiles) rather than only the augmentation.

**Rotation is off by default.** `aug_rotate_p` defaults to 0.0 and `aug_rotate_limit` to
15.0, but every measured run passed `--aug-rotate-p 0.5 --aug-rotate-limit 180.0`
explicitly. Tiles are cut at random rotations and the block-split separation math in
docs/process.md assumes the full circle, so omitting this does not merely lose a little
augmentation — it trains a different experiment, with no error, landing somewhere below
the published number. Recipes therefore set rotation explicitly.

**What a recipe does NOT set.** These are training-schedule settings rather than
augmentation, so they stay out — but the measured runs used them and a reproduction
needs them:

    --epochs 40 --batch-size 16 --seed 1 --deterministic

(lr 1e-3, dropout 0.1, patience 5, loss bce_soft_f1, block_order bn_relu, img_size 512,
encoder_weights none are all repo defaults already.) Without `--deterministic` a seed is
a label rather than a guarantee: two runs of the same code landed 2.0 F1 apart.

**Provenance caveat.** The exact invocations of the runs that produced the *published*
models-v2 artifacts (`scale-aug-20260813/jitter-s3`, `scale-aug-unet-20260813/jitter-s1`)
are **not recorded anywhere**. Those runs predate `write_run_config()`, and their
directories hold no run_config.json, no argv in the logs, and no shell history. The
values here are read from `runs/loso-scale-v2/*/run_config.json` (2026-08-14, one day
later, same 666 px / 19.512 m tiles, git 5412533, torch 2.4.1+cu121). They are the best
available evidence and are consistent with the published numbers, but they are not a
transcript of the released runs. Do not describe a run from these recipes as identical
to the release.
"""

# Rotation: set by every measured arm, off by default. Named once so the arms cannot
# drift apart from one another.
_ROTATE = {"aug_rotate_p": 0.5, "aug_rotate_limit": 180.0}

# The multi-scale pipeline. Both scale arms use it; only the crop range differs.
_MULTISCALE = {"multiscale": True, "eval_tiling": True}

RECIPES = {
    # Repo defaults: flips, mild brightness/contrast, mild hue; no rotation, no
    # multi-scale crop. What every pre-scale-augmentation result was trained with. This
    # is NOT the control arm of the scale experiment — for that, see `fixed`.
    "baseline": {},

    # The control arm of docs/scale-augmentation-results.md: the multi-scale pipeline
    # pinned to one scale (crop 512-512), so the only difference from `jitter` is the
    # crop range. Spread across the GSD sweep: 0.0334 (UNet).
    "fixed": {**_MULTISCALE, **_ROTATE, "crop_min_px": 512, "crop_max_px": 512},

    # The treatment arm: crop 394-666 px of a native-res tile, resized to 512 — a +/-30%
    # footprint, i.e. +/-30% effective GSD. Flattens the accuracy-vs-scale curve (spread
    # 0.0334 -> 0.0146 on UNet) at no measurable cost at the serving scale. Needs a
    # native-resolution dataset: `prepare.py --native-px`.
    "jitter": {**_MULTISCALE, **_ROTATE, "crop_min_px": 394, "crop_max_px": 666},

    # jitter + the hue recommendation from docs/scale-augmentation-loso.md. Bachsee_north
    # as a held-out site goes from F1 0.042 to 0.688; in-domain cost is -1.7 F1. The two
    # ingredients are not separable — a full hue circle at the default probability reaches
    # only 0.261, and half the range at full probability only 0.203. That document's
    # wording is "do not soften it", so these are not values to nudge.
    "robust": {**_MULTISCALE, **_ROTATE, "crop_min_px": 394, "crop_max_px": 666,
               "aug_hue_shift": 90, "aug_sat_shift": 60, "aug_val_shift": 30,
               "aug_hsv_p": 0.9},

    # UNEVALUATED. Mosaic has no measurement in this repo — no paired run, no LOSO fold,
    # no results document. It is here so the knob is reachable, not because it is
    # recommended. Before any claim rests on it, run it through the winmol-experiment
    # procedure: paired seeds against `robust`, leave-one-site-out, reported as a fold
    # count rather than a mean.
    "mosaic": {**_MULTISCALE, **_ROTATE, "crop_min_px": 394, "crop_max_px": 666,
               "aug_hue_shift": 90, "aug_sat_shift": 60, "aug_val_shift": 30,
               "aug_hsv_p": 0.9, "mosaic_p": 0.5},
}

UNEVALUATED = {"mosaic"}

# Printed alongside any recipe: leaving these at their defaults is the likeliest way a
# "reproduction" quietly measures something else.
SCHEDULE_NOTE = ("the measured runs also used --epochs 40 --batch-size 16 --seed 1 "
                 "--deterministic; recipes cover augmentation only")


def explicit_flags(parser, argv):
    """Names of the options actually present in `argv`.

    Comparing a parsed value against the parser's default cannot answer this: passing
    `--aug-hsv-p 0.5` when 0.5 *is* the default is indistinguishable from passing
    nothing, so a recipe would silently overwrite a value the user typed on purpose.

    Instead the same parser is re-run with every default replaced by None, so anything
    that comes back not-None was supplied on the command line. This is exact for both
    value options and store_true/store_false flags.
    """
    import copy

    probe = copy.deepcopy(parser)
    for action in probe._actions:
        if action.dest != "help":
            action.default = None
    # A required= or error-raising check would fire on the probe pass; ignore extras.
    parsed, _ = probe.parse_known_args(argv)
    return {dest for dest, value in vars(parsed).items() if value is not None}


def apply(recipe_name, args, parser, argv=None):
    """Fold a recipe into parsed `args`, letting explicitly-passed CLI flags win.

    `argv` is what makes "explicit" mean explicit rather than "differs from the default"
    — see `explicit_flags`. It is optional only so existing callers keep working; omit it
    and a flag set to its own default value loses to the recipe.

    Returns (args, notes); `notes` records what was applied and what the user overrode,
    so the caller can print it and the run log keeps the resolution.
    """
    if not recipe_name:
        return args, []
    try:
        recipe = RECIPES[recipe_name]
    except KeyError:
        raise SystemExit(
            f"unknown recipe {recipe_name!r}; choose from {', '.join(sorted(RECIPES))}")

    given = explicit_flags(parser, argv) if argv is not None else None

    applied, overridden = [], []
    for key, value in sorted(recipe.items()):
        current = getattr(args, key, None)
        was_given = (key in given) if given is not None else (current != parser.get_default(key))
        if was_given:
            overridden.append(f"{key}={current} (recipe wanted {value})")
            continue
        setattr(args, key, value)
        applied.append(f"{key}={value}")
    return args, applied + [f"OVERRIDDEN {o}" for o in overridden]


def describe(recipe_name):
    """One-line summary, for --help and for the run log."""
    body = RECIPES.get(recipe_name, {})
    if not body:
        return "repo defaults (flips, mild brightness/contrast, mild hue; no rotation)"
    parts = ", ".join(f"{k}={v}" for k, v in sorted(body.items()))
    flag = " [UNEVALUATED]" if recipe_name in UNEVALUATED else ""
    return parts + flag
