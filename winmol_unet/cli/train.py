"""`train.py` — train a segmentation model, single-stage or two-stage.

A thin layer over `winmol_unet.training.run_train`, which owns every flag. What this adds
is `--recipe`: a named set of defaults for the augmentation configurations this repo
actually measured, so reproducing a published result does not depend on remembering seven
numbers. Explicit flags always beat the recipe, and the resolved configuration is printed
and written to the run directory, so a run is auditable from its own log.

    train.py --data-dir DS --out-dir runs/a --recipe robust --arch hrnet
    train.py --gen-data-dir GEN --spec-data-dir SPEC --recipe jitter    # two-stage
    train.py --print-recipe robust                                      # resolve and exit
"""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Imported here, not at module scope: `import winmol_unet.cli` must stay torch-free
    # for the analyzer's install (see tests/test_import_boundary.py).
    from winmol_unet.training import recipes
    from winmol_unet.training.run_train import (
        build_parser, config_from_parsed, run_training, run_two_stage, write_run_config,
    )

    parser = build_parser()
    parser.add_argument(
        "--recipe", choices=sorted(recipes.RECIPES), default=None,
        help="named augmentation preset; explicit flags override it. "
             + "; ".join(f"{n}: {recipes.describe(n)}" for n in sorted(recipes.RECIPES)))
    parser.add_argument(
        "--print-recipe", metavar="NAME", default=None,
        help="print the settings NAME would apply, then exit without training")

    args = parser.parse_args(argv)

    if args.print_recipe:
        name = args.print_recipe
        if name not in recipes.RECIPES:
            parser.error(f"unknown recipe {name!r}; "
                         f"choose from {', '.join(sorted(recipes.RECIPES))}")
        print(f"recipe {name}: {recipes.describe(name)}")
        for key, value in sorted(recipes.RECIPES[name].items()):
            print(f"  {key} = {value}")
        print(f"\nNote: {recipes.SCHEDULE_NOTE}")
        if name in recipes.UNEVALUATED:
            print("\nUNEVALUATED: no measurement in this repo backs this recipe. "
                  "Do not report a result from it as a comparison without running it "
                  "through the winmol-experiment procedure first.")
        return 0

    # argv is passed so "explicit" means "present on the command line", not "differs
    # from the default" — otherwise `--aug-hsv-p 0.5` (which is also the default) would
    # lose to `--recipe robust` and silently become the strong-hue arm.
    args, applied = recipes.apply(args.recipe, args, parser, argv)
    if args.recipe:
        print(f"recipe {args.recipe}: " + (", ".join(applied) if applied else "no changes"))
        print(f"note: {recipes.SCHEDULE_NOTE}")
        if args.recipe in recipes.UNEVALUATED:
            print(f"WARNING: recipe {args.recipe!r} is UNEVALUATED — nothing in docs/ "
                  f"measures it. Treat any result from it as provisional.")

    cfg = config_from_parsed(args, parser)
    write_run_config(cfg, argv=argv)

    two_stage = bool(cfg.gen_data_dir and cfg.spec_data_dir)
    result = run_two_stage(cfg) if two_stage else run_training(cfg)
    print(result)
    return 0
