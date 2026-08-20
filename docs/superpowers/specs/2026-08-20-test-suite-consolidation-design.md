# Test-suite consolidation for public release

**Date:** 2026-08-20
**Status:** approved, not yet implemented
**Repos touched:** `WINMOL_segmentor_pt` (public), `winmol_segmentror_pt_helper` (private)

## Why

The public repo carries **56 test files, 280 tests, 4,039 lines, 585s of runtime**. That is
not a coverage problem — it is a legibility problem, and this repo is being prepared for
public release. The goal is a public test suite that reads as a deliberate executable spec
rather than as accumulated experiment scaffolding.

Three measurements drove the design. All were taken on this branch, not estimated:

1. **86% of runtime is ~15 tests.** The 40 slowest tests account for 505s of 585s; the
   other 240 tests finish in ~80s.
2. **Twelve `run_training()` calls across 8 files cost ~330s and assert the same three
   facts** — `set(metrics) == {"loss","precision","recall","f1"}`, plus "onnx and pt
   exist" — under different config flags. `test_val_data_dir.py` is the clearest case: it
   makes the precise, fast assertion first (`_build_loaders` yields 6 train / 4 val
   items), then spends 50s on a `run_training()` that adds no new claim.
3. **There is no `conftest.py`.** 22 of 56 files hand-roll the same synthetic stem dataset
   (`train{k}.jpeg` / `mask{k}.gif`); 15 repeat the same `TrainConfig(...)` block. This
   duplication, not the assertions, is what makes the suite 4,039 lines.

## What is explicitly *not* changing

The public repo keeps all of its code: the four entry points, `winmol_unet/training/`,
and `winmol_unet/geo/`. This is a test-layout change only. No production module moves.

## Design

### Public repo: 56 files to 10 (plus a conftest)

A new `tests/conftest.py` holds the shared fixtures every merged file draws on:

- `stem_dataset(tmp_path, n=..., size=...)` — the synthetic image/mask pair builder
  currently duplicated 22 times.
- `train_config(tmp_path, **overrides)` — the `TrainConfig` factory currently duplicated
  15 times, defaulting to `epochs=1, batch_size=2, device="cpu", encoder_weights=None`
  so merged tests stay hermetic and download nothing.
- CPU execution-provider pinning via `WINMOL_ONNX_FORCE_CPU`, so ONNX parity assertions
  stay bit-exact (CoreML and CUDA compute in fp16).

| public file | absorbs |
|---|---|
| `conftest.py` *(new)* | the 22 duplicated dataset builders, the 15 `TrainConfig` blocks |
| `test_contract.py` | `test_contract`, `test_published_models`, `test_preprocess`, `test_smoke` |
| `test_import_boundary.py` | unchanged |
| `test_export_serve.py` | `test_export`, `test_arch_onnx`, `test_runtime`, `test_quantize` |
| `test_keras_bridge.py` | `test_export_keras`, `test_export_multiformat`, `test_keras_model`, `test_no_eager_tensorflow`, `test_onnx_tf_coexist`, `test_deploy_models` |
| `test_model.py` | `test_model`, `test_model_width`, `test_init_weights`, `test_model_factory`, `test_width_wiring` (the `build_model` half) |
| `test_training.py` | `test_train_loop` (incl. the overfit test), `test_train_losses`, `test_train_dataset`, `test_dataset_cache`, `test_loss_selection`, `test_focal_loss`, `test_augment`, `test_colour_aug`, `test_run_config`, `test_run_logger` |
| `test_config_wiring.py` | `test_train_config`, `test_train_device`, `test_recipes`, plus the flag-to-config half of `test_arch_wiring`, `test_aug_wiring`, `test_width_wiring`, `test_val_data_dir`, `test_multiscale`, `test_mosaic`, `test_wandb_wiring`, `test_test_stage` |
| `test_data_prep.py` | `test_build_dataset`, `test_split_dataset`, `test_coco_to_dataset` |
| `test_geo.py` | `test_sample_training_tiles`, `test_block_splits`, `test_make_splits`, `test_compose_folds`, `test_rasterize_annotations`, `test_extract_parallel`, `test_plot_inference` |
| `test_cli.py` | `test_cli` (trimmed), `test_eval_checkpoint`; holds the inference test |

### The two tests that anchor the public suite

Per the brief, the public suite's training coverage rests on exactly two things:

- **The overfit test** — `test_training.py::test_overfit_loss_decreases`. A fake training
  run on a handful of synthetic tiles, asserting loss falls. This is the existing
  `test_train_loop.py::test_overfit_loss_decreases` (27.4s), kept as-is.
- **The inference test** — `test_cli.py::test_infer_writes_a_georeferenced_raster` (5.9s),
  which exercises `infer.py` end to end and asserts a georeferenced raster lands on disk.

### `test_config_wiring.py`: the parametrized table

The 12 wiring files each pair fast `config_from_args` assertions with a slow
`run_training()` smoke. The fast halves collapse into a single table:

```python
@pytest.mark.parametrize("flags,expected", [
    (["--arch", "deeplabv3plus", "--encoder", "resnet18"],
     {"arch": "deeplabv3plus", "encoder": "resnet18"}),
    (["--aug-rotate-p", "0.3", "--aug-rotate-limit", "20"],
     {"aug_rotate_p": 0.3, "aug_rotate_limit": 20}),
    ([], {"arch": "unet", "aug_hflip_p": 0.5, "aug_rotate_p": 0.0, "width_mult": 1.0}),
    ...
])
def test_cli_flag_reaches_config(flags, expected):
    cfg = config_from_args(["--data-dir", "d", *flags])
    for field, value in expected.items():
        assert getattr(cfg, field) == value
```

Defaults get their own rows, so the "rotation is off by default" trap documented in
`recipes.py` stays pinned. The table runs in milliseconds.

`test_recipes.py`'s nine assertions are **kept whole, not folded into the table**. They pin
the two traps CLAUDE.md calls out by name — that `multiscale=True` is not the jitter switch
(the `fixed` control sets it too; the crop *range* is the variable), and that rotation is off
by default while every measured run passed `--aug-rotate-p 0.5 --aug-rotate-limit 180`. They
are already fast, and rewriting them as parameter rows would cost the explanatory test names
that make a wrong reproduction obvious.

Where a wiring test makes a *behavioural* claim that config inspection cannot reach, the
cheap direct assertion is kept rather than dropped. `test_val_data_dir`'s
`_build_loaders` check (6 train / 4 val, no re-split) moves into `test_config_wiring.py`
as a real test; only its 50s `run_training()` tail leaves.

### Helper repo: the slow full-training halves

Moved to `winmol_segmentror_pt_helper/tests/`, following the convention already set by
`test_locate_tile.py` — a module docstring recording that the file moved in the release
split, and `pytest.importorskip` guards so a checkout without the public package installed
skips rather than errors:

`test_multiformat_e2e`, the `run_training()` half of `test_val_data_dir`, `test_test_stage`,
`test_train_e2e`, `test_small_dataset_trains`, `test_two_stage`, and the training halves of
`test_arch_wiring`, `test_aug_wiring`, `test_wandb_wiring`, `test_multiscale`, `test_mosaic`.

Fixtures are **copied**, not imported across the repo boundary. The helper repo's existing
`conftest.py` already states the rationale: test suites are not importable packages, and
coupling one repo's tests to another's test internals is worse than duplicated fixture code.

### Helper repo: the new integration test

`tests/test_full_training_integration.py` — the real full-training run the public suite no
longer attempts:

- trains for real epochs on a real prepared dataset (not synthetic tiles),
- asserts the resulting F1 clears a floor,
- asserts the exported ONNX passes `validate_onnx_model()`,
- asserts two runs at the same seed with `--deterministic` agree.

Marked `@pytest.mark.slow` and skipped unless its dataset path is present, so the helper
repo's own fast suite stays fast.

## What stays public against the brief

`test_contract.py` and `test_import_boundary.py` stay in the public repo. Both are fast,
and both guard the cross-repo boundary rather than the training code:

- `contract.py` is the frozen ONNX interface the WINMOL Analyzer depends on. A change here
  is a breaking change for another repo.
- `test_import_boundary.py` is the only thing preventing an eager import in any `__init__`
  from turning a 20 MB analyzer install into a 2 GB one. It must run in a subprocess,
  because by the time pytest reaches it torch is already in `sys.modules` and an in-process
  assertion would pass for the wrong reason.

Neither belongs in a private repo, and a public repo that cannot verify its own published
interface is not "clean".

## Expected outcome

| metric | before | after (public) |
|---|---|---|
| test files | 56 | 10 (+ `conftest.py`) |
| lines | 4,039 | ~1,600 |
| tests | 280 | ~200 |
| runtime | 585s | ~240s |

The test-count drop is deduplicated postconditions and merged fixtures, not removed
assertions. Every behavioural claim in the current suite is either kept public, made
cheaper, or relocated with a docstring recording where it went.

## Sequencing constraint

The branch has uncommitted work in `winmol_unet/contract.py`, `training/losses.py`,
`training/run_train.py` and their tests (`test_contract.py`, `test_loss_selection.py`, and
the untracked `test_focal_loss.py`). Those three test files are merge inputs. **Land the
in-flight focal-loss and contract work first**, then restructure, so the merge operates on
settled content.

## Verification

- Full public suite green before and after; test count and runtime recorded from real runs.
- `pytest --collect-only -q` diffed before and after, and every dropped test id accounted
  for in writing: merged, relocated, or deliberately deduplicated.
- Helper suite green with the public package installed editable.
- `tests/test_import_boundary.py` still passes in a subprocess.
