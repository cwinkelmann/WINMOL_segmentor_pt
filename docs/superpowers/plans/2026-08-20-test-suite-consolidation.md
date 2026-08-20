# Test-Suite Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the public repo's 56 test files to 10 plus a shared `conftest.py`, relocate the slow full-training halves to the private helper repo, and add a real full-training integration test there.

**Architecture:** This is a test-layout change only — no production module moves. Merging is mechanical: test bodies already exist on disk and are copied into subject-named files, with locally duplicated helpers swapped for two shared fixtures. Safety comes from a ledger that accounts for every collected test id before and after, so no assertion disappears silently.

**Tech Stack:** pytest, PyTorch, onnxruntime, albumentations, segmentation-models-pytorch, TensorFlow (Keras export only), rasterio/fiona/shapely (geo).

**Spec:** `docs/superpowers/specs/2026-08-20-test-suite-consolidation-design.md`

## Global Constraints

- Interpreter is the conda env: `~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python` (Python 3.11). Never use `.venv/` — it was removed.
- Tests stay **hermetic**: synthetic data in `tmp_path`, `encoder_weights=None` for smp archs so nothing downloads.
- ONNX parity tests pin the CPU execution provider via `WINMOL_ONNX_FORCE_CPU=1`. CoreML and CUDA compute in fp16 and are not bit-exact.
- `winmol_unet/__init__.py`, `cli/__init__.py` and `geo/__init__.py` import nothing eagerly. `conftest.py` must not import torch at module scope.
- Commit messages carry **no** Claude/AI references, co-author trailers, or session links.
- The helper repo lives at `/Users/christian/work/winmol_segmentror_pt_helper` and has **no hard dependency** on the public package. Relocated tests guard imports with `pytest.importorskip`.
- Helper-repo fixtures are **copied**, never imported across the repo boundary.

---

### Task 1: Land in-flight work and record the baseline ledger

The branch has uncommitted focal-loss and contract work whose test files are merge inputs. Merging on top of unsettled content would make the ledger meaningless.

**Files:**
- Modify: `winmol_unet/contract.py`, `winmol_unet/training/losses.py`, `winmol_unet/training/run_train.py`
- Modify: `tests/test_contract.py`, `tests/test_loss_selection.py`
- Create: `tests/test_focal_loss.py` (currently untracked)
- Create: `docs/superpowers/plans/consolidation-ledger.md`

**Interfaces:**
- Produces: `docs/superpowers/plans/consolidation-ledger.md` — the before/after accounting every later task appends to.

- [ ] **Step 1: Confirm the in-flight work is green on its own**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest \
  tests/test_contract.py tests/test_loss_selection.py tests/test_focal_loss.py -q
```

Expected: all pass. If any fail, stop — finish that work before consolidating.

- [ ] **Step 2: Commit the in-flight work**

```bash
git add winmol_unet/contract.py winmol_unet/training/losses.py \
        winmol_unet/training/run_train.py tests/test_contract.py \
        tests/test_loss_selection.py tests/test_focal_loss.py CLAUDE.md README.md
git commit -m "feat: add focal loss selection and widen contract validation"
```

- [ ] **Step 3: Record the baseline test ids**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q 2>/dev/null \
  | grep "::" | sort > /tmp/winmol-baseline-ids.txt
wc -l /tmp/winmol-baseline-ids.txt
git rev-parse HEAD > /tmp/winmol-baseline-sha.txt
```

Expected: 280 ids at the time of writing. Whatever this prints is the number every later
task reconciles against — record it in the ledger header rather than trusting 280.

- [ ] **Step 4: Create the ledger with its header**

```markdown
# Consolidation ledger

Baseline: 280 collected test ids at <sha from step 3>.
Every id below is accounted for as one of: **merged** (same assertion, new file),
**relocated** (moved to the helper repo), or **deduplicated** (removed because a
named surviving test already asserts it).

| old test id | disposition | new location / justification |
|---|---|---|
```

- [ ] **Step 5: Commit the ledger**

```bash
git add docs/superpowers/plans/consolidation-ledger.md
git commit -m "docs: open the test consolidation ledger"
```

---

### Task 2: Add the shared conftest and prove it against one file

22 files hand-roll the same synthetic dataset and 15 repeat the same `TrainConfig` block. Both become fixtures here. Porting one file in the same task proves the fixtures fit real call sites before nine more files depend on them.

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_train_dataset.py` (the proof-of-fit port)

**Interfaces:**
- Produces: `stem_dataset(root, n=6, size=32, stem_frac=0.5) -> pathlib.Path` — writes `root/train/train{k}.jpeg` and `root/mask/mask{k}.gif`, returns `root`.
- Produces: `train_config(data_dir=None, **overrides) -> TrainConfig` — hermetic defaults, all paths under `tmp_path`.
- Produces: `force_cpu_onnx` — autouse-on-request fixture setting `WINMOL_ONNX_FORCE_CPU=1`.

- [ ] **Step 1: Write `tests/conftest.py`**

Torch is imported inside the factory, not at module scope, so `conftest.py` stays cheap and does not undermine the import-boundary discipline.

```python
"""Shared fixtures for the WINMOL test suite.

Replaces the synthetic-dataset builder that was duplicated across 22 test modules
and the TrainConfig block duplicated across 15. Both are factories rather than
plain fixtures because call sites need several datasets (train + val + test) or
several configs (stage 1 + stage 2) inside one test.
"""
import os
import pathlib

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def stem_dataset():
    """Build a StemDataset-shaped folder: train/train{k}.jpeg + mask/mask{k}.gif.

    The image is a hard vertical split -- white where the mask is white -- so a
    model can actually fit it, which is what the overfit test needs.
    """
    def _make(root, n=6, size=32, stem_frac=0.5):
        root = pathlib.Path(root)
        img_dir, mask_dir = root / "train", root / "mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        split = max(1, int(size * stem_frac))
        for k in range(1, n + 1):
            rgb = np.zeros((size, size, 3), np.uint8)
            rgb[:, :split, :] = 255
            Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
            m = np.zeros((size, size), np.uint8)
            m[:, :split] = 255
            Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")
        return root
    return _make


@pytest.fixture
def train_config(tmp_path):
    """A hermetic TrainConfig: one epoch, CPU, no encoder download."""
    def _make(data_dir=None, **overrides):
        from winmol_unet.training.config import TrainConfig   # lazy: pulls torch
        out = tmp_path / "out"
        base = dict(
            data_dir=str(data_dir if data_dir is not None else tmp_path),
            checkpoint_dir=str(tmp_path / "ck"),
            log_dir=str(tmp_path / "log"),
            hdf5_out=str(out / "m.hdf5"),
            onnx_out=str(out / "m.onnx"),
            pt_out=str(out / "m.pt"),
            keras_out=str(out / "m.keras"),
            epochs=1,
            batch_size=2,
            patience=999,
            device="cpu",
            encoder_weights=None,
        )
        base.update(overrides)
        return TrainConfig(**base)
    return _make


@pytest.fixture
def force_cpu_onnx(monkeypatch):
    """Pin the CPU EP so ONNX parity assertions are bit-exact.

    CoreML and CUDA compute in fp16 and drift past a tight epsilon.
    """
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    yield
    monkeypatch.delenv("WINMOL_ONNX_FORCE_CPU", raising=False)
```

- [ ] **Step 2: Verify the fixtures load without importing torch**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_smoke.py -q
```

Expected: PASS. Collection must not error on `conftest.py`.

- [ ] **Step 3: Port `tests/test_train_dataset.py` onto the fixtures**

Replace its local dataset builder with the `stem_dataset` fixture. Keep every assertion and every test name byte-identical — only the setup changes.

- [ ] **Step 4: Run the ported file**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_train_dataset.py -v
```

Expected: the same 5 tests pass, same names.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_train_dataset.py
git commit -m "test: add shared dataset and config fixtures"
```

---

## The merge recipe (Tasks 3-11)

Every merge task follows the identical procedure. It is written once here; each task below states only its inputs, output, and any case-specific note.

1. **Snapshot the inputs.** `pytest --collect-only -q <input files>` and save the ids.
2. **Create the output file** with a module docstring naming the files it absorbs.
3. **Copy each test body verbatim.** Do not rewrite assertions. Where two inputs define the same local helper, delete both and use the `stem_dataset` / `train_config` fixture.
4. **Resolve name collisions** by prefixing with the former module's subject, e.g. two `test_output_shape` functions become `test_unet_output_shape` and `test_keras_unet_output_shape`. Record every rename in the ledger.
5. **Delete the input files** with `git rm`.
6. **Run the output file** and confirm the test count matches the input snapshot minus any row explicitly marked *deduplicated* in the ledger.
7. **Append one ledger row per input test id.**
8. **Commit** the output, the deletions, and the ledger together, so any single task can be reverted cleanly.

A test may only be marked *deduplicated* if the ledger row names the surviving test that makes the same assertion. "Looks redundant" is not a justification.

---

### Task 3: Merge `test_contract.py`

**Files:**
- Modify: `tests/test_contract.py` (absorbs the others)
- Delete: `tests/test_published_models.py`, `tests/test_preprocess.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: `force_cpu_onnx` from Task 2.

- [ ] **Step 1: Snapshot inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q \
  tests/test_contract.py tests/test_published_models.py \
  tests/test_preprocess.py tests/test_smoke.py 2>/dev/null | grep "::" | sort
```

Record the printed count. It is this task's reconciliation target — do not
compute it from `def test_` counts, because parametrized tests expand into several
ids each (`test_arch_onnx.py` has 2 defs but collects 5).

- [ ] **Step 2: Apply the merge recipe**

Docstring: `"""The frozen cross-repo ONNX contract, the preprocessing it assumes, and the published models that must satisfy it."""`

Swap `test_published_models.py`'s hand-set `monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")` for the `force_cpu_onnx` fixture.

- [ ] **Step 3: Run and verify**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_contract.py -v
git rm tests/test_published_models.py tests/test_preprocess.py tests/test_smoke.py
```

Expected: the count matches Step 1's snapshot.

- [ ] **Step 4: Append 15 ledger rows and commit**

```bash
git add tests/test_contract.py docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: fold preprocessing and published-model checks into the contract suite"
```

---

### Task 4: Merge `test_export_serve.py`

**Files:**
- Create: `tests/test_export_serve.py`
- Delete: `tests/test_export.py`, `tests/test_arch_onnx.py`, `tests/test_runtime.py`, `tests/test_quantize.py`

**Interfaces:**
- Consumes: `force_cpu_onnx`, `train_config` from Task 2.

**Case-specific note:** `test_runtime.py` manipulates `WINMOL_ONNX_PROVIDERS` and `WINMOL_ONNX_FORCE_CPU` directly to test provider selection itself. Those tests must **not** adopt `force_cpu_onnx` — the env var is their subject, not their setup. Keep their manual handling and note it in a comment.

- [ ] **Step 1: Snapshot inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q \
  tests/test_export.py tests/test_arch_onnx.py tests/test_runtime.py \
  tests/test_quantize.py 2>/dev/null | grep "::" | sort
```

Record the printed count. It is this task's reconciliation target — do not
compute it from `def test_` counts, because parametrized tests expand into several
ids each (`test_arch_onnx.py` has 2 defs but collects 5).

- [ ] **Step 2: Apply the merge recipe**

Docstring: `"""Export any nn.Module to ONNX and serve it: contract shape, torch-vs-ONNX parity across architectures, provider selection, and fp16 quantization."""`

- [ ] **Step 3: Run and verify**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_export_serve.py -v
git rm tests/test_export.py tests/test_arch_onnx.py tests/test_runtime.py tests/test_quantize.py
```

Expected: the count matches Step 1's snapshot, including the parametrized `test_arch_exports_and_serves_nhwc[hrnet-None]` and the `test_dpt_cannot_yet_export_onnx` negative case.

- [ ] **Step 4: Append ledger rows and commit**

```bash
git add tests/test_export_serve.py docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: merge export, arch-ONNX, runtime and quantize into one serving suite"
```

---

### Task 5: Merge `test_keras_bridge.py`

**Files:**
- Create: `tests/test_keras_bridge.py`
- Delete: `tests/test_export_keras.py`, `tests/test_export_multiformat.py`, `tests/test_keras_model.py`, `tests/test_no_eager_tensorflow.py`, `tests/test_onnx_tf_coexist.py`, `tests/test_deploy_models.py`

**Case-specific note:** this whole file is the UNet-only Keras path. Guard the module with `pytest.importorskip("tensorflow")` at the top **except** for the tests that assert TensorFlow is *not* imported eagerly (`test_no_eager_tensorflow.py`) — those must run without TF present, so they keep their own guards and must not sit behind a module-level skip. If that proves awkward, keep `test_no_eager_tensorflow.py`'s single test in `test_import_boundary.py` instead and record that in the ledger.

- [ ] **Step 1: Snapshot inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q \
  tests/test_export_keras.py tests/test_export_multiformat.py tests/test_keras_model.py \
  tests/test_no_eager_tensorflow.py tests/test_onnx_tf_coexist.py \
  tests/test_deploy_models.py 2>/dev/null | grep "::" | sort
```

Record the printed count. It is this task's reconciliation target — do not
compute it from `def test_` counts, because parametrized tests expand into several
ids each (`test_arch_onnx.py` has 2 defs but collects 5).

- [ ] **Step 2: Apply the merge recipe**

Docstring: `"""The UNet-only Keras HDF5 drop-in: hand-built mirror, layer-by-layer weight transfer, torch-vs-Keras parity, and coexistence with onnxruntime in one process."""`

- [ ] **Step 3: Run and verify**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_keras_bridge.py -v
git rm tests/test_export_keras.py tests/test_export_multiformat.py tests/test_keras_model.py \
       tests/test_no_eager_tensorflow.py tests/test_onnx_tf_coexist.py tests/test_deploy_models.py
```

Expected: the count matches Step 1's snapshot (or snapshot minus 1, if the eager-TensorFlow test was relocated to `test_import_boundary.py` per the note).

- [ ] **Step 4: Append ledger rows and commit**

```bash
git add tests/test_keras_bridge.py tests/test_import_boundary.py \
        docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: gather the UNet Keras bridge into one suite"
```

---

### Task 6: Merge `test_model.py`

**Files:**
- Modify: `tests/test_model.py` (absorbs the others)
- Delete: `tests/test_model_width.py`, `tests/test_init_weights.py`, `tests/test_model_factory.py`
- Modify: `tests/test_width_wiring.py` — only its two `build_model` tests move here; its two `config_from_args` tests are Task 8's input. Do **not** delete the file yet.

**Case-specific note:** `test_width_wiring.py` is split across two tasks. Leave the file in place holding only its CLI tests after this task, and let Task 8 delete it.

- [ ] **Step 1: Snapshot inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q \
  tests/test_model.py tests/test_model_width.py tests/test_init_weights.py \
  tests/test_model_factory.py tests/test_width_wiring.py 2>/dev/null | grep "::" | sort
```

Record the printed count. All of it lands here except exactly two ids —
`test_cli_parses_width_mult` and `test_cli_width_mult_defaults_to_one` — which stay in
`test_width_wiring.py` for Task 8. Note `test_model_factory.py` is parametrized: 3 defs
collect as 6 ids.

- [ ] **Step 2: Apply the merge recipe**

Docstring: `"""The UNet definition and the architecture factory: output shape, channel width, block order, weight init, and contract-shaped logits from every supported arch."""`

Preserve `test_block_order_switches_conv_norm_activation` — it pins the documented Conv-BN-ReLU deviation from the R original.

- [ ] **Step 3: Run and verify**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_model.py tests/test_width_wiring.py -v
git rm tests/test_model_width.py tests/test_init_weights.py tests/test_model_factory.py
```

Expected: Step 1's count minus 2 in `test_model.py`, and exactly 2 remaining in `test_width_wiring.py`.

- [ ] **Step 4: Append ledger rows and commit**

```bash
git add tests/test_model.py tests/test_width_wiring.py docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: merge model width, init and factory into the model suite"
```

---

### Task 7: Merge `test_training.py`

This file carries **the overfit test** — one of the two anchors the public suite keeps by design.

**Files:**
- Create: `tests/test_training.py`
- Delete: `tests/test_train_loop.py`, `tests/test_train_losses.py`, `tests/test_train_dataset.py`, `tests/test_dataset_cache.py`, `tests/test_loss_selection.py`, `tests/test_focal_loss.py`, `tests/test_augment.py`, `tests/test_colour_aug.py`, `tests/test_run_config.py`, `tests/test_run_logger.py`

**Case-specific note:** `test_colour_aug.py` imports `run_train`; keep only its augmentation-pipeline assertions here. If it contains a `run_training()` call, that call is Task 12's input, not this task's.

- [ ] **Step 1: Snapshot inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q \
  tests/test_train_loop.py tests/test_train_losses.py tests/test_train_dataset.py \
  tests/test_dataset_cache.py tests/test_loss_selection.py tests/test_focal_loss.py \
  tests/test_augment.py tests/test_colour_aug.py tests/test_run_config.py \
  tests/test_run_logger.py 2>/dev/null | grep "::" | sort
```

Record the printed count. It is this task's reconciliation target — do not
compute it from `def test_` counts, because parametrized tests expand into several
ids each (`test_arch_onnx.py` has 2 defs but collects 5).

- [ ] **Step 2: Apply the merge recipe**

Docstring: `"""The training loop and everything it consumes: the dataset and its resize cache, augmentation, loss selection, metrics, the run manifest, and the loggers. test_overfit_loss_decreases is the public suite's fake-training anchor."""`

- [ ] **Step 3: Verify the overfit anchor explicitly**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest \
  tests/test_training.py::test_overfit_loss_decreases -v
```

Expected: PASS. This test must survive under exactly this name — the spec names it as a public anchor.

- [ ] **Step 4: Run the whole file and delete inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_training.py -v
git rm tests/test_train_loop.py tests/test_train_losses.py tests/test_train_dataset.py \
       tests/test_dataset_cache.py tests/test_loss_selection.py tests/test_focal_loss.py \
       tests/test_augment.py tests/test_colour_aug.py tests/test_run_config.py \
       tests/test_run_logger.py
```

Expected: the count matches Step 1's snapshot.

- [ ] **Step 5: Append ledger rows and commit**

```bash
git add tests/test_training.py docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: gather the training loop, losses, augmentation and logging into one suite"
```

---

### Task 8: Create `test_config_wiring.py`

The largest single reduction. Twelve files each pair fast `config_from_args` assertions with a slow `run_training()` smoke; the fast halves collapse into one table and the slow halves become Task 12's input.

**Files:**
- Create: `tests/test_config_wiring.py`
- Delete: `tests/test_train_config.py`, `tests/test_train_device.py`, `tests/test_width_wiring.py`
- Modify (strip to their training halves, leaving them as Task 12 inputs): `tests/test_arch_wiring.py`, `tests/test_aug_wiring.py`, `tests/test_val_data_dir.py`, `tests/test_multiscale.py`, `tests/test_mosaic.py`, `tests/test_wandb_wiring.py`, `tests/test_test_stage.py`
- Move whole: `tests/test_recipes.py` content into this file

**Interfaces:**
- Consumes: `config_from_args` from `winmol_unet.training.run_train`.

- [ ] **Step 1: Write the parametrized flag table**

```python
"""Every CLI flag reaches the config field it claims to set.

Collapses the flag-parsing half of twelve former wiring modules. The training
half of each -- a full run_training() call asserting the same three
postconditions -- moved to the helper repo's integration suite.
"""
import pytest

from winmol_unet.training.run_train import config_from_args


@pytest.mark.parametrize("flags,expected", [
    # architecture
    (["--arch", "deeplabv3plus", "--encoder", "resnet18", "--encoder-weights", "imagenet"],
     {"arch": "deeplabv3plus", "encoder": "resnet18", "encoder_weights": "imagenet"}),
    # augmentation
    (["--aug-rotate-p", "0.3", "--aug-rotate-limit", "20", "--aug-hflip-p", "0.25"],
     {"aug_rotate_p": 0.3, "aug_rotate_limit": 20.0, "aug_hflip_p": 0.25}),
    # width
    (["--width-mult", "0.5"], {"width_mult": 0.5}),
    # fixed validation split
    (["--val-data-dir", "v"], {"val_data_dir": "v"}),
    # multiscale crop range
    (["--multiscale", "--crop-min-px", "400", "--crop-max-px", "900"],
     {"multiscale": True, "crop_min_px": 400, "crop_max_px": 900}),
    # defaults -- these rows pin the traps documented in recipes.py
    ([], {"arch": "unet", "width_mult": 1.0, "aug_hflip_p": 0.5,
          "aug_rotate_p": 0.0, "multiscale": False, "val_data_dir": None}),
])
def test_cli_flag_reaches_config(flags, expected):
    cfg = config_from_args(["--data-dir", "d", *flags])
    for field, value in expected.items():
        assert getattr(cfg, field) == value, f"{field} did not reach the config"
```

Add one row per flag group actually asserted by the twelve input files. Read each input file and transcribe its assertions — do not guess flag names.

- [ ] **Step 2: Move the behavioural checks that config inspection cannot reach**

`test_val_data_dir.py`'s loader assertion is a real behavioural claim and is fast. Move it whole:

```python
def test_fixed_val_split_is_used_not_resplit(tmp_path, stem_dataset, train_config):
    """A fixed --val-data-dir is used as given: no 80/20 re-split of either set."""
    from winmol_unet.training.augment import build_augmentation
    from winmol_unet.training.run_train import _build_loaders

    train_d, val_d = tmp_path / "tr", tmp_path / "va"
    stem_dataset(train_d, n=6)
    stem_dataset(val_d, n=4)
    cfg = train_config(data_dir=train_d, val_data_dir=str(val_d))

    tl, vl = _build_loaders(cfg.image_dir, cfg.mask_dir, cfg, build_augmentation(cfg),
                            val_image_dir=cfg.val_image_dir, val_mask_dir=cfg.val_mask_dir)
    assert len(tl.dataset) == 6 and len(vl.dataset) == 4
```

The 50s `run_training()` tail this test used to carry is dropped — record it in the ledger as *deduplicated*, justified by the helper repo's integration test, which exercises `run_training` end to end.

- [ ] **Step 3: Move `test_recipes.py` in whole**

Copy all nine tests unchanged, keeping their names. They pin the two traps CLAUDE.md calls out — that `multiscale=True` is not the jitter switch, and that rotation is off by default — and their explanatory names are the point.

- [ ] **Step 4: Add device resolution and config defaults**

Copy `test_train_device.py`'s two tests and `test_train_config.py`'s one test verbatim.

- [ ] **Step 5: Run and verify**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_config_wiring.py -v --durations=5
git rm tests/test_train_config.py tests/test_train_device.py tests/test_width_wiring.py tests/test_recipes.py
```

Expected: all pass, and the **slowest test is under 2 seconds** except the one `_build_loaders` case. If anything takes 20s+, a `run_training()` call was copied in by mistake.

- [ ] **Step 6: Append ledger rows and commit**

```bash
git add tests/test_config_wiring.py tests/test_arch_wiring.py tests/test_aug_wiring.py \
        tests/test_val_data_dir.py tests/test_multiscale.py tests/test_mosaic.py \
        tests/test_wandb_wiring.py tests/test_test_stage.py \
        docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: collapse twelve wiring modules into one parametrized flag table"
```

---

### Task 9: Create `test_data_prep.py`

**Files:**
- Create: `tests/test_data_prep.py`
- Delete: `tests/test_build_dataset.py`, `tests/test_split_dataset.py`, `tests/test_coco_to_dataset.py`

- [ ] **Step 1: Snapshot inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q \
  tests/test_build_dataset.py tests/test_split_dataset.py \
  tests/test_coco_to_dataset.py 2>/dev/null | grep "::" | sort
```

Record the printed count. It is this task's reconciliation target — do not
compute it from `def test_` counts, because parametrized tests expand into several
ids each (`test_arch_onnx.py` has 2 defs but collects 5).

- [ ] **Step 2: Apply the merge recipe**

Docstring: `"""prepare.py --from-folder: converting arbitrary folders (non-integer names, palette and instance masks, COCO annotations) into the train{N}.jpeg / mask{N}.gif convention StemDataset expects."""`

- [ ] **Step 3: Run and verify**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_data_prep.py -v
git rm tests/test_build_dataset.py tests/test_split_dataset.py tests/test_coco_to_dataset.py
```

Expected: the count matches Step 1's snapshot.

- [ ] **Step 4: Append ledger rows and commit**

```bash
git add tests/test_data_prep.py docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: merge dataset building, splitting and COCO conversion"
```

---

### Task 10: Create `test_geo.py`

**Files:**
- Create: `tests/test_geo.py`
- Delete: `tests/test_sample_training_tiles.py`, `tests/test_block_splits.py`, `tests/test_make_splits.py`, `tests/test_compose_folds.py`, `tests/test_rasterize_annotations.py`, `tests/test_extract_parallel.py`, `tests/test_plot_inference.py`

**Case-specific note:** `test_sample_training_tiles.py` is 410 lines and 19 tests — the largest input in the plan, and its synthetic-site fixture (`_write_ortho`, `_write_polygons`) is the one the helper repo already copied into its own `conftest.py`. Move that fixture into `tests/conftest.py` as a `geo_site` fixture guarded by `pytest.importorskip` for rasterio/fiona/shapely, so `test_geo.py` and any future geo test share it. Do **not** delete the helper repo's copy — its docstring explains why the duplication is deliberate.

- [ ] **Step 1: Snapshot inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q \
  tests/test_sample_training_tiles.py tests/test_block_splits.py tests/test_make_splits.py \
  tests/test_compose_folds.py tests/test_rasterize_annotations.py \
  tests/test_extract_parallel.py tests/test_plot_inference.py 2>/dev/null | grep "::" | sort
```

Record the printed count. It is this task's reconciliation target — do not
compute it from `def test_` counts, because parametrized tests expand into several
ids each (`test_arch_onnx.py` has 2 defs but collects 5).

- [ ] **Step 2: Add the `geo_site` fixture to `tests/conftest.py`**

Lift `_write_ortho` and `_write_polygons` from `test_sample_training_tiles.py` verbatim, guarded at fixture level:

```python
@pytest.fixture
def geo_site(tmp_path):
    """A tiny synthetic orthomosaic + stem/AOI shapefiles, in a real CRS.

    Lifted from test_sample_training_tiles.py. Everything is built in tmp_path,
    so nothing touches the real orthomosaics (100-1900 megapixels).
    """
    pytest.importorskip("rasterio")
    pytest.importorskip("fiona")
    pytest.importorskip("shapely")
    # Transcribe verbatim from tests/test_sample_training_tiles.py:
    #   lines 17-19  CRS / GSD / ORIGIN constants
    #   lines 22-34  _write_ortho
    #   lines 36-51  _write_polygons
    #   lines 53-..  the body of the existing `site` fixture
    # Return that fixture's dict unchanged, so call sites keep site["ortho"],
    # site["stems"], site["aoi"].
```

Transcribe the bodies from the source file rather than reinventing them — the CRS (`EPSG:25833`), GSD (0.02) and mid-grey noise range (60-200) matter, because nodata rejection is asserted against them.

- [ ] **Step 3: Apply the merge recipe**

Docstring: `"""The geo pipeline: tile sampling from an orthomosaic, leak-free block splits, fold composition, annotation rasterization, parallel extraction, and prediction plotting."""`

- [ ] **Step 4: Run and verify**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_geo.py -v
git rm tests/test_sample_training_tiles.py tests/test_block_splits.py tests/test_make_splits.py \
       tests/test_compose_folds.py tests/test_rasterize_annotations.py \
       tests/test_extract_parallel.py tests/test_plot_inference.py
```

Expected: the count matches Step 1's snapshot. `test_min_stem_frac_rejects_sparse_tiles` and `test_nodata_collar_is_rejected` must survive — they are the leak-freedom and nodata guards the experiment skill depends on.

- [ ] **Step 5: Append ledger rows and commit**

```bash
git add tests/test_geo.py tests/conftest.py docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: merge the geo pipeline suites and share the synthetic-site fixture"
```

---

### Task 11: Trim `test_cli.py`

This file carries **the inference test** — the second public anchor.

**Files:**
- Modify: `tests/test_cli.py`
- Delete: `tests/test_eval_checkpoint.py`

- [ ] **Step 1: Snapshot inputs**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q \
  tests/test_cli.py tests/test_eval_checkpoint.py 2>/dev/null | grep "::" | sort
```

Record the printed count. It is this task's reconciliation target — do not
compute it from `def test_` counts, because parametrized tests expand into several
ids each (`test_arch_onnx.py` has 2 defs but collects 5).

- [ ] **Step 2: Apply the merge recipe**

Docstring: `"""The four entry points end to end: prepare, train, infer, evaluate. test_infer_writes_a_georeferenced_raster is the public suite's inference anchor."""`

Port the local dataset builder to `stem_dataset` and the local config to `train_config`.

- [ ] **Step 3: Verify the inference anchor explicitly**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest \
  tests/test_cli.py::test_infer_writes_a_georeferenced_raster -v
```

Expected: PASS, under exactly this name.

- [ ] **Step 4: Run the file and delete the input**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_cli.py -v
git rm tests/test_eval_checkpoint.py
```

Expected: the count matches Step 1's snapshot.

- [ ] **Step 5: Append ledger rows and commit**

```bash
git add tests/test_cli.py docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: fold checkpoint scoring into the CLI suite"
```

---

### Task 12: Relocate the slow full-training halves to the helper repo

**Files:**
- Create: `/Users/christian/work/winmol_segmentror_pt_helper/tests/test_training_variants.py`
- Create: `/Users/christian/work/winmol_segmentror_pt_helper/tests/test_two_stage.py`
- Modify: `/Users/christian/work/winmol_segmentror_pt_helper/tests/conftest.py`
- Delete (public repo): `tests/test_arch_wiring.py`, `tests/test_aug_wiring.py`, `tests/test_val_data_dir.py`, `tests/test_multiscale.py`, `tests/test_mosaic.py`, `tests/test_wandb_wiring.py`, `tests/test_test_stage.py`, `tests/test_multiformat_e2e.py`, `tests/test_train_e2e.py`, `tests/test_small_dataset_trains.py`, `tests/test_two_stage.py`

**Interfaces:**
- Consumes: `run_training`, `run_two_stage`, `TrainConfig` from the public package, installed editable.

**Case-specific note:** follow the convention `test_locate_tile.py` already set — a module docstring recording that the file moved in the release split, plus `pytest.importorskip` guards so a checkout without the public package skips rather than errors.

- [ ] **Step 1: Confirm the public package is importable from the helper repo**

```bash
cd /Users/christian/work/winmol_segmentror_pt_helper
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -c "import winmol_unet.training.run_train as m; print(m.__file__)"
```

Expected: a path inside `/Users/christian/work/work/WINMOL_segmentor_pt`. If it fails:

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/pip install -e "/Users/christian/work/work/WINMOL_segmentor_pt[train,geo]"
```

- [ ] **Step 2: Copy the shared fixtures into the helper conftest**

Append `stem_dataset` and `train_config` from the public `tests/conftest.py` **verbatim as a copy**, under a comment stating why:

```python
# Copied, not imported, from the public repo's tests/conftest.py. Test suites are
# not importable packages, and coupling this repo's tests to another repo's test
# internals is worse than duplicated fixture code -- same rationale as the
# geo_site fixture above.
```

- [ ] **Step 3: Write `test_training_variants.py`**

Header:

```python
"""Full run_training() variants, moved here in the release split.

Each of these calls run_training() end to end to assert one config flag is
honoured. They cost ~330s between them, which is why they left the public repo;
the public suite asserts the same flags reach the config in
tests/test_config_wiring.py, and proves training works at all with a single
overfit test. Needs the public repo installed editable:

    pip install -e /Users/christian/work/work/WINMOL_segmentor_pt[train,geo]
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("albumentations")
```

Copy the `run_training()` tests from the seven stripped wiring files plus `test_multiformat_e2e.py`, `test_train_e2e.py` and `test_small_dataset_trains.py`, unchanged apart from fixture names.

- [ ] **Step 4: Write `test_two_stage.py`**

Copy all four tests from the public `tests/test_two_stage.py` verbatim, with the same docstring header. `test_two_stage_trains_both_stages_and_exports` must keep its name — CLAUDE.md cites it by name as the single-test example.

- [ ] **Step 5: Run the helper suite**

```bash
cd /Users/christian/work/winmol_segmentror_pt_helper
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_training_variants.py tests/test_two_stage.py -v
```

Expected: all pass.

- [ ] **Step 6: Delete the public originals and commit both repos**

```bash
cd /Users/christian/work/work/WINMOL_segmentor_pt
git rm tests/test_arch_wiring.py tests/test_aug_wiring.py tests/test_val_data_dir.py \
       tests/test_multiscale.py tests/test_mosaic.py tests/test_wandb_wiring.py \
       tests/test_test_stage.py tests/test_multiformat_e2e.py tests/test_train_e2e.py \
       tests/test_small_dataset_trains.py tests/test_two_stage.py
git add docs/superpowers/plans/consolidation-ledger.md
git commit -m "test: move the slow full-training variants to the helper repo"

cd /Users/christian/work/winmol_segmentror_pt_helper
git add tests/
git commit -m "test: take on the full-training variants from the release split"
```

---

### Task 13: Add the full-training integration test to the helper repo

**Files:**
- Create: `/Users/christian/work/winmol_segmentror_pt_helper/tests/test_full_training_integration.py`
- Modify: `/Users/christian/work/winmol_segmentror_pt_helper/pyproject.toml` (register the `slow` marker)

- [ ] **Step 1: Register the marker**

Add to `[tool.pytest.ini_options]` in the helper repo's `pyproject.toml`:

```toml
markers = [
    "slow: real training on a real dataset; needs WINMOL_INTEGRATION_DATA set",
]
```

- [ ] **Step 2: Write the failing test**

```python
"""A real training run, end to end, on a real prepared dataset.

The public repo proves training works with a synthetic overfit test. This proves
it works on real tiles: that the loss lands somewhere useful, that the exported
ONNX satisfies the frozen contract, and that a seeded deterministic run is
reproducible. Opt-in: set WINMOL_INTEGRATION_DATA to a prepared dataset
directory (one containing train/ and mask/).
"""
import os
import pathlib

import pytest

pytest.importorskip("torch")

DATA = os.environ.get("WINMOL_INTEGRATION_DATA")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DATA, reason="set WINMOL_INTEGRATION_DATA to run"),
]

F1_FLOOR = 0.35   # a real run on real tiles clears this comfortably; well under
                  # the ~0.7 the measured runs reach, so it fails on breakage,
                  # not on ordinary run-to-run variance


def _config(tmp_path, seed):
    from winmol_unet.training.config import TrainConfig
    out = tmp_path / f"out{seed}"
    return TrainConfig(
        data_dir=DATA,
        checkpoint_dir=str(tmp_path / f"ck{seed}"),
        log_dir=str(tmp_path / f"log{seed}"),
        hdf5_out=str(out / "m.hdf5"),
        onnx_out=str(out / "m.onnx"),
        pt_out=str(out / "m.pt"),
        epochs=3,
        batch_size=4,
        patience=999,
        device="cpu",
        seed=seed,
        deterministic=True,
        encoder_weights=None,
    )


def test_real_training_clears_the_f1_floor_and_exports_a_valid_model(tmp_path):
    from winmol_unet.contract import validate_onnx_model
    from winmol_unet.training.run_train import run_training

    cfg = _config(tmp_path, seed=1)
    metrics = run_training(cfg)

    assert set(metrics) >= {"loss", "precision", "recall", "f1"}
    assert metrics["f1"] >= F1_FLOOR, f"F1 {metrics['f1']:.3f} below floor {F1_FLOOR}"
    assert pathlib.Path(cfg.onnx_out).exists()
    validate_onnx_model(cfg.onnx_out)      # raises if the frozen contract is broken


def test_deterministic_runs_at_the_same_seed_agree(tmp_path):
    from winmol_unet.training.run_train import run_training

    a = run_training(_config(tmp_path / "a", seed=7))
    b = run_training(_config(tmp_path / "b", seed=7))
    assert a["f1"] == pytest.approx(b["f1"], abs=1e-6)
```

- [ ] **Step 3: Run it unset — it must skip, not fail**

```bash
cd /Users/christian/work/winmol_segmentror_pt_helper
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_full_training_integration.py -v
```

Expected: 2 skipped, reason "set WINMOL_INTEGRATION_DATA to run".

- [ ] **Step 4: Run it against a real dataset**

```bash
WINMOL_INTEGRATION_DATA=<path to a prepared dataset> \
  ~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest \
  tests/test_full_training_integration.py -v
```

Expected: both pass. If the F1 floor is not cleared at 3 epochs on the chosen dataset, lower `epochs` is wrong — raise epochs or lower the floor, and record in the docstring which dataset the floor was calibrated against.

- [ ] **Step 5: Commit**

```bash
git add tests/test_full_training_integration.py pyproject.toml
git commit -m "test: add the full-training integration test"
```

---

### Task 14: Reconcile the ledger, verify both suites, update the docs

**Files:**
- Modify: `docs/superpowers/plans/consolidation-ledger.md`
- Modify: `CLAUDE.md`
- Modify: `docs/README.md`

- [ ] **Step 1: Reconcile every baseline id**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest --collect-only -q 2>/dev/null \
  | grep "::" | sort > /tmp/winmol-after-ids.txt
wc -l /tmp/winmol-baseline-ids.txt /tmp/winmol-after-ids.txt
```

Then confirm every id in the baseline appears in the ledger exactly once:

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python - <<'EOF'
import pathlib, re
base = {l.strip() for l in open("/tmp/winmol-baseline-ids.txt") if "::" in l}
ledger = pathlib.Path("docs/superpowers/plans/consolidation-ledger.md").read_text()
missing = sorted(i for i in base if i.split("::")[-1] not in ledger)
print(f"baseline={len(base)} unaccounted={len(missing)}")
for m in missing:
    print("   ", m)
EOF
```

Expected: `unaccounted=0`. Any remaining id means an assertion was dropped without justification — go back and account for it before continuing.

- [ ] **Step 2: Run the full public suite**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest -q --durations=10
```

Expected: green. Record the test count and wall-clock. Target from the spec is ~200 tests in ~240s; if runtime is still above 400s, a `run_training()` call was left behind — find it in the durations output.

- [ ] **Step 3: Confirm the import boundary still holds**

```bash
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest tests/test_import_boundary.py -v
```

Expected: the count matches Step 1's snapshot. This one must still spawn a subprocess — an in-process assertion passes for the wrong reason, because torch is already in `sys.modules` by the time pytest reaches it.

- [ ] **Step 4: Run the helper suite**

```bash
cd /Users/christian/work/winmol_segmentror_pt_helper
~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python -m pytest -q
```

Expected: green, with the integration test skipped.

- [ ] **Step 5: Update `CLAUDE.md`**

Two edits:
- In **Commands**, replace the single-test example `pytest tests/test_two_stage.py::test_two_stage_trains_both_stages_and_exports` — that file now lives in the helper repo. Use `pytest tests/test_training.py::test_overfit_loss_decreases`.
- In **Conventions**, state the new layout: ten subject-named test files plus `conftest.py`; the slow full-training variants and the integration test live in the helper repo; `test_contract.py` and `test_import_boundary.py` stay public because they guard the cross-repo boundary.

- [ ] **Step 6: Update `docs/README.md`**

Add a row to the table of what moved to the private helper repo, matching the existing "(private helper repo)" convention: the full-training variants and the integration test.

- [ ] **Step 7: Final commit**

```bash
cd /Users/christian/work/work/WINMOL_segmentor_pt
git add CLAUDE.md docs/README.md docs/superpowers/plans/consolidation-ledger.md
git commit -m "docs: record the consolidated test layout"
```

---

## Done when

- Public repo has exactly 10 `test_*.py` files plus `conftest.py`.
- `pytest -q` is green and materially faster than the 585s baseline.
- The ledger accounts for all 280 baseline test ids, with every *deduplicated* row naming the surviving test that covers it.
- `test_training.py::test_overfit_loss_decreases` and `test_cli.py::test_infer_writes_a_georeferenced_raster` both pass under those exact names.
- The helper suite is green, and its integration test passes against a real dataset.
