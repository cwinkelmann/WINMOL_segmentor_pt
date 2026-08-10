# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`winmol_unet` re-implements the WINMOL tree-stem segmentation model (originally R + Keras/TensorFlow in the sibling `WINMOL_segmentor` repo) in **PyTorch**, and bridges trained models to the Python `WINMOL_Analyzer` via **ONNX** (and, for UNet, an optional Keras HDF5 drop-in). Cross-repo design docs live in `docs/` (start with `docs/2026-07-02-segmentor-pytorch-onnx-design.md` and the specs/plans under `docs/superpowers/`).

## Commands

```bash
pip install -e ".[train]"        # torch, torchvision, pillow, tensorboard, albumentations, segmentation-models-pytorch
pip install -e ".[keras]"        # + tensorflow (only for the UNet Keras .hdf5/.keras export)
pip install -e ".[wandb]"        # + wandb, python-dotenv (optional logging)
pip install -e ".[dev]"          # pytest

pytest                           # full suite
pytest tests/test_two_stage.py::test_two_stage_trains_both_stages_and_exports   # single test
pytest -k onnx                   # by keyword
```

The `.venv/` here runs Python 3.9 with torch 2.8 + tensorflow 2.20 installed. Many tests train small models and are slow (minutes); the suite runs several architectures.

### Training / tooling entry points

```bash
# single-stage
python -m training.run_train --data-dir <DS> --out-dir output/run --arch deeplabv3plus \
  --encoder resnet34 --encoder-weights imagenet --epochs 20 --device mps
# two-stage (GenDS -> SpecDS fine-tune)
python -m training.run_train --gen-data-dir <GEN> --spec-data-dir <SPEC> --arch deeplabv3plus ...
python scripts/build_dataset.py --src <raw> --dst <ready>          # convert to loader format
python scripts/benchmark_architectures.py --gen-data-dir <GEN> --spec-data-dir <SPEC> --out-dir results
```

`--device auto` prefers MPS → CUDA → CPU. For large datasets use `--no-cache-dataset --num-workers 4`.

## Architecture — the load-bearing ideas

**Two packages with a hard boundary.** `winmol_unet/` is the **shared** package the analyzer installs; `training/` is **dev-only and must never be imported by the analyzer** (it's excluded from the wheel — `pyproject.toml` `packages = ["winmol_unet"]`). Keep heavy training deps (torch, smp, albumentations, tensorflow) out of `winmol_unet` import paths: its analyzer-facing modules (`runtime`, `contract`) import only `onnxruntime`/`numpy`, so the analyzer can `pip install winmol_unet` and serve ONNX **without** torch/tf. The model factory (`training/model_factory.py`) imports `segmentation-models-pytorch` **lazily**, only for non-UNet archs.

**`contract.py` is the frozen cross-repo interface.** ONNX I/O: NCHW input `[batch,3,512,512]` → output `[batch,1,512,512]`, dynamic batch, opset 17, sigmoid baked in at export. `validate_onnx_model()` enforces it — but spatial dims may be **fixed 512 OR dynamic** (symbolic), so architectures like smp HRNet whose decoder exports symbolic shapes still pass (a *wrong* fixed size is still rejected). Any change here is a breaking change requiring coordinated analyzer updates.

**ONNX is the uniform bridge; Keras HDF5 is UNet-only.** `winmol_unet/export.py::export_to_onnx` is **architecture-agnostic** — it wraps *any* `nn.Module` (adding the sigmoid head via `_WithSigmoid`) and validates the contract; `winmol_unet/runtime.py::OnnxSegmenter` serves it, duck-typing the Keras model's `predict_on_batch(NHWC)→NHWC`. So every arch (unet/deeplabv3plus/hrnet) loads the **same way**. `export_keras.py`/`keras_model.py` are a hand-built UNet mirror + layer-by-layer weight transfer — **UNet-specific**, opt-in via `--export-keras`, and they fail loud on any other topology.

**`model.py` (UNet) is a modernized adaptation of the R `model_UNet.R`, not a layer-exact port.** Documented deviations (see the docstring + design-spec §7): Conv→BN→ReLU (vs R's Conv→ReLU→BN), no BatchNorm after ConvTranspose (18 vs 22 BN), 256 filters where R had a `265` typo, and 512×512 (vs the R script's 256).

**Training loop is architecture-agnostic.** `training/train.py::train_one_run` only needs `model.forward(x) → logits [N,1,512,512]`; `run_train.py::run_training` (single-stage) and `run_two_stage` (GenDS→SpecDS fine-tune: build one model, train stage 1, then fine-tune the SAME model on stage 2) wire it. Loss is `BCEWithLogits + (1 − soft_F1)` on logits; metrics are hard-rounded (sigmoid + 0.5), micro-averaged in `evaluate`.

**Dataset convention + scale.** `training/dataset.py::StemDataset` pairs `train/train{N}.jpeg` ↔ `mask/mask{N}.gif` by integer N, resizes to 512 via `winmol_unet.preprocess` (bicubic image / nearest mask), and binarizes the mask. It has an in-memory resize cache (default on, needs `num_workers=0`); for large sets use `cache=False` + `num_workers>0`. `scripts/build_dataset.py` converts arbitrary folders (non-integer names, palette/instance masks) into this format.

## Conventions

- Tests are TDD-first and are the executable spec (contract parity, export/serve, two-stage handoff). Add/adjust tests before changing behavior. Keep them **hermetic** (synthetic data in `tmp_path`, `encoder_weights=None` for smp archs to avoid downloads).
- ONNX parity/serve tests pin the CPU EP via the `WINMOL_ONNX_FORCE_CPU` env var — CoreML/CUDA compute in fp16 and are not bit-exact; use CPU for exact fp32 comparisons.
- `pyproject.toml` scopes filterwarnings; keep exports/warnings clean rather than re-adding noise.

## Running experiments — use the `winmol-experiment` skill

Any claim that one configuration beats another goes through
`.claude/skills/winmol-experiment/SKILL.md`. It is short, and every rule in it exists
because its absence already produced a wrong result in this repo:

- **Name the yardstick before training.** Arms trained on different data are scored on
  different exams. A modal- and an amodal-trained model each scored against their own
  labels are not comparable; three preprocessing arms with different footprints are not
  comparable on F1. For full-pipeline claims the yardstick is full-orthomosaic inference
  **masked to the AOI** — outside the windthrow polygon stems are real but undigitised, so
  scoring the whole raster punishes the better model hardest.
- **Verify leak-freedom numerically** from `tiles.jsonl`, not by assertion. Oversampled
  tiles overlap; a random tile split leaks.
- **Sanity-check metrics before trusting them.** AP outside [0,1], file-size comparisons
  that break on external data, and skeleton endpoints dominated by outline spurs have all
  produced confidently wrong conclusions here. Pin any metric you rely on with a test.
- **Cross-validate across data sources** — leave-one-site-out by default; report every
  fold; state superiority as a fold count, never a mean.
- **Random figures by default.** A selected figure must be labelled as selected and shown
  alongside a random sample.
- **Reports render from JSON copied verbatim from `test_results.md`** and compute no
  metrics, so they cannot drift from what training reported.

Datasets carry `tiles.jsonl` (source ortho, world centre, rotation, GSD, stem fraction);
`scripts/locate_tile.py --tile N --crop out.png` re-cuts a tile's footprint at native
resolution, which is what settles most label questions.
