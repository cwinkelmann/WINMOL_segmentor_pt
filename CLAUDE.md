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
python scripts/simulate_depth.py --dataset <DS>    # synth depth{N}.png from masks (RGBD)
python scripts/benchmark_architectures.py --gen-data-dir <GEN> --spec-data-dir <SPEC> --out-dir results
```

`--device auto` prefers MPS → CUDA → CPU. For large datasets use `--no-cache-dataset --num-workers 4`.

## Architecture — the load-bearing ideas

**Two packages with a hard boundary.** `winmol_unet/` is the **shared** package the analyzer installs; `training/` is **dev-only and must never be imported by the analyzer** (it's excluded from the wheel — `pyproject.toml` `packages = ["winmol_unet"]`). Keep heavy training deps (torch, smp, albumentations, tensorflow) out of `winmol_unet` import paths: its analyzer-facing modules (`runtime`, `contract`) import only `onnxruntime`/`numpy`, so the analyzer can `pip install winmol_unet` and serve ONNX **without** torch/tf. The model factory (`training/model_factory.py`) imports `segmentation-models-pytorch` **lazily**, only for non-UNet archs.

**`contract.py` is the frozen cross-repo interface.** ONNX I/O: NCHW input `[batch,3,512,512]` → output `[batch,1,512,512]`, dynamic batch, opset 17, sigmoid baked in at export. `validate_onnx_model()` enforces it — but spatial dims may be **fixed 512 OR dynamic** (symbolic), so architectures like smp HRNet whose decoder exports symbolic shapes still pass (a *wrong* fixed size is still rejected). Any change here is a breaking change requiring coordinated analyzer updates.

**ONNX is the uniform bridge; Keras HDF5 is UNet-only.** `winmol_unet/export.py::export_to_onnx` is **architecture-agnostic** — it wraps *any* `nn.Module` (adding the sigmoid head via `_WithSigmoid`) and validates the contract; `winmol_unet/runtime.py::OnnxSegmenter` serves it, duck-typing the Keras model's `predict_on_batch(NHWC)→NHWC`. So every arch (unet/deeplabv3plus/hrnet) loads the **same way**. `export_keras.py`/`keras_model.py` are a hand-built UNet mirror + layer-by-layer weight transfer — **UNet-specific**, opt-in via `--export-keras`, and they fail loud on any other topology.

**`model.py` (UNet) is a modernized adaptation of the R `model_UNet.R`, not a layer-exact port.** Documented deviations (see the docstring + design-spec §7): Conv→BN→ReLU (vs R's Conv→ReLU→BN), no BatchNorm after ConvTranspose (18 vs 22 BN), 256 filters where R had a `265` typo, and 512×512 (vs the R script's 256).

**Training loop is architecture-agnostic.** `training/train.py::train_one_run` only needs `model.forward(x) → logits [N,1,512,512]`; `run_train.py::run_training` (single-stage) and `run_two_stage` (GenDS→SpecDS fine-tune: build one model, train stage 1, then fine-tune the SAME model on stage 2) wire it. Loss is `BCEWithLogits + (1 − soft_F1)` on logits; metrics are hard-rounded (sigmoid + 0.5), micro-averaged in `evaluate`.

**Dataset convention + scale.** `training/dataset.py::StemDataset` pairs `train/train{N}.jpeg` ↔ `mask/mask{N}.gif` by integer N, resizes to 512 via `winmol_unet.preprocess` (bicubic image / nearest mask), and binarizes the mask. It has an in-memory resize cache (default on, needs `num_workers=0`); for large sets use `cache=False` + `num_workers>0`. `scripts/build_dataset.py` converts arbitrary folders (non-integer names, palette/instance masks) into this format. `--rgbd` adds an optional `depth/depth{N}.png|.tif` channel (per-image min-max normalized, geometric-aug only), producing 4-channel models; Keras export is RGB-only.

## Conventions

- Tests are TDD-first and are the executable spec (contract parity, export/serve, two-stage handoff). Add/adjust tests before changing behavior. Keep them **hermetic** (synthetic data in `tmp_path`, `encoder_weights=None` for smp archs to avoid downloads).
- ONNX parity/serve tests pin the CPU EP via the `WINMOL_ONNX_FORCE_CPU` env var — CoreML/CUDA compute in fp16 and are not bit-exact; use CPU for exact fp32 comparisons.
- `pyproject.toml` scopes filterwarnings; keep exports/warnings clean rather than re-adding noise.

# context-mode — MANDATORY routing rules

You have context-mode MCP tools available. These rules are NOT optional — they protect your context window from flooding. A single unrouted command can dump 56 KB into context and waste the entire session.

## BLOCKED commands — do NOT attempt these

### curl / wget — BLOCKED
Any Bash command containing `curl` or `wget` is intercepted and replaced with an error message. Do NOT retry.
Instead use:
- `ctx_fetch_and_index(url, source)` to fetch and index web pages
- `ctx_execute(language: "javascript", code: "const r = await fetch(...)")` to run HTTP calls in sandbox

### Inline HTTP — BLOCKED
Any Bash command containing `fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, or `http.request(` is intercepted and replaced with an error message. Do NOT retry with Bash.
Instead use:
- `ctx_execute(language, code)` to run HTTP calls in sandbox — only stdout enters context

### WebFetch — BLOCKED
WebFetch calls are denied entirely. The URL is extracted and you are told to use `ctx_fetch_and_index` instead.
Instead use:
- `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` to query the indexed content

## REDIRECTED tools — use sandbox equivalents

### Bash (>20 lines output)
Bash is ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`, and other short-output commands.
For everything else, use:
- `ctx_batch_execute(commands, queries)` — run multiple commands + search in ONE call
- `ctx_execute(language: "shell", code: "...")` — run in sandbox, only stdout enters context

### Read (for analysis)
If you are reading a file to **Edit** it → Read is correct (Edit needs content in context).
If you are reading to **analyze, explore, or summarize** → use `ctx_execute_file(path, language, code)` instead. Only your printed summary enters context. The raw file content stays in the sandbox.

### Grep (large results)
Grep results can flood context. Use `ctx_execute(language: "shell", code: "grep ...")` to run searches in sandbox. Only your printed summary enters context.

## Tool selection hierarchy

1. **GATHER**: `ctx_batch_execute(commands, queries)` — Primary tool. Runs all commands, auto-indexes output, returns search results. ONE call replaces 30+ individual calls.
2. **FOLLOW-UP**: `ctx_search(queries: ["q1", "q2", ...])` — Query indexed content. Pass ALL questions as array in ONE call.
3. **PROCESSING**: `ctx_execute(language, code)` | `ctx_execute_file(path, language, code)` — Sandbox execution. Only stdout enters context.
4. **WEB**: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` — Fetch, chunk, index, query. Raw HTML never enters context.
5. **INDEX**: `ctx_index(content, source)` — Store content in FTS5 knowledge base for later search.

## Subagent routing

When spawning subagents (Agent/Task tool), the routing block is automatically injected into their prompt. Bash-type subagents are upgraded to general-purpose so they have access to MCP tools. You do NOT need to manually instruct subagents about context-mode.

## Output constraints

- Keep responses under 500 words.
- Write artifacts (code, configs, PRDs) to FILES — never return them as inline text. Return only: file path + 1-line description.
- When indexing content, use descriptive source labels so others can `ctx_search(source: "label")` later.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call the `ctx_stats` MCP tool and display the full output verbatim |
| `ctx doctor` | Call the `ctx_doctor` MCP tool, run the returned shell command, display as checklist |
| `ctx upgrade` | Call the `ctx_upgrade` MCP tool, run the returned shell command, display as checklist |
