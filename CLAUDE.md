# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`winmol_unet` re-implements the WINMOL tree-stem segmentation model (originally R + Keras/TensorFlow in the sibling `WINMOL_segmentor` repo) in **PyTorch**, and bridges trained models to the Python `WINMOL_Analyzer` via **ONNX** (and, for UNet, an optional Keras HDF5 drop-in). Cross-repo design docs live in `docs/` (start with `docs/2026-07-02-segmentor-pytorch-onnx-design.md`, which specifies the frozen ONNX contract). The full experiment record — every results document including the superseded ones, the pre-registered designs, and the one-off experiment scripts — lives in the **private helper repo** alongside this one; `docs/README.md` says what stayed and what went.

## Commands

```bash
pip install -e "."               # serve ONNX only: onnxruntime + numpy. What the analyzer installs.
pip install -e ".[train]"        # torch, torchvision, pillow, tensorboard, albumentations, segmentation-models-pytorch
pip install -e ".[geo]"          # rasterio, fiona, shapely — prepare.py / infer.py / evaluate.py
pip install -e ".[optimize]"     # onnxconverter-common — rebuilding the fp16 release assets
pip install -e ".[keras]"        # + tensorflow (only for the UNet Keras .hdf5/.keras export)
pip install -e ".[wandb]"        # + wandb, python-dotenv (optional logging)
pip install -e ".[dev]"          # pytest

pytest                           # full suite
pytest tests/test_training.py::test_overfit_loss_decreases   # single test
pytest -k onnx                   # by keyword
```

Use the conda env **`WINMOL_segmentor_pt`** (`~/opt/anaconda3/envs/WINMOL_segmentor_pt/bin/python`, Python 3.11) — it runs the repo, the WINMOL Analyzer and Keras-2 `.hdf5` loads from one interpreter. The old `.venv/` was Python 3.9 and has been removed: it could not import the Analyzer at all (`utils/IO.py` uses `str | None`, needing 3.10+). Many tests train small models and are slow (minutes); the suite runs several architectures.

### Entry points

Four scripts at the repo root are the whole public surface. Each is a thin wrapper over
`winmol_unet/cli/<name>.py`, so the installed console scripts (`winmol-prepare`,
`winmol-train`, `winmol-infer`, `winmol-evaluate`) run identical code.

```bash
# ortho -> training tiles. Writes tiles.jsonl, which is what makes a split auditable.
python prepare.py --config configs/sites.example.json --out <DS> --strategy blocks [--jobs 4]
python prepare.py --ortho <O> --stems <S> --aoi <A> --out <DS> [--native-px 1024]

# train. --recipe applies a measured augmentation preset; explicit flags always win.
python train.py --data-dir <DS> --out-dir output/run --arch hrnet --recipe robust \
  --epochs 40 --batch-size 16 --deterministic --device cuda
python train.py --gen-data-dir <GEN> --spec-data-dir <SPEC> ...   # two-stage fine-tune
python train.py --print-recipe robust                             # resolve and exit

# apply a model to an orthomosaic -> georeferenced stem map
python infer.py --model model.onnx --ortho <O> --aoi <A> --out pred.tif

# score: on tiles, on the ground, or an existing Analyzer stem map
python evaluate.py --model model.onnx --data-dir <DS>/test
python evaluate.py --model model.onnx --ortho <O> --aoi <A> --stems <S>
python evaluate.py --stem-map out.tif --aoi <A> --stems <S>

python prepare.py --from-folder --src <raw> --out <ready>          # convert to loader format
```

`--device auto` prefers MPS → CUDA → CPU. For large datasets use `--no-cache-dataset --num-workers 4`.

**Recipes are augmentation-only.** `baseline`, `fixed`, `jitter`, `robust`, and `mosaic`
(UNEVALUATED). Two traps are documented at length in `winmol_unet/training/recipes.py` and
are worth knowing before quoting any reproduction: `multiscale=True` is *not* the jitter
switch (the `fixed` control sets it too — the crop **range** is the variable), and rotation
is off by default while every measured run used `--aug-rotate-p 0.5 --aug-rotate-limit 180`.
The schedule those runs used (`--epochs 40 --batch-size 16 --deterministic`, seed varying
by family) is *not* part of a recipe and must be passed explicitly.

## Architecture — the load-bearing ideas

**One package, one hard boundary inside it.** Everything ships as `winmol_unet`: the
analyzer-facing core, plus the `training`, `geo` and `cli` subpackages. The boundary is no
longer *which package* but **what gets imported eagerly** — the analyzer runs
`pip install winmol_unet` with no torch, no TensorFlow and no GDAL, and serves ONNX
through `runtime`/`contract`, which import only `onnxruntime`/`numpy`.

So: **`winmol_unet/__init__.py` imports nothing**, and neither do `cli/__init__.py` or
`geo/__init__.py`. Each CLI imports its heavy dependencies inside `main()`; the model
factory (`winmol_unet/training/model_factory.py`) imports `segmentation-models-pytorch` lazily, only
for non-UNet archs; `_export` imports `export_keras` lazily because it pulls in TensorFlow.
`tests/test_import_boundary.py` pins this in a **subprocess** — by the time pytest reaches
it, torch is already in `sys.modules`, so an in-process assertion would pass for the wrong
reason. Adding one eager import to any `__init__` turns a 20 MB install into a 2 GB one
that fails on every machine without torch, and nothing else would catch it.

**`contract.py` is the frozen cross-repo interface.** ONNX I/O: NCHW input `[batch,3,512,512]` → output `[batch,1,512,512]`, dynamic batch, opset 17, sigmoid baked in at export. `validate_onnx_model()` enforces it — but spatial dims may be **fixed 512 OR dynamic** (symbolic), so architectures like smp HRNet whose decoder exports symbolic shapes still pass (a *wrong* fixed size is still rejected). Any change here is a breaking change requiring coordinated analyzer updates.

**ONNX is the uniform bridge; Keras HDF5 is UNet-only.** `winmol_unet/export.py::export_to_onnx` is **architecture-agnostic** — it wraps *any* `nn.Module` (adding the sigmoid head via `_WithSigmoid`) and validates the contract; `winmol_unet/runtime.py::OnnxSegmenter` serves it, duck-typing the Keras model's `predict_on_batch(NHWC)→NHWC`. So every arch (unet/deeplabv3plus/hrnet) loads the **same way**. `export_keras.py`/`keras_model.py` are a hand-built UNet mirror + layer-by-layer weight transfer — **UNet-specific**, opt-in via `--export-keras`, and they fail loud on any other topology.

**`model.py` (UNet) is a modernized adaptation of the R `model_UNet.R`, not a layer-exact port.** Documented deviations (see the docstring + design-spec §7): Conv→BN→ReLU (vs R's Conv→ReLU→BN), no BatchNorm after ConvTranspose (18 vs 22 BN), 256 filters where R had a `265` typo, and 512×512 (vs the R script's 256).

**Training loop is architecture-agnostic.** `winmol_unet/training/train.py::train_one_run` only needs `model.forward(x) → logits [N,1,512,512]`; `run_train.py::run_training` (single-stage) and `run_two_stage` (GenDS→SpecDS fine-tune: build one model, train stage 1, then fine-tune the SAME model on stage 2) wire it. Loss is `BCEWithLogits + (1 − soft_F1)` on logits; metrics are hard-rounded (sigmoid + 0.5), micro-averaged in `evaluate`.

**Dataset convention + scale.** `winmol_unet/training/dataset.py::StemDataset` pairs `train/train{N}.jpeg` ↔ `mask/mask{N}.gif` by integer N, resizes to 512 via `winmol_unet.preprocess` (bicubic image / nearest mask), and binarizes the mask. It has an in-memory resize cache (default on, needs `num_workers=0`); for large sets use `cache=False` + `num_workers>0`. `winmol_unet/data/build.py` (`prepare.py --from-folder`) converts arbitrary folders (non-integer names, palette/instance masks) into this format.

## Conventions

- Tests are TDD-first and are the executable spec (contract parity, export/serve, two-stage handoff). Add/adjust tests before changing behavior. Keep them **hermetic** (synthetic data in `tmp_path`, `encoder_weights=None` for smp archs to avoid downloads).
- The public suite is ten subject-named files (`test_cli.py`, `test_config_wiring.py`,
  `test_contract.py`, `test_data_prep.py`, `test_export_serve.py`, `test_geo.py`,
  `test_import_boundary.py`, `test_keras_bridge.py`, `test_model.py`, `test_training.py`)
  plus `tests/conftest.py`, which holds the shared fixtures (`stem_dataset`, `train_config`,
  `force_cpu_onnx`, `geo_site`). The slow full-training variants (two-stage fine-tuning,
  rotation/multiscale/wandb/export sweeps) and the real-data integration test now live in
  the private helper repo. `test_contract.py` and `test_import_boundary.py` stay public
  because they guard the cross-repo boundary — the frozen ONNX interface and the
  no-eager-heavy-imports rule that keeps the analyzer install small.
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
`locate_tile.py --tile N --crop out.png` re-cuts a tile's footprint at native resolution,
which is what settles most label questions. It lives in the private helper repo alongside
the rest of the experiment tooling, as do the architecture and latency benchmarks, the
scale sweeps and the figure generators.

## Running the Analyzer — use the `winmol-analyzer` skill

`.claude/skills/winmol-analyzer/SKILL.md` covers the sibling repo at
`/Users/christian/hnee/WINMOL_Analyzer`: the five-positional-argument CLI contract, handing
a trained ONNX over, and full-orthomosaic evaluation against a rasterised stem map.

Two things from it are worth knowing even without reading it:

- **`tile_size` is a scale knob, not a performance knob.** The Analyzer cuts
  `ceil(tile_size / pixel_size)` pixels and resizes to 512, so the model's effective ground
  resolution is `tile_size / 512` — 2.93 cm/px at the default 15 m, independent of the
  orthomosaic's own resolution. A fixed-scale model loses 5.2 F1 across ±30% zoom, so check
  this before blaming a model for inconsistent results.
- **Evaluate inside the AOI only.** Outside the windthrow polygon stems are real but
  undigitised; scoring the whole raster counts correct detections as false positives.

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
