# UNet CPU-inference speedup — design

**Date:** 2026-07-21
**Branch:** `feat/cpu-inference-speedup` (off `feat/bamforests-benchmark`)
**Goal:** Make the shared `winmol_unet.model.UNet` faster to run on **pure CPU** (ONNX Runtime
CPU EP) without breaking the frozen ONNX contract and without giving up more than **0.03 F1**
against the reference. On GPU the model is already fast enough; CPU is the target.

## 1. Success criteria

- **Accuracy budget:** held-out **beech TestDS** (117 tiles) F1 must stay **≥ 0.730**
  (reference fp32 UNet = 0.760; budget = −0.03). Report P/R too.
- **Speed:** primary metric is **CPU-EP latency at batch 1** (single 512² tile, the CPU-bound
  case), median + p90 over ≥20 timed runs after warmup, at a pinned thread count. Batch 4
  reported as a secondary number. Report **speedup ×** vs the fp32 baseline on the *same* CPU.
- **Contract:** every candidate must export/validate through `winmol_unet.export` (or the
  quantizer) as `[batch,3,512,512]→[batch,1,512,512]`, opset 17, sigmoid in graph, and load in
  `OnnxSegmenter`. A quantized graph keeps float32 I/O with Q/DQ at the edges — contract holds.
- **Deliverable:** a table of {candidate → latency, speedup, F1, ΔF1, size} + a recommendation,
  plus the reusable benchmark rig and the winning ONNX artifact(s).

## 2. Baselines (already exist — no retraining)

- **fp32 reference (deployable regime):** `results/r_vs_pytorch_lrfix/unet/model.onnx`
  (two-stage lrfix, TestDS F1 0.760, 124 MB). This is the accuracy/speed anchor.
- **fp32 single-stage:** `results/r_vs_pytorch_single_stage/unet/model.onnx` (SpecDS-only,
  TestDS F1 0.762). Single-stage ≈ two-stage (±0.002) but trains in ~15 min vs ~150 min, so it
  is the **fast iteration regime** for width variants. Width variants are compared to this 0.762
  single-stage full-width baseline (apples-to-apples: same regime, only width differs).

## 3. Measurement rig (build first — nothing is trustworthy without it)

`scripts/benchmark_cpu_latency.py`:
- Input: one or more `.onnx` paths (+ optional `--test-data-dir`, `--threads`, `--runs`,
  `--warmup`, `--batch`).
- **Latency:** `OnnxSegmenter` forced to CPU EP (`WINMOL_ONNX_FORCE_CPU=1`), fixed
  `intra_op_num_threads`; feed a real preprocessed tile (or zeros) NHWC→NCHW; warmup N, time M,
  report min/median/p90/mean in ms for batch 1 and batch 4.
- **Accuracy:** reuse the exact eval path from `benchmark_architectures.py::_onnx_test_metrics`
  (StemDataset → `predict_on_batch` → threshold 0.5 → micro-averaged TP/FP/FN) so F1 is
  identical in definition to every earlier report.
- **Thread sweep helper:** measure batch-1 latency at intra-op threads {1,2,4,8} to pick the
  operating point and expose the free runtime win. Report a single pinned thread count for all
  cross-candidate comparisons (default: the machine's best, recorded in the output).
- Emits a machine-readable JSON row per model + a markdown table.
- Input scale note: the model expects **float32 in [0,1]** NCHW (training uses
  `preprocess.to_float01`; `runtime.predict_on_batch` does NOT rescale). Calibration and latency
  inputs must be [0,1] tiles, not 0–255.

Runs in Docker `winmol-test` (has torch+ort+pytest) with the repo mounted, CPU only.

## 4. Optimization families (each measured for latency AND F1)

Ordered by effort/risk. Per the scope decision, pruning + depthwise are **conditional** — run
only if quant+width miss a compelling point on the trade curve.

### 4a. Runtime-only (free, no accuracy change)
ORT graph-optimization level (`ORT_ENABLE_ALL`) + the intra-op thread sweep from §3. Establishes
the true baseline latency and best thread count. No new artifact, just the right session config.

### 4b. Quantization — no retrain, operates on the existing fp32 ONNX
`scripts/quantize_unet.py`:
- **Dynamic int8** (`quantize_dynamic`, weights int8). Cheap, no calibration; modest for conv.
- **Static int8** (`quantize_static`, QDQ, per-channel weights, int8 activations) with a
  `CalibrationDataReader` over ~128 real training tiles ([0,1] NCHW). This is the expected
  headline win on this **AVX-VNNI** CPU (hardware int8). Compare QDQ vs QOperator if it matters.
- fp16 is **not** pursued (no CPU benefit); noted for completeness.
- Report latency + ΔF1 for each. Static int8 must clear the −0.03 gate to be recommended.

### 4c. Smaller architecture — retrain
Add `width_mult` to `winmol_unet.model.UNet` (default 1.0 → unchanged; base widths scale
`round(w*base/8)*8` for clean channels). Train **width 0.5 and 0.25** single-stage (SpecDS-only,
512², from scratch, R-matched aug, seed 1 — the 0.762 regime). Width 0.5 ≈ 4× fewer FLOPs.
Measure the F1/latency trade curve; whichever widths clear the gate advance.

### 4d. Structured channel pruning + fine-tune — CONDITIONAL
Only if 4b+4c leave an obvious gap. L1-norm channel pruning of the full-width model + short
fine-tune, compared head-to-head with the equal-size from-scratch width model (which usually
wins). One honest data point, timeboxed — not an exhaustive sweep.

### 4e. Stack the winners
Best width variant **+ static int8** (retrain → export → quantize → measure). Width (≈4×) and
int8 (≈2–3×) act on different axes and should compound. Final recommendation comes from this
combined trade curve.

## 5. Promotion to a deployable artifact
The single recommended configuration (best F1-above-gate per unit latency) is retrained in the
**full two-stage lrfix regime** (matching the 0.760 reference/deployed model), re-exported,
re-quantized if applicable, and its ONNX validated + round-tripped through `OnnxSegmenter`.
Saved alongside a short provenance note. `width_mult` ships as a normal `UNet` kwarg so the
analyzer-facing package stays unchanged for the default (1.0) path.

## 6. Testing (TDD-first, hermetic)
- `test_width_mult`: `UNet(width_mult=0.5)` builds, forward gives `[N,1,512,512]`, param count
  ~¼ of full, `width_mult=1.0` is byte-identical to today's `UNet`.
- `test_export_width`: a width-scaled UNet exports through `export_to_onnx` and passes
  `validate_onnx_model` (contract holds at reduced width).
- `test_quantize_roundtrip`: quantized ONNX loads in `OnnxSegmenter`, output shape/scale sane,
  contract validator passes.
- Latency rig: a tiny self-check (runs on a 1-tile synthetic set, asserts it returns finite ms
  and an F1 in [0,1]).
All hermetic (`tmp_path`, `encoder_weights=None` where relevant), run in `winmol-test`.

## 7. Risks / notes
- **CPU variability:** pin thread count, warm up, use median/p90, run all candidates back-to-back
  on the same idle machine. Report the CPU model in the output.
- **Static-quant accuracy cliff:** conv-heavy int8 can lose F1; per-channel weights + enough
  calibration tiles mitigate. If it fails the gate, dynamic int8 / width scaling carry the result.
- **`width_mult` must not change the default model** — 1.0 path stays byte-identical (test-gated).
- **ConvTranspose int8** support in ORT is weaker than QLinearConv; if the decoder up-convs don't
  quantize cleanly, fall back to quantizing only the convs (mixed) and report it.
- All heavy runs are GPU-gated behind whatever training is live; latency runs are CPU-only and
  independent.

## 8. Out of scope
Changing the ONNX contract (512², opset, I/O names); GPU-side optimization; the Keras HDF5 path
(UNet-specific, not CPU-latency relevant); non-UNet archs (deeplab/hrnet) — the ask is the UNet.
