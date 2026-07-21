# UNet CPU-inference speedup — results

**Goal:** make `winmol_unet.model.UNet` faster on **pure CPU** (ONNX Runtime CPU EP) within a
**≤0.03 F1** budget, contract unchanged. Branch `feat/cpu-inference-speedup`. Design:
`docs/superpowers/specs/2026-07-21-cpu-inference-speedup-design.md`.

**Setup.** CPU = 12th Gen i7-1255U (AVX2 + **AVX-VNNI** → hardware int8), ORT CPU EP, 4 intra-op
threads, batch 1 (the CPU-bound single-tile case). Latency = median of 30 timed `session.run`
after 5 warmup. Accuracy = **beech TestDS** (117 tiles), ONNX-served, same F1 definition as every
earlier report. Reference = the deployable two-stage lrfix UNet (fp32, TestDS F1 0.760).

## Headline table (4 threads, batch 1)

| Model | latency (ms) | **speedup** | F1 | ΔF1 | size |
|-------|-------------:|------------:|----:|----:|-----:|
| **fp32** (reference) | 2520.8 | 1.00× | 0.7604 | — | 124 MB |
| fp32 + dynamic int8 | 7659.9 | **0.33×** ⚠ | 0.7602 | −0.0002 | 39 MB |
| fp32 + static int8 | 850.8 | 2.96× | 0.7597 | −0.0007 | 31 MB |
| **width 0.5** (fp32) | 635.5 | 3.97× | 0.7603 | −0.0001 | 31 MB |
| **width 0.5 + static int8** | 248.7 | **10.1×** | 0.7601 | −0.0003 | 8 MB |
| width 0.25 (fp32) | 185.5 | 13.6× | 0.7547 | −0.0057 | 8 MB |
| **width 0.25 + static int8** | 86.0 | **29.3×** | 0.7546 | −0.0058 | 2 MB |

(`width 1.0 + static int8` = 856 ms / 0.7538 — the freshly-retrained full-width control quantized;
its slightly lower F1 vs `fp32+static` is because it comes from a single-stage retrain, F1 0.754,
not the 0.760 reference. Width variants are mutually comparable — same single-stage regime.)

## What each lever buys

1. **Static int8 quantization ≈ 3× faster, lossless** (−0.0007 F1). On this AVX-VNNI CPU the
   int8 QLinearConv path is hardware-accelerated. No retraining. This is the free win.
2. **Dynamic int8 is a trap here — 3× *slower*.** ORT's dynamic path quantizes activations at
   runtime and has no fast dynamic conv kernel, so a conv-heavy UNet regresses badly. Avoid.
3. **Width scaling is essentially free on this task.** Single-stage TestDS F1 is *flat* across
   width: 1.0 → 0.754, 0.5 → **0.760**, 0.25 → 0.755 (all within noise). The stem-segmentation
   signal doesn't need 31M params. width 0.5 (fp32) is already ~4× faster at full accuracy.
4. **The two levers multiply.** width-0.5 + static int8 = **10× faster, lossless, 15× smaller**.
   width-0.25 + static int8 = **29× faster** for −0.006 F1.

## Thread scaling (batch 1)

| threads | fp32 (ms) | width-0.25 int8 (ms) |
|--------:|----------:|---------------------:|
| 1 | 3929 | 115 |
| 2 | 2745 | 86 |
| 4 | 2464 | 85 |
| 8 | 2079 | 83 |

fp32 barely scales (1.9× from 1→8 threads — memory-bandwidth bound); the int8 narrow model is so
cheap it's overhead-bound and essentially flat. **On a single-threaded CPU the gap is largest:**
fp32 3929 ms vs width-0.25-int8 115 ms = **34×**. The realistic "pure CPU load" the study targeted
is exactly this low-thread regime, where the win is biggest.

## Recommendation

- **Default deployable: width-0.5 + static int8** — **10× faster CPU inference, F1 0.760 (lossless
  vs the reference), 8 MB (15× smaller).** Comfortably inside the 0.03 budget with zero accuracy cost.
- **Aggressive option: width-0.25 + static int8** — 29× faster, 2 MB, for −0.006 F1. Use when CPU
  latency dominates and a hair of F1 is acceptable.
- Pruning / depthwise-separable were **not needed**: a from-scratch narrow model already matches
  full-width accuracy, so structured pruning (which at best recovers a smaller model's accuracy)
  has nothing to add here.

## Artifacts

- `results/cpu_speedup/train/{w10,w05,w025}/model.onnx` + `.pt` — retrained width variants (fp32).
- `results/cpu_speedup/models/{w05_static,w025_static,fp32unet_static,fp32unet_dyn}.onnx` — int8.
- `results/cpu_speedup/{sweep.md,sweep.json,thread_sweep.json}` — raw measurements.
- Tooling: `scripts/benchmark_cpu_latency.py`, `scripts/quantize_unet.py`,
  `scripts/run_cpu_speedup_benchmark.sh`; `UNet(width_mult=)` + `run_train --width-mult`.

All quantized models pass `validate_onnx_model` and serve through `OnnxSegmenter` (dynamic batch,
sigmoid output) — the analyzer loads them unchanged.
