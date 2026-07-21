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

## Does it port to GPU? (RTX 4080 SUPER, PyTorch eager CUDA)

ONNX Runtime here is CPU-only, so GPU latency is measured with PyTorch eager on CUDA (the actual
on-GPU compute; an onnxruntime-gpu/TensorRT deployment is typically no slower). The GPU analogue of
the CPU int8 lever is **fp16** (Tensor Cores). Raw sweep: `results/cpu_speedup/gpu_sweep.md`.

Batch-1 latency, speedup vs fp32 full-width:

| config | latency | speedup | note |
|--------|--------:|--------:|------|
| fp32 width-1.0 | 13.8 ms | 1.0× | baseline (already ~180× faster than CPU fp32) |
| fp16 width-1.0 | 7.2 ms | 1.9× | Tensor Cores, **F1 lossless** (0.7603=0.7603 measured) |
| fp32 width-0.5 | 4.2 ms | 3.3× | width scaling helps GPU more than expected |
| **fp16 width-0.5** | **2.6 ms** | **5.4×** | the CPU-recommended model, on GPU |
| fp16 width-0.25 | 1.4 ms | 9.8× (14.6× at batch 4) | aggressive |

**Yes, it ports — better than expected.** Both levers carry to GPU:
- **Width scaling helps substantially on GPU** (2.5–3.3×), not the modest gain first guessed — the
  4080 is compute-bound enough at 512² for this UNet. The gain shrinks slightly at larger batch
  (b1 3.3× → b16 2.5×) as the full model uses the GPU more efficiently.
- **fp16 stacks ~1.9× and is bit-lossless here** (measured, not assumed).
- Combined **width-0.5 + fp16 = 5.4×** (lossless); width-0.25 + fp16 peaks at **14.6×** (batch 4).

**Caveat — GPU is already fast:** fp32 full-width is 13.8 ms/tile. The optimization takes that to
2.6 ms, which matters for **throughput/batched** workloads (73 → 390 img/s) and power, not because
13.8 ms is slow. On CPU the same model went 2521 → 249 ms, where it actually removes a bottleneck.
**MPS/CoreML not measured** (no Apple hardware); CoreML already computes in fp16, so expect the
fp16 lever to be largely automatic there and width scaling to add on top.

### Production path: ONNX Runtime CUDA EP (not torch eager)

The numbers above are torch eager. The analyzer serves **ONNX**, so I built `winmol-onnxgpu`
(`Dockerfile.onnxgpu`, onnxruntime-gpu CUDA-12 build) and re-measured the actual served path.
Batch-1 median, speedup vs w1.0-fp32, and img/s at batch 8 (throughput regime):

| model | b1 latency | speedup | b8 img/s | ONNX size |
|-------|-----------:|--------:|---------:|----------:|
| w1.0 fp32 | 14.9 ms | 1.0× | 62 | 124 MB |
| w1.0 fp16 | 8.6 ms | 1.7× | 105 | 62 MB |
| w0.5 fp32 | 5.7 ms | 2.6× | 143 | 31 MB |
| **w0.5 fp16** | 4.3 ms | 3.4× | 216 | 16 MB |
| w0.25 fp32 | 3.5 ms | 4.3× | 260 | 8 MB |
| **w0.25 fp16** | 3.1 ms | 4.9× (5.9× at b4) | 346 | 4 MB |

**ONNX-CUDA gains are real but *more modest* than torch eager** (fp16 1.7× vs 1.9×; w0.25+fp16
5–6× vs 10–14×). ORT's CUDA EP carries higher per-call overhead (input/output binding + H2D/D2H
copies, less kernel fusion than eager cudnn) which dominates for these tiny/narrow models — so the
speedup plateaus around 5–6×. This is the honest deployment figure. **fp16 is bit-lossless through
this served path** (measured: w0.5 ONNX-CUDA fp32 F1 0.7603 = fp16 0.7603). The `TensorrtExecutionProvider`
is available in the image and would likely recover eager-level (or better) via per-shape engine
builds + INT8 — a worthwhile follow-up if GPU throughput ever becomes the constraint.

Bottom line for GPU: the levers port, fp16 is free and lossless, but the payoff (~5×, on already-
single-digit-ms latency) is a throughput/power win, not the bottleneck-removal it is on CPU.

### Going further on GPU: TensorRT EP (w05 UNet, RTX 4080S)

The ORT CUDA EP leaves performance on the table (per-call overhead, less fusion). The
**TensorRT EP** compiles a fused, autotuned engine and recovers it. Built `winmol-onnxgpu` with
TensorRT 10; measured on the w05 UNet:

| provider | b1 | b8 throughput | vs CUDA EP (b1) |
|----------|---:|--------------:|----------------:|
| CUDA fp32 | 5.98 ms | 142 img/s | 1.0× |
| **TRT fp16** | **3.09 ms** | **377 img/s** | **1.9× (2.7× throughput)** |
| TRT int8 (QDQ) | 6.58 ms | 123 img/s | 0.9× — slower |

- **TensorRT fp16 is the GPU winner** — ~1.9× over the CUDA EP at batch-1 and **2.7× throughput**
  at batch-8; it also beats the plain CUDA-EP fp16 (4.3 ms → 3.09 ms).
- **TensorRT int8 is not worth it here** — *slower* than fp16, and TRT's ONNX parser rejects the
  ORT-QDQ quantize-on-bias nodes (falls back / mis-optimises). GPU int8 wants a native TRT int8
  calibration, and for a model this small it's overhead-bound anyway. **int8 stays a CPU lever.**

**How to use it (no new artifact):** the ONNX we ship *is* what TensorRT consumes. Serve through
the TRT EP — e.g. `WINMOL_ONNX_PROVIDERS="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"`
plus `trt_fp16_enable` — and TensorRT builds+caches the engine on the target GPU. The engine is
device/TRT-version specific (built at run time, never distributed). Requires `onnxruntime-gpu`
**and** TensorRT libs (`libnvinfer`) — the analyzer env needs both. To flip on fp16/int8 the
session must pass provider *options*, so `OnnxSegmenter` would need a small change to accept them.

## Artifacts

- `results/cpu_speedup/train/{w10,w05,w025}/model.onnx` + `.pt` — retrained width variants (fp32).
- `results/cpu_speedup/models/{w05_static,w025_static,fp32unet_static,fp32unet_dyn}.onnx` — int8.
- `results/cpu_speedup/{sweep.md,sweep.json,thread_sweep.json}` — raw measurements.
- Tooling: `scripts/benchmark_cpu_latency.py`, `scripts/quantize_unet.py`,
  `scripts/run_cpu_speedup_benchmark.sh`; `UNet(width_mult=)` + `run_train --width-mult`.

All quantized models pass `validate_onnx_model` and serve through `OnnxSegmenter` (dynamic batch,
sigmoid output) — the analyzer loads them unchanged.
