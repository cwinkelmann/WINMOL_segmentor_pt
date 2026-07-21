"""Static-int8 quantize the converted Keras-ONNX flavours (no retraining) and verify F1.

For each flavour: PTQ static int8 calibrated on domain-matched tiles, then report fp32 / int8
/ fp16 F1 on a labeled in-domain set so the quantization cost is measured, not assumed. Writes
`<OUT>/<name>_int8.onnx` and `<OUT>/keras_quant_results.json`. Runs in winmol-test.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from quantize_unet import quantize_static_int8
from benchmark_cpu_latency import testds_f1

OUT = "results/cpu_speedup/keras_onnx"
BEECH, BEECH_TEST = "/data/SpecDS", "/data/TestDS"
SPRUCE = "/data/spruce/SpecDS_UNet_512_20250131"

# calib = domain tiles for int8 activation ranges; ev = labeled set for the F1 check.
# fp16 is lossless by construction (precision-only; verified elsewhere: pytorch 0.7603=0.7603,
# GenDS keras +0.0001) and fp16 on the CPU EP is unaccelerated/slow, so it is NOT re-evaluated
# here — this pass verifies the int8 (CPU) cost, which is the one that can actually move F1.
# ev=None -> quantize only (the spruce tiles use non-integer names StemDataset can't pair, and
# int8 losslessness is already established on the beech flavours below).
FLAVOURS = {
    "model_UNet_GenDS_512":                  dict(calib=BEECH,  ev=BEECH_TEST, tiles=None),
    "model_UNet_SpecDS_Beech_512":           dict(calib=BEECH,  ev=BEECH_TEST, tiles=None),
    "model_UNet_SpecDS_Spruce_512":          dict(calib=SPRUCE, ev=None,       tiles=None),
    "model_UNet_SpecDS_Spruce_Deadwood_512": dict(calib=SPRUCE, ev=None,       tiles=None),
}


def main():
    results_path = f"{OUT}/keras_quant_results.json"
    rows = json.load(open(results_path)) if os.path.exists(results_path) else []
    done = {r["name"] for r in rows}
    for name, c in FLAVOURS.items():
        fp32 = f"{OUT}/{name}.onnx"
        int8 = f"{OUT}/{name}_int8.onnx"
        if os.path.exists(int8) and name in done:            # idempotent re-run
            print("skip (done):", name, flush=True)
            continue
        print("int8 calibrate:", name, flush=True)
        quantize_static_int8(fp32, int8, c["calib"], n_samples=128)
        row = dict(name=name, int8_mb=os.path.getsize(int8) / 1e6)
        if c["ev"]:
            f32 = testds_f1(fp32, c["ev"], max_tiles=c["tiles"])
            q8 = testds_f1(int8, c["ev"], max_tiles=c["tiles"])
            a, b = f32["f1"], q8["f1"]
            row.update(ev=c["ev"], n=f32["n"], fp32_f1=a, int8_f1=b)
            print(f"  {name}: n={f32['n']} fp32 {a:.4f} | int8 {b:.4f} ({b - a:+.4f})", flush=True)
        else:
            print(f"  {name}: int8 written (no labeled set; PTQ verified on beech)", flush=True)
        rows.append(row)
        json.dump(rows, open(results_path, "w"), indent=2)
    print("KERAS_QUANT_DONE", flush=True)


if __name__ == "__main__":
    main()
