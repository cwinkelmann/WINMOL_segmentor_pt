"""Post-training int8 quantization of a contract-conformant UNet ONNX for CPU speedup.

Two entry points (see the CPU-speedup design doc):

  quantize_dynamic_int8(src, dst)             -- weights int8, activations quantized at
                                                 run time. No calibration data. Cheapest.
  quantize_static_int8(src, dst, calib_dir)   -- weights + activations int8 (QDQ), calibrated
                                                 on real tiles. Best CPU speedup on AVX-VNNI.

Both keep the ONNX I/O float32 with Quantize/Dequantize at the edges, so the frozen
[batch,3,512,512] -> [batch,1,512,512] contract still validates and OnnxSegmenter loads it
unchanged. Calibration tiles come from a StemDataset dir (train/ + mask/) as [0,1] NCHW --
the exact scale the model was trained/exported at.
"""
import argparse
import os
import tempfile

import numpy as np

from winmol_unet.contract import INPUT_NAME


def _preprocess(src, dst):
    """ORT's shape-inference + graph clean-up; recommended before static quant."""
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(src, dst)


def quantize_dynamic_int8(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    with tempfile.TemporaryDirectory() as td:
        pre = os.path.join(td, "pre.onnx")
        _preprocess(src, pre)
        quantize_dynamic(pre, dst, weight_type=QuantType.QInt8)
    return dst


class _TileCalibrationReader:
    """Feeds [0,1] NCHW tiles from a StemDataset dir to the static quantizer."""

    def __init__(self, calib_dir, input_name, n_samples, img_size):
        from training.dataset import StemDataset
        ds = StemDataset(os.path.join(calib_dir, "train"),
                         os.path.join(calib_dir, "mask"),
                         img_size, transform=None, cache=False)
        n = min(n_samples, len(ds))
        self._input_name = input_name
        self._data = (
            {input_name: ds[i][0].numpy()[None].astype(np.float32)}  # 1CHW
            for i in range(n)
        )

    def get_next(self):
        return next(self._data, None)


def quantize_static_int8(src, dst, calib_dir, n_samples=128, img_size=512,
                         per_channel=True):
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    with tempfile.TemporaryDirectory() as td:
        pre = os.path.join(td, "pre.onnx")
        _preprocess(src, pre)
        reader = _TileCalibrationReader(calib_dir, INPUT_NAME, n_samples, img_size)
        quantize_static(
            pre, dst, reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            # VNNI-friendly combo: uint8 activations, int8 (per-channel) weights.
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", choices=["dynamic", "static"], default="static")
    ap.add_argument("--calib-dir", default=None, help="StemDataset dir (required for static)")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--no-per-channel", action="store_true")
    args = ap.parse_args()

    if args.mode == "dynamic":
        quantize_dynamic_int8(args.src, args.dst)
    else:
        if not args.calib_dir:
            ap.error("--calib-dir is required for static quantization")
        quantize_static_int8(args.src, args.dst, args.calib_dir, args.n_samples,
                             args.img_size, not args.no_per_channel)
    print(f"wrote {args.dst} ({os.path.getsize(args.dst) / 1e6:.1f} MB) "
          f"from {args.src} ({os.path.getsize(args.src) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
