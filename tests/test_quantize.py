import os
import sys

import numpy as np
import onnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from quantize_unet import quantize_dynamic_int8, quantize_static_int8

from winmol_unet.contract import validate_onnx_model
from winmol_unet.export import export_to_onnx
from winmol_unet.model import UNet
from winmol_unet.runtime import OnnxSegmenter


def _tiny_unet_onnx(path):
    # small width -> fast to export/quantize, still contract-shaped (512x512x3->512x512x1)
    export_to_onnx(UNet(width_mult=0.25).eval(), path)
    return path


def test_dynamic_int8_roundtrips_and_keeps_contract(tmp_path):
    src = _tiny_unet_onnx(str(tmp_path / "m.onnx"))
    dst = str(tmp_path / "m.int8.onnx")
    quantize_dynamic_int8(src, dst)

    validate_onnx_model(onnx.load(dst))              # contract still holds
    assert os.path.getsize(dst) < os.path.getsize(src)  # int8 weights are smaller

    os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"
    seg = OnnxSegmenter(dst)
    y = seg.predict_on_batch(np.zeros((1, 512, 512, 3), np.float32))
    assert y.shape == (1, 512, 512, 1)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0   # sigmoid output


def test_static_int8_calibrates_from_a_tile_dir(tmp_path):
    from PIL import Image
    # a StemDataset-shaped calibration dir
    d = tmp_path / "calib"
    (d / "train").mkdir(parents=True); (d / "mask").mkdir(parents=True)
    for k in range(1, 5):
        Image.fromarray((np.random.rand(40, 40, 3) * 255).astype(np.uint8), "RGB").save(
            d / "train" / f"train{k}.jpeg")
        Image.fromarray(np.zeros((40, 40), np.uint8), "L").save(d / "mask" / f"mask{k}.gif")

    src = _tiny_unet_onnx(str(tmp_path / "m.onnx"))
    dst = str(tmp_path / "m.static.onnx")
    quantize_static_int8(src, dst, str(d), n_samples=4)

    validate_onnx_model(onnx.load(dst))
    os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"
    y = OnnxSegmenter(dst).predict_on_batch(np.zeros((1, 512, 512, 3), np.float32))
    assert y.shape == (1, 512, 512, 1)
