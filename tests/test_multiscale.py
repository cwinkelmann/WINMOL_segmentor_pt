"""Multi-scale native-fidelity dataloader: the full run_training + eval-tiling integration.

The native-load-mode, rotate->crop-order, and tiling-index tests that used to live here
moved to tests/test_training.py (task 7 of the 2026-08-20 test-suite consolidation); this
one remains because it drives a full `run_training()` call, and is owned by a later task
in that consolidation.
"""
import glob

import numpy as np
from PIL import Image

from winmol_unet.training.config import TrainConfig
from winmol_unet.training.run_train import run_training


def _native_ds(d, n=6, size=64):
    """Write n native-resolution RGB jpeg / binary gif pairs of `size`x`size`."""
    (d / "train").mkdir(parents=True)
    (d / "mask").mkdir(parents=True)
    for k in range(1, n + 1):
        rgb = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        Image.fromarray(rgb, "RGB").save(d / "train" / f"train{k}.jpeg", quality=95)
        m = np.zeros((size, size), np.uint8)
        m[: size // 2, : size // 2] = 255                 # a quadrant of foreground
        Image.fromarray(m, "L").save(d / "mask" / f"mask{k}.gif")


def test_run_training_multiscale_and_eval_tiling(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _native_ds(data, n=6, size=64)
    cfg = TrainConfig(
        data_dir=str(data), checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "log"),
        hdf5_out="x", onnx_out=str(out / "m.onnx"), pt_out=str(out / "m.pt"),
        epochs=1, batch_size=2, patience=999, device="cpu", img_size=32,
        multiscale=True, crop_min_px=32, crop_max_px=64, eval_tiling=True,
        aug_rotate_p=1.0, aug_rotate_limit=180.0, cache_dataset=False,
    )
    metrics = run_training(cfg)
    assert set(metrics) >= {"loss", "precision", "recall", "f1"}
    assert (out / "m.onnx").exists() and (out / "m.pt").exists()
    assert not glob.glob(str(out / "*.hdf5"))            # non-keras path
