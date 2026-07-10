import os

import numpy as np
from PIL import Image

from training.config import TrainConfig
from training.run_train import config_from_args, run_two_stage


def _ds(d, n=6):
    (d / "train").mkdir(parents=True, exist_ok=True)
    (d / "mask").mkdir(parents=True, exist_ok=True)
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(d / "train" / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(d / "mask" / f"mask{k}.gif")


def test_cli_two_stage_flags_and_dirs():
    cfg = config_from_args([
        "--gen-data-dir", "g", "--spec-data-dir", "s",
        "--patience-stage1", "2", "--patience-stage2", "4",
        "--no-cache-dataset", "--num-workers", "2", "--arch", "deeplabv3plus"])
    assert cfg.gen_data_dir == "g" and cfg.spec_data_dir == "s"
    assert cfg.patience_stage1 == 2 and cfg.patience_stage2 == 4
    assert cfg.cache_dataset is False and cfg.num_workers == 2
    assert cfg.gen_image_dir == os.path.join("g", "train")
    assert cfg.spec_mask_dir == os.path.join("s", "mask")


def test_missing_dirs_errors():
    import pytest
    with pytest.raises(SystemExit):            # neither --data-dir nor gen+spec
        config_from_args(["--epochs", "1"])


def test_two_stage_trains_both_stages_and_exports(tmp_path):
    gen, spec = tmp_path / "gen", tmp_path / "spec"
    _ds(gen); _ds(spec)
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir="", gen_data_dir=str(gen), spec_data_dir=str(spec),
        checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "log"),
        pt_out=str(out / "m.pt"), hdf5_out=str(out / "m.hdf5"),
        keras_out=str(out / "m.keras"), onnx_out=str(out / "m.onnx"),
        epochs=1, batch_size=2, device="cpu", arch="deeplabv3plus",
        encoder_weights=None, cache_dataset=False,
    )
    metrics = run_two_stage(cfg)
    assert set(metrics) == {"loss", "precision", "recall", "f1"}
    # per-stage checkpoints — stage 2 did not clobber stage 1
    assert os.path.exists(tmp_path / "ck" / "best_stage1.pt")
    assert os.path.exists(tmp_path / "ck" / "best_stage2.pt")
    # non-UNet -> ONNX + pt only
    assert os.path.exists(cfg.onnx_out) and os.path.exists(cfg.pt_out)
    assert not os.path.exists(cfg.hdf5_out) and not os.path.exists(cfg.keras_out)
    # separate per-stage TensorBoard dirs
    assert os.path.isdir(tmp_path / "log" / "stage1")
    assert os.path.isdir(tmp_path / "log" / "stage2")
