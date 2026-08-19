import os

import numpy as np
from PIL import Image

from winmol_unet.training.augment import build_augmentation
from winmol_unet.training.config import TrainConfig
from winmol_unet.training.run_train import _build_loaders, config_from_args, run_training


def _ds(d, n):
    (d / "train").mkdir(parents=True); (d / "mask").mkdir(parents=True)
    for k in range(1, n + 1):
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(d / "train" / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(d / "mask" / f"mask{k}.gif")


def test_cli_parses_val_data_dir():
    cfg = config_from_args(["--data-dir", "t", "--val-data-dir", "v"])
    assert cfg.val_data_dir == "v"
    assert cfg.val_image_dir == os.path.join("v", "train")
    assert config_from_args(["--data-dir", "t"]).val_data_dir is None


def test_fixed_val_split_is_used_not_resplit(tmp_path):
    train_d, val_d, out = tmp_path / "tr", tmp_path / "va", tmp_path / "out"
    _ds(train_d, 6); _ds(val_d, 4)
    cfg = TrainConfig(
        data_dir=str(train_d), val_data_dir=str(val_d),
        checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "log"),
        hdf5_out="x", onnx_out=str(out / "m.onnx"), epochs=1, batch_size=2,
        patience=999, device="cpu",
    )
    # train uses ALL 6 (no held-out); val uses ALL 4 (the fixed set), no re-split
    tl, vl = _build_loaders(cfg.image_dir, cfg.mask_dir, cfg, build_augmentation(cfg),
                            val_image_dir=cfg.val_image_dir, val_mask_dir=cfg.val_mask_dir)
    assert len(tl.dataset) == 6 and len(vl.dataset) == 4

    metrics = run_training(cfg)
    assert set(metrics) == {"loss", "precision", "recall", "f1"}
    assert os.path.exists(cfg.onnx_out)
