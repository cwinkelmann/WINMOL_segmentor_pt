import os
from winmol_unet.training.config import TrainConfig


def test_config_defaults_and_derived_dirs():
    c = TrainConfig(
        data_dir="/data/TestDS", checkpoint_dir="ck", log_dir="log",
        hdf5_out="out/model.hdf5", onnx_out="out/model.onnx",
    )
    assert c.batch_size == 4
    assert c.epochs == 100
    assert c.lr == 1e-3
    assert c.img_size == 512
    assert c.val_fraction == 0.2
    assert c.patience == 5
    assert c.seed == 1
    assert c.image_dir == os.path.join("/data/TestDS", "train")
    assert c.mask_dir == os.path.join("/data/TestDS", "mask")
