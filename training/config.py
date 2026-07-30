"""Hyperparameters + paths for single-stage training (mirrors R controlling.R)."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    data_dir: str          # contains train/ (jpeg) and mask/ (gif)
    checkpoint_dir: str
    log_dir: str
    hdf5_out: str
    onnx_out: str
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-3
    dropout: float = 0.1
    img_size: int = 512
    val_fraction: float = 0.2
    patience: int = 5
    seed: int = 1
    device: str = "auto"   # auto -> mps, then cuda, then cpu
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    aug_hflip_p: float = 0.5
    aug_vflip_p: float = 0.5
    aug_rotate_p: float = 0.0
    aug_rotate_limit: float = 15.0
    aug_bc_p: float = 0.5
    aug_brightness_limit: float = 0.2
    aug_contrast_limit: float = 0.2
    aug_hsv_p: float = 0.5
    # HueSaturationValue magnitudes (albumentations defaults; the benchmark lowers
    # them to mirror the R input_pipeline.R random_hue/random_saturation ranges).
    aug_hue_shift: int = 20
    aug_sat_shift: int = 30
    aug_val_shift: int = 20
    keras_out: Optional[str] = None
    pt_out: Optional[str] = None
    arch: str = "unet"                       # unet | deeplabv3plus | hrnet
    encoder: str = "resnet34"                # smp encoder (deeplabv3plus)
    encoder_weights: Optional[str] = None    # None (no download) or "imagenet"
    export_keras: bool = False               # also emit Keras .hdf5/.keras (UNet only)
    cache_dataset: bool = True               # in-memory resize cache; off for large sets
    num_workers: int = 0                     # DataLoader workers (>0 only with cache off)
    gen_data_dir: Optional[str] = None       # two-stage: stage-1 general dataset
    spec_data_dir: Optional[str] = None      # two-stage: stage-2 species fine-tune dataset
    patience_stage1: int = 3                 # early-stop patience, stage 1 (R value)
    patience_stage2: int = 5                 # early-stop patience, stage 2 (R value)
    val_data_dir: Optional[str] = None       # single-stage: fixed val set (else 80/20 split of data_dir)
    test_data_dir: Optional[str] = None      # held-out test set evaluated after training (R cost_eval)
    rgbd: bool = False                       # 4-channel RGBD input; needs depth/ in each dataset dir

    def _depth_for(self, data_dir):
        return os.path.join(data_dir, "depth") if (self.rgbd and data_dir) else None

    @property
    def in_channels(self) -> int:
        from winmol_unet.contract import IN_CHANNELS, RGBD_IN_CHANNELS
        return RGBD_IN_CHANNELS if self.rgbd else IN_CHANNELS

    @property
    def depth_dir(self) -> Optional[str]:
        return self._depth_for(self.data_dir)

    @property
    def val_depth_dir(self) -> Optional[str]:
        return self._depth_for(self.val_data_dir)

    @property
    def gen_depth_dir(self) -> Optional[str]:
        return self._depth_for(self.gen_data_dir)

    @property
    def spec_depth_dir(self) -> Optional[str]:
        return self._depth_for(self.spec_data_dir)

    @property
    def test_depth_dir(self) -> Optional[str]:
        return self._depth_for(self.test_data_dir)

    @property
    def image_dir(self) -> str:
        return os.path.join(self.data_dir, "train")

    @property
    def mask_dir(self) -> str:
        return os.path.join(self.data_dir, "mask")

    @property
    def val_image_dir(self) -> Optional[str]:
        return os.path.join(self.val_data_dir, "train") if self.val_data_dir else None

    @property
    def val_mask_dir(self) -> Optional[str]:
        return os.path.join(self.val_data_dir, "mask") if self.val_data_dir else None

    @property
    def gen_image_dir(self) -> str:
        return os.path.join(self.gen_data_dir, "train")

    @property
    def gen_mask_dir(self) -> str:
        return os.path.join(self.gen_data_dir, "mask")

    @property
    def spec_image_dir(self) -> str:
        return os.path.join(self.spec_data_dir, "train")

    @property
    def spec_mask_dir(self) -> str:
        return os.path.join(self.spec_data_dir, "mask")
