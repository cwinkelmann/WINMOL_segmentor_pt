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
    keras_out: Optional[str] = None
    pt_out: Optional[str] = None
    arch: str = "unet"                       # unet | deeplabv3plus | hrnet
    encoder: str = "resnet34"                # smp encoder (deeplabv3plus)
    encoder_weights: Optional[str] = None    # None (no download) or "imagenet"
    cache_dataset: bool = True               # in-memory resize cache; off for large sets
    num_workers: int = 0                     # DataLoader workers (>0 only with cache off)

    @property
    def image_dir(self) -> str:
        return os.path.join(self.data_dir, "train")

    @property
    def mask_dir(self) -> str:
        return os.path.join(self.data_dir, "mask")
