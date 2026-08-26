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
    # Multiclass species segmentation: 1 = binary stem/background (default, unchanged);
    # C > 1 = C softmax channels (0 = background, 1..C-1 = species). class_names names
    # channels 1..C-1 (len == num_classes - 1) and drives per-species contract-slice
    # exports next to onnx_out.
    num_classes: int = 1
    class_names: Optional[list] = None
    val_fraction: float = 0.2
    patience: int = 5
    loss: str = "bce_soft_f1"   # or 'bce' — what R effectively optimises
    num_classes: int = 1        # >1: species segmentation (index masks, ce_soft_f1, softmax export)
    block_order: str = "bn_relu"  # or 'relu_bn' — R's Conv->ReLU->BN order
    label_smoothing: float = 0.0  # pull targets toward 0.5 near mask edges
    smooth_band_px: int = 2       # 0 = global smoothing instead of edge-only
    seed: int = 1
    deterministic: bool = False   # cuDNN deterministic kernels; needed for replicates
    device: str = "auto"   # auto -> mps, then cuda, then cpu
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_notes: Optional[str] = None   # set at wandb.init so a run is born documented
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
    init_weights: str | None = None
    aug_hue_shift: int = 20
    aug_sat_shift: int = 30
    aug_val_shift: int = 20
    keras_out: Optional[str] = None
    pt_out: Optional[str] = None
    arch: str = "unet"                       # unet | deeplabv3plus | hrnet
    width_mult: float = 1.0                  # UNet channel-width scale (1.0 = full; <1 = smaller/faster)
    encoder: Optional[str] = None            # smp encoder; None = per-arch default
    encoder_weights: Optional[str] = None    # None (no download) or "imagenet"
    export_keras: bool = False               # also emit Keras .hdf5/.keras (UNet only)
    plots: bool = False                      # write static curve/prediction PNGs after training
    cache_dataset: bool = True               # in-memory resize cache; off for large sets
    num_workers: int = 0                     # DataLoader workers (>0 only with cache off)
    gen_data_dir: Optional[str] = None       # two-stage: stage-1 general dataset
    spec_data_dir: Optional[str] = None      # two-stage: stage-2 species fine-tune dataset
    patience_stage1: int = 3                 # early-stop patience, stage 1 (R value)
    patience_stage2: int = 5                 # early-stop patience, stage 2 (R value)
    val_data_dir: Optional[str] = None       # single-stage: fixed val set (else 80/20 split of data_dir)
    test_data_dir: Optional[str] = None      # held-out test set evaluated after training (R cost_eval)
    multiscale: bool = False                 # multi-scale RandomSizedCrop on native-res tiles
    crop_min_px: int = 400                   # multiscale: min crop side sampled from the tile
    crop_max_px: int = 1024                  # multiscale: max crop side (>= native tile -> full)
    eval_tiling: bool = False                # deterministic grid tiling for val/test (native res)
    # Mosaic: stitch a grid of tiles into one training image. UNEVALUATED here — no
    # paired run, no LOSO fold, no results document — so it is off by default and must
    # stay off until winmol-experiment measures it.
    mosaic_p: float = 0.0
    mosaic_grid_yx: tuple = (2, 2)

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
