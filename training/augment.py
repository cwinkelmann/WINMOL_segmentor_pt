"""Build an albumentations pipeline from TrainConfig knobs (online augmentation).

Geometric transforms carry image+mask; photometric carry the image only. Applied
after the winmol_unet.preprocess resize, on 512x512 float32 [0,1] arrays.
border_mode=0 is cv2.BORDER_CONSTANT; fill/fill_mask=0 pad rotated borders with 0
(no phantom stems). A transform is included only when its probability > 0. The
Compose is seeded with cfg.seed so runs are reproducible (albumentations 2.x uses
its own RNG per Compose, NOT the global numpy RNG).
"""
import albumentations as A


def build_augmentation(cfg):
    transforms = []
    if cfg.aug_hflip_p > 0:
        transforms.append(A.HorizontalFlip(p=cfg.aug_hflip_p))
    if cfg.aug_vflip_p > 0:
        transforms.append(A.VerticalFlip(p=cfg.aug_vflip_p))
    if cfg.aug_rotate_p > 0:
        transforms.append(A.Rotate(limit=cfg.aug_rotate_limit, p=cfg.aug_rotate_p,
                                   border_mode=0, fill=0, fill_mask=0))
    if cfg.aug_bc_p > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=cfg.aug_brightness_limit,
            contrast_limit=cfg.aug_contrast_limit, p=cfg.aug_bc_p))
    if cfg.aug_hsv_p > 0:
        transforms.append(A.HueSaturationValue(p=cfg.aug_hsv_p))
    return A.Compose(transforms, seed=cfg.seed)   # seed the Compose (global numpy seed is ignored)
