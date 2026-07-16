"""Build an albumentations pipeline from TrainConfig knobs (online augmentation).

Geometric transforms carry image+mask; photometric carry the image only.
border_mode=0 is cv2.BORDER_CONSTANT; fill/fill_mask=0 pad rotated/exposed borders with 0
(no phantom stems). A transform is included only when its probability > 0. The Compose is
seeded with cfg.seed so runs are reproducible (albumentations 2.x uses its own RNG per
Compose, NOT the global numpy RNG).

Ordering depends on cfg.multiscale:
  * multiscale (native-res input): Rotate (full tile) -> RandomSizedCrop (-> img_size) ->
    flips -> photometric. Rotating the FULL tile before the crop means the crop lands on
    valid interior pixels, so black padding only appears near the tile edge (vs. rotating an
    already-cropped 512 tile, which always blackens the corners).
  * non-multiscale (img_size input): flips -> Rotate -> photometric (unchanged prior order).
"""
import albumentations as A


def _rotate(cfg):
    """Arbitrary-angle rotation, image+mask, exposed corners padded with black."""
    return A.Rotate(limit=cfg.aug_rotate_limit, p=cfg.aug_rotate_p,
                    border_mode=0, fill=0, fill_mask=0)


def build_augmentation(cfg):
    multiscale = getattr(cfg, "multiscale", False)
    transforms = []

    if multiscale:
        # Rotate the full native tile FIRST (any angle in +/- aug_rotate_limit; set it to 180
        # for full 360 deg coverage), then the multi-scale crop samples a window and resizes
        # to img_size. crop_max ~= tile -> coarse downscale, crop_min -> fine zoom-in, native
        # tile size -> 1:1 — this is what makes the model scale-robust across the analyzer's
        # user-set tile_size (GSD). Needs a native-res dataset (StemDataset(resize=False)).
        if cfg.aug_rotate_p > 0:
            transforms.append(_rotate(cfg))
        transforms.append(A.RandomSizedCrop(
            min_max_height=(cfg.crop_min_px, cfg.crop_max_px),
            size=(cfg.img_size, cfg.img_size), w2h_ratio=1.0,
            mask_interpolation=0, p=1.0))          # 0 = cv2.INTER_NEAREST (binary mask safe)

    if cfg.aug_hflip_p > 0:
        transforms.append(A.HorizontalFlip(p=cfg.aug_hflip_p))
    if cfg.aug_vflip_p > 0:
        transforms.append(A.VerticalFlip(p=cfg.aug_vflip_p))
    if not multiscale and cfg.aug_rotate_p > 0:
        transforms.append(_rotate(cfg))            # non-multiscale: rotate inline on img_size
    if cfg.aug_bc_p > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=cfg.aug_brightness_limit,
            contrast_limit=cfg.aug_contrast_limit, p=cfg.aug_bc_p))
    if cfg.aug_hsv_p > 0:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=cfg.aug_hue_shift, sat_shift_limit=cfg.aug_sat_shift,
            val_shift_limit=cfg.aug_val_shift, p=cfg.aug_hsv_p))
    return A.Compose(transforms, seed=cfg.seed)   # seed the Compose (global numpy seed is ignored)
