import numpy as np
import albumentations as A

from winmol_unet.training.config import TrainConfig
from winmol_unet.training.augment import build_augmentation


def _cfg(**kw):
    base = dict(data_dir="d", checkpoint_dir="c", log_dir="l", hdf5_out="h", onnx_out="o")
    base.update(kw)
    return TrainConfig(**base)


def test_only_enabled_transforms_included():
    cfg = _cfg(aug_hflip_p=0.5, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=0.0, aug_hsv_p=0.0)
    names = [type(t).__name__ for t in build_augmentation(cfg).transforms]
    assert names == ["HorizontalFlip"]


def test_augmentation_reproducible_by_seed():
    # albumentations 2.x uses its own per-Compose RNG (NOT the global numpy seed),
    # so build_augmentation must seed the Compose with cfg.seed for reproducibility.
    img = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)
    mask = np.zeros((64, 64), np.float32); mask[:, :32] = 1.0
    knobs = dict(aug_hflip_p=0.5, aug_vflip_p=0.5, aug_rotate_p=1.0,
                 aug_rotate_limit=25, aug_bc_p=1.0, aug_hsv_p=1.0)
    a = build_augmentation(_cfg(seed=1, **knobs))(image=img.copy(), mask=mask.copy())
    b = build_augmentation(_cfg(seed=1, **knobs))(image=img.copy(), mask=mask.copy())
    c = build_augmentation(_cfg(seed=2, **knobs))(image=img.copy(), mask=mask.copy())
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["mask"], b["mask"])
    assert not np.array_equal(a["image"], c["image"])   # different seed -> different aug


def test_all_transforms_when_enabled():
    cfg = _cfg(aug_hflip_p=0.5, aug_vflip_p=0.5, aug_rotate_p=0.3,
               aug_bc_p=0.5, aug_hsv_p=0.5)
    names = {type(t).__name__ for t in build_augmentation(cfg).transforms}
    assert names == {"HorizontalFlip", "VerticalFlip", "Rotate",
                     "RandomBrightnessContrast", "HueSaturationValue"}


def test_paired_geometric_and_photometric_image_only():
    # Left-bright image + left-half mask; forced hflip must move both together.
    img = np.zeros((512, 512, 3), np.float32); img[:, :256, :] = 1.0
    mask = np.zeros((512, 512), np.float32); mask[:, :256] = 1.0
    cfg = _cfg(aug_hflip_p=1.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=0.0, aug_hsv_p=0.0)
    out = build_augmentation(cfg)(image=img, mask=mask)
    # after hflip the bright half and the mask==1 half are both on the right, still aligned
    assert out["image"][:, 256:, :].mean() > out["image"][:, :256, :].mean()
    assert out["mask"][:, 256:].mean() > out["mask"][:, :256].mean()
    assert np.allclose((out["image"].mean(axis=2) > 0.5), out["mask"] > 0.5)

    # photometric changes the image but never the mask
    cfg2 = _cfg(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=1.0, aug_hsv_p=0.0)
    out2 = build_augmentation(cfg2)(image=img, mask=mask)
    assert not np.allclose(out2["image"], img)
    assert np.allclose(out2["mask"], mask)


def test_rotate_keeps_mask_binary():
    img = np.random.rand(512, 512, 3).astype(np.float32)
    mask = np.zeros((512, 512), np.float32); mask[100:400, 100:400] = 1.0
    cfg = _cfg(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=1.0, aug_rotate_limit=30,
               aug_bc_p=0.0, aug_hsv_p=0.0)
    out = build_augmentation(cfg)(image=img, mask=mask)
    assert set(np.unique(out["mask"]).tolist()) <= {0.0, 1.0}
