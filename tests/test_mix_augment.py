import numpy as np
import torch

from training.mix_augment import MixAugmentDataset, copy_paste, cutout, mosaic


class _FakeDS(torch.utils.data.Dataset):
    """Each sample is a constant image with a distinct value and a horizontal stem band."""

    def __init__(self, n=8, size=32):
        self.n, self.size = n, size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = torch.full((3, self.size, self.size), float(i + 1))
        mask = torch.zeros((1, self.size, self.size))
        mask[0, (i % self.size), :] = 1.0          # one row of stem per sample
        return img, mask


def test_mosaic_keeps_shape_and_draws_from_all_four_tiles():
    ds = _FakeDS()
    rng = np.random.default_rng(0)
    img, mask = mosaic([ds[0], ds[1], ds[2], ds[3]], rng)
    assert img.shape == (3, 32, 32) and mask.shape == (1, 32, 32)
    # each donor has a unique constant value, so every quadrant should be traceable
    vals = set(torch.unique(img).tolist())
    assert vals == {1.0, 2.0, 3.0, 4.0}, f"expected all four donors, got {vals}"


def test_mosaic_does_not_rescale():
    """Scale is fixed by the 15/512 m/px ground resolution; mosaic must crop, not resize.

    A resized donor would smear its single 1-px stem row into a fractional-valued band.
    """
    ds = _FakeDS()
    _, mask = mosaic([ds[0], ds[1], ds[2], ds[3]], np.random.default_rng(1))
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}, "mask picked up interpolated values"


def test_copy_paste_adds_foreground_and_keeps_mask_binary():
    ds = _FakeDS()
    dst, src = ds[0], ds[5]
    out_img, out_mask = copy_paste(dst, src, np.random.default_rng(0))
    assert out_mask.sum() >= dst[1].sum(), "pasting stems must not reduce foreground"
    assert set(torch.unique(out_mask).tolist()) <= {0.0, 1.0}
    # pasted pixels carry the donor's appearance
    assert float(out_img.max()) == 6.0 and float(out_img.min()) == 1.0


def test_cutout_erases_image_and_mask_together():
    """Erasing the image but keeping the label would teach the model to hallucinate stems."""
    img = torch.ones((3, 32, 32))
    mask = torch.ones((1, 32, 32))
    out_img, out_mask = cutout(img, mask, np.random.default_rng(0),
                               n_holes=(4, 4), size_frac=(0.3, 0.4))
    holes = (out_img[0] == 0)
    assert holes.any(), "expected some erased pixels"
    assert float(out_mask[0][holes].max()) == 0.0, "mask must be erased wherever the image is"


def test_mix_dataset_is_deterministic_and_shape_preserving():
    ds = MixAugmentDataset(_FakeDS(), mosaic_p=1.0, copypaste_p=1.0, cutout_p=1.0, seed=3)
    a_img, a_mask = ds[2]
    b_img, b_mask = ds[2]
    assert torch.equal(a_img, b_img) and torch.equal(a_mask, b_mask)
    assert a_img.shape == (3, 32, 32) and a_mask.shape == (1, 32, 32)
    assert set(torch.unique(a_mask).tolist()) <= {0.0, 1.0}


def test_mix_dataset_disabled_is_a_passthrough():
    base = _FakeDS()
    ds = MixAugmentDataset(base, mosaic_p=0.0, copypaste_p=0.0, cutout_p=0.0)
    for i in range(len(base)):
        assert torch.equal(ds[i][0], base[i][0])
        assert torch.equal(ds[i][1], base[i][1])


def test_wrapper_exposes_base_transform_for_worker_reseeding():
    """run_train._worker_init reseeds dataset.transform per DataLoader worker; if the
    wrapper hides it, every worker replays identical photometric augmentation."""
    class _WithTf(_FakeDS):
        transform = "sentinel"

    assert MixAugmentDataset(_WithTf()).transform == "sentinel"
    assert MixAugmentDataset(_FakeDS()).transform is None
