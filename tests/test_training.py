"""The training loop and everything it consumes: the dataset and its resize cache,
augmentation, loss selection, metrics, the run manifest, and the loggers.
test_overfit_loss_decreases is the public suite's fake-training anchor.

Absorbs test_train_loop.py, test_train_losses.py, test_train_dataset.py,
test_dataset_cache.py, test_loss_selection.py, test_focal_loss.py, test_augment.py,
test_colour_aug.py, test_run_config.py, test_run_logger.py, plus nine of
test_mosaic.py's eleven tests and seven of test_multiscale.py's eight (the CLI-flag
and full-training-run tests from those two files stay behind for later tasks).
"""
import glob
import json
import sys
import types

import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from winmol_unet.model import UNet
from winmol_unet.training.augment import build_augmentation
from winmol_unet.training.config import TrainConfig
from winmol_unet.training.dataset import (
    StemDataset,
    TilingStemDataset,
    _tile_starts,
    split_ids,
    train_val_split,
)
from winmol_unet.training.evaluate import evaluate
from winmol_unet.training.losses import (
    LOSSES,
    bce_loss,
    bce_soft_f1_loss,
    focal_loss,
    soften_targets,
)
from winmol_unet.training.metrics import f1, precision, recall
from winmol_unet.training.run_train import config_from_args, write_run_config, _worker_init
from winmol_unet.training.train import train_one_run


# --- the training loop (test_overfit_loss_decreases is a public anchor) ------


def _train_mode_loss(model, loader):
    # The overfit smoke test asks "does optimization reduce the loss?" — a
    # train-mode question. Eval-mode BatchNorm uses running stats that are
    # meaningless on a 2-sample toy set, so we measure the loss the training
    # loop actually optimizes (train mode, batch stats).
    model.train()
    device = next(model.parameters()).device   # model may be on MPS/CUDA after train_one_run
    with torch.no_grad():
        img, mask = next(iter(loader))
        img, mask = img.to(device), mask.to(device)
        return bce_soft_f1_loss(model(img), mask).item()


def test_overfit_loss_decreases(tmp_path, stem_dataset, train_config):
    torch.manual_seed(0)
    root = stem_dataset(tmp_path, n=2)
    ds = StemDataset(str(root / "train"), str(root / "mask"))
    loader = DataLoader(ds, batch_size=2)
    # "auto" (MPS/CUDA where available) restores this test's pre-consolidation runtime;
    # its assertion is relational (after < before), so it is device-independent.
    cfg = train_config(epochs=8, lr=1e-2, device="auto")
    model = UNet(dropout=0.0)
    before = _train_mode_loss(model, loader)
    train_one_run(model, loader, loader, cfg)
    after = _train_mode_loss(model, loader)
    assert after < before


def test_evaluate_returns_metric_keys(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=2)
    ds = StemDataset(str(root / "train"), str(root / "mask"))
    loader = DataLoader(ds, batch_size=2)
    out = evaluate(UNet(dropout=0.0), loader)
    # Exact four metric keys plus the val-loss components (train: log the parts).
    assert {"loss", "precision", "recall", "f1"} <= set(out)
    assert {"loss_bce", "loss_soft_f1_term"} <= set(out)


# --- loss + metrics on hand-computed tensors ----------------------------------


def _logits_for(target, strong=12.0):
    # logits whose sigmoid ~ target (near 0 or 1)
    return (target * 2 - 1) * strong


def test_perfect_prediction_low_loss_high_f1():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = _logits_for(target)
    loss = bce_soft_f1_loss(logits, target)
    assert loss.item() < 1e-2
    assert f1(logits, target) == 1.0
    assert precision(logits, target) == 1.0
    assert recall(logits, target) == 1.0


def test_loss_is_differentiable_through_soft_f1():
    target = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.zeros_like(target, requires_grad=True)
    bce_soft_f1_loss(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_metrics_match_hand_computed():
    # pred: [1,1,0,0]  target: [1,0,0,0]  -> tp=1 fp=1 fn=0 tn=2
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    logits = _logits_for(torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]))
    assert abs(precision(logits, target) - 0.5) < 1e-6   # tp/(tp+fp)=1/2
    assert abs(recall(logits, target) - 1.0) < 1e-6      # tp/(tp+fn)=1/1
    assert abs(f1(logits, target) - (2 / 3)) < 1e-6      # 2*.5*1/(1.5)


# --- StemDataset: shapes, pairing, transforms, split --------------------------


def test_getitem_shapes_ranges_and_pairing(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, ids=(1, 2, 10))
    img_dir, mask_dir = root / "train", root / "mask"
    ds = StemDataset(str(img_dir), str(mask_dir))       # transform=None
    assert len(ds) == 3
    img, mask = ds[0]
    assert img.shape == (3, 512, 512) and img.dtype == torch.float32
    assert mask.shape == (1, 512, 512)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_transform_paired_flip_keeps_alignment(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=1, size=40, stem_frac=0.5)
    img_dir, mask_dir = root / "train", root / "mask"
    transform = A.Compose([A.HorizontalFlip(p=1.0)])    # always flip
    ds = StemDataset(str(img_dir), str(mask_dir), transform=transform)
    img, mask = ds[0]
    bright = img[:, mask[0] == 1].mean()
    dark = img[:, mask[0] == 0].mean()
    assert bright > dark        # image and mask flipped together


def test_transform_photometric_leaves_mask(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=1)
    img_dir, mask_dir = root / "train", root / "mask"
    base = StemDataset(str(img_dir), str(mask_dir))[0][1]        # mask, no transform
    tf = A.Compose([A.RandomBrightnessContrast(p=1.0)])
    _, mask = StemDataset(str(img_dir), str(mask_dir), transform=tf)[0]
    assert torch.equal(base, mask)                              # mask unchanged by photometric


def test_cache_not_mutated_by_transform(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=1)
    img_dir, mask_dir = root / "train", root / "mask"
    tf = A.Compose([A.HorizontalFlip(p=1.0)])
    ds = StemDataset(str(img_dir), str(mask_dir), transform=tf)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)      # deterministic (p=1 flip) -> cache base intact each call


def test_split_disjoint_and_val_has_no_transform(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=10)
    img_dir, mask_dir = root / "train", root / "mask"
    tf = A.Compose([A.HorizontalFlip(p=1.0)])
    tr, va = train_val_split(str(img_dir), str(mask_dir), 0.2, 1, transform=tf)
    assert len(tr) == 8 and len(va) == 2
    assert set(tr.ids).isdisjoint(set(va.ids))
    assert tr.transform is tf and va.transform is None


# --- the in-memory resize cache ------------------------------------------------


def test_cache_off_matches_cache_on_and_stores_nothing(tmp_path, stem_dataset):
    root = stem_dataset(tmp_path, n=2)
    img_dir, mask_dir = root / "train", root / "mask"

    cached = StemDataset(str(img_dir), str(mask_dir), cache=True)
    uncached = StemDataset(str(img_dir), str(mask_dir), cache=False)

    ci, cm = cached[0]
    ui, um = uncached[0]
    assert torch.equal(ci, ui) and torch.equal(cm, um)   # identical output

    _ = uncached[0]; _ = uncached[1]
    assert uncached._cache == {}                          # nothing cached when off
    assert cached._cache != {}                            # cached when on


def test_loader_with_workers_and_transform(tmp_path, stem_dataset):
    # num_workers>0 + an albumentations transform must work (Compose picklable under
    # spawn; per-worker reseed via run_train._worker_init). Locks the large-dataset path.
    root = stem_dataset(tmp_path, n=4)
    img_dir, mask_dir = root / "train", root / "mask"
    ds = StemDataset(str(img_dir), str(mask_dir), transform=A.Compose([A.HorizontalFlip(p=0.5)], seed=1),
                     cache=False)
    loader = DataLoader(ds, batch_size=2, num_workers=2, worker_init_fn=_worker_init)
    batches = [b for b in loader]
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 3, 512, 512)


# --- loss selection: `bce` is the R-equivalent one -----------------------------
#
# R's loss is `BCE + (1 - F1)` with `k_round` applied to the prediction. A rounded
# value has zero gradient almost everywhere, so R's F1 term never reaches the
# optimiser and R trains on BCE alone. `--loss bce` reproduces that, which makes
# the soft-F1 term ablatable.


def test_both_losses_registered_and_distinct():
    assert set(LOSSES) == {"bce_soft_f1", "bce", "bce_hard_f1",
                           "focal", "focal_soft_f1"}
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    assert abs(bce_soft_f1_loss(logits, target) - bce_loss(logits, target)) > 1e-3


def test_soft_f1_term_is_the_only_difference():
    """bce_soft_f1 == bce + (1 - soft_F1), exactly."""
    logits = torch.randn(4, 1, 8, 8)
    target = (torch.rand(4, 1, 8, 8) > 0.5).float()
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum()
    fp = (probs * (1 - target)).sum()
    fn = ((1 - probs) * target).sum()
    soft_f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    assert torch.allclose(bce_soft_f1_loss(logits, target),
                          bce_loss(logits, target) + (1 - soft_f1), atol=1e-6)


def test_a_rounded_f1_term_has_no_gradient():
    """Why `bce` is the honest stand-in for R: the rounded term cannot train anything."""
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    hard = torch.round(torch.sigmoid(logits))          # R's k_round(y_pred)
    tp = (hard * target).sum()
    fp = (hard * (1 - target)).sum()
    fn = ((1 - hard) * target).sum()
    (1 - (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).backward()
    assert logits.grad is None or torch.count_nonzero(logits.grad) == 0


def test_cli_selects_the_loss():
    assert config_from_args(["--data-dir", "x"]).loss == "bce_soft_f1"
    assert config_from_args(["--data-dir", "x", "--loss", "bce"]).loss == "bce"


def test_r_literal_loss_has_the_same_gradient_as_plain_bce():
    """`bce_hard_f1` is R's F1Score_loss transcribed. It reports a bigger number than
    `bce` and produces a bit-identical gradient, which is the whole finding."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.9).float()

    grads = {}
    values = {}
    for name in ("bce", "bce_hard_f1", "bce_soft_f1"):
        x = logits.clone().requires_grad_(True)
        v = LOSSES[name](x, target)
        v.backward()
        grads[name] = x.grad.clone()
        values[name] = float(v.detach())

    # R's term inflates the reported loss...
    assert values["bce_hard_f1"] > values["bce"] + 0.5
    # ...and changes nothing the optimiser sees
    assert torch.equal(grads["bce_hard_f1"], grads["bce"])
    # while the soft version genuinely does
    assert not torch.allclose(grads["bce_soft_f1"], grads["bce"])


def test_label_smoothing_touches_only_the_edge_band():
    """Boundary-aware smoothing must leave interiors and far background alone.

    That is the whole point: false negatives are 1.59x enriched within 2 px of an
    annotation edge, while interiors are pixels the annotators were sure about.
    """
    t = torch.zeros(1, 1, 11, 11)
    t[0, 0, 4:7, 4:7] = 1.0                        # a 3x3 stem

    s = soften_targets(t, eps=0.1, band_px=1)
    assert s[0, 0, 5, 5].item() == 1.0             # interior untouched
    assert abs(s[0, 0, 4, 4].item() - 0.9) < 1e-5  # stem edge pulled down
    assert abs(s[0, 0, 3, 5].item() - 0.1) < 1e-5  # background beside it pulled up
    assert s[0, 0, 0, 0].item() == 0.0             # far background untouched

    # off by default, and exactly the identity when eps=0
    assert torch.equal(soften_targets(t, 0.0, 2), t)

    # band_px=0 is the classic global variant: every pixel moves
    g = soften_targets(t, eps=0.1, band_px=0)
    assert abs(g[0, 0, 5, 5].item() - 0.9) < 1e-5
    assert abs(g[0, 0, 0, 0].item() - 0.1) < 1e-5


# --- focal loss: a third arm beside `bce` and `bce_soft_f1` --------------------
#
# Focal (Lin et al. 2017) is `-alpha_t * (1 - p_t)^gamma * log(p_t)`: it keeps
# BCE's shape but down-weights pixels the model already gets right, so the
# gradient concentrates on the hard ones. Motivation here is the same asymmetry
# the label-smoothing attempt targeted and failed to fix -- false negatives are
# 1.59x enriched within 2 px of an annotation edge -- except focal reweights by
# *difficulty* rather than by distance to an edge.
#
# `alpha` defaults to None (no class re-weighting) so that a focal-vs-bce
# difference is attributable to the focusing term alone, matching how every
# other arm in this repo isolates exactly one change.


def test_gamma_zero_without_alpha_is_exactly_bce():
    """The focusing term is the only thing focal adds; switch it off and BCE remains."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    assert torch.allclose(focal_loss(logits, target, gamma=0.0),
                          bce_loss(logits, target), atol=1e-6)


def test_easy_pixels_are_down_weighted_relative_to_bce():
    """The defining property: on confident-correct pixels focal must fall below BCE."""
    target = torch.ones(1, 1, 8, 8)
    easy = torch.full_like(target, 4.0)      # sigmoid ~ 0.982, already correct
    assert focal_loss(easy, target).item() < 0.25 * bce_loss(easy, target).item()


def test_hard_pixels_keep_almost_all_their_weight():
    """A pixel the model gets wrong is barely modulated -- (1-p_t)^gamma -> 1."""
    target = torch.ones(1, 1, 8, 8)
    hard = torch.full_like(target, -4.0)     # sigmoid ~ 0.018, confidently wrong
    ratio = focal_loss(hard, target).item() / bce_loss(hard, target).item()
    assert ratio > 0.95


def test_focal_is_differentiable():
    target = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.zeros_like(target, requires_grad=True)
    focal_loss(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_alpha_reweights_the_positive_class():
    """alpha is available but off by default, so it never silently confounds an arm."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    positives = torch.ones(2, 1, 16, 16)
    unweighted = focal_loss(logits, positives, alpha=None).item()
    assert abs(focal_loss(logits, positives, alpha=0.25).item()
               - 0.25 * unweighted) < 1e-6


def test_focal_arms_are_registered_and_selectable():
    assert {"focal", "focal_soft_f1"} <= set(LOSSES)
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    # focal_soft_f1 adds the same (1 - soft_F1) term bce_soft_f1 does
    assert LOSSES["focal_soft_f1"](logits, target) > LOSSES["focal"](logits, target)


def test_cli_accepts_the_focal_arms():
    for name in ("focal", "focal_soft_f1"):
        assert config_from_args(["--data-dir", "x", "--loss", name]).loss == name


# --- augmentation pipeline: which transforms, reproducibility, alignment ------


def test_only_enabled_transforms_included(train_config):
    cfg = train_config(aug_hflip_p=0.5, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=0.0, aug_hsv_p=0.0)
    names = [type(t).__name__ for t in build_augmentation(cfg).transforms]
    assert names == ["HorizontalFlip"]


def test_augmentation_reproducible_by_seed(train_config):
    # albumentations 2.x uses its own per-Compose RNG (NOT the global numpy seed),
    # so build_augmentation must seed the Compose with cfg.seed for reproducibility.
    img = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)
    mask = np.zeros((64, 64), np.float32); mask[:, :32] = 1.0
    knobs = dict(aug_hflip_p=0.5, aug_vflip_p=0.5, aug_rotate_p=1.0,
                 aug_rotate_limit=25, aug_bc_p=1.0, aug_hsv_p=1.0)
    a = build_augmentation(train_config(seed=1, **knobs))(image=img.copy(), mask=mask.copy())
    b = build_augmentation(train_config(seed=1, **knobs))(image=img.copy(), mask=mask.copy())
    c = build_augmentation(train_config(seed=2, **knobs))(image=img.copy(), mask=mask.copy())
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["mask"], b["mask"])
    assert not np.array_equal(a["image"], c["image"])   # different seed -> different aug


def test_all_transforms_when_enabled(train_config):
    cfg = train_config(aug_hflip_p=0.5, aug_vflip_p=0.5, aug_rotate_p=0.3,
               aug_bc_p=0.5, aug_hsv_p=0.5)
    names = {type(t).__name__ for t in build_augmentation(cfg).transforms}
    assert names == {"HorizontalFlip", "VerticalFlip", "Rotate",
                     "RandomBrightnessContrast", "HueSaturationValue"}


def test_paired_geometric_and_photometric_image_only(train_config):
    # Left-bright image + left-half mask; forced hflip must move both together.
    img = np.zeros((512, 512, 3), np.float32); img[:, :256, :] = 1.0
    mask = np.zeros((512, 512), np.float32); mask[:, :256] = 1.0
    cfg = train_config(aug_hflip_p=1.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=0.0, aug_hsv_p=0.0)
    out = build_augmentation(cfg)(image=img, mask=mask)
    # after hflip the bright half and the mask==1 half are both on the right, still aligned
    assert out["image"][:, 256:, :].mean() > out["image"][:, :256, :].mean()
    assert out["mask"][:, 256:].mean() > out["mask"][:, :256].mean()
    assert np.allclose((out["image"].mean(axis=2) > 0.5), out["mask"] > 0.5)

    # photometric changes the image but never the mask
    cfg2 = train_config(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=1.0, aug_hsv_p=0.0)
    out2 = build_augmentation(cfg2)(image=img, mask=mask)
    assert not np.allclose(out2["image"], img)
    assert np.allclose(out2["mask"], mask)


def test_rotate_keeps_mask_binary(train_config):
    img = np.random.rand(512, 512, 3).astype(np.float32)
    mask = np.zeros((512, 512), np.float32); mask[100:400, 100:400] = 1.0
    cfg = train_config(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=1.0, aug_rotate_limit=30,
               aug_bc_p=0.0, aug_hsv_p=0.0)
    out = build_augmentation(cfg)(image=img, mask=mask)
    assert set(np.unique(out["mask"]).tolist()) <= {0.0, 1.0}


# --- colour augmentation: limits must reach the pipeline, not just the config -
#
# Colour is the axis the beech sites differ on most, and the one autumn site is
# unlearnable from the others. A knob that silently keeps its default would make
# that experiment a no-op that still produces plausible numbers.


def _colour_args(tmp_path, *extra):
    return config_from_args(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path),
                             *extra])


def test_cli_defaults_match_the_config_defaults(tmp_path):
    """Arm A of the colour experiment reuses earlier runs, so defaults must not move."""
    cfg = _colour_args(tmp_path)
    assert (cfg.aug_hue_shift, cfg.aug_sat_shift, cfg.aug_val_shift) == (20, 30, 20)


def test_limits_reach_the_transform(tmp_path):
    cfg = _colour_args(tmp_path, "--aug-hue-shift", "90", "--aug-sat-shift", "60",
                "--aug-val-shift", "30", "--aug-hsv-p", "0.9")
    hsv = [t for t in build_augmentation(cfg).transforms
           if type(t).__name__ == "HueSaturationValue"]
    assert len(hsv) == 1
    h = hsv[0]
    assert tuple(h.hue_shift_limit) == (-90, 90)
    assert tuple(h.sat_shift_limit) == (-60, 60)
    assert h.p == pytest.approx(0.9)


def test_strong_hue_actually_moves_colour(tmp_path):
    """A full-circle hue shift must change hue far more than the default does."""
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = 200                       # a strongly red image
    img[..., 1] = 60
    mask = np.zeros((64, 64), np.uint8)

    def spread(hue_shift):
        cfg = _colour_args(tmp_path, "--aug-hue-shift", str(hue_shift), "--aug-hsv-p", "1.0",
                    "--aug-rotate-p", "0", "--aug-hflip-p", "0", "--aug-vflip-p", "0",
                    "--aug-bc-p", "0")
        aug = build_augmentation(cfg)
        outs = [aug(image=img, mask=mask)["image"].astype(int) for _ in range(12)]
        # channel ordering scatters as hue rotates; measure how far from the original
        return float(np.mean([np.abs(o - img.astype(int)).mean() for o in outs]))

    assert spread(90) > spread(5) * 2, "strong hue shift barely moved the image"


# --- the run manifest: a run must record what it was, not just what it scored -
#
# `test_results.md` names the dataset and the arch and nothing else, so a
# finished run could not be audited after the fact — the crop range and seed
# lived only in the shell that launched it. These tests pin the fields that
# answer "was this run what it claimed".


def _run_config(train_config, tmp_path, **kw):
    # onnx_out/hdf5_out/pt_out are placed directly in tmp_path (not the fixture's
    # default out/ subdirectory) so write_run_config's "beside the model" path lands
    # exactly at tmp_path/run_config.json, which test_lands_beside_the_model_it_describes
    # pins.
    return train_config(
        data_dir="/ds/train",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        pt_out=str(tmp_path / "model.pt"),
        hdf5_out=str(tmp_path / "model.hdf5"),
        onnx_out=str(tmp_path / "model.onnx"),
        **kw)


def test_records_the_knobs_that_define_an_arm(tmp_path, train_config):
    """Crop range and seed are the whole experiment — they must survive the shell."""
    cfg = _run_config(train_config, tmp_path, multiscale=True, crop_min_px=394, crop_max_px=666,
               seed=1, deterministic=True, arch="hrnet")
    p = write_run_config(cfg, argv=["run_train", "--crop-min-px", "394"])
    d = json.loads(open(p).read())
    assert d["config"]["crop_min_px"] == 394 and d["config"]["crop_max_px"] == 666
    assert d["config"]["seed"] == 1 and d["config"]["deterministic"] is True
    assert d["config"]["arch"] == "hrnet" and d["config"]["multiscale"] is True
    assert d["argv"] == ["run_train", "--crop-min-px", "394"]


def test_lands_beside_the_model_it_describes(tmp_path, train_config):
    cfg = _run_config(train_config, tmp_path)
    assert write_run_config(cfg) == str(tmp_path / "run_config.json")


def test_records_provenance(tmp_path, train_config):
    cfg = _run_config(train_config, tmp_path)
    d = json.loads(open(write_run_config(cfg)).read())
    for k in ("started_utc", "host", "torch", "git_commit", "git_dirty"):
        assert k in d
    assert d["host"] and d["torch"]


def test_survives_a_missing_git_checkout(tmp_path, train_config, monkeypatch):
    """A run outside a checkout is still a valid run — provenance degrades, not fails."""
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    cfg = _run_config(train_config, tmp_path)
    d = json.loads(open(write_run_config(cfg)).read())
    assert d["git_commit"] is None and d["git_dirty"] is None
    assert d["config"]["data_dir"] == "/ds/train"


# --- the run logger: TensorBoard always, wandb only when asked ----------------


def _fake_wandb():
    calls = {}
    m = types.ModuleType("wandb")
    m.init = lambda **kw: calls.__setitem__("init", kw)
    m.log = lambda d, step=None: calls.__setitem__("log", (dict(d), step))
    m.finish = lambda: calls.__setitem__("finish", True)
    return m, calls


def _fake_dotenv():
    m = types.ModuleType("dotenv")
    m.load_dotenv = lambda *a, **k: True
    return m


def test_tensorboard_only_when_wandb_off(tmp_path, monkeypatch):
    # Ensure wandb is never needed when off.
    monkeypatch.setitem(sys.modules, "wandb", None)   # import wandb -> ImportError if touched
    from winmol_unet.training.run_logger import RunLogger
    lg = RunLogger(str(tmp_path / "log"), use_wandb=False)
    lg.log_scalars({"val/f1": 0.5}, 1)
    lg.close()
    assert glob.glob(str(tmp_path / "log" / "events*"))   # TB wrote something


def test_wandb_init_log_finish(tmp_path, monkeypatch):
    fake, calls = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "dotenv", _fake_dotenv())
    from winmol_unet.training.run_logger import RunLogger
    lg = RunLogger(str(tmp_path / "log"), use_wandb=True, project="P", run_name="R")
    assert calls["init"] == {"project": "P", "name": "R", "notes": None}
    lg.log_scalars({"val/f1": 0.5}, 3)
    assert calls["log"] == ({"val/f1": 0.5}, 3)
    lg.close()
    assert calls["finish"] is True


def test_missing_wandb_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)   # forces ImportError on `import wandb`
    from winmol_unet.training.run_logger import RunLogger
    with pytest.raises(RuntimeError, match=r"\[wandb\]"):
        RunLogger(str(tmp_path / "log"), use_wandb=True)


# --- mosaic augmentation: off by default, binary-safe, and split-safe ---------
#
# Mosaic is UNEVALUATED in this repo — no paired run, no LOSO fold, no results
# document — so the most important property is that it is genuinely inert unless
# asked for. The second is that it cannot reach across a split: tiles are
# oversampled and overlap, so leakage is already the failure mode the whole split
# machinery exists to prevent, and an augmentation that stitches four tiles
# together is an obvious way to reintroduce it.
#
# Nine of test_mosaic.py's eleven tests: the CLI-flag test and the "recipe is
# labelled unevaluated" test stay in tests/test_mosaic.py for a later task.


def _mosaic_cfg(**kw):
    return TrainConfig(data_dir="", checkpoint_dir="", log_dir="", onnx_out="",
                       hdf5_out="", **kw)


def _mosaic_dataset(tmp_path, n=8, **kw):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype="uint8")).save(
            img_dir / f"train{i}.jpeg")
        # A DISTINCT mask per tile: the stripe moves with i. Tiles that are byte-identical
        # make every provenance assertion vacuous -- a leak, a duplicated primary and a
        # correct draw all look the same. This is what lets the tests below check identity
        # rather than just count.
        m = np.zeros((64, 64), dtype="uint8")
        m[10:40, (2 * i) % 50:(2 * i) % 50 + 6] = 255
        Image.fromarray(m).save(mask_dir / f"mask{i}.gif")
    return StemDataset(str(img_dir), str(mask_dir), img_size=64, **kw)


def test_mosaic_is_off_by_default():
    assert _mosaic_cfg().mosaic_p == 0.0


def test_default_pipeline_contains_no_mosaic():
    """p=0 must not merely disable it — the transform should not be in the pipeline."""
    names = [t.__class__.__name__ for t in build_augmentation(_mosaic_cfg()).transforms]
    assert "Mosaic" not in names


def test_mosaic_is_first_in_the_pipeline():
    """Anything after mosaic sees one coherent image; a crop before it would undo it."""
    names = [t.__class__.__name__ for t in build_augmentation(_mosaic_cfg(mosaic_p=0.5)).transforms]
    assert names[0] == "Mosaic", names


def test_mosaic_is_first_even_with_the_multiscale_crop():
    cfg = _mosaic_cfg(mosaic_p=0.5, multiscale=True, crop_min_px=32, crop_max_px=64, img_size=64)
    names = [t.__class__.__name__ for t in build_augmentation(cfg).transforms]
    assert names[0] == "Mosaic"
    assert names.index("Mosaic") < names.index("RandomSizedCrop")


def test_output_stays_binary_and_the_right_shape(tmp_path):
    cfg = _mosaic_cfg(mosaic_p=1.0, img_size=64)
    ds = _mosaic_dataset(tmp_path, transform=build_augmentation(cfg), mosaic_p=1.0, seed=1)
    img, mask = ds[0]
    assert img.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)
    # a mask that came back as a blend of cells would have intermediate values
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}


def test_partners_never_leave_this_datasets_ids(tmp_path):
    """The leak boundary: a val dataset holds its own ids, so mosaic cannot reach train.

    Asserted on the mechanism rather than on outputs, because a leak here would be
    invisible in the tensors and would show up only as an inflated validation score.
    """
    ds = _mosaic_dataset(tmp_path, n=8, mosaic_p=1.0, seed=1)
    held_out = set(ds.ids[4:])             # pretend these are the train half
    ds.ids = ds.ids[:4]                    # ...and this dataset is the val half

    # Compare the ACTUAL returned partners against the tiles of the held-out half. Each
    # fixture tile has a distinct mask, so identity is checkable from the pixels.
    forbidden = {ds._load_mask(n).tobytes() for n in held_out}
    seen = 0
    for _ in range(30):
        for part in ds._mosaic_partners(0):
            assert part["mask"].tobytes() not in forbidden, \
                "a mosaic partner came from outside this dataset's split"
            seen += 1
    assert seen > 0, "no partners were drawn, so nothing was actually checked"


def test_partners_exclude_the_primary_tile(tmp_path):
    """The primary tile must not also appear as one of its own partners.

    Checked on the returned arrays, not on the count: a 2x2 mosaic of four copies of one
    tile is a zoom-out, not an augmentation, and would be invisible in a length assertion.
    """
    ds = _mosaic_dataset(tmp_path, n=8, mosaic_p=1.0, seed=1)
    for i in range(len(ds.ids)):
        primary = ds._load_mask(ds.ids[i]).tobytes()
        for _ in range(10):
            picks = ds._mosaic_partners(i)
            assert len(picks) == 3
            assert all(p["mask"].tobytes() != primary for p in picks), \
                f"tile {i} was returned as its own mosaic partner"


def test_a_single_tile_dataset_does_not_crash(tmp_path):
    """`sample` on an empty population raises; one tile has no partners to draw."""
    cfg = _mosaic_cfg(mosaic_p=1.0, img_size=64)
    ds = _mosaic_dataset(tmp_path, n=1, transform=build_augmentation(cfg), mosaic_p=1.0, seed=1)
    img, mask = ds[0]
    assert img.shape == (3, 64, 64)


def test_evaluation_datasets_never_mosaic(tmp_path):
    """Val/test are built with transform=None, so mosaic cannot apply there at all."""
    ds = _mosaic_dataset(tmp_path, transform=None, mosaic_p=1.0, seed=1)
    img, mask = ds[0]
    assert img.shape == (3, 64, 64)
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}


# --- multi-scale native-fidelity dataloader: native load, rotate->crop, tiling
#
# Seven of test_multiscale.py's eight tests: the full run_training + eval-tiling
# integration test stays in tests/test_multiscale.py for a later task.


def _native_ds(d, n=6, size=64):
    """Write n native-resolution RGB jpeg / binary gif pairs of `size`x`size`."""
    (d / "train").mkdir(parents=True)
    (d / "mask").mkdir(parents=True)
    for k in range(1, n + 1):
        rgb = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        Image.fromarray(rgb, "RGB").save(d / "train" / f"train{k}.jpeg", quality=95)
        m = np.zeros((size, size), np.uint8)
        m[: size // 2, : size // 2] = 255                 # a quadrant of foreground
        Image.fromarray(m, "L").save(d / "mask" / f"mask{k}.gif")


class _MultiscaleCfg:
    """Minimal config stand-in for build_augmentation (only the aug knobs it reads)."""
    def __init__(self, **kw):
        d = dict(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_rotate_limit=15.0,
                 aug_bc_p=0.0, aug_brightness_limit=0.2, aug_contrast_limit=0.2, aug_hsv_p=0.0,
                 aug_hue_shift=20, aug_sat_shift=30, aug_val_shift=20, seed=1, img_size=16,
                 multiscale=False, crop_min_px=8, crop_max_px=40)
        d.update(kw)
        self.__dict__.update(d)


def test_native_mode_skips_resize(tmp_path):
    _native_ds(tmp_path, n=2, size=40)
    native = StemDataset(str(tmp_path / "train"), str(tmp_path / "mask"),
                         img_size=16, resize=False, cache=False)
    img, mask = native[0]
    assert img.shape == (3, 40, 40) and mask.shape == (1, 40, 40)      # untouched native size
    resized = StemDataset(str(tmp_path / "train"), str(tmp_path / "mask"),
                          img_size=16, resize=True, cache=False)
    img2, _ = resized[0]
    assert img2.shape == (3, 16, 16)                                   # forced to img_size


def test_multiscale_transform_outputs_img_size_and_binary_mask(tmp_path):
    _native_ds(tmp_path, n=2, size=40)
    tf = build_augmentation(_MultiscaleCfg(multiscale=True, img_size=16, crop_min_px=8, crop_max_px=40))
    ds = StemDataset(str(tmp_path / "train"), str(tmp_path / "mask"),
                     img_size=16, transform=tf, resize=False, cache=False)
    img, mask = ds[0]
    assert img.shape == (3, 16, 16) and mask.shape == (1, 16, 16)      # crop resized to img_size
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}             # mask stays binary


def test_multiscale_rotates_full_tile_before_crop():
    # order matters: Rotate (full native tile) must precede RandomSizedCrop so the crop
    # lands on interior pixels; flips/photometric come after.
    tf = build_augmentation(_MultiscaleCfg(multiscale=True, aug_rotate_p=1.0, aug_rotate_limit=180.0,
                                 aug_hflip_p=0.5))
    names = [type(t).__name__ for t in tf.transforms]
    assert names[0] == "Rotate" and names[1] == "RandomSizedCrop"
    assert names.index("Rotate") < names.index("RandomSizedCrop") < names.index("HorizontalFlip")


def test_non_multiscale_keeps_prior_order():
    tf = build_augmentation(_MultiscaleCfg(multiscale=False, aug_hflip_p=0.5, aug_rotate_p=1.0))
    names = [type(t).__name__ for t in tf.transforms]
    assert "RandomSizedCrop" not in names                # no crop without multiscale
    assert names.index("HorizontalFlip") < names.index("Rotate")   # flips before rotate


def test_tile_starts():
    assert _tile_starts(1024, 512) == [0, 512]           # clean 2x2
    assert _tile_starts(512, 512) == [0]
    assert _tile_starts(400, 512) == [0]                 # smaller than tile -> single
    assert _tile_starts(1000, 512) == [0, 488]           # last clamped to edge (dim - tile)


def test_tiling_dataset_covers_and_is_deterministic(tmp_path):
    _native_ds(tmp_path, n=3, size=32)
    ds = TilingStemDataset(str(tmp_path / "train"), str(tmp_path / "mask"), tile=16, cache=True)
    assert len(ds) == 3 * 4                               # 2x2 tiles per 32px image
    img, mask = ds[0]
    assert img.shape == (3, 16, 16) and mask.shape == (1, 16, 16)

    # the 4 tiles of image 0 reconstruct the full native image (top-left, top-right, ...)
    native = np.asarray(Image.open(tmp_path / "train" / "train1.jpeg").convert("RGB")) / 255.0
    tiles = [ds[j][0].numpy().transpose(1, 2, 0) for j in range(4)]
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    recon = np.concatenate([top, bot], axis=0)
    assert np.allclose(recon, native, atol=1e-4)

    ds2 = TilingStemDataset(str(tmp_path / "train"), str(tmp_path / "mask"), tile=16, cache=False)
    assert ds._index == ds2._index                       # deterministic index (cache-independent)
    assert torch.equal(ds2[5][0], ds[5][0])              # same tile regardless of cache mode


def test_split_ids_partitions_disjointly(tmp_path):
    _native_ds(tmp_path, n=10, size=16)
    train_ids, val_ids = split_ids(str(tmp_path / "train"), str(tmp_path / "mask"), 0.2, 1)
    assert set(train_ids).isdisjoint(val_ids)
    # ids are the raw filename stems (strings) so published sets like `train_100_1.jpeg`
    # load unrenamed; the partition property is what matters, not the type.
    assert sorted(train_ids + val_ids, key=int) == [str(i) for i in range(1, 11)]
    assert len(val_ids) == 2                              # round(10 * 0.2)


# --- metrics history: the per-epoch numbers, kept as data ----------------------------
# RunLogger already receives every scalar train.py computes. It forwards them to
# TensorBoard and forgets them, so nothing outside TB can re-plot or diff a run.


def test_run_logger_writes_metrics_history_on_close(tmp_path):
    """Every logged scalar dict is retained and written as JSON, one entry per step."""
    import json

    from winmol_unet.training.run_logger import RunLogger

    log_dir = tmp_path / "log"
    lg = RunLogger(str(log_dir), use_wandb=False)
    lg.log_scalars({"train/loss": 1.0, "val/loss": 2.0, "val/f1": 0.1}, 0)
    lg.log_scalars({"train/loss": 0.5, "val/loss": 1.5, "val/f1": 0.4}, 1)
    lg.close()

    history = json.loads((log_dir / "metrics_history.json").read_text())
    assert [e["step"] for e in history] == [0, 1]
    assert [e["train/loss"] for e in history] == [1.0, 0.5]
    assert [e["val/f1"] for e in history] == [0.1, 0.4]


# --- static plots on disk ------------------------------------------------------------
# The curves exist in TensorBoard already; these are the openable, attachable versions.


def _history(n=4):
    return [{"step": i, "train/loss": 1.0 - 0.1 * i, "val/loss": 1.2 - 0.1 * i,
             "val/precision": 0.1 * i, "val/recall": 0.05 * i, "val/f1": 0.07 * i,
             "lr": 1e-3} for i in range(n)]


def test_plot_history_writes_loss_and_metric_curves(tmp_path):
    from winmol_unet.training.plots import plot_history

    written = plot_history(_history(), str(tmp_path))

    names = sorted(p.name for p in written)
    assert names == ["loss_curves.png", "metric_curves.png"]
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_plot_history_survives_a_history_missing_optional_series(tmp_path):
    """A run with no per-component losses and no lr must still produce curves."""
    from winmol_unet.training.plots import plot_history

    minimal = [{"step": i, "train/loss": 1.0 - 0.1 * i, "val/loss": 1.1 - 0.1 * i}
               for i in range(3)]
    written = plot_history(minimal, str(tmp_path))

    assert [p.name for p in written] == ["loss_curves.png"]   # no metrics to draw


# --- prediction panels ---------------------------------------------------------------


class _EchoModel(torch.nn.Module):
    """Predicts stem wherever the red channel is bright.

    stem_dataset paints the image white exactly where the mask is white, so this
    scores near-perfectly on an aligned tile and near-zero on one whose mask has
    been inverted -- which is what makes worst-F1 selection testable.
    """

    def forward(self, x):
        return (x[:, :1] - 0.5) * 20.0


def _aligned_and_inverted(tmp_path, stem_dataset):
    """Three tiles: two the echo model gets right, one (index 2) it gets wrong."""
    from PIL import Image

    root = stem_dataset(tmp_path / "ds", n=3)
    bad = np.zeros((32, 32), np.uint8)
    bad[:, 16:] = 255                      # mask on the right, image is white on the left
    Image.fromarray(bad, "L").save(root / "mask" / "mask3.gif")
    return root


def test_worst_f1_indices_ranks_the_badly_predicted_tile_first(tmp_path, stem_dataset):
    from winmol_unet.training.dataset import StemDataset
    from winmol_unet.training.plots import worst_f1_indices

    root = _aligned_and_inverted(tmp_path, stem_dataset)
    ds = StemDataset(str(root / "train"), str(root / "mask"))

    worst = worst_f1_indices(_EchoModel(), ds, k=1)

    assert worst == [2], f"expected the inverted tile first, got {worst}"


def test_plot_predictions_writes_a_random_panel_and_a_labelled_selected_panel(
        tmp_path, stem_dataset):
    from winmol_unet.training.dataset import StemDataset
    from winmol_unet.training.plots import plot_predictions

    root = _aligned_and_inverted(tmp_path, stem_dataset)
    ds = StemDataset(str(root / "train"), str(root / "mask"))

    written = plot_predictions(_EchoModel(), ds, str(tmp_path / "out"), n=2)

    names = sorted(p.name for p in written)
    assert names == ["predictions_random.png", "predictions_selected_worst_f1.png"]
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_run_training_writes_plots_only_when_asked(tmp_path, stem_dataset, train_config):
    """--plots renders curves and panels beside the model; without it, nothing."""
    from winmol_unet.training.run_train import run_training

    root = stem_dataset(tmp_path / "ds", n=4)
    off = train_config(data_dir=root, onnx_out=str(tmp_path / "off" / "m.onnx"),
                       width_mult=0.125)
    run_training(off)
    assert not (tmp_path / "off" / "plots").exists()

    on = train_config(data_dir=root, onnx_out=str(tmp_path / "on" / "m.onnx"),
                      log_dir=str(tmp_path / "on_log"), plots=True, width_mult=0.125)
    run_training(on)

    produced = sorted(p.name for p in (tmp_path / "on" / "plots").glob("*.png"))
    assert produced == ["loss_curves.png", "metric_curves.png",
                        "predictions_random.png",
                        "predictions_selected_worst_f1.png"]


def test_two_stage_plots_curves_per_stage_and_predictions_from_the_final_model(
        tmp_path, stem_dataset, train_config):
    """Both stages get curves; only the stage-2 model can produce prediction panels."""
    from winmol_unet.training.run_train import run_two_stage

    gen = stem_dataset(tmp_path / "gen", n=4)
    spec = stem_dataset(tmp_path / "spec", n=4)
    cfg = train_config(gen_data_dir=str(gen), spec_data_dir=str(spec),
                       onnx_out=str(tmp_path / "out" / "m.onnx"),
                       log_dir=str(tmp_path / "log"), plots=True, width_mult=0.125)
    run_two_stage(cfg)

    produced = sorted(p.name for p in (tmp_path / "out" / "plots").glob("*.png"))
    assert produced == [
        "predictions_random.png",
        "predictions_selected_worst_f1.png",
        "stage1_loss_curves.png", "stage1_metric_curves.png",
        "stage2_loss_curves.png", "stage2_metric_curves.png",
    ]


# ---------------------------------------------------------------------------
# multiclass (species) segmentation: loss, metrics, dataset index masks
# ---------------------------------------------------------------------------

def test_ce_soft_f1_gradient_flows_and_ignore_index_is_inert():
    import torch
    from winmol_unet.training.losses import ce_soft_f1_loss
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    target[0, :2, :2] = 255                       # ignored region
    loss = ce_soft_f1_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    # flipping the class under an ignored pixel must not change the loss
    flipped = target.clone()
    flipped[0, :2, :2] = 255                      # stays ignored, same value
    other = target.clone()
    assert ce_soft_f1_loss(logits.detach(), other).item() == \
        ce_soft_f1_loss(logits.detach(), flipped).item()


def test_ce_soft_f1_near_zero_for_confident_correct_prediction():
    import torch
    from winmol_unet.training.losses import ce_soft_f1_loss
    target = torch.randint(0, 3, (1, 8, 8))
    logits = torch.full((1, 3, 8, 8), -20.0)
    logits.scatter_(1, target.unsqueeze(1), 20.0)  # +20 on the true class
    assert ce_soft_f1_loss(logits, target).item() < 0.01


def test_multiclass_counts_per_class_and_ignore():
    import torch
    from winmol_unet.training.metrics import multiclass_counts, prf
    # 1x3x2x2 logits: argmax = [[1, 2], [0, 1]]; target = [[1, 0], [255, 1]]
    logits = torch.zeros(1, 3, 2, 2)
    logits[0, 1, 0, 0] = 5; logits[0, 2, 0, 1] = 5
    logits[0, 0, 1, 0] = 5; logits[0, 1, 1, 1] = 5
    target = torch.tensor([[[1, 0], [255, 1]]])
    per = multiclass_counts(logits, target, num_classes=3)
    assert per[1] == (2, 0, 0)                     # class 1: both predicted hits
    assert per[2] == (0, 1, 0)                     # class 2: one false positive
    assert prf(*per[1])[2] == 1.0                  # class-1 F1 perfect


def test_stem_dataset_multiclass_returns_index_mask(tmp_path):
    import numpy as np
    from PIL import Image
    from winmol_unet.training.dataset import StemDataset
    (tmp_path / "train").mkdir(); (tmp_path / "mask").mkdir()
    rng = np.random.default_rng(0)
    for n in (1, 2):
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
            tmp_path / "train" / f"train{n}.jpeg")
        idx = np.zeros((32, 32), dtype=np.uint8)
        idx[4:12, 4:12] = 1; idx[20:28, 20:28] = 2
        Image.fromarray(idx, mode="P").save(tmp_path / "mask" / f"mask{n}.gif")
    ds = StemDataset(str(tmp_path / "train"), str(tmp_path / "mask"),
                     img_size=32, num_classes=3, cache=False)
    img, mask = ds[0]
    import torch
    assert mask.dtype == torch.int64
    assert set(torch.as_tensor(mask).unique().tolist()) <= {0, 1, 2}
    assert tuple(torch.as_tensor(mask).shape) == (32, 32)
