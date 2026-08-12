"""CLI: single-stage or two-stage (GenDS -> SpecDS) training + export.

Single-stage: --data-dir. Two-stage fine-tune: --gen-data-dir + --spec-data-dir
(stage 1 on the general set, stage 2 fine-tunes the SAME model on the species set).
"""
import argparse
import os
import warnings

import torch
from torch.utils.data import DataLoader

from winmol_unet.export import export_to_onnx, export_to_pt
# NOTE: winmol_unet.export_keras is imported lazily inside _export (only when
# cfg.export_keras is set) because it pulls in TensorFlow, which the lean training
# image (WITH_KERAS=0) does not install — importing it at module level would break
# `python -m training.run_train` on any TF-less environment. Mirrors the lazy smp
# import in model_factory.

from .augment import build_augmentation
from .config import TrainConfig
from .dataset import StemDataset, TilingStemDataset, split_ids
from .evaluate import evaluate
from .model_factory import build_model
from .train import train_one_run


def _worker_init(_):
    # Reseed the albumentations transform per worker so augmentation streams are
    # distinct across workers (its per-Compose RNG is not reached by torch's base seed).
    info = torch.utils.data.get_worker_info()
    tf = getattr(info.dataset, "transform", None)
    if tf is not None:
        tf.set_random_seed(info.seed % (2 ** 31 - 1))


def _eval_dataset(image_dir, mask_dir, cfg, ids=None):
    """Validation/test dataset: deterministic native-res grid tiling when cfg.eval_tiling,
    else the plain resize-to-img_size StemDataset. Never augments."""
    if cfg.eval_tiling:
        return TilingStemDataset(image_dir, mask_dir, tile=cfg.img_size, ids=ids,
                                 cache=cfg.cache_dataset)
    return StemDataset(image_dir, mask_dir, cfg.img_size, transform=None, ids=ids,
                       cache=cfg.cache_dataset)


def _build_loaders(image_dir, mask_dir, cfg, transform, val_image_dir=None, val_mask_dir=None):
    if cfg.num_workers > 0 and cfg.cache_dataset:
        warnings.warn("num_workers>0 with cache_dataset=True caches per-worker and discards "
                      "it each epoch; use --no-cache-dataset for large datasets.", stacklevel=2)
    # multiscale trains on native-res tiles (the transform's RandomSizedCrop resizes to
    # img_size); otherwise images are resized to img_size at load time as before.
    train_resize = not cfg.multiscale
    if val_image_dir is not None:
        # pre-materialized fixed split: train on all of image_dir, validate on the given dir
        train_ds = StemDataset(image_dir, mask_dir, cfg.img_size, transform=transform,
                               cache=cfg.cache_dataset, resize=train_resize)
        val_ds = _eval_dataset(val_image_dir, val_mask_dir, cfg)
    else:
        train_ids, val_ids = split_ids(image_dir, mask_dir, cfg.val_fraction, cfg.seed)
        train_ds = StemDataset(image_dir, mask_dir, cfg.img_size, transform=transform,
                               ids=train_ids, cache=cfg.cache_dataset, resize=train_resize)
        val_ds = _eval_dataset(image_dir, mask_dir, cfg, ids=val_ids)
    # drop_last avoids a trailing batch of 1 (breaks BatchNorm in DeepLabV3+ ASPP
    # [N,C,1,1]) — only when there is more than one batch's worth, so a tiny set
    # isn't zeroed out. num_workers>0 is only safe with cache_dataset=False.
    drop_last = len(train_ds) > cfg.batch_size
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=drop_last,
                              worker_init_fn=_worker_init)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                            worker_init_fn=_worker_init)
    return train_loader, val_loader


def _validate_export(cfg):
    # Fail fast (before any training) rather than after a full run.
    if cfg.export_keras and cfg.arch != "unet":
        raise ValueError(
            f"--export-keras is UNet-only (the Keras mirror is UNet-specific); "
            f"arch={cfg.arch!r} exports ONNX + .pt")


def _export(model, cfg):
    model.cpu()                                 # exporters read weights via CPU numpy
    for p in (cfg.pt_out, cfg.hdf5_out, cfg.keras_out, cfg.onnx_out):
        if p:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    if cfg.pt_out:
        export_to_pt(model, cfg.pt_out)
    # ONNX is the uniform path — every architecture loads the same way via OnnxSegmenter.
    # The Keras .hdf5/.keras mirror is UNet-specific and opt-in (validated upfront).
    if cfg.export_keras:
        from winmol_unet.export_keras import export_to_keras, export_to_keras_hdf5  # lazy: pulls in TF
        export_to_keras_hdf5(model, cfg.hdf5_out, dropout=cfg.dropout)
        if cfg.keras_out:
            export_to_keras(model, cfg.keras_out, dropout=cfg.dropout)
    export_to_onnx(model, cfg.onnx_out)


def _run_test(model, cfg):
    """R cost_eval: evaluate the trained model on a held-out TestDS (no augmentation).
    Prints, logs to TensorBoard, and writes test_results.md. Returns metrics or None."""
    if not cfg.test_data_dir:
        return None
    from torch.utils.tensorboard import SummaryWriter
    test_ds = _eval_dataset(os.path.join(cfg.test_data_dir, "train"),
                            os.path.join(cfg.test_data_dir, "mask"), cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    m = evaluate(model, test_loader)             # model on its current device (no aug)
    writer = SummaryWriter(os.path.join(cfg.log_dir, "test"))
    for k, v in m.items():
        writer.add_scalar(f"test/{k}", v, 0)
    writer.close()
    out_dir = os.path.dirname(cfg.onnx_out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_results.md"), "w") as f:
        f.write(f"# Test results\n\n**TestDS:** `{cfg.test_data_dir}` "
                f"({len(test_ds)} tiles) | **arch:** {cfg.arch}\n\n"
                "| metric | value |\n|--------|------:|\n")
        for k in ("f1", "precision", "recall", "loss"):
            f.write(f"| {k} | {m[k]:.4f} |\n")
    print(f"TEST ({len(test_ds)} tiles): F1={m['f1']:.4f} P={m['precision']:.4f} "
          f"R={m['recall']:.4f} loss={m['loss']:.4f}")
    return m


def run_training(cfg):
    _validate_export(cfg)                        # fail fast before training
    torch.manual_seed(cfg.seed)
    transform = build_augmentation(cfg)         # seeded internally via cfg.seed
    train_loader, val_loader = _build_loaders(
        cfg.image_dir, cfg.mask_dir, cfg, transform,
        val_image_dir=cfg.val_image_dir, val_mask_dir=cfg.val_mask_dir)
    model = build_model(cfg.arch, dropout=cfg.dropout, encoder=cfg.encoder,
                        encoder_weights=cfg.encoder_weights, width_mult=cfg.width_mult)
    train_one_run(model, train_loader, val_loader, cfg)
    val_metrics = evaluate(model, val_loader)   # on training device
    _run_test(model, cfg)                        # held-out TestDS eval (if --test-data-dir)
    _export(model, cfg)
    return val_metrics


def run_two_stage(cfg):
    """Two-stage fine-tune: train on GenDS (stage 1), then fine-tune the same model
    on SpecDS (stage 2). Final metrics + export come from the stage-2 (species) model."""
    _validate_export(cfg)                        # fail fast before either stage
    torch.manual_seed(cfg.seed)
    transform = build_augmentation(cfg)
    model = build_model(cfg.arch, dropout=cfg.dropout, encoder=cfg.encoder,
                        encoder_weights=cfg.encoder_weights, width_mult=cfg.width_mult)
    # ONE optimizer shared across both stages (Adam moment estimates carry over, as they do in
    # the R pipeline's single compiled optimizer), BUT the learning rate is reset to cfg.lr at
    # the start of stage 2 — mirroring the R fix `k_set_value(optimizer$lr, BASE_LR)`. Without
    # the reset, the LR that stage-1's ReduceLROnPlateau decayed (down to ~1e-6) carries into
    # stage 2 and cripples fine-tuning. Move params to the device first so the optimizer binds
    # device tensors.
    from .device import resolve_device
    model.to(resolve_device(cfg.device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    gen_train, gen_val = _build_loaders(cfg.gen_image_dir, cfg.gen_mask_dir, cfg, transform)
    train_one_run(model, gen_train, gen_val, cfg, patience=cfg.patience_stage1,
                  ckpt_name="best_stage1.pt", log_dir=os.path.join(cfg.log_dir, "stage1"),
                  optimizer=opt)

    for g in opt.param_groups:                   # reset LR for stage 2 (mirrors R's k_set_value)
        g["lr"] = cfg.lr
    # Stage 2 must validate on the SAME held-out split as single-stage runs, or its
    # checkpoint selection is not comparable to theirs. Omitting these fell back to
    # split_ids() — a random split of the spec tiles, which is both a different signal
    # (in-training-distribution, so ~+5 F1 higher) and a leaky one, since the samplers
    # oversample and adjacent tiles overlap. `--val-data-dir` was silently ignored here.
    spec_train, spec_val = _build_loaders(cfg.spec_image_dir, cfg.spec_mask_dir, cfg, transform,
                                          val_image_dir=cfg.val_image_dir,
                                          val_mask_dir=cfg.val_mask_dir)
    train_one_run(model, spec_train, spec_val, cfg, patience=cfg.patience_stage2,
                  ckpt_name="best_stage2.pt", log_dir=os.path.join(cfg.log_dir, "stage2"),
                  optimizer=opt)

    val_metrics = evaluate(model, spec_val)     # final = species val split
    _run_test(model, cfg)                        # held-out TestDS eval (if --test-data-dir)
    _export(model, cfg)
    return val_metrics


def config_from_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="single-stage dataset dir")
    p.add_argument("--val-data-dir", default=None,
                   help="single-stage: fixed val set (else 80/20 split of --data-dir)")
    p.add_argument("--test-data-dir", default=None,
                   help="held-out test set evaluated after training (writes test_results.md)")
    p.add_argument("--gen-data-dir", default=None, help="two-stage: general (stage 1) dir")
    p.add_argument("--spec-data-dir", default=None, help="two-stage: species (stage 2) dir")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--patience-stage1", type=int, default=3)
    p.add_argument("--patience-stage2", type=int, default=5)
    p.add_argument("--cache-dataset", dest="cache_dataset", action="store_true", default=True)
    p.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--aug-hflip-p", type=float, default=0.5)
    p.add_argument("--aug-vflip-p", type=float, default=0.5)
    p.add_argument("--aug-rotate-p", type=float, default=0.0)
    p.add_argument("--aug-rotate-limit", type=float, default=15.0)
    p.add_argument("--aug-bc-p", type=float, default=0.5)
    p.add_argument("--aug-brightness-limit", type=float, default=0.2)
    p.add_argument("--aug-contrast-limit", type=float, default=0.2)
    p.add_argument("--aug-hsv-p", type=float, default=0.5)
    p.add_argument("--multiscale", action="store_true",
                   help="multi-scale RandomSizedCrop on native-res tiles (needs a native-res "
                        "dataset, e.g. --native-px 1024 from sample_training_tiles.py)")
    p.add_argument("--crop-min-px", type=int, default=400, help="multiscale: min crop side")
    p.add_argument("--crop-max-px", type=int, default=1024, help="multiscale: max crop side")
    p.add_argument("--eval-tiling", action="store_true",
                   help="deterministic native-res grid tiling for val/test (full coverage)")
    p.add_argument("--arch", default="unet",
                   help="unet|deeplabv3plus|hrnet|segformer|convnext|fpn|pan|dpt")
    p.add_argument("--width-mult", type=float, default=1.0,
                   help="UNet channel-width scale (1.0=full; e.g. 0.5 = ~1/4 params, faster CPU)")
    p.add_argument("--encoder", default=None,
                   help="smp encoder; default is per-arch (e.g. mit_b0 for segformer, mit_b2/b3/b5 for larger)")
    p.add_argument("--loss", default="bce_soft_f1", choices=("bce_soft_f1", "bce"),
                   help="bce = what R effectively optimises (its F1 term is rounded, "
                        "so it has no gradient)")
    p.add_argument("--encoder-weights", default=None, help="None or 'imagenet' (needs network)")
    p.add_argument("--export-keras", action="store_true",
                   help="also emit Keras .hdf5/.keras (UNet only; ONNX is always exported)")
    a = p.parse_args(argv)
    two_stage = a.gen_data_dir and a.spec_data_dir
    if not a.data_dir and not two_stage:
        p.error("provide --data-dir (single-stage) or both --gen-data-dir and --spec-data-dir")
    return TrainConfig(
        data_dir=a.data_dir or "", val_data_dir=a.val_data_dir,
        test_data_dir=a.test_data_dir,
        gen_data_dir=a.gen_data_dir, spec_data_dir=a.spec_data_dir,
        checkpoint_dir=os.path.join(a.out_dir, "checkpoints"),
        log_dir=os.path.join(a.out_dir, "logs"),
        pt_out=os.path.join(a.out_dir, "model.pt"),
        hdf5_out=os.path.join(a.out_dir, "model.hdf5"),
        keras_out=os.path.join(a.out_dir, "model.keras"),
        onnx_out=os.path.join(a.out_dir, "model.onnx"),
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
        patience_stage1=a.patience_stage1, patience_stage2=a.patience_stage2,
        cache_dataset=a.cache_dataset, num_workers=a.num_workers,
        wandb=a.wandb, wandb_project=a.wandb_project, wandb_run_name=a.wandb_run_name,
        aug_hflip_p=a.aug_hflip_p, aug_vflip_p=a.aug_vflip_p,
        aug_rotate_p=a.aug_rotate_p, aug_rotate_limit=a.aug_rotate_limit,
        aug_bc_p=a.aug_bc_p, aug_brightness_limit=a.aug_brightness_limit,
        aug_contrast_limit=a.aug_contrast_limit, aug_hsv_p=a.aug_hsv_p,
        multiscale=a.multiscale, crop_min_px=a.crop_min_px, crop_max_px=a.crop_max_px,
        eval_tiling=a.eval_tiling,
        arch=a.arch, width_mult=a.width_mult, loss=a.loss,
        encoder=a.encoder, encoder_weights=a.encoder_weights,
        export_keras=a.export_keras,
    )


def main():
    cfg = config_from_args()
    result = run_two_stage(cfg) if (cfg.gen_data_dir and cfg.spec_data_dir) else run_training(cfg)
    print(result)


if __name__ == "__main__":
    main()
