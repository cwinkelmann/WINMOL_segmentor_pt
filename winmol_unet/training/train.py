"""Single-stage train/validate loop: Adam, best-val-loss checkpoint, early stop,
TensorBoard logging."""
import os

import torch

from .device import resolve_device
from .evaluate import evaluate
from .losses import LOSS_COMPONENTS, LOSSES, bce_soft_f1_loss, soften_targets
from .run_logger import RunLogger


def train_one_run(model, train_loader, val_loader, cfg, patience=None,
                  ckpt_name="best.pt", log_dir=None, optimizer=None):
    # patience/ckpt_name/log_dir default to the single-stage config; two-stage passes
    # per-stage values so stage 2 doesn't clobber stage 1's checkpoint or TB curves.
    # optimizer: pass a shared optimizer to CARRY its state (and decayed lr) across stages,
    # mirroring the R pipeline which compiles one optimizer once for both cost_train stages.
    # When None (single-stage) a fresh Adam is created.
    patience = cfg.patience if patience is None else patience
    loss_fn = LOSSES[getattr(cfg, "loss", "bce_soft_f1")]
    eps = getattr(cfg, "label_smoothing", 0.0)
    band = getattr(cfg, "smooth_band_px", 2)
    log_dir = cfg.log_dir if log_dir is None else log_dir
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(cfg.checkpoint_dir, ckpt_name)
    logger = RunLogger(log_dir, cfg.wandb, cfg.wandb_project, cfg.wandb_run_name)
    device = resolve_device(cfg.device)
    model.to(device)
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # Fresh ReduceLROnPlateau per stage (matches R's fresh callbacks_train1/2), but it reads
    # and writes the shared optimizer's lr, so a decayed lr carries into the next stage.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=2, threshold=1e-4)

    def _val_panel(n=5):
        """n fixed val tiles: image / ground truth / predicted probability.

        Indices are evenly spaced over the val set, so a val split assembled from
        several sources (beech blocks first, Tegel appended after) shows tiles from
        each rather than n neighbours from one corner of one site.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None                      # panels are optional, never fatal
        ds = val_loader.dataset
        if len(ds) == 0:
            return None
        idxs = {int(round(i * (len(ds) - 1) / max(1, n - 1))) for i in range(n)}
        # Pin extra tiles of interest (e.g. suspected under-annotation cases) via
        # WINMOL_VAL_PANEL_IDS="2072,2146" -- dataset indices, comma-separated.
        extra = os.environ.get("WINMOL_VAL_PANEL_IDS", "")
        idxs |= {int(x) for x in extra.split(",") if x.strip().isdigit()
                 and int(x) < len(ds)}
        idxs = sorted(idxs)
        fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)),
                                 squeeze=False)
        model.eval()
        with torch.no_grad():
            for r, i in enumerate(idxs):
                img, mask = ds[i]
                prob = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu()
                axes[r][0].imshow(img.permute(1, 2, 0).cpu().numpy())
                axes[r][1].imshow(mask.squeeze().cpu().numpy(), cmap="gray",
                                  vmin=0, vmax=1)
                axes[r][2].imshow(prob.numpy(), cmap="magma", vmin=0, vmax=1)
                axes[r][0].set_ylabel(f"val[{i}]", fontsize=8)
                for ax in axes[r]:
                    ax.set_xticks([]); ax.set_yticks([])
        for ax, t in zip(axes[0], ("image", "ground truth", "prediction")):
            ax.set_title(t, fontsize=9)
        fig.tight_layout()
        return fig

    best_val = float("inf")
    since_improve = 0
    try:
        for epoch in range(cfg.epochs):
            model.train()
            running, nb = 0.0, 0
            comp_fn = LOSS_COMPONENTS.get(getattr(cfg, "loss", None))
            comp_sums = {}
            for img, mask in train_loader:
                img, mask = img.to(device), mask.to(device)
                opt.zero_grad()
                # soft targets for the loss only; metrics stay on the hard mask
                logits = model(img)
                soft = soften_targets(mask, eps, band)
                loss = loss_fn(logits, soft)
                loss.backward()
                opt.step()
                running += loss.item()
                nb += 1
                if comp_fn is not None:
                    # Same logits, no second forward pass; elementwise ops only.
                    with torch.no_grad():
                        for k, v in comp_fn(logits, soft).items():
                            comp_sums[k] = comp_sums.get(k, 0.0) + v.item()
            train_loss = running / nb if nb else 0.0

            val = evaluate(model, val_loader)
            sched.step(val["loss"])              # reduce LR on val-loss plateau (R schedule)
            comp_scalars = {f"train/loss_{k}": v / nb for k, v in comp_sums.items()} if nb else {}
            logger.log_scalars({
                **comp_scalars,
                "train/loss": train_loss,
                **{f"val/{k}": v for k, v in val.items()},
                "lr": opt.param_groups[0]["lr"],
            }, epoch)

            fig = _val_panel()
            if fig is not None:
                logger.log_figure("val/examples", fig, epoch)

            if val["loss"] < best_val:
                best_val = val["loss"]
                since_improve = 0
                torch.save(model.state_dict(), ckpt)
            else:
                since_improve += 1
                if since_improve >= patience:
                    break
    finally:
        logger.close()   # always flush TB + wandb.finish, even on error/early exit

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model
