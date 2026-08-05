"""Train a Mask R-CNN stem-instance model on generated tiles; score on real TestDS.

    python scripts/train_instance.py --data <DS> --labels <LABELDIR> --mode modal|amodal \
        --test-data <TestDS> --out <RUNDIR> [--epochs 20]

Exists to compare label *formulations* on identical imagery: binary, modal
instances, and amodal instances all derive from one SceneSpec, so any
difference in results is the method, not label noise.

Scoring is deliberately binary. Real TestDS carries only binary annotation, so
instance predictions are unioned into one foreground mask and scored with the
same F1 the segmentation runs report — otherwise the three formulations would
not be comparable on real data at all. Instance quality itself is only
measurable on synthetic held-out tiles, where instance ground truth exists.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _load_amodal(path):
    d = np.load(path)
    shape = d["shape"]
    stack = np.unpackbits(d["masks"], axis=-1).astype(bool)
    return stack[:, : shape[1], : shape[2]]


class StemInstances(Dataset):
    """Generated tile -> per-stem binary masks + their boxes."""

    def __init__(self, data_dir, label_dir, mode, ids):
        self.data_dir, self.label_dir, self.mode, self.ids = data_dir, label_dir, mode, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        n = self.ids[i]
        img = Image.open(os.path.join(self.data_dir, "train", f"train{n}.jpeg")).convert("RGB")
        t = torch.from_numpy(np.asarray(img).transpose(2, 0, 1)).float() / 255.0

        if self.mode == "amodal":
            stack = _load_amodal(os.path.join(self.label_dir, f"amodal{n}.npz"))
        else:
            inst = np.asarray(Image.open(os.path.join(self.label_dir, f"inst{n}.png")))
            stack = np.stack([inst == v for v in np.unique(inst) if v > 0]) if inst.max() else \
                np.zeros((0,) + inst.shape, bool)

        keep, boxes = [], []
        for m in stack:
            ys, xs = np.where(m)
            # Mask R-CNN rejects degenerate boxes; a sliver clipped by the tile
            # edge can be 1px wide, so drop those rather than crash mid-epoch
            if len(ys) < 20 or xs.max() - xs.min() < 2 or ys.max() - ys.min() < 2:
                continue
            keep.append(m)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        if not keep:
            keep = [np.zeros(t.shape[1:], bool)]
            boxes = [[0, 0, 1, 1]]
        target = {
            "boxes": torch.as_tensor(np.array(boxes), dtype=torch.float32),
            "labels": torch.ones(len(keep), dtype=torch.int64),
            "masks": torch.as_tensor(np.array(keep), dtype=torch.uint8),
        }
        return t, target


def _collate(batch):
    return tuple(zip(*batch))


def build_model():
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, 2)          # background + stem
    inf_m = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(inf_m, 256, 2)
    m.roi_heads.detections_per_img = 60                            # dense windthrow tiles
    return m


@torch.no_grad()
def evaluate_binary(model, test_dir, device, score_thresh=0.5):
    """Union instance predictions and score against binary annotation."""
    model.eval()
    tp = fp = fn = 0
    names = sorted(glob.glob(os.path.join(test_dir, "train", "*.jpeg")))
    for p in names:
        n = os.path.splitext(os.path.basename(p))[0][len("train"):]
        img = Image.open(p).convert("RGB").resize((512, 512))
        t = torch.from_numpy(np.asarray(img).transpose(2, 0, 1)).float().div(255).to(device)
        out = model([t])[0]
        pred = np.zeros((512, 512), bool)
        for m, s in zip(out["masks"].cpu().numpy(), out["scores"].cpu().numpy()):
            if s >= score_thresh:
                pred |= m[0] >= 0.5
        gt_p = os.path.join(test_dir, "mask", f"mask{n}.gif")
        gt_im = Image.open(gt_p)
        gt_im.seek(0)
        gt = np.asarray(gt_im.convert("L").resize((512, 512), Image.NEAREST)) >= 128
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"f1": f1, "precision": prec, "recall": rec, "tiles": len(names)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="generated dataset (train/ + mask/)")
    p.add_argument("--labels", required=True, help="dir holding inst{N}.png / amodal{N}.npz")
    p.add_argument("--mode", choices=("modal", "amodal"), required=True)
    p.add_argument("--test-data", required=True, help="real held-out set with binary masks")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_tiles = len(glob.glob(os.path.join(a.data, "train", "*.jpeg")))
    ids = list(range(1, n_tiles + 1))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * a.val_fraction))
    train_ids, val_ids = ids[n_val:], ids[:n_val]

    ds = StemInstances(a.data, a.labels, a.mode, train_ids)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=8,
                    collate_fn=_collate)
    model = build_model().to(device)
    opt = torch.optim.SGD([p_ for p_ in model.parameters() if p_.requires_grad],
                          lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    os.makedirs(a.out, exist_ok=True)
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for imgs, targets in dl:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        sched.step()
        print(f"epoch {epoch + 1}/{a.epochs} loss {total / max(len(dl), 1):.4f}", flush=True)

    m = evaluate_binary(model, a.test_data, device)
    print(f"TEST ({m['tiles']} tiles) mode={a.mode}: F1={m['f1']:.4f} "
          f"P={m['precision']:.4f} R={m['recall']:.4f}")
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "test_results.md"), "w") as f:
        f.write(f"# Instance ({a.mode})\n\n| metric | value |\n|---|---:|\n")
        for k in ("f1", "precision", "recall"):
            f.write(f"| {k} | {m[k]:.4f} |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
