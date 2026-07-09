"""Mean loss + hard-rounded metrics over a loader (no grad)."""
import torch

from .losses import bce_soft_f1_loss
from .metrics import precision, recall, f1


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    tot = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        tot["loss"] += bce_soft_f1_loss(logits, mask).item()
        tot["precision"] += precision(logits, mask)
        tot["recall"] += recall(logits, mask)
        tot["f1"] += f1(logits, mask)
        n += 1
    return {k: (v / n if n else 0.0) for k, v in tot.items()}
