"""Device selection for training. 'auto' prefers Apple Metal (MPS), then CUDA,
then CPU. Export always happens on CPU (the exporters read weights via numpy)."""
import torch


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
