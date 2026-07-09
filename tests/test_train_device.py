import torch
from training.device import resolve_device


def test_resolve_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_returns_available_device():
    d = resolve_device("auto")
    assert d.type in {"mps", "cuda", "cpu"}
