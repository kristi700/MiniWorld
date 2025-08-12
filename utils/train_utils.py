import torch

from torchvision import transforms as T

def setup_device():
    """Setup and return the appropriate device (CUDA/CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_transforms(config):
    transform_cfg = config["transforms"]
    transforms_dict = {}
    for key, sequence in transform_cfg.items():
        transforms_dict[key] = _make_transforms(sequence)
    return transforms_dict

def _make_transforms(sequence):
    ops = []
    for entry in sequence:
        cls = getattr(T, entry["name"])
        params = entry.get("params") or {}
        ops.append(cls(**params))
    return T.Compose(ops)