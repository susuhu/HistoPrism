import os
import numpy as np
import torch
import random
from typing import Any


def get_nested(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current = config
    for i, key in enumerate(keys):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current if current is not None else default


def seed_everything(seed: int = 2025):
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): Random seed to use.
    """
    # Python built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For deterministic behavior (slower, but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables (for hash-based sources of randomness)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # CUDA >= 10.2 for full determinism
    )

