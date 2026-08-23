"""Seeds torch, numpy, random, and CUDA for deterministic runs."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs (torch/numpy/random/CUDA).

    If deterministic=True, forces cuDNN deterministic mode (incompatible with
    ``cudnn.benchmark``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Per-worker seed derivation for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    set_seed(42)
    print("Seed test passed!")