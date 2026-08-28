"""Seeds torch, numpy, random, and CUDA for deterministic runs."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs (torch/numpy/random/CUDA).

    If deterministic=True, forces cuDNN deterministic mode and requests
    deterministic algorithms (best-effort: ``warn_only`` avoids hard failures
    for ops with no deterministic CUDA implementation) and disables
    ``cudnn.benchmark`` (the two are incompatible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required by cuBLAS when deterministic algorithms are requested.
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        # Reset both flags so callers get a known state regardless of what a
        # previous deterministic run left behind.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """Per-worker seed derivation for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    set_seed(42)
    print("Seed test passed!")
