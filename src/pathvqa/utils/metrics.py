"""Evaluation metrics: calibration (ECE), uncertainty, and statistical significance."""

import torch
import numpy as np
from typing import Dict, List, Tuple


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor,
                                n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """Compute Expected Calibration Error (ECE) across n_bins confidence bins."""
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0  # ECE = Σ_k (|acc_k - conf_k| * n_k) / N
    bin_stats = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > low) & (confidences <= high)
        if mask.any():
            bin_acc = accuracies[mask].float().mean().item()
            bin_conf = confidences[mask].mean().item()
            bin_size = mask.sum().item()
            ece += abs(bin_acc - bin_conf) * bin_size / len(labels)
            bin_stats.append({
                "bin_lower": low.item(),
                "bin_upper": high.item(),
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "count": bin_size,
            })
        else:
            bin_stats.append({
                "bin_lower": low.item(),
                "bin_upper": high.item(),
                "accuracy": 0.0,
                "confidence": 0.0,
                "count": 0,
            })

    return ece, bin_stats


def temperature_scaling(logits: torch.Tensor, labels: torch.Tensor,
                         lr: float = 0.01, max_iter: int = 100) -> float:
    """Learn optimal temperature for post-hoc temperature scaling.

    Optimizes T so that softmax(logits / T) is better calibrated.
    Uses L-BFGS on cross-entropy loss.
    """
    T = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled_logits = logits / T
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    return T.item()


def compute_uncertainty(logits: torch.Tensor, method: str = "max_softmax") -> torch.Tensor:
    """Prediction uncertainty via max-softmax or entropy."""
    probs = torch.softmax(logits, dim=1)

    if method == "max_softmax":
        # U = 1 - max(p)  →  0 when confident, ~1 when uniform
        return 1.0 - probs.max(dim=1).values
    elif method == "entropy":
        # H(p) = -Σ p(c) log p(c)  →  0 when certain, log(C) when uniform
        log_probs = torch.log(probs + 1e-10)
        return -(probs * log_probs).sum(dim=1)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


def paired_bootstrap_test(acc1: float, acc2: float, n1: int, n2: int,
                           n_bootstrap: int = 10000, seed: int = 42) -> float:
    """Paired bootstrap test for comparing two accuracy values (approximate, conservative)."""
    rng = np.random.RandomState(seed)
    diff_observed = acc1 - acc2
    count = 0
    # p-value: fraction of bootstrap samples where |Δ_boot| ≥ |Δ_observed|
    for _ in range(n_bootstrap):
        boot1 = rng.binomial(n1, acc1) / n1
        boot2 = rng.binomial(n2, acc2) / n2
        if abs(boot1 - boot2) >= abs(diff_observed):
            count += 1

    return count / n_bootstrap
