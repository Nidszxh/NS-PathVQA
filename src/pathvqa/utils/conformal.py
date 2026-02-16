"""Split conformal prediction for coverage-guaranteed prediction sets.

Provides formal uncertainty guarantees: given a significance level α,
the prediction set C(x) contains the true label with probability ≥ 1-α.
"""

from typing import Dict, List, Optional
import torch
import numpy as np


class ConformalPredictor:
    """Split conformal predictor with calibrated prediction sets.

    Calibration: compute nonconformity scores s_i = 1 - P(y_true | x_i)
    on a held-out calibration set, then derive threshold τ.
    """

    def __init__(self, alpha: float = 0.1):
        """Args:
            alpha: significance level (e.g., 0.1 → 90% coverage guarantee).
        """
        self.alpha = alpha
        self.tau: Optional[float] = None
        self.calibration_stats: Dict = {}

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute conformal threshold from calibration data.

        Args:
            logits: (N, C) raw logits from model on calibration set
            labels: (N,) true label indices

        Returns:
            τ threshold such that P(y_true ∈ C(x)) ≥ 1 - α
        """
        probs = torch.softmax(logits, dim=1)
        n = probs.size(0)

        # Nonconformity scores: s_i = 1 - P(y_true | x_i)
        probs_true = probs[torch.arange(n), labels]
        scores = 1.0 - probs_true  # (N,)

        # Quantile: τ = ceil((1-α)(n+1)) / n quantile of scores
        # This is the standard split conformal threshold (Vovk et al. 2005)
        quantile_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        quantile_level = min(quantile_level, 1.0)
        tau = torch.quantile(scores, quantile_level).item()

        self.tau = tau

        # Compute coverage on calibration set for verification
        pred_sets = self.predict_set(logits, tau)
        coverage = sum(
            labels[i].item() in pred_sets[i]
            for i in range(n)
        ) / n
        avg_set_size = np.mean([len(s) for s in pred_sets])

        self.calibration_stats = {
            "tau": tau,
            "alpha": self.alpha,
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "n_calibration": n,
            "quantile_level": quantile_level,
        }
        return tau

    def predict_set(self, logits: torch.Tensor, tau: Optional[float] = None) -> List[set]:
        """Construct prediction sets C(x) = {a : softmax(logits)[a] ≥ 1 - τ}.

        Args:
            logits: (B, C) raw logits
            tau: threshold (uses self.tau if None)

        Returns:
            list of sets, each containing the predicted class indices
        """
        if tau is None:
            tau = self.tau
        if tau is None:
            raise ValueError("No threshold τ — call calibrate() first or provide tau")

        probs = torch.softmax(logits, dim=1)
        threshold = 1.0 - tau
        # Each set = {c : p(c) >= threshold}
        pred_sets = []
        for i in range(probs.size(0)):
            mask = probs[i] >= threshold
            pred_sets.append(set(mask.nonzero(as_tuple=True)[0].tolist()))
        return pred_sets

    def is_uncertain(self, logits: torch.Tensor, tau: Optional[float] = None,
                     max_set_size: int = 1) -> torch.Tensor:
        """Check if predictions are uncertain (prediction set too large).

        Args:
            logits: (B, C) raw logits
            tau: threshold (uses self.tau if None)
            max_set_size: if |C(x)| > this, mark as uncertain

        Returns:
            (B,) boolean tensor, True where uncertain
        """
        pred_sets = self.predict_set(logits, tau)
        return torch.tensor([len(s) > max_set_size for s in pred_sets])

    def get_route_signal(self, logits: torch.Tensor, tau: Optional[float] = None,
                         max_set_size: int = 1) -> Dict[str, torch.Tensor]:
        """Get routing signals for uncertainty-aware fusion.

        Returns:
            dict with:
                prediction_sets: list of sets
                uncertain: (B,) boolean tensor
                set_sizes: (B,) tensor of prediction set sizes
                max_prob: (B,) tensor of max probability
        """
        probs = torch.softmax(logits, dim=1)
        max_prob, max_pred = probs.max(dim=1)
        pred_sets = self.predict_set(logits, tau)
        set_sizes = torch.tensor([len(s) for s in pred_sets])
        uncertain = set_sizes > max_set_size

        return {
            "prediction_sets": pred_sets,
            "uncertain": uncertain,
            "set_sizes": set_sizes,
            "max_prob": max_prob,
            "max_pred": max_pred,
        }


def compute_conformal_metrics(logits: torch.Tensor, labels: torch.Tensor,
                              alphas: List[float] = None) -> Dict[str, Dict]:
    """Compute conformal prediction metrics at multiple alpha levels.

    Args:
        logits: (N, C) raw logits on calibration set
        labels: (N,) true labels
        alphas: list of significance levels to evaluate

    Returns:
        dict mapping alpha → {tau, coverage, avg_set_size}
    """
    if alphas is None:
        alphas = [0.05, 0.1, 0.2]

    results = {}
    for alpha in alphas:
        cp = ConformalPredictor(alpha=alpha)
        cp.calibrate(logits, labels)
        results[alpha] = cp.calibration_stats.copy()
    return results
