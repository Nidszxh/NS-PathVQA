"""Split conformal prediction for coverage-guaranteed prediction sets.

Provides formal uncertainty guarantees: given a significance level α,
the prediction set C(x) contains the true label with probability ≥ 1-α.
"""

from typing import Any, Dict, List, Optional, Set
import torch
import numpy as np


class ConformalPredictor:
    """Split conformal predictor with calibrated prediction sets.

    Calibration: compute nonconformity scores s_i = 1 - P(y_true | x_i)
    on a held-out calibration set, then derive threshold τ.
    """

    def __init__(self, alpha: float = 0.1, max_set_size: int = 1):
        """Configure the significance level and the set-size uncertainty flag.

        ``alpha`` is the conformal significance level (0.1 → 90% coverage).
        """
        self.alpha = alpha
        self.max_set_size = max_set_size
        self.tau: Optional[float] = None
        self.calibration_stats: Dict[str, Any] = {}

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

        # Exact split-conformal threshold (Vovk et al. 2005):
        # τ = the ⌈(n+1)(1-α)⌉-th smallest nonconformity score (order statistic,
        # no interpolation). This guarantees P(y_true ∈ C(x)) ≥ 1 - α.
        k = int(np.ceil((1 - self.alpha) * (n + 1)))
        k = min(k, n)
        tau = torch.sort(scores).values[k - 1].item()

        self.tau = tau

        # Compute coverage on calibration set for verification
        pred_sets = self.predict_set(logits, tau)
        coverage = sum(labels[i].item() in pred_sets[i] for i in range(n)) / n
        avg_set_size = np.mean([len(s) for s in pred_sets])

        self.calibration_stats = {
            "tau": tau,
            "alpha": self.alpha,
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "n_calibration": n,
            "quantile_rank": k,
        }
        return tau

    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Measure coverage / set size on a held-out split (no recalibration).

        Used to report the *out-of-sample* conformal guarantee: calibrating and
        measuring on the same data yields in-sample coverage that overstates the
        real guarantee.
        """
        if self.tau is None:
            raise ValueError("No threshold τ — call calibrate() first")
        n = logits.size(0)
        pred_sets = self.predict_set(logits)
        coverage = sum(labels[i].item() in pred_sets[i] for i in range(n)) / max(1, n)
        avg_set_size = float(np.mean([len(s) for s in pred_sets])) if pred_sets else 0.0
        set_sizes = [len(s) for s in pred_sets]
        return {
            "tau": self.tau,
            "alpha": self.alpha,
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "n_evaluation": n,
            "uncertain_rate": (sum(1 for s in set_sizes if s > self.max_set_size) / n if n else 0.0),
        }

    def predict_set(self, logits: torch.Tensor, tau: Optional[float] = None) -> List[Set[int]]:
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

    def is_uncertain(
        self, logits: torch.Tensor, tau: Optional[float] = None, max_set_size: Optional[int] = None
    ) -> torch.Tensor:
        """Check if predictions are uncertain (prediction set too large).

        Args:
            logits: (B, C) raw logits
            tau: threshold (uses self.tau if None)
            max_set_size: if |C(x)| > this, mark as uncertain (uses
                ``self.max_set_size`` when None)

        Returns:
            (B,) boolean tensor, True where uncertain
        """
        if max_set_size is None:
            max_set_size = self.max_set_size
        pred_sets = self.predict_set(logits, tau)
        # Keep the flag on the logits' device so callers can combine it with
        # CUDA-resident tensors without a device-mismatch error.
        return torch.tensor([len(s) > max_set_size for s in pred_sets], device=logits.device)

    def get_route_signal(
        self, logits: torch.Tensor, tau: Optional[float] = None, max_set_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get routing signals for uncertainty-aware fusion.

        Returns:
            dict with:
                prediction_sets: list of sets
                uncertain: (B,) boolean tensor
                set_sizes: (B,) tensor of prediction set sizes
                max_prob: (B,) tensor of max probability
                max_pred: (B,) most likely class index
        """
        if max_set_size is None:
            max_set_size = self.max_set_size
        probs = torch.softmax(logits, dim=1)
        max_prob, max_pred = probs.max(dim=1)
        pred_sets = self.predict_set(logits, tau)
        set_sizes = torch.tensor([len(s) for s in pred_sets], device=logits.device)
        uncertain = set_sizes > max_set_size

        return {
            "prediction_sets": pred_sets,
            "uncertain": uncertain,
            "set_sizes": set_sizes,
            "max_prob": max_prob,
            "max_pred": max_pred,
        }


def compute_conformal_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alphas: Optional[List[float]] = None,
    max_set_size: int = 1,
    eval_logits: Optional[torch.Tensor] = None,
    eval_labels: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> Dict[float, Dict[str, Any]]:
    """Compute *out-of-sample* conformal prediction metrics at multiple alphas.

    Calibration and coverage measurement MUST use disjoint data; measuring
    coverage on the same points used to fit τ overstates the guarantee. When
    ``eval_logits``/``eval_labels`` are not supplied, the inputs are split
    deterministically into two halves (calibration / evaluation).

    Args:
        logits: (N, C) raw logits available for calibration + evaluation
        labels: (N,) true labels
        alphas: list of significance levels to evaluate
        max_set_size: set-size limit for the ``uncertain_rate`` statistic
        eval_logits/eval_labels: optional explicit held-out split
        seed: shuffle seed for the internal split

    Returns:
        dict mapping alpha → {tau, coverage, avg_set_size, n_calibration,
        n_evaluation, calibration_coverage, uncertain_rate}
    """
    if alphas is None:
        alphas = [0.05, 0.1, 0.2]

    if eval_logits is None or eval_labels is None:
        n = logits.size(0)
        if n < 2:
            raise ValueError("Need at least 2 samples to split calibration/evaluation")
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)
        half = n // 2
        cal_idx, eval_idx = perm[:half], perm[half:]
        cal_logits, cal_labels = logits[cal_idx], labels[cal_idx]
        holdout_logits, holdout_labels = logits[eval_idx], labels[eval_idx]
    else:
        cal_logits, cal_labels = logits, labels
        holdout_logits, holdout_labels = eval_logits, eval_labels

    results: Dict[float, Dict[str, Any]] = {}
    for alpha in alphas:
        cp = ConformalPredictor(alpha=alpha, max_set_size=max_set_size)
        cp.calibrate(cal_logits, cal_labels)
        metrics = cp.evaluate(holdout_logits, holdout_labels)
        # Keep the in-sample calibration figure for reference/monitoring.
        metrics["calibration_coverage"] = cp.calibration_stats["coverage"]
        results[alpha] = metrics
    return results
