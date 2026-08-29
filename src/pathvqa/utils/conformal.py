"""Split conformal prediction sets for exchangeable examples or groups.

The finite-sample marginal coverage statement requires exchangeability of the
calibration and future units. For grouped observations, callers can calibrate
on a maximum nonconformity score per group to target simultaneous coverage of
all rows in a new group. The guarantee does not apply automatically under
distribution shift or to arbitrary dependent rows.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import torch
import numpy as np


def _validate_logits_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Validate classification inputs and align labels with the logits device."""
    if logits.ndim != 2 or logits.size(1) == 0:
        raise ValueError("logits must have shape (examples, classes) with at least one class")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("labels must have shape (examples,) matching logits")
    if labels.is_floating_point() or labels.is_complex():
        raise ValueError("labels must contain integer class indices")
    labels = labels.to(device=logits.device, dtype=torch.long)
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= logits.size(1)):
        raise ValueError("labels contain a class index outside the logits vocabulary")
    return labels


def groupwise_split_indices(group_ids: Sequence[str], seed: int = 0) -> Tuple[List[int], List[int]]:
    """Deterministically split examples while keeping every group intact.

    The split is approximately half by number of groups, not number of rows.
    This is appropriate when groups (PathVQA image contents) are the
    exchangeable sampling units and individual groups contain different
    numbers of question-answer pairs.
    """
    if len(group_ids) < 2:
        raise ValueError("At least two examples and two groups are required")
    normalized_groups = [str(group_id) for group_id in group_ids]
    unique_groups = list(dict.fromkeys(normalized_groups))
    if len(unique_groups) < 2:
        raise ValueError("At least two distinct groups are required")
    permuted_groups = np.random.default_rng(seed).permutation(unique_groups)
    cal_group_set = set(permuted_groups[: len(unique_groups) // 2])
    cal_indices = [i for i, group in enumerate(normalized_groups) if group in cal_group_set]
    eval_indices = [i for i, group in enumerate(normalized_groups) if group not in cal_group_set]
    return cal_indices, eval_indices


class ConformalPredictor:
    """Split conformal predictor with calibrated prediction sets.

    Calibration computes nonconformity scores ``s_i = 1 - P(y_true | x_i)``
    on a held-out calibration set, then derives threshold τ. With group IDs,
    the score for each group is the maximum question-level score.
    """

    def __init__(self, alpha: float = 0.1, max_set_size: int = 1):
        """Configure the significance level and the set-size uncertainty flag.

        ``alpha`` is the conformal significance level (0.1 → 90% coverage).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_set_size < 1:
            raise ValueError(f"max_set_size must be positive, got {max_set_size}")
        self.alpha = alpha
        self.max_set_size = max_set_size
        self.tau: Optional[float] = None
        self.calibration_stats: Dict[str, Any] = {}

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, group_ids: Optional[Sequence[str]] = None) -> float:
        """Compute conformal threshold from calibration data.

        Args:
            logits: (N, C) raw logits from model on calibration set
            labels: (N,) true label indices

        Returns:
            τ threshold for marginal coverage at level ``1-alpha``. If
            ``group_ids`` are given, the covered unit is a whole group and its
            simultaneous collection of rows.
        """
        labels = _validate_logits_labels(logits, labels)
        probs = torch.softmax(logits, dim=1)
        n = probs.size(0)

        # Nonconformity scores: s_i = 1 - P(y_true | x_i)
        probs_true = probs[torch.arange(n), labels]
        scores = 1.0 - probs_true  # (N,)

        if n == 0:
            raise ValueError("Calibration data must contain at least one example")
        if group_ids is not None:
            if len(group_ids) != n:
                raise ValueError("group_ids must contain one identifier per calibration example")
            grouped_scores: Dict[str, List[torch.Tensor]] = {}
            for group_id, score in zip(group_ids, scores):
                grouped_scores.setdefault(str(group_id), []).append(score)
            if not grouped_scores:
                raise ValueError("Calibration data must contain at least one group")
            # A group's score is the worst (largest) question-level
            # nonconformity. Calibrating this maximum targets simultaneous
            # coverage of all evaluated questions from a new exchangeable image.
            scores = torch.stack([torch.stack(values).max() for values in grouped_scores.values()])
        n_calibration_units = scores.numel()

        # Exact split-conformal threshold (Vovk et al. 2005):
        # τ = the ⌈(n+1)(1-α)⌉-th smallest nonconformity score (order statistic,
        # no interpolation). This guarantees P(y_true ∈ C(x)) ≥ 1 - α.
        k = int(np.ceil((1 - self.alpha) * (n_calibration_units + 1)))
        if k > n_calibration_units:
            # The finite-sample order statistic falls beyond the calibration
            # sample; τ=1 includes every class and preserves the guarantee.
            tau = 1.0
        else:
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
            "n_calibration_units": n_calibration_units,
            "calibration_unit": "group maximum" if group_ids is not None else "example",
            "quantile_rank": k,
        }
        if group_ids is not None:
            self.calibration_stats["n_calibration_groups"] = n_calibration_units
        return tau

    def evaluate(
        self, logits: torch.Tensor, labels: torch.Tensor, group_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Measure coverage / set size on a held-out split (no recalibration).

        Used to report the *out-of-sample* conformal guarantee: calibrating and
        measuring on the same data yields in-sample coverage that overstates the
        real guarantee.
        """
        if self.tau is None:
            raise ValueError("No threshold τ — call calibrate() first")
        labels = _validate_logits_labels(logits, labels)
        n = logits.size(0)
        if n == 0:
            raise ValueError("Evaluation data must contain at least one example")
        if group_ids is not None and len(group_ids) != n:
            raise ValueError("group_ids must contain one identifier per evaluation example")
        pred_sets = self.predict_set(logits)
        coverage = sum(labels[i].item() in pred_sets[i] for i in range(n)) / max(1, n)
        avg_set_size = float(np.mean([len(s) for s in pred_sets])) if pred_sets else 0.0
        set_sizes = [len(s) for s in pred_sets]
        metrics: Dict[str, Any] = {
            "tau": self.tau,
            "alpha": self.alpha,
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "n_evaluation": n,
            "uncertain_rate": (sum(1 for s in set_sizes if s > self.max_set_size) / n if n else 0.0),
        }
        if group_ids is not None:
            grouped_covered: Dict[str, bool] = {}
            for group_id, label, pred_set in zip(group_ids, labels, pred_sets):
                key = str(group_id)
                grouped_covered[key] = grouped_covered.get(key, True) and label.item() in pred_set
            metrics["n_evaluation_groups"] = len(grouped_covered)
            metrics["group_simultaneous_coverage"] = (
                sum(grouped_covered.values()) / len(grouped_covered) if grouped_covered else 0.0
            )
            metrics["coverage_unit"] = "question-answer example; threshold calibrated on group maxima"
        return metrics

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
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        if logits.ndim != 2 or logits.size(1) == 0:
            raise ValueError("logits must have shape (batch, classes) with at least one class")

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
    group_ids: Optional[Sequence[str]] = None,
    eval_group_ids: Optional[Sequence[str]] = None,
) -> Dict[float, Dict[str, Any]]:
    """Compute *out-of-sample* conformal prediction metrics at multiple alphas.

    Calibration and coverage measurement MUST use disjoint data; measuring
    coverage on the same points used to fit τ overstates the guarantee. When
    ``group_ids`` are supplied and no explicit evaluation split is given, whole
    groups are deterministically split between calibration and evaluation. The
    resulting group-level guarantee assumes exchangeability across groups.
    Without group IDs, rows are split and must themselves be exchangeable.

    Args:
        logits: (N, C) raw logits available for calibration + evaluation
        labels: (N,) true labels
        alphas: list of significance levels to evaluate
        max_set_size: set-size limit for the ``uncertain_rate`` statistic
        eval_logits/eval_labels: optional explicit held-out split
        seed: shuffle seed for the internal split
        group_ids: group identifier per input row. Calibration uses each
            group's maximum nonconformity score and splits whole groups.
        eval_group_ids: group identifiers for ``eval_logits`` when an explicit
            evaluation split is supplied. Both group arguments must be given.

    Returns:
        dict mapping alpha → {tau, coverage, avg_set_size, n_calibration,
        n_evaluation, calibration_coverage, uncertain_rate}
    """
    if alphas is None:
        alphas = [0.05, 0.1, 0.2]
    labels = _validate_logits_labels(logits, labels)

    if (eval_logits is None) != (eval_labels is None):
        raise ValueError("eval_logits and eval_labels must be provided together")
    if any(not 0.0 < alpha < 1.0 for alpha in alphas):
        raise ValueError("every alpha must be strictly between 0 and 1")
    cal_group_ids: Optional[Sequence[str]] = None
    holdout_group_ids: Optional[Sequence[str]] = None

    if eval_logits is None:
        n = logits.size(0)
        if n < 2:
            raise ValueError("Need at least 2 samples to split calibration/evaluation")
        if group_ids is not None:
            if len(group_ids) != n:
                raise ValueError("group_ids must contain one identifier per example")
            normalized_groups = [str(group_id) for group_id in group_ids]
            cal_indices, eval_indices = groupwise_split_indices(normalized_groups, seed)
            cal_idx = torch.tensor(cal_indices, dtype=torch.long)
            eval_idx = torch.tensor(eval_indices, dtype=torch.long)
            cal_group_ids = [normalized_groups[i] for i in cal_indices]
            holdout_group_ids = [normalized_groups[i] for i in eval_indices]
        else:
            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=generator)
            half = n // 2
            cal_idx, eval_idx = perm[:half], perm[half:]
            cal_group_ids = None
            holdout_group_ids = None
        cal_logits, cal_labels = logits[cal_idx], labels[cal_idx]
        holdout_logits, holdout_labels = logits[eval_idx], labels[eval_idx]
    else:
        assert eval_logits is not None and eval_labels is not None
        eval_labels = _validate_logits_labels(eval_logits, eval_labels)
        cal_logits, cal_labels = logits, labels
        holdout_logits, holdout_labels = eval_logits, eval_labels
        cal_group_ids = group_ids
        holdout_group_ids = eval_group_ids
        if cal_group_ids is not None and len(cal_group_ids) != len(cal_logits):
            raise ValueError("group_ids must match calibration examples")
        if holdout_group_ids is not None and len(holdout_group_ids) != len(holdout_logits):
            raise ValueError("eval_group_ids must match evaluation examples")
        if (cal_group_ids is None) != (holdout_group_ids is None):
            raise ValueError("Provide both group_ids and eval_group_ids for explicit grouped evaluation")

    results: Dict[float, Dict[str, Any]] = {}
    for alpha in alphas:
        cp = ConformalPredictor(alpha=alpha, max_set_size=max_set_size)
        cp.calibrate(cal_logits, cal_labels, group_ids=cal_group_ids)
        metrics = cp.evaluate(holdout_logits, holdout_labels, group_ids=holdout_group_ids)
        # Keep the in-sample calibration figure for reference/monitoring.
        metrics["calibration_coverage"] = cp.calibration_stats["coverage"]
        metrics["n_calibration_units"] = cp.calibration_stats["n_calibration_units"]
        metrics["calibration_unit"] = cp.calibration_stats["calibration_unit"]
        if "n_calibration_groups" in cp.calibration_stats:
            metrics["n_calibration_groups"] = cp.calibration_stats["n_calibration_groups"]
        results[alpha] = metrics
    return results
