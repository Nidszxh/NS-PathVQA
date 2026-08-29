"""Tests for conformal prediction."""

import torch
import pytest
from pathvqa.utils.conformal import ConformalPredictor, compute_conformal_metrics


def test_conformal_calibration():
    """Calibration should produce valid tau and coverage stats."""
    torch.manual_seed(42)
    logits = torch.randn(200, 10)
    labels = torch.randint(0, 10, (200,))

    cp = ConformalPredictor(alpha=0.1)
    tau = cp.calibrate(logits, labels)

    assert 0.0 <= tau <= 2.0  # tau = 1 - p, so in [0, 1] typically
    assert cp.calibration_stats["alpha"] == 0.1
    assert cp.calibration_stats["n_calibration"] == 200
    assert 0.0 <= cp.calibration_stats["coverage"] <= 1.0
    assert cp.calibration_stats["avg_set_size"] >= 1.0


def test_conformal_prediction_sets():
    """Prediction sets should contain true label with high probability."""
    torch.manual_seed(42)
    logits = torch.randn(200, 10)
    labels = torch.randint(0, 10, (200,))

    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(logits, labels)

    pred_sets = cp.predict_set(logits)
    assert len(pred_sets) == 200
    for s in pred_sets:
        assert isinstance(s, set)
        assert len(s) <= 10  # never exceed the class count; empty sets are legal under high tau


def test_conformal_uncertain():
    """Uncertain should flag predictions with large sets."""
    torch.manual_seed(42)
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))

    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(logits, labels)

    uncertain = cp.is_uncertain(logits, max_set_size=1)
    assert uncertain.shape == (100,)
    assert uncertain.dtype == torch.bool


def test_conformal_route_signal():
    """Route signal should return all expected fields."""
    torch.manual_seed(42)
    logits = torch.randn(50, 10)
    labels = torch.randint(0, 10, (50,))

    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(logits, labels)

    signal = cp.get_route_signal(logits)
    assert "prediction_sets" in signal
    assert "uncertain" in signal
    assert "set_sizes" in signal
    assert "max_prob" in signal
    assert "max_pred" in signal
    assert signal["set_sizes"].shape == (50,)
    assert signal["max_prob"].shape == (50,)


def test_conformal_metrics_multiple_alphas():
    """compute_conformal_metrics should handle multiple alpha levels."""
    torch.manual_seed(42)
    logits = torch.randn(200, 10)
    labels = torch.randint(0, 10, (200,))

    results = compute_conformal_metrics(logits, labels, alphas=[0.05, 0.1, 0.2])
    assert len(results) == 3
    for alpha in [0.05, 0.1, 0.2]:
        assert alpha in results
        assert "tau" in results[alpha]
        assert "coverage" in results[alpha]
        assert "avg_set_size" in results[alpha]


def test_conformal_requires_calibration():
    """predict_set should raise if not calibrated."""
    cp = ConformalPredictor(alpha=0.1)
    logits = torch.randn(10, 5)
    try:
        cp.predict_set(logits)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_conformal_exact_order_statistic():
    """tau must be the ceil((n+1)(1-alpha))-th smallest score (no interpolation)."""
    torch.manual_seed(0)
    logits = torch.randn(50, 4)
    labels = torch.randint(0, 4, (50,))
    cp = ConformalPredictor(alpha=0.1)
    tau = cp.calibrate(logits, labels)

    probs = torch.softmax(logits, dim=1)
    scores = 1.0 - probs[torch.arange(50), labels]
    k = min(int(torch.ceil(torch.tensor(0.9 * 51)).item()), 50)
    expected = torch.sort(scores).values[k - 1].item()
    assert abs(tau - expected) < 1e-6
    assert cp.calibration_stats["quantile_rank"] == k


def test_conformal_small_calibration_set_uses_all_labels_when_rank_exceeds_n():
    """When k > n, the exact finite-sample quantile is tau=1 (all labels)."""
    logits = torch.randn(10, 3)
    labels = torch.randint(0, 3, (10,))
    cp = ConformalPredictor(alpha=1e-9)
    tau = cp.calibrate(logits, labels)
    assert tau == 1.0
    assert cp.calibration_stats["quantile_rank"] == 11
    assert all(len(prediction_set) == logits.size(1) for prediction_set in cp.predict_set(logits))


def test_grouped_conformal_uses_worst_question_score_per_image():
    """The calibration rank is based on independent image groups, not QA rows."""
    logits = torch.tensor([[3.0, 0.0], [0.2, 0.0], [0.1, 0.0], [0.2, 0.0]])
    labels = torch.tensor([0, 0, 0, 0])
    groups = ["image-a", "image-a", "image-b", "image-b"]
    cp = ConformalPredictor(alpha=0.5)

    tau = cp.calibrate(logits, labels, group_ids=groups)

    scores = 1.0 - torch.softmax(logits, dim=1)[:, 0]
    expected = torch.stack([scores[:2].max(), scores[2:].max()]).sort().values[1].item()
    assert tau == pytest.approx(expected)
    assert cp.calibration_stats["quantile_rank"] == 2
    assert cp.calibration_stats["n_calibration_groups"] == 2
    assert cp.calibration_stats["calibration_unit"] == "group maximum"
    result = cp.evaluate(logits, labels, group_ids=groups)
    assert result["n_evaluation_groups"] == 2
    assert result["group_simultaneous_coverage"] == 1.0


def test_grouped_conformal_splits_whole_groups():
    logits = torch.randn(12, 4)
    labels = torch.randint(0, 4, (12,))
    groups = [f"image-{i // 3}" for i in range(12)]

    result = compute_conformal_metrics(logits, labels, alphas=[0.2], group_ids=groups)[0.2]

    assert result["calibration_unit"] == "group maximum"
    assert result["n_calibration_groups"] + result["n_evaluation_groups"] == 4
    assert "group_simultaneous_coverage" in result


def test_conformal_requires_paired_explicit_eval_inputs():
    logits = torch.randn(4, 3)
    labels = torch.randint(0, 3, (4,))
    with pytest.raises(ValueError, match="provided together"):
        compute_conformal_metrics(logits, labels, eval_logits=torch.randn(2, 3))
