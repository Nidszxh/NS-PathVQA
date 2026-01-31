"""Tests for conformal prediction."""

import torch
from utils.conformal import ConformalPredictor, compute_conformal_metrics


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
        assert len(s) >= 1  # at least one class in set


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
