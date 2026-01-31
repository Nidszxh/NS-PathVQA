import torch
from utils.metrics import (
    expected_calibration_error,
    paired_bootstrap_test,
    compute_uncertainty,
)


class TestPairedBootstrapTest:
    def test_returns_float(self):
        p = paired_bootstrap_test(0.8, 0.7, 100, 100)
        assert isinstance(p, float)

    def test_p_value_range(self):
        p = paired_bootstrap_test(0.8, 0.7, 100, 100)
        assert 0.0 <= p <= 1.0

    def test_equal_accuracies_not_significant(self):
        p = paired_bootstrap_test(0.75, 0.75, 200, 200)
        assert p > 0.05

    def test_large_gap_returns_float(self):
        p = paired_bootstrap_test(0.90, 0.50, 2000, 2000, n_bootstrap=5000)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_deterministic(self):
        p1 = paired_bootstrap_test(0.8, 0.7, 100, 100, seed=42)
        p2 = paired_bootstrap_test(0.8, 0.7, 100, 100, seed=42)
        assert p1 == p2


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        labels = torch.tensor([0, 1, 2])
        ece, bins = expected_calibration_error(logits, labels, n_bins=3)
        assert ece < 0.01

    def test_perfectly_uncalibrated(self):
        logits = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
        labels = torch.tensor([0, 0])
        ece, bins = expected_calibration_error(logits, labels, n_bins=2)
        assert ece > 0.5


class TestComputeUncertainty:
    def test_confident_prediction_low_uncertainty(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        unc = compute_uncertainty(logits, method="max_softmax")
        assert (unc < 0.01).all()

    def test_uniform_distribution_high_uncertainty(self):
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        unc = compute_uncertainty(logits, method="entropy")
        assert (unc > 0.6).all()
