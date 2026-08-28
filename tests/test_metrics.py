"""Calibration, uncertainty, and bootstrap significance metrics."""

import torch
from pathvqa.utils.metrics import (
    expected_calibration_error,
    paired_bootstrap_test,
    two_sample_bootstrap_test,
    compute_uncertainty,
)


class TestTwoSampleBootstrapTest:
    def test_returns_float(self):
        p = two_sample_bootstrap_test(0.8, 0.7, 100, 100)
        assert isinstance(p, float)

    def test_p_value_range(self):
        p = two_sample_bootstrap_test(0.8, 0.7, 100, 100)
        assert 0.0 <= p <= 1.0

    def test_equal_accuracies_not_significant(self):
        p = two_sample_bootstrap_test(0.75, 0.75, 200, 200)
        assert p > 0.05

    def test_large_gap_is_significant(self):
        # Regression: a 40-point gap at n=2000 must be significant.
        # The old implementation compared uncentered bootstrap draws and
        # always returned p≈0.5 regardless of effect size.
        p = two_sample_bootstrap_test(0.90, 0.50, 2000, 2000, n_bootstrap=5000)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
        assert p < 0.001

    def test_small_gap_not_significant(self):
        p = two_sample_bootstrap_test(0.51, 0.50, 100, 100)
        assert p > 0.05

    def test_deterministic(self):
        p1 = two_sample_bootstrap_test(0.8, 0.7, 100, 100, seed=42)
        p2 = two_sample_bootstrap_test(0.8, 0.7, 100, 100, seed=42)
        assert p1 == p2


class TestPairedBootstrapTest:
    def test_identical_correctness_not_significant(self):
        a = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 20
        p = paired_bootstrap_test(a, a)
        assert p == 1.0

    def test_large_paired_gap_significant(self):
        # Model A always correct, model B always wrong (n=500).
        a = [1.0] * 500
        b = [0.0] * 500
        p = paired_bootstrap_test(a, b, n_bootstrap=2000)
        assert 0.0 <= p <= 1.0
        assert p < 0.001

    def test_small_paired_gap_not_significant(self):
        import numpy as np

        rng = np.random.RandomState(0)
        a = rng.binomial(1, 0.51, size=100).astype(float)
        b = rng.binomial(1, 0.50, size=100).astype(float)
        p = paired_bootstrap_test(a, b, n_bootstrap=2000, seed=1)
        assert p > 0.05

    def test_deterministic(self):
        a = [1, 0, 1, 1, 0] * 30
        b = [0, 0, 1, 0, 1] * 30
        assert paired_bootstrap_test(a, b, seed=7) == paired_bootstrap_test(a, b, seed=7)

    def test_misaligned_vectors_raise(self):
        import pytest

        with pytest.raises(ValueError):
            paired_bootstrap_test([1, 0, 1], [0, 1])


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
