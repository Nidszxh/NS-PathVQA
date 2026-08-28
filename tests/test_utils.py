"""Tests for seed utilities, structured logging, and temperature scaling."""

import random

import numpy as np
import torch

from pathvqa.utils.logging_utils import get_logger, setup_logging
from pathvqa.utils.metrics import temperature_scaling
from pathvqa.utils.seed import seed_worker, set_seed


class TestSetSeed:
    def test_reproducible(self):
        set_seed(1234)
        a = [torch.randn(3), np.random.rand(3), random.random()]
        set_seed(1234)
        b = [torch.randn(3), np.random.rand(3), random.random()]
        assert torch.equal(a[0], b[0])
        assert np.allclose(a[1], b[1])
        assert a[2] == b[2]

    def test_deterministic_flags(self):
        set_seed(0, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        # Restore benchmark mode for other tests
        set_seed(0)
        torch.backends.cudnn.benchmark = True


class TestSeedWorker:
    def test_worker_seeding(self):
        torch.manual_seed(7)
        expected = torch.initial_seed() % 2**32
        seed_worker(0)
        # seed_worker derives the numpy/torch RNGs from the worker's base seed;
        # after seeding, identical seeds must yield identical numpy streams.
        first_draw = np.random.randint(0, 2**31)
        np.random.seed(expected + 0)
        import random as _random

        _random.seed(expected)
        assert first_draw == np.random.randint(0, 2**31)

    def test_deterministic_flags_restored_after_test(self):
        set_seed(0, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        set_seed(0)
        # set_seed without deterministic must restore benchmark-style defaults
        assert torch.backends.cudnn.benchmark is True


class TestSetupLogging:
    def test_console_and_file(self, tmp_path):
        log_file = tmp_path / "logs" / "run.log"
        logger = setup_logging(log_file=log_file)
        logger.info("hello file")
        for h in logger.handlers:
            h.flush()
        assert log_file.exists()
        assert "hello file" in log_file.read_text()

    def test_get_logger_namespaced(self):
        assert get_logger().name == "pathvqa"
        assert get_logger("train").name == "pathvqa.train"


class TestTemperatureScaling:
    def test_returns_positive_scalar(self):
        torch.manual_seed(0)
        logits = torch.randn(64, 5)
        labels = torch.randint(0, 5, (64,))
        t = temperature_scaling(logits, labels)
        assert isinstance(t, float)
        assert t > 0

    def test_scaled_logits_improve_or_match_nll(self):
        """Overconfident logits should be softened by an optimal T."""
        torch.manual_seed(1)
        logits = torch.randn(128, 3) * 10  # deliberately overconfident
        labels = torch.randint(0, 3, (128,))
        t = temperature_scaling(logits, labels)
        ce_before = torch.nn.functional.cross_entropy(logits, labels).item()
        ce_after = torch.nn.functional.cross_entropy(logits / t, labels).item()
        assert ce_after <= ce_before + 1e-6
