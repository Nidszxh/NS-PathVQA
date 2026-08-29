"""Entry-point tests for train / evaluate / serve / ablation / figure scripts.

These exercise the CLI wiring (argument parsing, override application, artifact
naming) without running real training or loading a real checkpoint.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    """Import a ``scripts/*.py`` module by path (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _isolated_config(tmp_path, **overrides):
    """A default config whose writable paths all live under ``tmp_path``."""
    from pathvqa.utils.config import get_default_config

    cfg = get_default_config()
    cfg.paths.output_dir = tmp_path / "outputs"
    cfg.paths.checkpoint_dir = tmp_path / "checkpoints"
    cfg.paths.log_dir = tmp_path / "logs"
    cfg.paths.data_dir = tmp_path / "data"
    cfg.paths.best_model_path = tmp_path / "checkpoints" / "best_model.pt"
    cfg.paths.__post_init__()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------


class _CapturingTrainer:
    def __init__(self, config, resume_path=None, overrides=None):
        self.config = config
        self.resume_path = resume_path
        self.overrides = overrides
        type(self).last = self

    def train(self):
        self.trained = True


def test_train_main_applies_ablation_and_experiment(tmp_path, monkeypatch):
    import train

    cfg = _isolated_config(tmp_path)
    monkeypatch.setattr(train, "get_default_config", lambda: cfg)
    monkeypatch.setattr(train, "setup_logging", lambda **kwargs: MagicMock())
    monkeypatch.setattr(train, "Trainer", _CapturingTrainer)
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--ablation", "C", "--experiment", "reg_run", "--weighting-strategy", "learned"]
    )

    train.main()

    captured = _CapturingTrainer.last
    assert captured.config.experiment_name == "reg_run"
    assert captured.config.symbolic.query_parser == "learned"
    assert captured.config.symbolic.executor == "differentiable"
    assert captured.config.symbolic.weighting_strategy == "learned"
    # Overrides are handed to the Trainer so they can be re-applied on resume.
    assert captured.overrides["ablation"] == "C"
    assert captured.trained is True


def test_train_main_no_symbolic_override(tmp_path, monkeypatch):
    import train

    cfg = _isolated_config(tmp_path)
    monkeypatch.setattr(train, "get_default_config", lambda: cfg)
    monkeypatch.setattr(train, "setup_logging", lambda **kwargs: MagicMock())
    monkeypatch.setattr(train, "Trainer", _CapturingTrainer)
    monkeypatch.setattr(sys, "argv", ["train.py", "--no-symbolic"])

    train.main()
    assert _CapturingTrainer.last.config.symbolic.enabled is False


def test_train_main_debug_selects_isolated_paths(tmp_path, monkeypatch):
    import train

    debug_cfg = _isolated_config(tmp_path, debug=True)
    debug_cfg.experiment_name = "debug_run"
    debug_cfg.paths.checkpoint_dir = tmp_path / "checkpoints" / "debug"
    debug_cfg.paths.best_model_path = debug_cfg.paths.checkpoint_dir / "best_model.pt"
    debug_cfg.paths.__post_init__()
    monkeypatch.setattr(train, "get_debug_config", lambda: debug_cfg)
    monkeypatch.setattr(train, "setup_logging", lambda **kwargs: MagicMock())
    monkeypatch.setattr(train, "Trainer", _CapturingTrainer)
    monkeypatch.setattr(sys, "argv", ["train.py", "--debug"])

    train.main()
    captured = _CapturingTrainer.last
    assert captured.config.debug is True
    assert captured.config.paths.checkpoint_dir == tmp_path / "checkpoints" / "debug"


def test_train_main_resume_auto_without_checkpoint_exits(tmp_path, monkeypatch):
    import train

    cfg = _isolated_config(tmp_path)
    monkeypatch.setattr(train, "get_default_config", lambda: cfg)
    monkeypatch.setattr(train, "setup_logging", lambda **kwargs: MagicMock())
    monkeypatch.setattr(sys, "argv", ["train.py", "--resume", "auto"])
    with pytest.raises(SystemExit, match="no checkpoint_epoch"):
        train.main()


# ---------------------------------------------------------------------------
# evaluate.py
# ---------------------------------------------------------------------------


def test_evaluate_main_writes_derived_artifact(tmp_path, monkeypatch):
    import evaluate

    cfg = _isolated_config(tmp_path)

    results = {
        "accuracy": 55.0,
        "correct": 11,
        "total": 20,
        "per_type_accuracy": {"identity": {"accuracy": 60.0, "correct": 6, "total": 10}},
        "calibration": {"ece": 0.05, "temperature": 1.0, "reliability_bins": []},
        "uncertainty": {"mean": 0.4, "std": 0.1},
        "conformal": {},
    }

    class FakeEvaluator:
        def __init__(self, checkpoint_path, config_path=None, ablation=None):
            self.config = cfg
            self.checkpoint_path = checkpoint_path
            self.ablation = ablation

        def evaluate(self, split="val", max_samples=None, intervention="none"):
            return results

    monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--checkpoint", "checkpoints/best_model.pt", "--split", "val", "--ablation", "C"],
    )

    evaluate.main()

    # Artifact name derives from checkpoint stem + split + ablation.
    out = tmp_path / "outputs" / "eval_best_model_val_C.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["accuracy"] == 55.0


def test_evaluate_main_default_output_naming_without_ablation(tmp_path, monkeypatch):
    import evaluate

    cfg = _isolated_config(tmp_path)

    class FakeEvaluator:
        def __init__(self, checkpoint_path, config_path=None, ablation=None):
            self.config = cfg

        def evaluate(self, split="val", max_samples=None, intervention="none"):
            return {
                "accuracy": 1.0,
                "correct": 0,
                "total": 1,
                "per_type_accuracy": {},
                "calibration": {"ece": 0.0, "temperature": 1.0},
                "uncertainty": {"mean": 0.0, "std": 0.0},
                "conformal": {},
            }

    monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", "ckpt_x.pt", "--split", "test"])
    evaluate.main()
    assert (tmp_path / "outputs" / "eval_ckpt_x_test.json").exists()


# ---------------------------------------------------------------------------
# serve.py
# ---------------------------------------------------------------------------


def test_serve_main_sets_env_and_launches_uvicorn(monkeypatch):
    import serve
    import uvicorn

    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["serve.py", "--checkpoint", "ckpt.pt", "--host", "127.0.0.1", "--port", "9999"])

    serve.main()

    assert os.environ["CHECKPOINT_PATH"] == "ckpt.pt"
    assert captured["app"] is serve.app
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 9999


# ---------------------------------------------------------------------------
# scripts/run_ablations.py
# ---------------------------------------------------------------------------


def test_run_ablations_persists_full_matrix(tmp_path, monkeypatch):
    import evaluate
    import train
    from pathvqa.utils.ablation import ABLATION_CONFIGS, ABLATION_LABELS

    run_ablations = _load_script("run_ablations")

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self):
            self.cfg.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.paths.best_model_path.write_bytes(b"checkpoint")

    class FakeEvaluator:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

        def evaluate(self, split="val", max_samples=None):
            return {"accuracy": 61.0, "per_type_accuracy": {}}

    monkeypatch.setattr(train, "Trainer", FakeTrainer)
    monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_ablations.py"])

    run_ablations.main()

    out = tmp_path / "outputs" / "ablation_results_seed42.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["seeds"] == [42]
    assert data["deterministic"] is True
    assert data["matrix"] == ABLATION_CONFIGS
    assert data["labels"] == ABLATION_LABELS
    expected_names = {f"{k}_{ABLATION_LABELS[k]}" for k in ABLATION_CONFIGS}
    assert set(data["results"]) == expected_names
    for name, res in data["results"].items():
        assert res["status"] == "OK", name
        assert res["accuracy_mean"] == 61.0
        # A single seed yields no variance estimate; it must be None, not 0.0.
        assert res["accuracy_std"] is None
        assert set(res["runs"]) == {"42"}


def test_run_ablations_prunes_epoch_checkpoints_by_default(tmp_path, monkeypatch):
    import evaluate
    import train

    run_ablations = _load_script("run_ablations")

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self):
            self.cfg.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.paths.best_model_path.write_bytes(b"checkpoint")
            for epoch in (0, 5, 10):
                (self.cfg.paths.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt").write_bytes(b"epoch")

    class FakeEvaluator:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

        def evaluate(self, split="val", max_samples=None):
            return {"accuracy": 60.0, "per_type_accuracy": {}}

    monkeypatch.setattr(train, "Trainer", FakeTrainer)
    monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "checkpoints" / "A_regex_hard_static" / "seed_42"

    monkeypatch.setattr(sys, "argv", ["run_ablations.py", "--ablations", "A"])
    run_ablations.main()
    assert (run_dir / "best_model.pt").exists()
    # best+last is the default: only the highest-numbered snapshot survives.
    assert [path.name for path in run_dir.glob("checkpoint_epoch_*.pt")] == ["checkpoint_epoch_10.pt"]

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ablations.py", "--ablations", "A", "--keep-checkpoints", "all", "--output", "outputs/keep.json"],
    )
    run_ablations.main()
    assert len(list(run_dir.glob("checkpoint_epoch_*.pt"))) == 3

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ablations.py", "--ablations", "A", "--keep-checkpoints", "best", "--output", "outputs/best.json"],
    )
    run_ablations.main()
    assert list(run_dir.glob("checkpoint_epoch_*.pt")) == []


def test_run_ablations_persists_running_and_failed_seed_states(tmp_path, monkeypatch):
    import evaluate
    import train
    from pathvqa.utils.ablation import ABLATION_LABELS

    run_ablations = _load_script("run_ablations")
    output = tmp_path / "outputs" / "ablation_results_seeds17-42-123.json"

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self):
            if self.cfg.seed == 17:
                data = json.loads(output.read_text())
                name = f"A_{ABLATION_LABELS['A']}"
                assert data["results"][name]["runs"]["17"]["status"] == "RUNNING"
                assert data["results"][name]["runs"]["42"]["status"] == "PENDING"
            if self.cfg.seed == 42:
                raise RuntimeError("simulated training error")
            self.cfg.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.paths.best_model_path.write_bytes(b"checkpoint")

    class FakeEvaluator:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

        def evaluate(self, split="val", max_samples=None):
            return {"accuracy": 60.0, "per_type_accuracy": {}}

    monkeypatch.setattr(train, "Trainer", FakeTrainer)
    monkeypatch.setattr(evaluate, "Evaluator", FakeEvaluator)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_ablations.py", "--ablations", "A", "--seeds", "17", "42", "123"])

    run_ablations.main()

    data = json.loads(output.read_text())
    summary = data["results"][f"A_{ABLATION_LABELS['A']}"]
    assert summary["status"] == "PARTIAL"
    assert summary["runs"]["17"]["status"] == "OK"
    assert summary["runs"]["42"]["status"] == "FAILED"
    assert summary["runs"]["123"]["status"] == "OK"


# ---------------------------------------------------------------------------
# scripts/generate_paper_figures.py
# ---------------------------------------------------------------------------


def test_generate_figures_main_requires_eval_results(tmp_path, monkeypatch):
    figures = _load_script("generate_paper_figures")
    monkeypatch.setattr(sys, "argv", ["generate_paper_figures.py", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        figures.main()


def test_generate_figures_real_data_writes_expected_files(tmp_path):
    figures = _load_script("generate_paper_figures")
    data = {
        "per_type_accuracy": {"identity": {"accuracy": 80.0, "correct": 8, "total": 10}},
        "qtype_confusion": {
            "matrix": [[5, 0], [1, 4]],
            "labels": ["identity", "yes_no"],
        },
        "calibration": {
            "ece": 0.03,
            "reliability_bins": [
                {"confidence": 0.5, "accuracy": 0.5, "count": 10},
                {"confidence": 0.8, "accuracy": 0.75, "count": 5},
            ],
        },
        "failure_taxonomy": {"failure_counts": {"perception": 3, "parsing": 1}},
        "concept_interventions": {"probe_accuracy_delta": 2.5},
    }
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(data))

    generated = 0
    for plotter in figures.PLOTTERS:
        assert plotter(data, tmp_path, False) is True
        generated += 1
    assert generated == len(figures.PLOTTERS)
    assert len(list(tmp_path.glob("*.png"))) == len(figures.PLOTTERS)


def test_generate_figures_skip_missing_fields(tmp_path):
    figures = _load_script("generate_paper_figures")
    for plotter in figures.PLOTTERS:
        assert plotter({}, tmp_path, False) is False
    assert list(tmp_path.glob("*.png")) == []


def test_generate_figures_mock_is_watermarked(tmp_path, monkeypatch):
    figures = _load_script("generate_paper_figures")
    monkeypatch.setattr(sys, "argv", ["generate_paper_figures.py", "--mock", "--out-dir", str(tmp_path / "mock_figs")])
    figures.main()
    assert len(list((tmp_path / "mock_figs").glob("*.png"))) == len(figures.PLOTTERS)
