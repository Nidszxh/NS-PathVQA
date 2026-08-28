"""Pre-DSL legacy checkpoint compatibility.

Checkpoints trained before the differentiable DSL interpreter was wired into
the model carry no ``dsl_interpreter.*`` parameters. Loading them under the
current architecture failed ``strict=True`` (missing keys); random-initializing
the interpreter would silently mis-evaluate count queries. Loaders must detect
the pre-DSL state dict (``pathvqa._compat.is_pre_dsl_checkpoint``) and rebuild
the model without the interpreter, reproducing the architecture that saved the
checkpoint. Covers all three load sites: ``evaluate.Evaluator``,
``pathvqa.api.inference.PathVQAInference`` and ``train.Trainer`` resume.
"""

import base64
import copy
import io
from unittest.mock import MagicMock, patch

import pytest
import torch

from pathvqa._compat import is_pre_dsl_checkpoint
from pathvqa.models.pathvqa_model import build_model
from pathvqa.symbolic.executor import build_region_names
from pathvqa.utils.config import get_default_config

ANSWER_TO_IDX = {"yes": 0, "no": 1, "lung": 2, "red": 3}
IDX_TO_ANSWER = {v: k for k, v in ANSWER_TO_IDX.items()}
DSL_PREFIX = "dsl_interpreter."


def _tiny_config(tmp_path):
    """Minimal real config with every writable path inside ``tmp_path``."""
    cfg = get_default_config()
    cfg.visual.model_name = "openai/clip-vit-base-patch32"
    cfg.visual.num_object_features = 128
    cfg.question.model_name = "distilbert-base-uncased"
    # Must match DistilBERT's output dimension (768).
    cfg.question.hidden_dim = 768
    cfg.symbolic.enabled = True
    cfg.symbolic.weighting_strategy = "static"
    region_names = build_region_names(sorted(ANSWER_TO_IDX))
    cfg.symbolic.num_regions = len(region_names)
    cfg.paths.output_dir = tmp_path / "outputs"
    cfg.paths.checkpoint_dir = tmp_path / "checkpoints"
    cfg.paths.log_dir = tmp_path / "logs"
    cfg.paths.data_dir = tmp_path / "data"
    cfg.paths.best_model_path = tmp_path / "checkpoints" / "best_model.pt"
    cfg.paths.__post_init__()
    return cfg


def _b64_image():
    """Small PNG as base64, matching the API tests' request shape."""
    import PIL.Image

    img = PIL.Image.new("RGB", (64, 64), color=(120, 30, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def pre_dsl_checkpoint(tmp_path_factory):
    """A checkpoint whose state dicts predate the DSL interpreter.

    Built by training nothing: construct the normal (DSL-carrying) tiny model,
    then strip ``dsl_interpreter.*`` from both state dicts — exactly what a
    pre-DSL artifact looks like to today's loaders.
    """
    tmp = tmp_path_factory.mktemp("pre_dsl")
    cfg = _tiny_config(tmp)
    region_names = build_region_names(sorted(ANSWER_TO_IDX))
    torch.manual_seed(0)
    model = build_model(cfg, len(ANSWER_TO_IDX))
    full = model.state_dict()
    assert any(k.startswith(DSL_PREFIX) for k in full), "tiny model should build the DSL interpreter"
    stripped = {k: v for k, v in full.items() if not k.startswith(DSL_PREFIX)}
    assert is_pre_dsl_checkpoint(stripped) is True

    path = tmp / "pre_dsl_model.pt"
    torch.save(
        {
            "config": cfg,
            "model_state_dict": stripped,
            "ema_state_dict": dict(stripped),
            "answer_to_idx": dict(ANSWER_TO_IDX),
            "idx_to_answer": dict(IDX_TO_ANSWER),
            "answer_vocab": sorted(ANSWER_TO_IDX),
            "region_names": region_names,
            "best_val_acc": 42.0,
        },
        path,
    )
    return path


class TestIsPreDslCheckpoint:
    """Unit coverage for the detection predicate."""

    def test_no_state_dicts_is_not_pre_dsl(self):
        # Nothing to decide: never disable the interpreter on absent state.
        assert is_pre_dsl_checkpoint() is False

    def test_state_without_dsl_keys_is_pre_dsl(self):
        assert is_pre_dsl_checkpoint({"scene_parser.region.weight": torch.zeros(2)}) is True

    def test_state_with_dsl_keys_is_not_pre_dsl(self):
        state = {f"{DSL_PREFIX}count_head.0.weight": torch.zeros(2)}
        assert is_pre_dsl_checkpoint(state) is False

    def test_one_dsl_carrying_dict_overrides(self):
        plain = {"classifier.0.weight": torch.zeros(2)}
        modern = {f"{DSL_PREFIX}concept_filter.0.weight": torch.zeros(2)}
        assert is_pre_dsl_checkpoint(plain, modern) is False

    def test_both_missing_means_pre_dsl(self):
        a = {"classifier.0.weight": torch.zeros(2)}
        b = {"ema.shadow": torch.zeros(2)}
        assert is_pre_dsl_checkpoint(a, b) is True


class TestEvaluatorPreDSL:
    """evaluate.py: pre-DSL checkpoint must load, build without the DSL, and fuse."""

    def test_evaluator_loads_pre_dsl_checkpoint(self, pre_dsl_checkpoint):
        from evaluate import Evaluator

        e = Evaluator(str(pre_dsl_checkpoint))
        assert e.config.symbolic.dsl_enabled is False
        assert not hasattr(e.model, "dsl_interpreter")
        assert e.fusion.dsl_interpreter is None

        # End-to-end: forward through the rebuilt architecture + PoE fusion.
        device = next(e.model.parameters()).device
        enc = e.tokenizer(
            "is this normal?",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=e.config.question.max_seq_len,
        )
        images = torch.randn(1, 3, *e.config.data.image_size, device=device)
        with torch.no_grad():
            outputs = e.model(images, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        log_p, _trace = e.fusion(outputs, ["is this normal?"])
        assert log_p.shape == (1, len(e.answer_vocab))
        assert torch.isfinite(log_p).all()

    def test_modern_checkpoint_keeps_the_interpreter(self, tmp_path):
        """The detection must not fire for checkpoints that carry DSL params."""
        from evaluate import Evaluator

        cfg = _tiny_config(tmp_path)
        torch.manual_seed(0)
        model = build_model(cfg, len(ANSWER_TO_IDX))
        path = tmp_path / "modern_model.pt"
        torch.save(
            {
                "config": copy.deepcopy(cfg),
                "model_state_dict": model.state_dict(),
                "ema_state_dict": model.state_dict(),
                "answer_to_idx": dict(ANSWER_TO_IDX),
                "idx_to_answer": dict(IDX_TO_ANSWER),
                "region_names": build_region_names(sorted(ANSWER_TO_IDX)),
                "best_val_acc": 1.0,
            },
            path,
        )
        e = Evaluator(str(path))
        assert e.config.symbolic.dsl_enabled is True
        assert hasattr(e.model, "dsl_interpreter")
        assert e.fusion.dsl_interpreter is not None


class TestInferencePreDSL:
    """api/inference.py: the serving path must load the same legacy artifact."""

    def test_inference_predicts_from_pre_dsl_checkpoint(self, pre_dsl_checkpoint):
        from pathvqa.api.inference import PathVQAInference

        infer = PathVQAInference(str(pre_dsl_checkpoint), device="cpu")
        assert infer.config.symbolic.dsl_enabled is False
        assert not hasattr(infer.model, "dsl_interpreter")
        assert infer.fusion.dsl_interpreter is None

        result = infer.predict(_b64_image(), "What organ is shown?")
        assert result["answer"] in IDX_TO_ANSWER.values()
        assert 0.0 <= result["confidence"] <= 1.0

        explained = infer.explain(_b64_image(), "What organ is shown?")
        assert explained["answer"] in IDX_TO_ANSWER.values()


def _collate_batch(questions, targets):
    """Build a real ``CollateOutput`` batch for the mocked loaders."""
    from pathvqa.data.pathvqa_dataset import CollateOutput

    return CollateOutput(
        images=torch.randn(len(questions), 3, 224, 224),
        targets=torch.tensor(targets),
        questions=list(questions),
        answers=["yes"] * len(questions),
    )


class TestTrainerResumePreDSL:
    """train.py: resuming a pre-DSL checkpoint must rebuild without the DSL.

    The source checkpoint is produced by a real ``Trainer`` whose config omits
    the ``dsl_enabled`` field entirely (as an old pickle would), so the test
    exercises backfill (which restores the modern default ``True``) *and* the
    state-dict detection that must flip it back to ``False``.
    """

    def test_resume_builds_model_without_dsl(self, tmp_path):
        from train import Trainer

        cfg = _tiny_config(tmp_path)
        cfg.experiment_name = "pre_dsl_resume_test"
        cfg.symbolic.dsl_enabled = False  # the old architecture: no DSL module

        mock_dataset = MagicMock()
        mock_dataset.answers = ["yes", "no", "lung", "red"]
        mock_dataset.answer_to_idx = dict(ANSWER_TO_IDX)
        mock_dataset.idx_to_answer = dict(IDX_TO_ANSWER)
        questions = ["is this normal?", "what is the diagnosis?"]
        batch = _collate_batch(questions, [0, 1])

        with (
            patch("train.PathVQADataset", return_value=mock_dataset),
            patch("train.DataLoader", return_value=[batch]),
            patch("train.pathvqa_dataloader", return_value=[batch]),
        ):
            source = Trainer(cfg)
            assert not hasattr(source.model, "dsl_interpreter")

            # Simulate a pre-field pickle: no dsl_enabled stored in the config.
            saved_cfg = copy.deepcopy(source.config)
            saved_cfg.symbolic.__dict__.pop("dsl_enabled", None)

            ckpt_path = tmp_path / "pre_dsl_resume.pt"
            torch.save(
                {
                    "config": saved_cfg,
                    "model_state_dict": source.model.state_dict(),
                    "ema_state_dict": source.ema.state_dict() if source.ema is not None else None,
                    "optimizer_state_dict": source.optimizer.state_dict(),
                    "answer_vocab": source.answer_vocab,
                    "answer_to_idx": source.answer_to_idx,
                    "idx_to_answer": source.idx_to_answer,
                    "region_names": source.region_names,
                    "epoch": 0,
                    "best_val_acc": 0.0,
                },
                ckpt_path,
            )
            resumed = Trainer(cfg, resume_path=str(ckpt_path))

        # Backfill restored the modern default; detection must have flipped it.
        assert resumed.config.symbolic.dsl_enabled is False
        assert not hasattr(resumed.model, "dsl_interpreter")
        # Weights actually came from the checkpoint, not fresh init.
        assert torch.equal(
            resumed.model.classifier[-1].weight,
            source.model.classifier[-1].weight,
        )
        assert resumed.current_epoch == 0
