"""End-to-end integration: one train step, one eval pass, and a short Trainer loop."""

import pytest
import torch
import json
from unittest.mock import patch, MagicMock

from pathvqa.models.pathvqa_model import NeuroSymbolicPathVQA
from pathvqa.utils.config import get_default_config
from train import Trainer


@pytest.fixture
def mock_config(tmp_path):
    config = get_default_config()
    config.data.batch_size = 2
    config.data.max_train_samples = 4
    config.data.max_val_samples = 4
    config.training.num_epochs = 1
    config.training.log_every = 1
    config.training.save_every = 1
    # PathConfig has no base_dir field; isolate every writable location.
    config.paths.output_dir = tmp_path / "outputs"
    config.paths.checkpoint_dir = tmp_path / "checkpoints"
    config.paths.log_dir = tmp_path / "logs"
    config.paths.data_dir = tmp_path / "data"
    # best_model_path is a plain field (not derived from checkpoint_dir);
    # without this a training run would overwrite the real best checkpoint.
    config.paths.best_model_path = tmp_path / "checkpoints" / "best_model.pt"
    config.paths.__post_init__()
    config.visual.model_name = "openai/clip-vit-base-patch32"
    config.visual.num_object_features = 128
    config.question.model_name = "distilbert-base-uncased"
    # Must match DistilBERT's output dimension (768); the cross-modal
    # transformer and differentiable query heads consume that size directly.
    config.question.hidden_dim = 768
    config.experiment_name = "integration_test"
    return config


@pytest.fixture
def mock_batch():
    batch_size = 2
    return {
        "image": torch.randn(batch_size, 3, 224, 224),
        "input_ids": torch.randint(0, 1000, (batch_size, 10)),
        "attention_mask": torch.ones(batch_size, 10),
        "answer_idx": torch.randint(0, 10, (batch_size,)),
        "qtype_onehot": torch.zeros(batch_size, 5),
    }


def test_integration_single_train_step(mock_batch):
    """Test that a single forward + backward + optimizer step works."""
    model = NeuroSymbolicPathVQA(
        answer_vocab_size=10,
        num_object_features=128,
        question_hidden_dim=768,
        symbolic_enabled=True,
        num_regions=5,
        visual_config={"model_name": "openai/clip-vit-base-patch32", "num_object_features": 128},
        question_config={"model_name": "distilbert-base-uncased"},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    optimizer.zero_grad()

    outputs = model(
        images=mock_batch["image"],
        input_ids=mock_batch["input_ids"],
        attention_mask=mock_batch["attention_mask"],
        qtype_onehot=mock_batch["qtype_onehot"],
    )

    loss = torch.nn.functional.cross_entropy(outputs.answer_logits, mock_batch["answer_idx"])
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert outputs.answer_logits.shape == (2, 10)


def test_integration_eval_pass(mock_batch):
    """Test a single evaluation forward pass."""
    model = NeuroSymbolicPathVQA(
        answer_vocab_size=10,
        num_object_features=128,
        question_hidden_dim=768,
        symbolic_enabled=True,
        num_regions=5,
        visual_config={"model_name": "openai/clip-vit-base-patch32", "num_object_features": 128},
        question_config={"model_name": "distilbert-base-uncased"},
    )
    model.eval()
    with torch.no_grad():
        outputs = model(
            images=mock_batch["image"],
            input_ids=mock_batch["input_ids"],
            attention_mask=mock_batch["attention_mask"],
            qtype_onehot=mock_batch["qtype_onehot"],
        )
    assert outputs.answer_logits.shape == (2, 10)


def _collate_batch(questions, targets):
    """Build a real ``CollateOutput`` batch for the mocked loaders."""
    from pathvqa.data.pathvqa_dataset import CollateOutput

    return CollateOutput(
        images=torch.randn(len(questions), 3, 224, 224),
        targets=torch.tensor(targets),
        questions=list(questions),
        answers=["yes"] * len(questions),
    )


def test_trainer_integration_loop(mock_config):
    """Run one full ``Trainer.train()`` epoch end-to-end with mocked loaders.

    All three dataset entry points are patched: ``PathVQADataset`` (train
    split construction), ``DataLoader`` (train loader) and
    ``pathvqa_dataloader`` (val loader). Without the last one the Trainer
    silently constructs a *real* HuggingFace dataset (network/disk I/O),
    which dominated the suite runtime by minutes.

    The loaders return a single synthetic ``CollateOutput`` batch each so the
    real ``train_epoch`` → ``validate`` → ``save_checkpoint`` →
    ``_write_baseline`` path is exercised (not just the artifact writer).
    """
    mock_dataset = MagicMock()
    mock_dataset.answers = ["yes", "no", "lung", "red"]
    mock_dataset.answer_to_idx = {"yes": 0, "no": 1, "lung": 2, "red": 3}
    mock_dataset.idx_to_answer = {0: "yes", 1: "no", 2: "lung", 3: "red"}

    questions = ["what is the diagnosis?", "is this normal?"]
    train_batch = _collate_batch(questions, [0, 1])
    val_batch = _collate_batch(questions, [0, 1])

    with (
        patch("train.PathVQADataset") as mock_dataset_class,
        patch("train.DataLoader", return_value=[train_batch]),
        patch("train.pathvqa_dataloader", return_value=[val_batch]),
    ):
        mock_dataset_class.return_value = mock_dataset

        trainer = Trainer(mock_config)
        assert trainer.config.experiment_name == "integration_test"
        assert trainer.device.type in ("cuda", "cpu")

        # Execute a real training epoch: forward, backward, optimiser step,
        # validation, checkpointing and artifact emission.
        trainer.train()

        expected_output = mock_config.paths.output_dir / "integration_test.json"
        assert expected_output.exists()
        with open(expected_output, "r") as f:
            results = json.load(f)
        assert results["experiment_name"] == "integration_test"
        assert 0.0 <= results["best_val_acc"] <= 100.0
        assert results["resources"]["epochs_trained"] == 1
        assert results["config"]["symbolic_enabled"] is True
        assert "transform_version" in results["config"]

        # A rolling checkpoint must land in the isolated checkpoint dir, and the
        # real production best_model.pt must never be touched.
        assert any(mock_config.paths.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert mock_config.paths.best_model_path.parent == mock_config.paths.checkpoint_dir
