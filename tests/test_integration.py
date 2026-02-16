import pytest
import torch
import os
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
    config.paths.base_dir = tmp_path
    config.paths.__post_init__()
    config.visual.model_name = "openai/clip-vit-base-patch32"
    config.visual.num_object_features = 128
    config.question.model_name = "distilbert-base-uncased"
    config.question.hidden_dim = 128
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
        "qtype_onehot": torch.zeros(batch_size, 5)
    }

def test_integration_single_train_step(mock_batch):
    """Test that a single forward + backward + optimizer step works."""
    config = get_default_config()
    model = NeuroSymbolicPathVQA(
        answer_vocab_size=10,
        num_object_features=128,
        question_hidden_dim=128,
        symbolic_enabled=True,
        num_regions=5,
        visual_config={"model_name": "openai/clip-vit-base-patch32", "num_object_features": 128},
        question_config={"model_name": "distilbert-base-uncased", "hidden_dim": 128}
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    optimizer.zero_grad()
    
    outputs = model(
        images=mock_batch["image"],
        input_ids=mock_batch["input_ids"],
        attention_mask=mock_batch["attention_mask"],
        qtype_onehot=mock_batch["qtype_onehot"]
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
        question_hidden_dim=128,
        symbolic_enabled=True,
        num_regions=5,
        visual_config={"model_name": "openai/clip-vit-base-patch32", "num_object_features": 128},
        question_config={"model_name": "distilbert-base-uncased", "hidden_dim": 128}
    )
    model.eval()
    with torch.no_grad():
        outputs = model(
            images=mock_batch["image"],
            input_ids=mock_batch["input_ids"],
            attention_mask=mock_batch["attention_mask"],
            qtype_onehot=mock_batch["qtype_onehot"]
        )
    assert outputs.answer_logits.shape == (2, 10)

@patch("train.PathVQADataset")
def test_trainer_integration_loop(mock_dataset_class, mock_config):
    """Test the full Trainer loop with a mocked dataset."""
    mock_dataset = MagicMock()
    mock_dataset.answers = ["yes", "no", "lung", "red"]
    mock_dataset.answer_to_idx = {"yes": 0, "no": 1, "lung": 2, "red": 3}
    mock_dataset.idx_to_answer = {0: "yes", 1: "no", 2: "lung", 3: "red"}
    mock_dataset_class.return_value = mock_dataset

    with patch("train.DataLoader") as mock_dataloader:
        mock_dataloader.return_value = []
        trainer = Trainer(mock_config)
        assert trainer.config.experiment_name == "integration_test"
        assert trainer.device.type in ("cuda", "cpu")
        
        # Test that _write_baseline doesn't crash
        trainer.train_start_time = 0
        trainer.best_val_acc = 50.0
        trainer.best_val_epoch = 0
        trainer.best_val_per_type = {}
        trainer.epoch_times = [10.0]
        
        os.makedirs(mock_config.paths.output_dir, exist_ok=True)
        trainer._write_baseline()
        
        expected_output = mock_config.paths.output_dir / "integration_test.json"
        assert expected_output.exists()
        with open(expected_output, "r") as f:
            results = json.load(f)
            assert results["best_val_acc"] == 50.0
            assert results["experiment_name"] == "integration_test"
