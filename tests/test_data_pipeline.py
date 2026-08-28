"""Dataset loading on synthetic fixtures and batch preparation shapes."""

from PIL import Image
import torch
from unittest.mock import patch, MagicMock

from pathvqa.data.pathvqa_dataset import PathVQADataset, prepare_batch


def test_pathvqa_dataset_synthetic_fixtures(tmp_path):
    """Test PathVQADataset using a synthetic HF dataset fixture."""
    # Create synthetic dataset structure
    synthetic_hf_dataset = [
        {"image": Image.new("RGB", (320, 240), color="red"), "question": "what color is this?", "answer": "red"},
        {"image": Image.new("RGB", (224, 224), color="blue"), "question": "is this blue?", "answer": "yes"},
    ]

    # Mock the loader where it is bound; return a real (offline) HF Dataset
    import datasets

    with patch("pathvqa.data.pathvqa_dataset.load_dataset") as mock_load:
        mock_load.return_value = datasets.Dataset.from_list(synthetic_hf_dataset)

        # Initialize dataset (should not hit HF)
        ds = PathVQADataset(split="train", cache_dir=tmp_path / "cache", use_cache=False)

        assert len(ds) == 2

        # Test item retrieval
        item = ds[0]
        assert "image" in item
        assert "question" in item
        assert "answer_idx" in item

        # Verify shape of image tensor
        assert item["image"].shape == (3, 224, 224)


def test_prepare_batch_shapes():
    """Test the output shapes from prepare_batch."""
    from pathvqa.data.pathvqa_dataset import CollateOutput

    # Create a dummy CollateOutput
    batch = CollateOutput(
        images=torch.randn(2, 3, 224, 224),
        targets=torch.tensor([0, 1]),
        questions=["what is this?", "where is it?"],
        answers=["lung", "chest"],
    )

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.randint(0, 30000, (2, 10)), "attention_mask": torch.ones(2, 10)}

    prepared = prepare_batch(batch, tokenizer=tokenizer, device=torch.device("cpu"), max_seq_len=10)

    assert prepared.images.shape == (2, 3, 224, 224)
    assert prepared.input_ids.shape == (2, 10)
    assert prepared.attention_mask.shape == (2, 10)
    assert prepared.targets.shape == (2,)
    assert len(prepared.questions) == 2
