import pytest
import torch

from pathvqa.symbolic.diff_query_classifier import DifferentiableQueryClassifier
from pathvqa.symbolic.diff_target_extractor import DifferentiableTargetExtractor
from pathvqa.symbolic.diff_executor import DifferentiableExecutor
from pathvqa.models.pathvqa_model import build_model
from pathvqa.utils.config import get_default_config

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def hidden_dim():
    return 768

@pytest.fixture
def num_regions():
    return 50

def test_diff_query_classifier_shapes(batch_size, hidden_dim):
    classifier = DifferentiableQueryClassifier(hidden_dim=hidden_dim, num_classes=5)
    question_state = torch.randn(batch_size, hidden_dim)
    
    # Test soft probabilities
    probs = classifier(question_state, return_probs=True)
    assert probs.shape == (batch_size, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))
    assert (probs >= 0).all() and (probs <= 1).all()
    
    # Test raw logits
    logits = classifier(question_state, return_probs=False)
    assert logits.shape == (batch_size, 5)


def test_diff_target_extractor_shapes(batch_size, hidden_dim, num_regions):
    extractor = DifferentiableTargetExtractor(hidden_dim=hidden_dim, num_regions=num_regions)
    question_state = torch.randn(batch_size, hidden_dim)
    region_embeddings = torch.randn(num_regions, 512)
    
    probs = extractor(question_state, region_embeddings, return_probs=True)
    assert probs.shape == (batch_size, num_regions)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))
    
    logits = extractor(question_state, region_embeddings, return_probs=False)
    assert logits.shape == (batch_size, num_regions)


def test_diff_executor_shapes(batch_size, num_regions):
    answer_to_idx = {"yes": 0, "no": 1, "lung": 2, "liver": 3, "red": 4, "round": 5}
    region_names = [f"region_{i}" for i in range(num_regions)]
    region_names[0] = "lung"
    region_names[1] = "liver"
    
    executor = DifferentiableExecutor(answer_to_idx, region_names)
    
    vocab_size = len(answer_to_idx)
    soft_qtype = torch.randn(batch_size, 5).softmax(dim=-1)
    soft_target = torch.randn(batch_size, num_regions).softmax(dim=-1)
    
    scene_logits = {
        "scene_region_logits": torch.randn(batch_size, num_regions),
        "scene_color_logits": torch.randn(batch_size, 17),
        "scene_shape_logits": torch.randn(batch_size, 9),
        "scene_size_logits": torch.randn(batch_size, 11)
    }
    
    symbolic_logits = executor.execute(scene_logits, soft_qtype, soft_target)
    assert symbolic_logits.shape == (batch_size, vocab_size)


def test_diff_pipeline_integration(batch_size):
    config = get_default_config()
    config.symbolic.enabled = True
    config.symbolic.differentiable_pipeline = True
    config.symbolic.num_regions = 10
    
    model = build_model(config, answer_vocab_size=100)
    
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 10))
    attention_mask = torch.ones(batch_size, 10)
    
    with torch.no_grad():
        out = model(images, input_ids, attention_mask)
        
    assert hasattr(out, "soft_qtype")
    assert out.soft_qtype.shape == (batch_size, 5)
    
    assert hasattr(out, "soft_target")
    assert out.soft_target.shape == (batch_size, 10)
    
    assert hasattr(out, "qtype_logits")
    assert out.qtype_logits.shape == (batch_size, 5)
    
    assert hasattr(out, "target_logits")
    assert out.target_logits.shape == (batch_size, 10)
