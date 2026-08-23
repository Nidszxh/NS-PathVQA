import pytest
import torch
from unittest.mock import patch

from pathvqa.models.visual.visual_encoder import CLIPViTEncoder
from pathvqa.models.question.question_encoder import DistilBERTQuestionEncoder
from pathvqa.models.fusion import CrossModalTransformer
from pathvqa.symbolic.scene_parser import SceneParser
from pathvqa.symbolic.routing import LearnedGate

@pytest.fixture
def batch_size():
    return 2

def test_visual_encoder_shapes(batch_size):
    encoder = CLIPViTEncoder(model_name="openai/clip-vit-base-patch32", num_object_features=512)
    images = torch.randn(batch_size, 3, 224, 224)
    # Using torch.no_grad to avoid backward graph memory on a quick shape test
    with torch.no_grad():
        out = encoder(images)
    
    assert hasattr(out, "features")
    assert hasattr(out, "mask")
    assert out.features.shape == (batch_size, 49, 512)
    assert out.mask.shape == (batch_size, 49)

def test_question_encoder_shapes(batch_size):
    encoder = DistilBERTQuestionEncoder(model_name="distilbert-base-uncased")
    input_ids = torch.randint(0, 1000, (batch_size, 10))
    attention_mask = torch.ones(batch_size, 10)
    
    with torch.no_grad():
        out = encoder(input_ids, attention_mask)
        
    assert hasattr(out, "question_state")
    assert out.question_state.shape == (batch_size, 768)

def test_cross_modal_transformer_shapes(batch_size):
    fusion = CrossModalTransformer(
        question_dim=768, visual_dim=512, hidden_dim=512, num_heads=8
    )
    question_state = torch.randn(batch_size, 768)
    visual_features = torch.randn(batch_size, 49, 512)
    mask = torch.ones(batch_size, 49, dtype=torch.bool)
    
    with torch.no_grad():
        attended, weights = fusion(question_state, visual_features, mask)
        
    assert attended.shape == (batch_size, 512)
    assert weights.shape == (batch_size, 8, 1, 49)

def test_scene_parser_shapes(batch_size):
    num_regions = 50
    parser = SceneParser(visual_dim=512, num_regions=num_regions)
    attended_features = torch.randn(batch_size, 512)
    
    with torch.no_grad():
        out = parser(attended_features)
        
    assert out.region_logits.shape == (batch_size, num_regions)
    assert out.object_presence.shape == (batch_size, num_regions)
    assert out.color_logits.shape == (batch_size, 17)
    assert out.shape_logits.shape == (batch_size, 9)
    assert out.size_logits.shape == (batch_size, 11)

def test_learned_gate_shapes(batch_size):
    num_regions = 50
    gate = LearnedGate(visual_dim=512, num_regions=num_regions, num_qtypes=5)
    
    h_attn = torch.randn(batch_size, 512)
    c_scene = torch.randn(batch_size, 1)  # max across regions
    qtype_onehot = torch.randn(batch_size, 5)
    attn_weights = torch.randn(batch_size, 8, 1, 49)
    
    with torch.no_grad():
        g = gate(h_attn, c_scene, qtype_onehot, attn_weights)
        
    assert g.shape == (batch_size, 1)

def test_model_output_shapes(batch_size):
    from pathvqa.models.pathvqa_model import build_model
    from pathvqa.utils.config import get_default_config
    
    config = get_default_config()
    model = build_model(config, num_answers=100)
    
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 10))
    attention_mask = torch.ones(batch_size, 10)
    
    with torch.no_grad():
        out = model(images, input_ids, attention_mask)
        
    assert hasattr(out, "answer_logits")
    assert out.answer_logits.shape == (batch_size, 100)
    if config.symbolic.enabled:
        assert hasattr(out, "scene_region_logits")
        assert out.scene_region_logits is not None
