"""Shape contracts for every encoder, head, and fusion module."""

import pytest
import torch

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
    fusion = CrossModalTransformer(text_dim=768, visual_dim=512, hidden_dim=512, num_heads=8)
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
    model = build_model(config, 100)

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


def test_multiscale_encoder_crop_branch_changes_features():
    """The crop branch must actually contribute when enabled.

    Regression: the multi-scale encoder was constructed but every caller used
    the global-only default, so the high-resolution crop branch was dead code.
    """
    from pathvqa.models.visual.visual_encoder import MultiScaleVisualEncoder

    encoder = MultiScaleVisualEncoder(
        model_name="openai/clip-vit-base-patch32",
        num_object_features=64,
        num_objects=49,
    )
    encoder.eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        global_out = encoder(images, use_multiscale=False)
        ms_out = encoder(images, use_multiscale=True)
    assert global_out.features.shape == ms_out.features.shape == (2, 49, 64)
    # The learned fusion blend must change the representation.
    assert not torch.allclose(global_out.features, ms_out.features)


def test_model_executes_multiscale_branch_by_default():
    """A model built with use_multiscale=True must default to the crop branch.

    Regression: ``forward`` used to default to global-only regardless of how
    the model was constructed, so a configured multi-scale model silently
    evaluated the wrong architecture.
    """
    from pathvqa.models.pathvqa_model import NeuroSymbolicPathVQA

    torch.manual_seed(0)
    model = NeuroSymbolicPathVQA(
        answer_vocab_size=5,
        num_object_features=64,
        question_hidden_dim=768,
        symbolic_enabled=False,
        num_regions=4,
        visual_config={"model_name": "openai/clip-vit-base-patch32", "num_object_features": 64, "num_objects": 49},
        question_config={"model_name": "distilbert-base-uncased"},
        use_multiscale=True,
    )
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)

    with torch.no_grad():
        default_out = model(images, input_ids, attention_mask)
        global_out = model(images, input_ids, attention_mask, use_multiscale=False)

    # Default (crop branch) must differ from the explicit global-only path.
    assert not torch.allclose(default_out.answer_logits, global_out.answer_logits)
    assert model.use_multiscale is True
