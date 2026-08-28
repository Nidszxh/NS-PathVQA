"""Unit tests for Logic Tensor Networks and Fuzzy Logic Operators."""

import pytest
import torch
from pathvqa.symbolic.ltn import FuzzyLogicEngine, MedicalLogicTensorNetwork


def test_fuzzy_logic_operators_range():
    engine = FuzzyLogicEngine()
    a = torch.tensor([0.0, 0.5, 1.0])
    b = torch.tensor([1.0, 0.5, 0.0])

    # AND
    and_prod = engine.AND(a, b, "product")
    assert torch.all((and_prod >= 0.0) & (and_prod <= 1.0))
    assert and_prod[0].item() == 0.0
    assert and_prod[1].item() == 0.25
    assert and_prod[2].item() == 0.0

    # OR
    or_prod = engine.OR(a, b, "product")
    assert torch.all((or_prod >= 0.0) & (or_prod <= 1.0))
    assert or_prod[0].item() == 1.0
    assert or_prod[1].item() == 0.75
    assert or_prod[2].item() == 1.0

    # NOT
    not_a = engine.NOT(a)
    assert torch.allclose(not_a, torch.tensor([1.0, 0.5, 0.0]))

    # IMPLIES
    imp = engine.IMPLIES(a, b, "product")
    assert torch.all((imp >= 0.0) & (imp <= 1.0))
    assert imp[0].item() == 1.0  # 0 -> 1 is True
    assert imp[2].item() == 0.0  # 1 -> 0 is False


def test_fuzzy_quantifiers():
    engine = FuzzyLogicEngine()
    x = torch.tensor([[0.9, 0.95, 0.99], [0.1, 0.2, 0.9]])

    # Forall
    forall_val = engine.FORALL(x, dim=-1)
    assert forall_val.shape == (2,)
    assert forall_val[0] > forall_val[1]

    # Exists
    exists_val = engine.EXISTS(x, dim=-1)
    assert exists_val.shape == (2,)
    assert exists_val[0] > 0.8
    assert exists_val[1] > 0.5


def test_medical_ltn_eval_and_grad():
    ltn = MedicalLogicTensorNetwork(num_regions=3, region_names=["lung", "liver", "heart"])
    raw_obj = torch.randn(4, 3, requires_grad=True)
    scene_logits = {
        "scene_region_logits": torch.randn(4, 3, requires_grad=True),
        "scene_object_presence": torch.sigmoid(raw_obj),
        "scene_color_logits": torch.randn(4, 17, requires_grad=True),
    }
    loss, scores = ltn.evaluate_clauses(scene_logits)
    assert loss.ndim == 0
    assert "region_object_coherence" in scores
    assert "overall_satisfaction" in scores

    # Backpropagation check
    loss.backward()
    assert scene_logits["scene_region_logits"].grad is not None
    assert raw_obj.grad is not None


def test_ltn_pathology_clauses():
    """Verify new pathology-specific clauses are evaluated."""
    ltn = MedicalLogicTensorNetwork(num_regions=3, region_names=["lung", "liver", "heart"])
    scene_logits = {
        "scene_region_logits": torch.randn(4, 3, requires_grad=True),
        "scene_object_presence": torch.sigmoid(torch.randn(4, 3)),
        "scene_color_logits": torch.randn(4, 17, requires_grad=True),
        "scene_shape_logits": torch.randn(4, 9, requires_grad=True),
        "scene_size_logits": torch.randn(4, 11, requires_grad=True),
    }
    loss, scores = ltn.evaluate_clauses(scene_logits)

    # Existing clauses
    assert "region_object_coherence" in scores
    assert "attribute_certainty" in scores
    # New pathology-specific clauses
    assert "morphology_region_coherence" in scores
    assert "attribute_confidence_union" in scores
    assert "region_sparsity" in scores
    assert "overall_satisfaction" in scores

    # Loss in valid range
    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 1.0

    # Gradient flows through new clauses
    loss.backward()
    assert scene_logits["scene_region_logits"].grad is not None
    assert scene_logits["scene_shape_logits"].grad is not None


def test_fuzzy_t_norm_variants():
    """Łukasiewicz and Gödel t-norms plus unknown-norm errors."""
    engine = FuzzyLogicEngine()
    a = torch.tensor([0.8, 0.3, 0.6])
    b = torch.tensor([0.7, 0.9, 0.2])

    and_luka = engine.AND(a, b, "lukasiewicz")
    torch.testing.assert_close(and_luka, torch.clamp(a + b - 1.0, min=0.0))
    assert engine.AND(a, b, "goedel").equal(torch.min(a, b))
    with pytest.raises(ValueError):
        engine.AND(a, b, "bogus")

    or_luka = engine.OR(a, b, "lukasiewicz")
    torch.testing.assert_close(or_luka, torch.clamp(a + b, max=1.0))
    assert engine.OR(a, b, "goedel").equal(torch.max(a, b))
    with pytest.raises(ValueError):
        engine.OR(a, b, "bogus")

    imp_luka = engine.IMPLIES(a, b, "lukasiewicz")
    torch.testing.assert_close(imp_luka, torch.clamp(1.0 - a + b, min=0.0, max=1.0))
    imp_godel = engine.IMPLIES(a, b, "goedel")
    assert imp_godel[1].item() == 1.0  # a <= b → True
    assert imp_godel[0].item() == pytest.approx(b[0].item())  # a > b → b
    with pytest.raises(ValueError):
        engine.IMPLIES(a, b, "bogus")


def test_ltn_missing_inputs_guard():
    """Missing scene logits must yield a differentiable zero loss."""
    ltn = MedicalLogicTensorNetwork(num_regions=3, region_names=["lung", "liver", "heart"])
    loss, scores = ltn.evaluate_clauses({})
    assert loss.item() == 0.0
    assert scores == {}
    loss.backward()  # still part of a graph
