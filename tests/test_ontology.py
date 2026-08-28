"""Tests for ontology-conditioned executor and sibling regularizer."""

from pathvqa.data.dataset_adapter import AnatomicalOntology
from pathvqa.models.protocols import SceneParserOutput
from pathvqa.symbolic.executor import ontology_fallback, execute, build_region_mapping, build_attribute_mappings
from pathvqa.symbolic.ontology_loss import compute_sibling_regularization
from pathvqa.symbolic.query_parser import Query
import torch


def test_ontology_fallback_synonym():
    ontology = AnatomicalOntology()
    answer_to_idx = {"stomach": 0, "lung": 1, "liver": 2, "yes": 3, "no": 4}
    # "gastric" is a synonym for "stomach"
    idx = ontology_fallback("gastric", ontology, answer_to_idx)
    assert idx == 0


def test_ontology_fallback_parent():
    ontology = AnatomicalOntology()
    answer_to_idx = {"gastrointestinal system": 0, "lung": 1, "yes": 2}
    # "colon" is in the gastrointestinal system
    idx = ontology_fallback("colon", ontology, answer_to_idx)
    assert idx == 0


def test_ontology_fallback_none():
    ontology = AnatomicalOntology()
    answer_to_idx = {"lung": 0}
    idx = ontology_fallback("xyz_nonexistent", ontology, answer_to_idx)
    assert idx is None


def test_ontology_fallback_empty_target():
    ontology = AnatomicalOntology()
    idx = ontology_fallback("", ontology, {"lung": 0})
    assert idx is None


def test_execute_with_ontology():
    ontology = AnatomicalOntology()
    region_names = ["lung", "liver"]
    answer_vocab = ["yes", "no", "lung", "liver", "gastric", "stomach"]
    answer_to_idx = {a: i for i, a in enumerate(answer_vocab)}
    region_to_answer_idx = build_region_mapping(region_names, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)

    scene_logits = SceneParserOutput(
        region_logits=torch.randn(2, len(region_names)),
        object_presence=torch.sigmoid(torch.randn(2, len(region_names))),
        color_logits=torch.randn(2, 17),
        shape_logits=torch.randn(2, 9),
        size_logits=torch.randn(2, 11),
    )
    queries = [
        Query(qtype="identity", target="gastric"),  # not in region_names, but synonym of "stomach"
        Query(qtype="yes_no", target="lung"),
    ]
    neural_logits = torch.randn(2, len(answer_vocab))

    result = execute(
        scene_logits,
        queries,
        region_names,
        region_to_answer_idx,
        attr_mappings,
        answer_to_idx,
        len(answer_vocab),
        neural_logits,
        ontology=ontology,
    )
    # "gastric" should be mapped to "stomach" via ontology fallback
    # "stomach" is not in region_names but "gastric" maps to it
    assert result.symbolic_logits.shape == (2, len(answer_vocab))
    assert result.symbolic_logits[0, answer_to_idx["stomach"]] != 0.0


def test_sibling_regularization():
    ontology = AnatomicalOntology()
    # colon and stomach are both in gastrointestinal system → should be siblings
    region_names = ["colon", "stomach", "lung"]
    weight = torch.randn(3, 64, requires_grad=True)
    loss = compute_sibling_regularization(weight, region_names, ontology)
    assert loss.ndim == 0
    # colon and stomach are related → loss should be non-zero
    assert loss.item() > 0
    loss.backward()
    assert weight.grad is not None


def test_ontology_fallback_substring_match():
    """Direct substring match on the answer vocabulary."""
    ontology = AnatomicalOntology()
    answer_to_idx = {"lung carcinoma": 0, "yes": 1}
    idx = ontology_fallback("lung", ontology, answer_to_idx)
    assert idx == 0


def test_ontology_fallback_parent_partial_match():
    """Parent-system traversal resolves an organ to its system answer."""
    ontology = AnatomicalOntology()
    # "colon" belongs to the GI system; the system name is an answer entry.
    idx = ontology_fallback("colon", ontology, {"lung": 1, "gastrointestinal system": 0})
    assert idx == 0


def test_execute_identity_synonym_boosts_fallback_answer():
    """A synonym target (not a region) must be boosted via the ontology fallback."""
    ontology = AnatomicalOntology()
    region_names = ["lung", "liver"]
    answer_vocab = ["yes", "no", "lung", "liver", "stomach"]
    answer_to_idx = {a: i for i, a in enumerate(answer_vocab)}
    region_to_answer_idx = build_region_mapping(region_names, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)

    scene_logits = SceneParserOutput(
        region_logits=torch.randn(1, len(region_names)),
        object_presence=torch.sigmoid(torch.randn(1, len(region_names))),
        color_logits=torch.randn(1, 17),
        shape_logits=torch.randn(1, 9),
        size_logits=torch.randn(1, 11),
    )
    queries = [Query(qtype="identity", target="gastric")]
    neural_logits = torch.zeros(1, len(answer_vocab))

    result = execute(
        scene_logits,
        queries,
        region_names,
        region_to_answer_idx,
        attr_mappings,
        answer_to_idx,
        len(answer_vocab),
        neural_logits,
        ontology=ontology,
    )
    # "gastric" → "stomach" via ontology; its logit should be boosted above zeros
    stomach_idx = answer_to_idx["stomach"]
    assert result.symbolic_logits[0, stomach_idx] != 0.0


def test_execute_no_target_boosts_all_regions():
    """Empty target boosts every region-mapped answer."""
    region_names = ["lung", "liver"]
    answer_vocab = ["yes", "no", "lung", "liver"]
    answer_to_idx = {a: i for i, a in enumerate(answer_vocab)}
    region_to_answer_idx = build_region_mapping(region_names, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)

    scene_logits = SceneParserOutput(
        region_logits=torch.rand(2, len(region_names)),
        object_presence=torch.sigmoid(torch.randn(2, len(region_names))),
        color_logits=torch.randn(2, 17),
        shape_logits=torch.randn(2, 9),
        size_logits=torch.randn(2, 11),
    )
    queries = [Query(qtype="identity", target=""), Query(qtype="identity", target="")]
    neural_logits = torch.zeros(2, len(answer_vocab))

    result = execute(
        scene_logits,
        queries,
        region_names,
        region_to_answer_idx,
        attr_mappings,
        answer_to_idx,
        len(answer_vocab),
        neural_logits,
    )
    lung_idx = answer_to_idx["lung"]
    liver_idx = answer_to_idx["liver"]
    assert (result.symbolic_logits[:, lung_idx] > 0).all()
    assert (result.symbolic_logits[:, liver_idx] > 0).all()
