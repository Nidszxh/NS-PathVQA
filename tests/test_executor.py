import torch

from symbolic.executor import (
    COLOR_VALUES,
    SHAPE_VALUES,
    SIZE_VALUES,
    build_attribute_mappings,
    build_region_mapping,
    build_region_names,
    execute,
    VectorizedSymbolicExecutor,
)
from symbolic.query_parser import Query


def _make_scene_logits(batch_size, num_regions):
    return {
        "scene_region_logits": torch.randn(batch_size, num_regions),
        "scene_color_logits": torch.randn(batch_size, len(COLOR_VALUES)),
        "scene_shape_logits": torch.randn(batch_size, len(SHAPE_VALUES)),
        "scene_size_logits": torch.randn(batch_size, len(SIZE_VALUES)),
    }


def test_build_region_names_filters_vocab(answer_vocab):
    regions = build_region_names(answer_vocab)
    assert "gastrointestinal system" in regions
    assert "lung" in regions
    assert "yes" not in regions
    assert "red" not in regions


def test_build_region_mapping_missing_entries(answer_to_idx):
    mapping = build_region_mapping(["lung", "unknown"], answer_to_idx)
    assert mapping.tolist() == [
        answer_to_idx["lung"],
        -1,
    ]


def test_execute_identity_boosts_region_entries(answer_vocab, answer_to_idx):
    regions = build_region_names(answer_vocab)
    region_mapping = build_region_mapping(regions, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)
    scene_logits = _make_scene_logits(1, len(regions))
    queries = [Query(qtype="identity", target="lung")]
    neural = torch.zeros(1, len(answer_vocab))
    result = execute(
        scene_logits, queries, regions, region_mapping, attr_mappings,
        answer_to_idx, len(answer_vocab), neural,
    )
    assert result["trace"]["symbolic_used"] == [True]
    assert result["symbolic_logits"].shape == (1, len(answer_vocab))
    assert result["symbolic_logits"][0, answer_to_idx["lung"]] != 0


def test_execute_yes_no_boosts_binary_classes(answer_vocab, answer_to_idx):
    """Yes/no questions should boost yes/no indices, not the target word index."""
    regions = build_region_names(answer_vocab)
    region_mapping = build_region_mapping(regions, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)
    scene_logits = _make_scene_logits(1, len(regions))
    queries = [Query(qtype="yes_no", target="lung")]
    neural = torch.zeros(1, len(answer_vocab))
    result = execute(
        scene_logits, queries, regions, region_mapping, attr_mappings,
        answer_to_idx, len(answer_vocab), neural,
    )
    assert result["trace"]["symbolic_used"] == [True]
    assert result["symbolic_logits"].shape == (1, len(answer_vocab))
    # Should NOT boost the target word index (e.g., "lung")
    assert result["symbolic_logits"][0, answer_to_idx["lung"]] == 0
    # Should boost yes or no based on region confidence
    yn_val = result["symbolic_logits"][0, answer_to_idx["yes"]]
    no_val = result["symbolic_logits"][0, answer_to_idx["no"]]
    assert yn_val != 0 or no_val != 0, "yes or no should be boosted"


def test_execute_yes_no_uses_max_region_confidence(answer_vocab, answer_to_idx):
    regions = build_region_names(answer_vocab)
    region_mapping = build_region_mapping(regions, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)
    scene_logits = _make_scene_logits(1, len(regions))
    queries = [Query(qtype="yes_no", target="lung")]
    neural = torch.zeros(1, len(answer_vocab))
    result = execute(
        scene_logits, queries, regions, region_mapping, attr_mappings,
        answer_to_idx, len(answer_vocab), neural,
    )
    yn_val = result["symbolic_logits"][0, answer_to_idx["yes"]]
    no_val = result["symbolic_logits"][0, answer_to_idx["no"]]
    assert yn_val != 0 or no_val != 0


def test_execute_count_is_noop(answer_vocab, answer_to_idx):
    regions = build_region_names(answer_vocab)
    region_mapping = build_region_mapping(regions, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)
    scene_logits = _make_scene_logits(2, len(regions))
    queries = [Query(qtype="count", target="nuclei")] * 2
    neural = torch.zeros(2, len(answer_vocab))
    result = execute(
        scene_logits, queries, regions, region_mapping, attr_mappings,
        answer_to_idx, len(answer_vocab), neural,
    )
    assert result["trace"]["symbolic_used"] == [False, False]
    assert torch.equal(result["symbolic_logits"], torch.zeros_like(result["symbolic_logits"]))


def test_execute_missing_region_logits_returns_zeros(answer_vocab, answer_to_idx):
    regions = build_region_names(answer_vocab)
    region_mapping = build_region_mapping(regions, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)
    queries = [Query(qtype="identity")]
    neural = torch.zeros(1, len(answer_vocab))
    result = execute(
        {}, queries, regions, region_mapping, attr_mappings,
        answer_to_idx, len(answer_vocab), neural,
    )
    assert result["region_logits"] is None
    assert torch.equal(result["symbolic_logits"], torch.zeros_like(result["symbolic_logits"]))
    assert result["trace"]["symbolic_used"] == [False]


def test_execute_mixed_qtypes(answer_vocab, answer_to_idx):
    """Test batched execution with mixed question types."""
    regions = build_region_names(answer_vocab)
    region_mapping = build_region_mapping(regions, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)
    batch_size = 4
    scene_logits = _make_scene_logits(batch_size, len(regions))
    queries = [
        Query(qtype="identity", target="lung"),
        Query(qtype="yes_no", target="lung"),
        Query(qtype="attribute", attribute="color"),
        Query(qtype="count", target="nuclei"),
    ]
    neural = torch.zeros(batch_size, len(answer_vocab))
    result = execute(
        scene_logits, queries, regions, region_mapping, attr_mappings,
        answer_to_idx, len(answer_vocab), neural,
    )
    assert result["trace"]["symbolic_used"] == [True, True, True, False]
    assert result["symbolic_logits"].shape == (batch_size, len(answer_vocab))


def test_vectorized_executor_matches_execute(answer_vocab, answer_to_idx):
    """VectorizedSymbolicExecutor wrapper should produce same output as execute()."""
    regions = build_region_names(answer_vocab)
    region_mapping = build_region_mapping(regions, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)
    scene_logits = _make_scene_logits(2, len(regions))
    queries = [
        Query(qtype="identity", target="lung"),
        Query(qtype="yes_no", target="lung"),
    ]
    neural = torch.zeros(2, len(answer_vocab))

    result = execute(
        scene_logits, queries, regions, region_mapping, attr_mappings,
        answer_to_idx, len(answer_vocab), neural,
    )
    vec_executor = VectorizedSymbolicExecutor(regions, answer_to_idx, attr_mappings)
    vec_out = vec_executor(scene_logits, queries, neural)
    assert torch.allclose(result["symbolic_logits"], vec_out["symbolic_logits"])
