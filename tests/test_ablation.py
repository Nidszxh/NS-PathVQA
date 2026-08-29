"""Tests for the canonical full-factorial ablation matrix."""

import pytest

from pathvqa.utils.ablation import (
    ABLATION_CHOICES,
    ABLATION_CONFIGS,
    ABLATION_LABELS,
    apply_ablation,
)
from pathvqa.utils.config import get_default_config


def test_matrix_values_match_documented_semantics():
    """Factorial variants and matched learning controls must be present."""
    assert ABLATION_CONFIGS == {
        "A": {"query_parser": "regex", "executor": "hard", "weighting_strategy": "static"},
        "B": {"query_parser": "regex", "executor": "hard", "weighting_strategy": "learned"},
        "C": {"query_parser": "learned", "executor": "differentiable", "weighting_strategy": "learned"},
        "D": {"query_parser": "learned", "executor": "differentiable", "weighting_strategy": "static"},
        "E": {"query_parser": "regex", "executor": "differentiable", "weighting_strategy": "learned"},
        "F": {"query_parser": "learned", "executor": "hard", "weighting_strategy": "static"},
        "G": {"query_parser": "regex", "executor": "differentiable", "weighting_strategy": "static"},
        "H": {"query_parser": "learned", "executor": "hard", "weighting_strategy": "learned"},
        "I": {
            "query_parser": "learned",
            "executor": "differentiable",
            "weighting_strategy": "static",
            "enabled": False,
            "ltn_enabled": False,
            "aux_region_weight": 0.0,
            "aux_attr_weight": 0.0,
            "aux_yn_weight": 0.0,
            "ontology_sibling_weight": 0.0,
        },
        "J": {
            "query_parser": "learned",
            "executor": "differentiable",
            "weighting_strategy": "static",
            "ltn_enabled": False,
        },
        "K": {
            "query_parser": "learned",
            "executor": "differentiable",
            "weighting_strategy": "static",
            "ltn_enabled": False,
            "aux_region_weight": 0.0,
            "aux_attr_weight": 0.0,
            "aux_yn_weight": 0.0,
            "ontology_sibling_weight": 0.0,
        },
    }


def test_choices_and_labels_cover_every_key():
    assert ABLATION_CHOICES == sorted(ABLATION_CONFIGS)
    assert set(ABLATION_LABELS) == set(ABLATION_CONFIGS)
    assert len(set(ABLATION_LABELS.values())) == len(ABLATION_LABELS)


@pytest.mark.parametrize("key", sorted(ABLATION_CONFIGS))
def test_apply_ablation_sets_all_three_fields(key):
    cfg = get_default_config()
    apply_ablation(cfg, key)
    expected = ABLATION_CONFIGS[key]
    assert cfg.symbolic.query_parser == expected["query_parser"]
    assert cfg.symbolic.executor == expected["executor"]
    assert cfg.symbolic.weighting_strategy == expected["weighting_strategy"]


def test_apply_ablation_unknown_raises():
    cfg = get_default_config()
    with pytest.raises(ValueError, match="Unknown ablation"):
        apply_ablation(cfg, "Z")


def test_evaluate_reexports_canonical_matrix():
    """evaluate.py must re-export the canonical definitions, not a copy.

    The matrix used to be duplicated between train.py and evaluate.py and could
    drift; guard against a future re-introduction of a private copy.
    """
    import evaluate

    assert evaluate.ABLATION_CONFIGS is ABLATION_CONFIGS
    assert evaluate.apply_ablation is apply_ablation


def test_train_cli_ablation_override_uses_canonical_matrix():
    from train import apply_cli_overrides

    cfg = get_default_config()
    apply_cli_overrides(cfg, {"ablation": "E"})
    assert cfg.symbolic.query_parser == "regex"
    assert cfg.symbolic.executor == "differentiable"
    assert cfg.symbolic.weighting_strategy == "learned"
