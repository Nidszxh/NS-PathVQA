"""Canonical symbolic ablation matrix and matched controls.

Historically the matrix was duplicated between ``train.py`` and
``evaluate.py`` and could silently drift. Every entry point now imports these
definitions from here.

Variants A–H form the complete parser × executor × weighting factorial. I is
the matched neural-only control; J removes LTN while retaining the auxiliary
losses; K removes both LTN and auxiliary symbolic losses. See CHANGELOG.md §5.

Overrides must be applied *before* the model is constructed: ``query_parser``,
``executor`` and ``weighting_strategy`` all decide which modules are built.
"""

from typing import Any, Dict, List, Mapping

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
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

ABLATION_LABELS: Dict[str, str] = {
    "A": "regex_hard_static",
    "B": "regex_hard_learned_gate",
    "C": "learned_diff_learned_gate",
    "D": "learned_diff_static",
    "E": "regex_diff_learned_gate",
    "F": "learned_hard_static",
    "G": "regex_diff_static",
    "H": "learned_hard_learned_gate",
    "I": "neural_only",
    "J": "learned_diff_static_no_ltn",
    "K": "learned_diff_static_no_aux",
}

ABLATION_CHOICES: List[str] = sorted(ABLATION_CONFIGS)


def apply_ablation(config: Any, ablation: str) -> Any:
    """Apply the symbolic overrides for one of the eleven variants in place."""
    if ablation not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation {ablation!r}; expected one of {ABLATION_CHOICES}")
    overrides: Mapping[str, Any] = ABLATION_CONFIGS[ablation]
    for name in (
        "query_parser",
        "executor",
        "weighting_strategy",
        "enabled",
        "ltn_enabled",
        "aux_region_weight",
        "aux_attr_weight",
        "aux_yn_weight",
        "ontology_sibling_weight",
    ):
        if name in overrides:
            setattr(config.symbolic, name, overrides[name])
    return config
