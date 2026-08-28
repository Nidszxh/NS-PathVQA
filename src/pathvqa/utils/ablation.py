"""Canonical A-E ablation matrix (single source of truth).

Historically the matrix was duplicated between ``train.py`` and
``evaluate.py`` and could silently drift. Every entry point now imports these
definitions from here.

Matrix (see DOCS.md §5, "Ablation matrix A–E"):

    A  Baseline        Regex     Hard            Static
    B  SOTA            Regex     Hard            Learned
    C  Diff            Learned   Differentiable  Learned
    D  NoGate          Learned   Differentiable  Static
    E  Regex+Diff      Regex     Differentiable  Learned

Overrides must be applied *before* the model is constructed: ``query_parser``,
``executor`` and ``weighting_strategy`` all decide which modules are built.
"""

from typing import Any, Dict, List, Mapping

ABLATION_CONFIGS: Dict[str, Dict[str, str]] = {
    "A": {"query_parser": "regex", "executor": "hard", "weighting_strategy": "static"},
    "B": {"query_parser": "regex", "executor": "hard", "weighting_strategy": "learned"},
    "C": {"query_parser": "learned", "executor": "differentiable", "weighting_strategy": "learned"},
    "D": {"query_parser": "learned", "executor": "differentiable", "weighting_strategy": "static"},
    "E": {"query_parser": "regex", "executor": "differentiable", "weighting_strategy": "learned"},
}

ABLATION_LABELS: Dict[str, str] = {
    "A": "baseline",
    "B": "sota",
    "C": "differentiable",
    "D": "no_gate",
    "E": "regex_diffexec",
}

ABLATION_CHOICES: List[str] = sorted(ABLATION_CONFIGS)


def apply_ablation(config: Any, ablation: str) -> Any:
    """Apply the symbolic overrides for ``ablation`` (one of A-E) in place."""
    if ablation not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation {ablation!r}; expected one of {ABLATION_CHOICES}")
    overrides: Mapping[str, str] = ABLATION_CONFIGS[ablation]
    config.symbolic.query_parser = overrides["query_parser"]
    config.symbolic.executor = overrides["executor"]
    config.symbolic.weighting_strategy = overrides["weighting_strategy"]
    return config
