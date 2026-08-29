"""Freeze an ablation recipe using validation results only.

Selection must happen before the test split is touched, otherwise the reported
test accuracy is optimistically biased. This script:

1. reads a repeated-seed ablation aggregate (validation accuracies only);
2. keeps variants that completed every expected seed;
3. ranks by mean validation accuracy (default) or a conservative
   ``mean - std`` lower-confidence criterion;
4. writes a frozen-recipe artifact recording the aggregate hash, criterion,
   seeds, and per-variant evidence.

No test-split artifact is read or written. The chosen checkpoint is then
evaluated once on test with ``evaluate.py --split test``.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Optional


CRITERIA = ("mean", "mean_minus_std")


def _complete_accuracies(variant: Dict[str, Any]) -> Optional[list[float]]:
    """Return the accuracies if every expected seed finished, else ``None``."""
    runs = variant.get("runs") or {}
    if not runs:
        return None
    accuracies: list[float] = []
    for run in runs.values():
        if run.get("status") != "OK" or run.get("accuracy") is None:
            return None
        accuracies.append(float(run["accuracy"]))
    return accuracies


def rank_variants(aggregate: Dict[str, Any], criterion: str = "mean", min_seeds: int = 2) -> list[dict[str, Any]]:
    """Rank complete variants; ties break by variant name for determinism.

    ``min_seeds`` excludes variants with too few completed seeds. Ranking
    single-seed results would simply select the luckiest seed, so at least two
    seeds are required by default.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    ranked: list[dict[str, Any]] = []
    for name, variant in (aggregate.get("results") or {}).items():
        accuracies = _complete_accuracies(variant)
        if accuracies is None or len(accuracies) < min_seeds:
            continue
        mean = statistics.fmean(accuracies)
        std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        score = mean - std if criterion == "mean_minus_std" else mean
        ranked.append(
            {
                "variant": name,
                "ablation": variant.get("ablation"),
                "label": variant.get("label"),
                "seeds": len(accuracies),
                "mean_validation_accuracy": mean,
                "std_validation_accuracy": std if len(accuracies) > 1 else None,
                "score": score,
                "criterion": criterion,
            }
        )
    ranked.sort(key=lambda row: (-row["score"], row["variant"]))
    return ranked


def freeze_recipe(
    aggregate: Dict[str, Any],
    aggregate_path: Path,
    criterion: str = "mean",
    min_seeds: int = 2,
) -> Dict[str, Any]:
    """Build the frozen-recipe artifact; raises if no variant is complete."""
    ranked = rank_variants(aggregate, criterion, min_seeds=min_seeds)
    if not ranked:
        raise ValueError(
            f"No variant completed at least {min_seeds} seed(s); nothing can be frozen. "
            "Single-seed rankings select the luckiest run rather than a better recipe; "
            "use --min-seeds 1 only for a descriptive ranking that selects no reported model."
        )
    winner = ranked[0]
    return {
        "selection_split": "validation",
        "test_split_used_for_selection": False,
        "criterion": criterion,
        "aggregate": str(aggregate_path),
        "aggregate_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
        "seeds": aggregate.get("seeds"),
        "deterministic": aggregate.get("deterministic"),
        "debug": aggregate.get("debug"),
        "selected": winner["variant"],
        "selected_ablation": winner["ablation"],
        "selected_mean_validation_accuracy": winner["mean_validation_accuracy"],
        "selected_std_validation_accuracy": winner["std_validation_accuracy"],
        "ranking": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/frozen_recipe.json"))
    parser.add_argument("--criterion", choices=CRITERIA, default="mean")
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=2,
        help=(
            "Minimum completed seeds for a variant to be eligible (default: 2). "
            "Single-seed aggregates cannot be frozen; pass 1 only for a descriptive ranking."
        ),
    )
    args = parser.parse_args()

    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    recipe = freeze_recipe(aggregate, args.aggregate, criterion=args.criterion, min_seeds=args.min_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {recipe['selected']} on validation ({recipe['criterion']}).")
    print(f"Frozen recipe written to {args.output}")


if __name__ == "__main__":
    main()
