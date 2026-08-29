"""Summarise a repeated-seed A–K ablation artifact.

The runner (``scripts/run_ablations.py``) stores one validation artifact per
variant/seed plus an aggregate. This script turns that aggregate into the table
the manuscript reports:

* per-variant mean / sample standard deviation over successful seeds;
* paired per-seed differences against a reference variant (same seeds only);
* seed-level bootstrap intervals for the mean difference.

All inputs are validation results. Test-split numbers are intentionally not
read here: the recipe is frozen with ``scripts/select_recipe.py`` and only then
evaluated once on test.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

TERMINAL_OK = "OK"


def _variant_seeds(variant: Dict[str, Any]) -> Dict[int, float]:
    """Return ``{seed: validation_accuracy}`` for successful seeds only."""
    runs = variant.get("runs") or {}
    result: Dict[int, float] = {}
    for seed_key, run in runs.items():
        if run.get("status") != TERMINAL_OK:
            continue
        accuracy = run.get("accuracy")
        if accuracy is None:
            continue
        result[int(run.get("seed", seed_key))] = float(accuracy)
    return result


def summarize_variants(aggregate: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per-variant mean/std/count over successful seeds."""
    summaries: Dict[str, Dict[str, Any]] = {}
    for name, variant in (aggregate.get("results") or {}).items():
        accuracies = list(_variant_seeds(variant).values())
        entry: Dict[str, Any] = {
            "ablation": variant.get("ablation"),
            "label": variant.get("label"),
            "status": variant.get("status"),
            "seeds_ok": len(accuracies),
            "seeds_expected": len(variant.get("runs") or {}),
            "accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
            "accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else None,
            "accuracy_min": float(np.min(accuracies)) if accuracies else None,
            "accuracy_max": float(np.max(accuracies)) if accuracies else None,
        }
        summaries[name] = entry
    return summaries


def paired_difference(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    bootstrap_samples: int = 10000,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """Paired candidate-minus-reference accuracy difference over shared seeds.

    Returns ``None`` when the two variants share no successful seed. With three
    seeds the interval is descriptive rather than a substitute for a larger
    repeated-seed study; the number of shared seeds is always reported.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    reference_seeds = _variant_seeds(reference)
    candidate_seeds = _variant_seeds(candidate)
    shared = sorted(set(reference_seeds) & set(candidate_seeds))
    if not shared:
        return None

    deltas = np.asarray([candidate_seeds[s] - reference_seeds[s] for s in shared], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    chunk = min(500, bootstrap_samples)
    for start in range(0, bootstrap_samples, chunk):
        stop = min(start + chunk, bootstrap_samples)
        sampled = rng.integers(0, len(deltas), size=(stop - start, len(deltas)))
        bootstrap[start:stop] = deltas[sampled].mean(axis=1)
    bootstrap.sort()
    return {
        "shared_seeds": shared,
        "n_shared_seeds": len(shared),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
        "per_seed_delta": {str(s): float(candidate_seeds[s] - reference_seeds[s]) for s in shared},
        "candidate_better_seeds": int(np.sum(deltas > 0)),
        "reference_better_seeds": int(np.sum(deltas < 0)),
        "tied_seeds": int(np.sum(deltas == 0)),
        "bootstrap_95ci": [
            float(bootstrap[int(0.025 * bootstrap_samples)]),
            float(bootstrap[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]),
        ],
        "interval_note": "seed-level percentile bootstrap; descriptive for small seed counts",
    }


def build_report(
    aggregate: Dict[str, Any],
    reference: str,
    bootstrap_samples: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Assemble the full analysis report for one aggregate artifact."""
    summaries = summarize_variants(aggregate)
    if reference not in summaries:
        raise KeyError(f"Reference variant {reference!r} not in aggregate results")
    comparisons = {}
    for name, variant in (aggregate.get("results") or {}).items():
        if name == reference:
            continue
        comparison = paired_difference(
            aggregate["results"][reference], variant, bootstrap_samples=bootstrap_samples, seed=seed
        )
        if comparison is not None:
            comparisons[name] = comparison
    return {
        "source": aggregate.get("source"),
        "seeds": aggregate.get("seeds"),
        "deterministic": aggregate.get("deterministic"),
        "debug": aggregate.get("debug"),
        "reference": reference,
        "variants": summaries,
        "paired_vs_reference": comparisons,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    """Markdown table for the manuscript appendix."""
    lines = [
        "| Variant | Seeds | Validation accuracy (mean ± std) | Min | Max | Δ vs reference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["variants"].items():
        mean = summary["accuracy_mean"]
        std = summary["accuracy_std"]
        value = "—" if mean is None else f"{mean:.2f}" + ("" if std is None else f" ± {std:.2f}")
        minimum = "—" if summary["accuracy_min"] is None else f"{summary['accuracy_min']:.2f}"
        maximum = "—" if summary["accuracy_max"] is None else f"{summary['accuracy_max']:.2f}"
        comparison = report["paired_vs_reference"].get(name)
        delta = "—"
        if comparison is not None:
            delta = f"{comparison['delta_mean']:+.2f}"
        lines.append(
            f"| {name} | {summary['seeds_ok']}/{summary['seeds_expected']} | {value} | "
            f"{minimum} | {maximum} | {delta} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True, help="Output of scripts/run_ablations.py")
    parser.add_argument("--reference", type=str, default="A_regex_hard_static")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    aggregate.setdefault("source", str(args.aggregate))
    report = build_report(
        aggregate,
        reference=args.reference,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["aggregate_sha256"] = hashlib.sha256(args.aggregate.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    incomplete = [name for name, summary in report["variants"].items() if summary["status"] != "OK"]
    if incomplete:
        print(f"Warning: incomplete variants: {', '.join(incomplete)}")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
