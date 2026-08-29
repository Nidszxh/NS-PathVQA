"""Analysis and validation-only recipe freezing for repeated-seed ablations."""

import json

import pytest

from scripts.analyze_ablations import build_report, paired_difference, summarize_variants
from scripts.select_recipe import freeze_recipe, rank_variants


def _variant(ablation, accuracies, failed_seed=None):
    runs = {str(seed): {"status": "OK", "seed": seed, "accuracy": accuracy} for seed, accuracy in accuracies.items()}
    if failed_seed is not None:
        runs[str(failed_seed)] = {"status": "EVAL_FAILED", "seed": failed_seed, "error": "boom"}
    return {
        "status": "OK" if failed_seed is None else "PARTIAL",
        "ablation": ablation,
        "label": f"label-{ablation}",
        "runs": runs,
    }


def _aggregate():
    return {
        "seeds": [1, 2, 3],
        "deterministic": True,
        "debug": False,
        "results": {
            "A_regex_hard_static": _variant("A", {1: 60.0, 2: 61.0, 3: 62.0}),
            "B_regex_hard_learned": _variant("B", {1: 61.0, 2: 61.0, 3: 63.0}),
            "C_regex_diff_static": _variant("C", {1: 65.0, 2: 65.0}, failed_seed=3),
        },
    }


def test_summary_uses_only_successful_seeds():
    summaries = summarize_variants(_aggregate())

    assert summaries["A_regex_hard_static"]["seeds_ok"] == 3
    assert summaries["A_regex_hard_static"]["accuracy_mean"] == pytest.approx(61.0)
    assert summaries["A_regex_hard_static"]["accuracy_std"] == pytest.approx(1.0)
    # The failed third seed is not silently imputed or averaged in.
    assert summaries["C_regex_diff_static"]["seeds_ok"] == 2
    assert summaries["C_regex_diff_static"]["accuracy_std"] == pytest.approx(0.0)


def test_paired_difference_uses_shared_seeds_only():
    aggregate = _aggregate()
    comparison = paired_difference(
        aggregate["results"]["A_regex_hard_static"],
        aggregate["results"]["B_regex_hard_learned"],
        bootstrap_samples=200,
        seed=0,
    )

    assert comparison is not None
    assert comparison["shared_seeds"] == [1, 2, 3]
    assert comparison["delta_mean"] == pytest.approx(2.0 / 3.0)
    assert comparison["per_seed_delta"] == {"1": 1.0, "2": 0.0, "3": 1.0}
    assert comparison["candidate_better_seeds"] == 2
    assert comparison["tied_seeds"] == 1
    assert comparison["bootstrap_95ci"][0] <= comparison["delta_mean"] <= comparison["bootstrap_95ci"][1]

    disjoint = paired_difference(
        aggregate["results"]["A_regex_hard_static"],
        _variant("Z", {99: 50.0}),
        bootstrap_samples=50,
    )
    assert disjoint is None


def test_report_includes_pairings_against_reference():
    report = build_report(_aggregate(), reference="A_regex_hard_static", bootstrap_samples=100, seed=0)
    # C is compared on its two shared seeds and its incompleteness stays visible
    # in the per-variant table, so an interrupted variant cannot look complete.
    assert set(report["paired_vs_reference"]) == {"B_regex_hard_learned", "C_regex_diff_static"}
    assert report["paired_vs_reference"]["C_regex_diff_static"]["n_shared_seeds"] == 2
    assert report["variants"]["C_regex_diff_static"]["seeds_ok"] == 2
    assert report["variants"]["C_regex_diff_static"]["seeds_expected"] == 3


def test_ranking_keeps_only_complete_variants_and_breaks_ties_by_name():
    aggregate = {
        "seeds": [1, 2],
        "results": {
            "Z_tie": _variant("Z", {1: 60.0, 2: 60.0}),
            "A_tie": _variant("A", {1: 60.0, 2: 60.0}),
            "M_partial": _variant("M", {1: 70.0}, failed_seed=2),
        },
    }

    ranked = rank_variants(aggregate, criterion="mean")

    assert [row["variant"] for row in ranked] == ["A_tie", "Z_tie"]
    assert ranked[0]["mean_validation_accuracy"] == pytest.approx(60.0)


def test_mean_minus_std_prefers_the_stable_variant():
    aggregate = {
        "seeds": [1, 2],
        "results": {
            "High_volatile": _variant("H", {1: 67.0, 2: 60.0}),
            "Lower_stable": _variant("L", {1: 63.0, 2: 63.5}),
        },
    }
    assert rank_variants(aggregate, criterion="mean")[0]["variant"] == "High_volatile"
    assert rank_variants(aggregate, criterion="mean_minus_std")[0]["variant"] == "Lower_stable"


def test_freeze_recipe_records_validation_only_selection(tmp_path):
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(_aggregate()), encoding="utf-8")

    recipe = freeze_recipe(_aggregate(), aggregate_path, criterion="mean")

    assert recipe["selected"] == "B_regex_hard_learned"
    assert recipe["selection_split"] == "validation"
    assert recipe["test_split_used_for_selection"] is False
    assert len(recipe["aggregate_sha256"]) == 64
    assert [row["variant"] for row in recipe["ranking"]] == [
        "B_regex_hard_learned",
        "A_regex_hard_static",
    ]


def test_freeze_recipe_refuses_incomplete_matrix(tmp_path):
    aggregate = {
        "seeds": [1, 2],
        "results": {"Only": _variant("O", {1: 70.0}, failed_seed=2)},
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    with pytest.raises(ValueError, match="No variant completed"):
        freeze_recipe(aggregate, aggregate_path)
