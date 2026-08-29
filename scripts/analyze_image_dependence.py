"""Paired comparison of actual, blank, and shuffled-image evaluation artifacts."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _exact_binomial_two_sided_p(successes: int, trials: int) -> tuple[float, float]:
    """Exact two-sided binomial p-value and log10 value for p=0.5."""
    if trials == 0:
        return 1.0, 0.0
    tail = min(successes, trials - successes)
    log_terms = [
        math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1) - trials * math.log(2.0)
        for k in range(tail + 1)
    ]
    max_log_term = max(log_terms)
    log_lower_tail = max_log_term + math.log(sum(math.exp(value - max_log_term) for value in log_terms))
    log_p_value = min(0.0, math.log(2.0) + log_lower_tail)
    p_value = math.exp(log_p_value) if log_p_value > math.log(math.ulp(0.0)) else 0.0
    return p_value, log_p_value / math.log(10.0)


def load_predictions(path: Path, expected_condition: str) -> dict[int, dict[str, Any]]:
    artifact = json.loads(path.read_text())
    if artifact.get("image_condition") != expected_condition:
        raise ValueError(f"{path} has image_condition={artifact.get('image_condition')!r}")
    records = artifact.get("predictions")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} has no per-example predictions")
    result = {int(row["row_index"]): row for row in records}
    if len(result) != len(records):
        raise ValueError(f"{path} has duplicate row indices")
    return result


def attach_image_cluster_ids(
    predictions: dict[str, dict[int, dict[str, Any]]],
    cluster_ids: list[str],
) -> None:
    """Attach dataset-derived image IDs to aligned condition predictions."""
    for condition, rows in predictions.items():
        for row_index, row in rows.items():
            if row_index < 0 or row_index >= len(cluster_ids):
                raise ValueError(f"{condition} row {row_index} is outside the cluster-ID mapping")
            image_id = cluster_ids[row_index]
            existing = row.get("image_cluster_id")
            if existing is not None and existing != image_id:
                raise ValueError(f"{condition} image cluster mismatch at row {row_index}")
            row["image_cluster_id"] = image_id


def derive_pathvqa_image_cluster_ids(artifact: dict[str, Any]) -> list[str]:
    """Hash the pinned PathVQA split without running the model again."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluate import _dataset_image_cluster_ids
    from pathvqa.data.pathvqa_dataset import DATASET_REVISION, PathVQADataset

    dataset_info = artifact.get("dataset", {})
    if dataset_info.get("hub_revision") != DATASET_REVISION:
        raise ValueError("Artifact dataset revision does not match the pinned PathVQA revision")
    config = artifact.get("config", {})
    data_config = config.get("data", {})
    total = int(artifact["total"])
    full_counts = dataset_info.get("full_split_counts", {})
    split_key = "validation" if artifact["split"] == "val" else artifact["split"]
    split_count = full_counts.get(split_key)
    dataset = PathVQADataset(
        split=artifact["split"],
        image_size=tuple(data_config.get("image_size", (224, 224))),
        max_samples=None if split_count == total else total,
        answer_to_idx={"<UNK>": 0},
        use_randaugment=False,
        use_cache=False,
        norm="imagenet",
    )
    if len(dataset) != total:
        raise ValueError("Pinned dataset row count does not match the prediction artifact")
    expected_fingerprint = dataset_info.get("fingerprint")
    if expected_fingerprint and dataset.data._fingerprint != expected_fingerprint:
        raise ValueError("Pinned dataset fingerprint does not match the prediction artifact")
    return _dataset_image_cluster_ids(dataset)


def paired_summary(
    baseline: dict[int, dict[str, Any]],
    condition: dict[int, dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if baseline.keys() != condition.keys():
        raise ValueError("Evaluation artifacts do not contain the same row indices")
    indices = sorted(baseline)
    if not indices:
        raise ValueError("Evaluation artifacts contain no prediction rows")
    for index in indices:
        if baseline[index]["target_index"] != condition[index]["target_index"]:
            raise ValueError(f"Target mismatch at row {index}")

    base_correct = [bool(baseline[i]["correct"]) for i in indices]
    cond_correct = [bool(condition[i]["correct"]) for i in indices]
    baseline_only = sum(a and not b for a, b in zip(base_correct, cond_correct))
    condition_only = sum(not a and b for a, b in zip(base_correct, cond_correct))
    delta_values = [float(b) - float(a) for a, b in zip(base_correct, cond_correct)]
    rng = np.random.default_rng(seed)
    values = np.asarray(delta_values, dtype=np.float64)
    boot = np.empty(bootstrap_samples, dtype=np.float64)
    chunk_size = min(500, bootstrap_samples)
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        sampled_indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        boot[start:stop] = values[sampled_indices].mean(axis=1)
    boot.sort()
    lo = boot[int(0.025 * bootstrap_samples)]
    hi = boot[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]
    discordant = baseline_only + condition_only
    p_value, log10_p_value = _exact_binomial_two_sided_p(baseline_only, discordant)

    cluster_ids = [baseline[i].get("image_cluster_id") for i in indices]
    cluster_ci = None
    cluster_sign_p = None
    cluster_sign_log10_p = None
    cluster_count = None
    cluster_nonzero_count = None
    if all(value is not None for value in cluster_ids):
        for index, cluster_id in zip(indices, cluster_ids):
            if condition[index].get("image_cluster_id") != cluster_id:
                raise ValueError(f"Image cluster mismatch at row {index}")
        grouped_delta: dict[str, list[float]] = {}
        for cluster_id, delta in zip(cluster_ids, delta_values):
            grouped_delta.setdefault(str(cluster_id), []).append(delta)
        cluster_sums = np.asarray([sum(values) for values in grouped_delta.values()], dtype=np.float64)
        cluster_sizes = np.asarray([len(values) for values in grouped_delta.values()], dtype=np.float64)
        cluster_count = len(cluster_sums)
        cluster_nonzero_count = int(np.count_nonzero(cluster_sums))

        # Resample images as clusters; all QA pairs from a sampled image stay
        # together. The ratio preserves the QA-pair-weighted estimand.
        cluster_boot = np.empty(bootstrap_samples, dtype=np.float64)
        chunk_size = min(500, bootstrap_samples)
        for start in range(0, bootstrap_samples, chunk_size):
            stop = min(start + chunk_size, bootstrap_samples)
            sampled = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
            numerators = cluster_sums[sampled].sum(axis=1)
            denominators = cluster_sizes[sampled].sum(axis=1)
            cluster_boot[start:stop] = numerators / denominators
        cluster_boot.sort()
        cluster_ci = [
            100 * cluster_boot[int(0.025 * bootstrap_samples)],
            100 * cluster_boot[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))],
        ]
        positive_clusters = int(np.count_nonzero(cluster_sums > 0))
        negative_clusters = int(np.count_nonzero(cluster_sums < 0))
        cluster_sign_p, cluster_sign_log10_p = _exact_binomial_two_sided_p(
            positive_clusters, positive_clusters + negative_clusters
        )

    return {
        "n": len(indices),
        "bootstrap_unit": "question-answer example",
        "baseline_accuracy": 100 * sum(base_correct) / len(indices),
        "condition_accuracy": 100 * sum(cond_correct) / len(indices),
        "condition_minus_baseline_percentage_points": 100 * sum(delta_values) / len(indices),
        "paired_bootstrap_95ci_percentage_points": [100 * lo, 100 * hi],
        "image_cluster_count": cluster_count,
        "image_cluster_uncertainty_available": cluster_ci is not None,
        "image_cluster_bootstrap_95ci_percentage_points": cluster_ci,
        "image_cluster_sign_test_non_tied_clusters": cluster_nonzero_count,
        "image_cluster_sign_test_two_sided_p": cluster_sign_p,
        "image_cluster_sign_test_log10_p": cluster_sign_log10_p,
        "baseline_correct_condition_wrong": baseline_only,
        "baseline_wrong_condition_correct": condition_only,
        "mcnemar_exact_two_sided_p": p_value,
        "mcnemar_exact_two_sided_log10_p": log10_p_value,
        "mcnemar_p_underflow": p_value == 0.0 and discordant > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--blank", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--image-only", type=Path)
    parser.add_argument(
        "--derive-image-clusters",
        action="store_true",
        help="Hash the pinned PathVQA images and report cluster-aware uncertainty without model inference",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    actual = load_predictions(args.actual, "actual")
    blank = load_predictions(args.blank, "blank")
    shuffled = load_predictions(args.shuffled, "shuffled")
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in (args.actual, args.blank, args.shuffled)]
    if args.image_only is not None:
        artifacts.append(json.loads(args.image_only.read_text(encoding="utf-8")))
    identity = (artifacts[0].get("checkpoint_sha256"), artifacts[0].get("dataset"), artifacts[0].get("split"))
    for artifact in artifacts[1:]:
        if (
            artifact.get("checkpoint_sha256"),
            artifact.get("dataset"),
            artifact.get("split"),
        ) != identity:
            raise ValueError("Paired artifacts must share checkpoint, dataset provenance, and split")
    predictions = {"actual": actual, "blank": blank, "shuffled": shuffled}
    image_only = None
    if args.image_only is not None:
        image_only = load_predictions(args.image_only, "image_only")
        predictions["image_only"] = image_only
    has_cluster_ids = all(
        row.get("image_cluster_id") is not None for rows in predictions.values() for row in rows.values()
    )
    cluster_source = "artifact" if has_cluster_ids else None
    if args.derive_image_clusters and not has_cluster_ids:
        cluster_ids = derive_pathvqa_image_cluster_ids(artifacts[0])
        attach_image_cluster_ids(predictions, cluster_ids)
        has_cluster_ids = True
        cluster_source = "pinned PathVQA image content, derived without model inference"
    report = {
        "actual_artifact": str(args.actual),
        "blank_artifact": str(args.blank),
        "shuffled_artifact": str(args.shuffled),
        "checkpoint_sha256": identity[0],
        "dataset": identity[1],
        "split": identity[2],
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "image_cluster_ids_available": has_cluster_ids,
        "image_cluster_id_source": cluster_source,
        "blank_vs_actual": paired_summary(actual, blank, args.bootstrap_samples, args.seed),
        "shuffled_vs_actual": paired_summary(actual, shuffled, args.bootstrap_samples, args.seed),
    }
    if image_only is not None:
        report["image_only_vs_actual"] = paired_summary(actual, image_only, args.bootstrap_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
