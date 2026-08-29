"""Export a blinded, stratified PathVQA sample for independent expert review.

The reviewer sheet omits model predictions and reference answers. A separate
key file supports later adjudication. Sampling is stratified by question type
and model correctness, so estimates from this sample must be reported as
stratum-weighted rather than unweighted population estimates.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from pathvqa.data.pathvqa_dataset import PathVQADataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/final_repro_val.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/expert_review"))
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error("--per-stratum must be positive")

    artifact = json.loads(args.predictions.read_text())
    if artifact.get("split") != "val" or artifact.get("image_condition") != "actual":
        parser.error("Expert review sampling requires an actual-image validation artifact")
    predictions = artifact.get("predictions", [])
    if len(predictions) != artifact.get("total"):
        parser.error("Prediction artifact must contain every validation example")

    rng = random.Random(args.seed)
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in predictions:
        strata.setdefault((row["qtype"], bool(row["correct"])), []).append(row)
    selected = []
    stratum_manifest = []
    for stratum in sorted(strata, key=lambda key: (key[0], key[1])):
        rows = strata[stratum]
        rng.shuffle(rows)
        sampled_rows = rows[: args.per_stratum]
        selected.extend(sampled_rows)
        stratum_manifest.append(
            {
                "qtype": stratum[0],
                "correctness_stratum": "correct" if stratum[1] else "incorrect",
                "population_count": len(rows),
                "sample_count": len(sampled_rows),
                "population_weight": len(rows) / len(sampled_rows),
            }
        )
    selected.sort(key=lambda row: int(row["row_index"]))

    dataset = PathVQADataset(
        split="val",
        image_size=tuple(artifact["config"]["data"]["image_size"]),
        max_samples=artifact["total"],
        use_randaugment=False,
        use_cache=False,
        norm=artifact["config"]["data"]["norm"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.out_dir / "review_blinded.csv"
    key_path = args.out_dir / "adjudication_key.csv"
    manifest_path = args.out_dir / "adjudication_manifest.json"

    (args.out_dir / "review_instructions.md").write_text(
        "# Blinded PathVQA case review\n\n"
        "Review the image and question in `review_blinded.csv`. Do not open or share "
        "`adjudication_key.csv` or `adjudication_manifest.json` with reviewers before they submit their independent "
        "answers. For each case, record an answer in your own words, whether the "
        "image is sufficient, whether the question is answerable, whether the "
        "dataset reference answer is valid (after unblinding), and a short rationale.\n\n"
        "Cases were sampled by question type and model correctness. Report "
        "population estimates using stratum weights `N_h / n_h`, where `N_h` is "
        "the full validation count and `n_h` is the reviewed count for that "
        "question-type × correctness stratum. Do not report the raw sample mean "
        "as a population estimate. Record reviewer qualifications, disagreements, "
        "adjudication procedure, and missing responses. This is research review, "
        "not clinical validation.\n",
        encoding="utf-8",
    )

    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "image_file",
                "question",
                "expert_answer",
                "image_sufficient",
                "question_answerable",
                "reference_answer_valid",
                "rationale",
            ],
        )
        writer.writeheader()
        for row in selected:
            case_id = f"val_{int(row['row_index']):05d}"
            image = dataset.data[int(row["row_index"])]["image"]
            if image is None:
                raise ValueError(f"Missing image for row {row['row_index']}")
            filename = f"{case_id}.jpg"
            image.convert("RGB").save(image_dir / filename, quality=95)
            writer.writerow(
                {
                    "case_id": case_id,
                    "image_file": f"images/{filename}",
                    "question": row["question"],
                    "expert_answer": "",
                    "image_sufficient": "",
                    "question_answerable": "",
                    "reference_answer_valid": "",
                    "rationale": "",
                }
            )

    with key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "row_index",
                "qtype",
                "correctness_stratum",
                "reference_answer",
                "model_answer",
                "model_confidence",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "case_id": f"val_{int(row['row_index']):05d}",
                    "row_index": row["row_index"],
                    "qtype": row["qtype"],
                    "correctness_stratum": "correct" if row["correct"] else "incorrect",
                    "reference_answer": row["target_answer"],
                    "model_answer": row["predicted_answer"],
                    "model_confidence": row["confidence"],
                }
            )
    manifest = {
        "source_artifact": str(args.predictions),
        "checkpoint_sha256": artifact.get("checkpoint_sha256"),
        "dataset": artifact.get("dataset"),
        "sampling_seed": args.seed,
        "requested_per_stratum": args.per_stratum,
        "selected_cases": len(selected),
        "strata": stratum_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(selected)} cases to {args.out_dir}")
    print(f"Blinded form: {review_path}")
    print(f"Adjudication key: {key_path}")
    print(f"Restricted sampling manifest: {manifest_path}")


if __name__ == "__main__":
    main()
