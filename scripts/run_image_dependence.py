"""Run paired image-dependence controls with one loaded checkpoint.

Writes actual, blank-image, shuffled-image and image-only validation artifacts
for the same ordered split, then computes paired bootstrap and exact McNemar
comparisons. No model selection is performed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import Evaluator
from scripts.analyze_image_dependence import paired_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/image_dependence"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.num_workers < 0 or args.bootstrap_samples < 1:
        parser.error("--num-workers must be nonnegative and --bootstrap-samples positive")

    evaluator = Evaluator(str(args.checkpoint))
    evaluator.config.seed = args.seed
    evaluator.config.data.num_workers = args.num_workers
    artifacts = {}
    for condition in ("actual", "blank", "shuffled", "image_only"):
        metrics = evaluator.evaluate(split=args.split, image_condition=condition)
        path = args.output_dir / f"{args.split}_{condition}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
        artifacts[condition] = metrics
        print(f"{condition}: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']}) -> {path}")

    actual = {int(row["row_index"]): row for row in artifacts["actual"]["predictions"]}
    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": artifacts["actual"]["checkpoint_sha256"],
        "dataset": artifacts["actual"]["dataset"],
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "comparisons": {},
    }
    for condition in ("blank", "shuffled", "image_only"):
        rows = {int(row["row_index"]): row for row in artifacts[condition]["predictions"]}
        report["comparisons"][condition] = paired_summary(actual, rows, args.bootstrap_samples, args.seed)

    report_path = args.output_dir / f"{args.split}_paired_comparisons.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Paired comparison report: {report_path}")


if __name__ == "__main__":
    main()
