"""Run the A–E ablation matrix end to end (train + evaluate each variant).

Each variant gets its own checkpoint directory and its result is consolidated
into a single JSON artifact (kept even for failed runs).
"""

import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the A–E ablation matrix")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fast)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Aggregate results artifact (default: outputs/ablation_results[_debug].json)",
    )
    args = parser.parse_args()

    # Make repo-root imports (train.py, evaluate.py) work when launched
    # as `python scripts/run_ablations.py`.
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    import train
    import evaluate
    from pathvqa.utils.config import get_default_config, get_debug_config
    from pathvqa.utils.ablation import ABLATION_CONFIGS, ABLATION_LABELS, ABLATION_CHOICES

    print(f"Starting Ablation runs... (Debug: {args.debug})")
    results = {}

    for key in ABLATION_CHOICES:
        label = ABLATION_LABELS[key]
        name = f"{key}_{label}"
        print(f"\n{'=' * 50}\nRunning Ablation: {name}\n{'=' * 50}")

        cfg = get_debug_config() if args.debug else get_default_config()
        cfg.experiment_name = f"ablation_{name}"

        # Overrides must be applied before Trainer builds the model.
        evaluate.apply_ablation(cfg, key)

        # Adjust checkpoint directory so they don't overwrite each other
        base_dir = "checkpoints/debug" if args.debug else "checkpoints"
        ckpt_dir = Path(base_dir) / name
        cfg.paths.checkpoint_dir = ckpt_dir
        cfg.paths.best_model_path = ckpt_dir / "best_model.pt"
        cfg.paths.__post_init__()

        # 1. Train
        print(f"[{name}] Starting Training...")
        try:
            trainer = train.Trainer(cfg)
            trainer.train()
        except Exception as e:
            print(f"[{name}] Training failed: {e}")
            results[name] = {"status": "FAILED", "error": str(e)}
            continue

        # 2. Evaluate
        print(f"[{name}] Starting Evaluation...")
        try:
            # Evaluator rebuilds from the checkpoint's saved config, which
            # already carries this ablation's symbolic overrides.
            if not cfg.paths.best_model_path.exists():
                print(f"[{name}] Checkpoint not found at {cfg.paths.best_model_path}")
                results[name] = {"status": "NO_CHECKPOINT"}
                continue

            evaluator = evaluate.Evaluator(checkpoint_path=str(cfg.paths.best_model_path))
            r = evaluator.evaluate(split="val", max_samples=cfg.data.max_val_samples)
            results[name] = {
                "status": "OK",
                "ablation": key,
                "label": label,
                "accuracy": r["accuracy"],
                "per_type_accuracy": r.get("per_type_accuracy", {}),
            }
            print(f"[{name}] Validation Accuracy: {r['accuracy']:.2f}%")
        except Exception as e:
            print(f"[{name}] Evaluation failed: {e}")
            results[name] = {"status": "EVAL_FAILED", "error": str(e)}

    print("\n" + "=" * 50)
    print("Ablation Results Summary")
    print("=" * 50)
    for name, res in results.items():
        if res.get("status") == "OK":
            print(f"{name}: {res['accuracy']:.2f}%")
        else:
            print(f"{name}: {res['status']}")

    # Persist a consolidated artifact so results survive process exit,
    # including failed runs.
    out_path = args.output
    if out_path is None:
        suffix = "_debug" if args.debug else ""
        out_path = str(Path("outputs") / f"ablation_results{suffix}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "debug": args.debug,
                "matrix": ABLATION_CONFIGS,
                "labels": ABLATION_LABELS,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Aggregate results saved to {out_path}")


if __name__ == "__main__":
    main()
