"""Run the A–K factorial matrix and controls end to end (train + evaluate).

Each variant gets its own checkpoint directory and its result is consolidated
into a single JSON artifact (kept even for failed runs).
"""

import sys
import json
import argparse
import os
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the full 2x2x2 symbolic ablation matrix")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fast)")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help=(
            "Random seeds to repeat every selected variant (default: 42). The documented "
            "protocol is single-seed: the matrix is descriptive and makes no seed-variance "
            "claim. Pass several seeds (e.g. --seeds 17 42 123) only for a deliberate "
            "repeated-seed study."
        ),
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=list("ABCDEFGHIJK"),
        default=list("ABCDEFGHIJK"),
        help="A-H factorial plus neural-only and auxiliary-loss controls (default: all eleven)",
    )
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Allow nondeterministic kernels; deterministic runs are the research default",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Aggregate results artifact (default filename includes the selected seeds)",
    )
    parser.add_argument(
        "--keep-checkpoints",
        choices=("best", "best+last", "all"),
        default="best+last",
        help=(
            "Per-run checkpoint retention. 'best' keeps only best_model.pt (~1.3 GB); "
            "'best+last' also keeps the final-epoch snapshot, so a crashed run can be "
            "resumed or audited (~2.6 GB); 'all' keeps every save_every snapshot "
            "(8-13 GB, enough to exhaust the disk across a sweep). Default: best+last."
        ),
    )
    args = parser.parse_args()

    # Make repo-root imports (train.py, evaluate.py) work when launched
    # as `python scripts/run_ablations.py`.
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    import train
    import evaluate
    from pathvqa.utils.config import get_default_config, get_debug_config
    from pathvqa.utils.ablation import ABLATION_CONFIGS, ABLATION_LABELS

    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        parser.error("at least one seed is required")
    selected_ablations = list(dict.fromkeys(args.ablations))
    suffix = "_debug" if args.debug else ""
    if len(seeds) > 1:
        suffix += "_seeds" + "-".join(str(seed) for seed in seeds)
    else:
        suffix += f"_seed{seeds[0]}"
    out_path = Path(args.output) if args.output is not None else Path("outputs") / f"ablation_results{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting Ablation runs... (Debug: {args.debug}; seeds: {seeds})")
    results = {}

    for key in selected_ablations:
        name = f"{key}_{ABLATION_LABELS[key]}"
        results[name] = {
            "status": "PENDING",
            "ablation": key,
            "label": ABLATION_LABELS[key],
            "runs": {str(seed): {"status": "PENDING", "seed": seed} for seed in seeds},
            "accuracy_mean": None,
            "accuracy_std": None,
        }

    def persist_results():
        """Atomically preserve progress so an interrupted matrix stays auditable."""
        payload = {
            "debug": args.debug,
            "seeds": seeds,
            "deterministic": not args.nondeterministic,
            "selected_ablations": selected_ablations,
            "matrix": ABLATION_CONFIGS,
            "labels": ABLATION_LABELS,
            "results": results,
        }
        temporary_path = out_path.with_name(out_path.name + ".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, out_path)

    def record_run(summary, seed, run):
        summary["runs"][str(seed)] = run
        successful_runs = [entry for entry in summary["runs"].values() if entry.get("status") == "OK"]
        terminal_statuses = {"OK", "FAILED", "EVAL_FAILED", "NO_CHECKPOINT"}
        complete = all(entry.get("status") in terminal_statuses for entry in summary["runs"].values())
        summary["status"] = (
            "OK"
            if complete and len(successful_runs) == len(seeds)
            else "PARTIAL"
            if complete and successful_runs
            else "FAILED"
            if complete
            else "RUNNING"
        )
        summary["accuracy_mean"] = (
            statistics.mean(entry["accuracy"] for entry in successful_runs) if successful_runs else None
        )
        summary["accuracy_std"] = (
            statistics.stdev(entry["accuracy"] for entry in successful_runs) if len(successful_runs) > 1 else None
        )
        if complete and len(seeds) == 1:
            summary.update(run)
            summary["runs"] = {str(seed): run}
            summary["accuracy_mean"] = run.get("accuracy")
            summary["accuracy_std"] = None
        persist_results()

    def prune_epoch_checkpoints(cfg):
        """Apply the retention policy to this run's checkpoints.

        Every ``checkpoint_epoch_N.pt`` is ~1.3 GB. Keeping all of them costs
        8-13 GB per run, enough to exhaust the disk across a sweep, so the
        default keeps the best weights plus the final-epoch snapshot (resume and
        last-epoch audit remain possible) and drops the intermediate ones.
        """
        if args.keep_checkpoints == "all":
            return
        snapshots = sorted(
            cfg.paths.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if args.keep_checkpoints == "best+last" and snapshots:
            snapshots = snapshots[:-1]
        removed = 0
        for stale in snapshots:
            stale.unlink()
            removed += 1
        if removed:
            print(f"[{cfg.experiment_name}] pruned {removed} intermediate checkpoint(s)")

    persist_results()

    for key in selected_ablations:
        label = ABLATION_LABELS[key]
        name = f"{key}_{label}"
        summary = results[name]
        print(f"\n{'=' * 50}\nRunning Ablation: {name}\n{'=' * 50}")
        for seed in seeds:
            run_name = f"{name}_seed_{seed}"
            cfg = get_debug_config() if args.debug else get_default_config()
            cfg.experiment_name = f"ablation_{run_name}"
            cfg.seed = seed
            cfg.training.deterministic = not args.nondeterministic

            # Overrides must be applied before Trainer builds the model.
            evaluate.apply_ablation(cfg, key)

            # Isolate every seed/variant so checkpoints and logs cannot overwrite.
            base_dir = Path("checkpoints/debug" if args.debug else "checkpoints")
            ckpt_dir = base_dir / name / f"seed_{seed}"
            cfg.paths.checkpoint_dir = ckpt_dir
            cfg.paths.best_model_path = ckpt_dir / "best_model.pt"
            cfg.paths.log_dir = Path("logs") / run_name
            cfg.paths.__post_init__()

            record_run(summary, seed, {"status": "RUNNING", "seed": seed})

            print(f"[{run_name}] Starting Training (deterministic={cfg.training.deterministic})...")
            try:
                # The runner constructs Trainer directly rather than through
                # train.main(), so seed all RNGs before model initialization.
                train.set_seed(seed, deterministic=cfg.training.deterministic)
                trainer = train.Trainer(cfg)
                trainer.train()
            except Exception as e:
                print(f"[{run_name}] Training failed: {e}")
                record_run(summary, seed, {"status": "FAILED", "seed": seed, "error": str(e)})
                prune_epoch_checkpoints(cfg)
                continue

            print(f"[{run_name}] Starting validation...")
            try:
                if not cfg.paths.best_model_path.exists():
                    print(f"[{run_name}] Checkpoint not found at {cfg.paths.best_model_path}")
                    record_run(summary, seed, {"status": "NO_CHECKPOINT", "seed": seed})
                    prune_epoch_checkpoints(cfg)
                    continue

                # Model selection is validation-only; the test split is kept
                # untouched until the final recipe and checkpoint are frozen.
                evaluator = evaluate.Evaluator(checkpoint_path=str(cfg.paths.best_model_path))
                metrics = evaluator.evaluate(split="val", max_samples=cfg.data.max_val_samples)
                metrics_path = Path("outputs") / "ablations" / name / f"seed_{seed}_val.json"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("w", encoding="utf-8") as metrics_file:
                    json.dump(metrics, metrics_file, indent=2)
                record_run(
                    summary,
                    seed,
                    {
                        "status": "OK",
                        "seed": seed,
                        "checkpoint": str(cfg.paths.best_model_path),
                        "validation_artifact": str(metrics_path),
                        "accuracy": metrics["accuracy"],
                        "correct": metrics.get("correct"),
                        "total": metrics.get("total"),
                        "per_type_accuracy": metrics.get("per_type_accuracy", {}),
                    },
                )
                print(f"[{run_name}] Validation Accuracy: {metrics['accuracy']:.2f}%")
            except Exception as e:
                print(f"[{run_name}] Evaluation failed: {e}")
                record_run(summary, seed, {"status": "EVAL_FAILED", "seed": seed, "error": str(e)})
            prune_epoch_checkpoints(cfg)

    print("\n" + "=" * 50)
    print("Ablation Results Summary")
    print("=" * 50)
    for name, res in results.items():
        if res.get("status") == "OK":
            value = res.get("accuracy", res.get("accuracy_mean"))
            print(f"{name}: {value:.2f}%")
        else:
            print(f"{name}: {res['status']}")

    persist_results()
    print(f"Aggregate results saved to {out_path}")


if __name__ == "__main__":
    main()
