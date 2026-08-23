import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Ablations (Phase E)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fast)")
    args = parser.parse_args()

    # The 5 Ablation configurations as per NOVELTIES.md matrix:
    # A (baseline): Regex | Hard-coded | Static
    # B (current SOTA): Regex | Hard-coded | Learned
    # C (differentiable): Learned | Differentiable | Learned
    # D (no gate): Learned | Differentiable | Static (no-gate implies fallback to static weighting)
    # E (regex + diff exec): Regex | Differentiable | Learned

    ablations = {
        "A_baseline": {
            "query_parser": "regex",
            "executor": "hard",
            "weighting_strategy": "static"
        },
        "B_sota": {
            "query_parser": "regex",
            "executor": "hard",
            "weighting_strategy": "learned"
        },
        "C_differentiable": {
            "query_parser": "learned",
            "executor": "differentiable",
            "weighting_strategy": "learned"
        },
        "D_no_gate": {
            "query_parser": "learned",
            "executor": "differentiable",
            "weighting_strategy": "static"
        },
        "E_regex_diffexec": {
            "query_parser": "regex",
            "executor": "differentiable",
            "weighting_strategy": "learned"
        }
    }

    print(f"Starting Ablation runs... (Debug: {args.debug})")

    # To run programmatically, we can generate a config JSON for each and pass to train.py
    # Since train.py does not currently support --config directly as an argument for overriding all properties easily without a JSON file, 
    # we will generate a config JSON and invoke train.py (assuming we can pass config or we just override it in a script).
    # Wait, train.py does not have a --config argument! It uses `get_default_config()` or `get_debug_config()`.
    # Let's import train and evaluate directly to run them in process.

    # Modify sys.path so we can import train.py and evaluate.py
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    import train
    import evaluate
    from pathvqa.utils.config import get_default_config, get_debug_config

    results = {}

    for name, config_overrides in ablations.items():
        print(f"\n{'='*50}\nRunning Ablation: {name}\n{'='*50}")
        
        cfg = get_debug_config() if args.debug else get_default_config()
        cfg.experiment_name = f"ablation_{name}"
        
        cfg.symbolic.query_parser = config_overrides["query_parser"]
        cfg.symbolic.executor = config_overrides["executor"]
        cfg.symbolic.weighting_strategy = config_overrides["weighting_strategy"]
        
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
            results[name] = "FAILED"
            continue

        # 2. Evaluate
        print(f"[{name}] Starting Evaluation...")
        try:
            # We use evaluate.py's Evaluator on the best checkpoint
            if not cfg.paths.best_model_path.exists():
                print(f"[{name}] Checkpoint not found at {cfg.paths.best_model_path}")
                results[name] = "NO_CHECKPOINT"
                continue

            # Need to save the config as json so Evaluator can load it, or just pass it in?
            # Evaluator loads from checkpoint's config, so we just pass the checkpoint path
            evaluator = evaluate.Evaluator(checkpoint_path=str(cfg.paths.best_model_path))
            # evaluate on val split
            acc, per_type = evaluator.evaluate(split="val", max_samples=cfg.data.max_val_samples)
            results[name] = acc
            print(f"[{name}] Validation Accuracy: {acc:.2f}%")
        except Exception as e:
            print(f"[{name}] Evaluation failed: {e}")
            results[name] = "EVAL_FAILED"

    print("\n" + "="*50)
    print("Ablation Results Summary")
    print("="*50)
    for name, acc in results.items():
        if isinstance(acc, float):
            print(f"{name}: {acc:.2f}%")
        else:
            print(f"{name}: {acc}")


if __name__ == "__main__":
    main()
