"""Evaluation script for Neuro-Symbolic PathVQA.

Loads a trained checkpoint and evaluates on the specified split.
Supports neuro-symbolic and neural-only modes, auto-detected from checkpoint.

Outputs accuracy, per-question-type breakdown, calibration (ECE, temperature),
and uncertainty estimates.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --checkpoint checkpoints/best_model.pt --split test
    python evaluate.py --checkpoint checkpoints/best_model.pt --max_samples 100
"""

import torch
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, "src")

from data.pathvqa_dataset import pathvqa_dataloader, prepare_batch
from models.pathvqa_model import build_model
from models.question.question_encoder import get_question_tokenizer
from utils.config import Config, backfill_config, validate_config
from utils.logging_utils import setup_logging, get_logger
from utils.seed import set_seed
from utils.metrics import expected_calibration_error, temperature_scaling, compute_uncertainty
from utils.conformal import ConformalPredictor, compute_conformal_metrics
from symbolic.query_parser import parse_question
from symbolic.executor import (
    execute, build_region_names, build_region_mapping,
    build_attribute_mappings,
)
from symbolic.routing import encode_question_types
from data.dataset_adapter import AnatomicalOntology


class Evaluator:
    """Loads a trained checkpoint and evaluates on PathVQA data."""

    def __init__(self, checkpoint_path: str, config_path: str = None):
        """Load checkpoint, rebuild model and vocab, prepare symbolic mappings."""
        self.logger = get_logger("eval")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = Config.load(config_path) if config_path else backfill_config(ckpt["config"])
        self.config.paths.__post_init__()
        self.idx_to_answer = ckpt.get("idx_to_answer")
        self.answer_to_idx = ckpt.get("answer_to_idx")
        self.region_names = ckpt.get("region_names", [])

        # Fall back to vocab_info.json if checkpoint lacks metadata
        if not self.region_names and not self.idx_to_answer:
            with open(self.config.paths.data_dir / "vocab_info.json") as f:
                info = json.load(f)
            self.region_names = info.get("region_names", [])
            self.idx_to_answer = {int(k): v for k, v in info["idx_to_answer"].items()}
            self.answer_to_idx = info["answer_to_idx"]
        self.answer_vocab = list(self.answer_to_idx.keys())

        # Build symbolic mappings if region names are available
        if not self.region_names:
            self.region_names = build_region_names(self.answer_vocab)
        self.region_to_answer_idx = build_region_mapping(self.region_names, self.answer_to_idx).to(self.device)
        self.attribute_mappings = build_attribute_mappings(self.answer_to_idx)
        self.ontology = AnatomicalOntology(getattr(self.config.symbolic, "ontology_path", None))
        self.conformal = ConformalPredictor(alpha=getattr(self.config.symbolic, "conformal_alpha", 0.1))

        # HF tokenizer replaces the old question-vocab dependency (Phase 3):
        # evaluation no longer requires a prior training run.
        self.tokenizer = get_question_tokenizer(self.config.question.model_name)
        self.model = build_model(self.config, len(self.answer_vocab))
        if ckpt.get("ema_state_dict") is not None:
            # EMA snapshot is the eval checkpoint (Phase 4); best_val_acc was
            # measured with EMA weights, so use them for parity.
            self.model.load_state_dict(ckpt["ema_state_dict"])
            self.logger.info("Using EMA weights from checkpoint (Phase 4)")
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.amp_enabled = self.config.training.use_amp and self.device.type == "cuda"
        self.logger.info(f"Model loaded. Best val acc: {ckpt['best_val_acc']:.2f}%")
        self.logger.info(f"Symbolic: {'enabled' if self.config.symbolic.enabled else 'disabled'}, "
                         f"{len(self.region_names)} regions")
        attr_sizes = {k: len(v) for k, v in self.attribute_mappings.items()}
        self.logger.info(f"Attribute heads: {attr_sizes}")

    def _compute_symbolic_logits(self, outputs, questions):
        """Combine neural and symbolic logits via Product-of-Experts (mirrors trainer logic)."""
        if not self.config.symbolic.enabled or "scene_region_logits" not in outputs:
            return outputs["answer_logits"], {}
        queries = [parse_question(q, self.answer_vocab) for q in questions]
        exec_out = execute(
            scene_logits=outputs,
            queries=queries,
            region_names=self.region_names,
            region_to_answer_idx=self.region_to_answer_idx,
            attribute_mappings=self.attribute_mappings,
            answer_to_idx=self.answer_to_idx,
            answer_vocab_size=len(self.answer_vocab),
            neural_logits=outputs["answer_logits"],
            ontology=self.ontology,
        )
        log_p_neural = torch.log_softmax(outputs["answer_logits"], dim=-1)
        log_p_symbolic = torch.log_softmax(exec_out["symbolic_logits"] + 1e-8, dim=-1)
        if self.config.symbolic.weighting_strategy == "learned" and "gate_values" in outputs:
            g = outputs["gate_values"].squeeze(-1)
            log_p_combined = (1 - g).unsqueeze(1) * log_p_neural + g.unsqueeze(1) * log_p_symbolic
        else:
            w = self.config.symbolic.symbolic_weight
            log_p_combined = (1 - w) * log_p_neural + w * log_p_symbolic
        return log_p_combined, exec_out.get("trace", {})

    def evaluate(self, split: str = "val", max_samples: int = None):
        """Run evaluation on the specified split, including per-question-type accuracy."""
        loader = pathvqa_dataloader(split=split, batch_size=self.config.data.batch_size,
            shuffle=False, num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size, max_samples=max_samples,
            answer_to_idx=self.answer_to_idx,
            use_cache=self.config.data.use_cache,
            cache_dir=self.config.paths.cache_dir,
            norm=self.config.data.norm)
        correct, total = 0, 0
        per_type = {}  # qtype -> [correct, total]
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                input_ids, attn_mask, targets, images, questions = prepare_batch(
                    batch, self.tokenizer, self.device,
                    max_seq_len=self.config.question.max_seq_len)
                qtype_onehot = encode_question_types(questions).to(self.device) if self.config.symbolic.weighting_strategy == "learned" else None
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    outputs = self.model(images, input_ids, attn_mask, qtype_onehot)
                    logits, _ = self._compute_symbolic_logits(outputs, questions)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                for i, question in enumerate(questions):
                    qtype = parse_question(question, self.answer_vocab).qtype
                    stats = per_type.setdefault(qtype, [0, 0])
                    stats[1] += 1
                    if preds[i] == targets[i]:
                        stats[0] += 1
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        accuracy = 100.0 * correct / total
        per_type_accuracy = {
            qtype: {"accuracy": 100.0 * c / t, "correct": c, "total": t}
            for qtype, (c, t) in sorted(per_type.items())
        }
        ece, bin_stats = expected_calibration_error(all_logits, all_targets)
        temp = temperature_scaling(all_logits, all_targets)
        uncertainty = compute_uncertainty(all_logits, method="max_softmax")
        # Conformal prediction metrics at multiple alpha levels
        conformal_metrics = compute_conformal_metrics(all_logits, all_targets, alphas=[0.05, 0.1, 0.2])
        return {
            "accuracy": accuracy, "correct": correct, "total": total,
            "per_type_accuracy": per_type_accuracy,
            "calibration": {
                "ece": ece,
                "temperature": temp,
                "reliability_bins": bin_stats,
            },
            "uncertainty": {
                "mean": uncertainty.mean().item(),
                "std": uncertainty.std().item(),
            },
            "conformal": {
                str(k): v for k, v in conformal_metrics.items()
            },
        }


def main():
    """Parse CLI args, load checkpoint, and run evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Neuro-Symbolic PathVQA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()
    setup_logging(log_file=Path(args.output).parent / "evaluate.log")
    e = Evaluator(args.checkpoint, args.config)
    validate_config(e.config)
    set_seed(42)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    r = e.evaluate(args.split, args.max_samples)
    print(f"\nAccuracy: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
    print(f"ECE: {r['calibration']['ece']:.4f}  Temperature: {r['calibration']['temperature']:.3f}")
    print(f"Mean uncertainty: {r['uncertainty']['mean']:.4f} (std: {r['uncertainty']['std']:.4f})")
    for qtype in sorted(r.get("per_type_accuracy", {})):
        st = r["per_type_accuracy"][qtype]
        print(f"  {qtype:10s}: {st['accuracy']:.2f}% ({st['correct']}/{st['total']})")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
