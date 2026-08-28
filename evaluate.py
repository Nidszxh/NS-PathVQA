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
from dataclasses import asdict
from typing import Any, Dict, Optional
from tqdm import tqdm


from pathvqa.data.pathvqa_dataset import pathvqa_dataloader, prepare_batch
from pathvqa.models.pathvqa_model import build_model
from pathvqa.models.question.question_encoder import get_question_tokenizer
from pathvqa._compat import is_pre_dsl_checkpoint, migrate_legacy_scene_head
from pathvqa.utils.config import Config, backfill_config, validate_config
from pathvqa.utils.logging_utils import setup_logging, get_logger
from pathvqa.utils.seed import set_seed
from pathvqa.utils.metrics import expected_calibration_error, temperature_scaling, compute_uncertainty
from pathvqa.utils.conformal import ConformalPredictor, compute_conformal_metrics
from pathvqa.utils.interpretability import classify_failure_type
from pathvqa.symbolic.query_parser import parse_question
from pathvqa.symbolic.routing import QTYPE_LIST, QTYPE_TO_IDX, encode_question_types
from pathvqa.symbolic.poe import SymbolicFusionEngine
from pathvqa.symbolic.executor import (
    build_region_names,
    build_region_mapping,
    build_attribute_mappings,
)
from pathvqa.data.dataset_adapter import AnatomicalOntology
from pathvqa.utils.ablation import ABLATION_CONFIGS, ABLATION_CHOICES, apply_ablation

__all__ = ["ABLATION_CONFIGS", "ABLATION_CHOICES", "apply_ablation", "Evaluator", "main"]


# Architecture-defining fields that must match the checkpoint. An evaluation
# ``--config`` may tune evaluation-time knobs (batch size, cache, ...) but must
# never silently swap the architecture the weights were trained under.
_ARCHITECTURE_FIELDS = [
    ("visual", "use_multiscale"),
    ("visual", "num_object_features"),
    ("visual", "num_objects"),
    ("question", "hidden_dim"),
    ("question", "model_name"),
    ("visual", "model_name"),
    ("symbolic", "enabled"),
    ("symbolic", "query_parser"),
    ("symbolic", "executor"),
    ("symbolic", "weighting_strategy"),
]


def _protect_checkpoint_architecture(ckpt_config: Config, override_config: Config, logger: object) -> Config:
    """Overlay an evaluation config while pinning checkpoint architecture fields.

    Loading a checkpoint into a differently shaped model fails at
    ``load_state_dict`` anyway; this surfaces the conflict early and keeps the
    checkpoint's architecture so evaluation is never silently incorrect.
    """
    for sub_name, field in _ARCHITECTURE_FIELDS:
        ckpt_sub = getattr(ckpt_config, sub_name)
        ov_sub = getattr(override_config, sub_name)
        ckpt_val = getattr(ckpt_sub, field)
        ov_val = getattr(ov_sub, field)
        if ckpt_val != ov_val:
            logger.warning(  # type: ignore[attr-defined]
                f"--config changes architecture field {sub_name}.{field} "
                f"({ov_val!r} -> {ckpt_val!r}); keeping the checkpoint value."
            )
            setattr(ov_sub, field, ckpt_val)
    return override_config


def _config_to_dict(cfg: Config) -> Dict[str, Any]:
    """Serializable config snapshot for provenance in evaluation artifacts."""
    raw = asdict(cfg)

    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    result: Dict[str, Any] = convert(raw)
    return result


class Evaluator:
    """Loads a trained checkpoint and evaluates on PathVQA data."""

    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None, ablation: Optional[str] = None):
        """Load checkpoint, rebuild model and vocab, prepare symbolic mappings."""
        self.logger = get_logger("eval")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        self.checkpoint_path = str(checkpoint_path)
        self.ablation = ablation
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_config = backfill_config(ckpt["config"]) if ckpt.get("config") is not None else None
        if config_path:
            file_config = Config.load(config_path)
            if ckpt_config is None:
                self.config = file_config
            else:
                self.config = _protect_checkpoint_architecture(ckpt_config, file_config, self.logger)
        elif ckpt_config is not None:
            self.config = ckpt_config
        else:
            raise SystemExit("Checkpoint carries no config and no --config was provided; cannot build the model.")
        if ablation is not None:
            apply_ablation(self.config, ablation)
        self.config.paths.__post_init__()
        # Checkpoints trained before the DSL interpreter was wired in carry no
        # dsl_interpreter.* params; rebuild the model without it (reproducing
        # the architecture that saved the checkpoint) instead of failing
        # strict=True loading or random-initializing weights never trained.
        legacy_states = [s for s in (ckpt.get("ema_state_dict"), ckpt.get("model_state_dict")) if s]
        if self.config.symbolic.enabled and legacy_states and is_pre_dsl_checkpoint(*legacy_states):
            self.config.symbolic.dsl_enabled = False
            self.logger.warning("Pre-DSL checkpoint: building model without the DSL interpreter")
        self.idx_to_answer = ckpt.get("idx_to_answer")
        self.answer_to_idx = ckpt.get("answer_to_idx")
        self.region_names = ckpt.get("region_names", [])

        # Fall back to vocab_info.json for any missing metadata key
        if not self.idx_to_answer or not self.answer_to_idx or not self.region_names:
            vocab_file = self.config.paths.data_dir / "vocab_info.json"
            if vocab_file.exists():
                with open(vocab_file) as f:
                    info = json.load(f)
                if not self.region_names:
                    self.region_names = info.get("region_names", [])
                if not self.idx_to_answer:
                    self.idx_to_answer = {int(k): v for k, v in info["idx_to_answer"].items()}
                if not self.answer_to_idx:
                    self.answer_to_idx = info["answer_to_idx"]
        missing = [
            k for k, v in (("idx_to_answer", self.idx_to_answer), ("answer_to_idx", self.answer_to_idx)) if not v
        ]
        if missing:
            raise SystemExit(
                f"Checkpoint is missing required metadata: {missing}. "
                "Re-save the checkpoint from training or provide vocab_info.json."
            )
        self.answer_vocab = list(self.answer_to_idx.keys())

        # Build symbolic mappings if region names are available
        if not self.region_names:
            self.region_names = build_region_names(self.answer_vocab)
        self.region_to_answer_idx = build_region_mapping(self.region_names, self.answer_to_idx).to(self.device)
        self.attribute_mappings = build_attribute_mappings(self.answer_to_idx)
        self.ontology = AnatomicalOntology(getattr(self.config.symbolic, "ontology_path", None))
        self.conformal = ConformalPredictor(
            alpha=getattr(self.config.symbolic, "conformal_alpha", 0.1),
            max_set_size=getattr(self.config.symbolic, "conformal_max_set_size", 1),
        )

        # HF tokenizer replaces the old question-vocab dependency (Phase 3):
        # evaluation no longer requires a prior training run.
        self.tokenizer = get_question_tokenizer(self.config.question.model_name)
        # Pass the attribute mappings so the DSL interpreter matches training
        # (the interpreter consumes them inside the model).
        self.model = build_model(self.config, len(self.answer_vocab), self.attribute_mappings)

        if getattr(self.config.symbolic, "executor", "hard") == "differentiable":
            from pathvqa.symbolic.diff_executor import DifferentiableExecutor

            self.diff_executor: Optional[DifferentiableExecutor] = DifferentiableExecutor(
                self.answer_to_idx, self.region_names
            )
        else:
            self.diff_executor = None

        # Shared PoE fusion (single implementation for train/eval/API — see
        # pathvqa.symbolic.poe).
        self.fusion = SymbolicFusionEngine(
            config_symbolic=self.config.symbolic,
            answer_to_idx=self.answer_to_idx,
            region_names=self.region_names,
            device=self.device,
            ontology=self.ontology,
            diff_executor=self.diff_executor,
            dsl_interpreter=getattr(self.model, "dsl_interpreter", None),
            region_to_answer_idx=self.region_to_answer_idx,
            attribute_mappings=self.attribute_mappings,
        )

        if ckpt.get("ema_state_dict") is not None:
            # EMA snapshot is the eval checkpoint (Phase 4); best_val_acc was
            # measured with EMA weights, so use them for parity.
            state, removed = migrate_legacy_scene_head(ckpt["ema_state_dict"])
            if removed:
                self.logger.warning(f"Dropping {len(removed)} legacy density-head params from checkpoint")
            self.model.load_state_dict(state)
            self.logger.info("Using EMA weights from checkpoint (Phase 4)")
        else:
            state, removed = migrate_legacy_scene_head(ckpt["model_state_dict"])
            if removed:
                self.logger.warning(f"Dropping {len(removed)} legacy density-head params from checkpoint")
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.amp_enabled = self.config.training.use_amp and self.device.type == "cuda"
        self.logger.info(f"Model loaded. Best val acc: {ckpt.get('best_val_acc', 'n/a')}%")
        self.logger.info(
            f"Symbolic: {'enabled' if self.config.symbolic.enabled else 'disabled'}, {len(self.region_names)} regions"
        )
        attr_sizes = {k: len(v) for k, v in self.attribute_mappings.items()}
        self.logger.info(f"Attribute heads: {attr_sizes}")

    def _compute_symbolic_logits(self, outputs, questions):
        """Combine neural and symbolic logits via Product-of-Experts (PoE).

        Delegates to the shared fusion engine (``pathvqa.symbolic.poe``) so
        training, evaluation, and serving execute byte-identical fusion logic.
        """
        return self.fusion(outputs, questions)

    def evaluate(self, split: str = "val", max_samples: Optional[int] = None, intervention: str = "none"):
        """Run evaluation on the specified split, including per-question-type accuracy."""
        loader = pathvqa_dataloader(
            split=split,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size,
            max_samples=max_samples,
            answer_to_idx=self.answer_to_idx,
            use_cache=self.config.data.use_cache,
            cache_dir=self.config.paths.cache_dir,
            norm=self.config.data.norm,
        )
        correct, total = 0, 0
        per_type = {}  # qtype -> [correct, total]
        all_logits = []
        all_targets = []
        # Regex qtype (row) vs learned soft-qtype argmax (column), for paper fig 2.
        qtype_confusion = [[0 for _ in QTYPE_LIST] for _ in QTYPE_LIST]
        qtype_confusion_available = False
        failure_counts: Dict[str, int] = {}

        intervention_results = {
            "probe_baseline_correct": 0,
            "probe_corrected_correct": 0,
            "probe_total": 0,
            "attributions": [],
            "counterfactuals": [],
        }
        intervention_samples = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                batch_data = prepare_batch(batch, self.tokenizer, self.device, self.config.question.max_seq_len)
                input_ids = batch_data.input_ids
                attn_mask = batch_data.attention_mask
                targets = batch_data.targets
                images = batch_data.images
                questions = batch_data.questions
                qtype_onehot = (
                    encode_question_types(questions).to(self.device)
                    if self.config.symbolic.weighting_strategy == "learned"
                    else None
                )
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
                    is_correct = preds[i] == targets[i]
                    if is_correct:
                        stats[0] += 1

                    # Learned qtype confusion (only when the learned parser ran).
                    if outputs.soft_qtype is not None:
                        qtype_confusion_available = True
                        true_q = QTYPE_TO_IDX.get(qtype, 0)
                        pred_q = int(outputs.soft_qtype[i].argmax().item())
                        qtype_confusion[true_q][pred_q] += 1

                    # Failure taxonomy (C7): attribute errors to perception/
                    # parsing/execution using region *names* when available.
                    if not is_correct:
                        top_r = (
                            outputs.scene_region_logits[i].argmax().item()
                            if outputs.scene_region_logits is not None
                            else None
                        )
                        sample: Dict[str, Any] = {
                            "qtype": qtype,
                            "target": targets[i].item(),
                            "scene_region_logits": outputs.scene_region_logits[i]
                            if outputs.scene_region_logits is not None
                            else None,
                        }
                        if top_r is not None and self.region_names:
                            sample["top_region_name"] = (
                                self.region_names[top_r] if top_r < len(self.region_names) else f"region_{top_r}"
                            )
                            sample["target_answer"] = self.idx_to_answer.get(targets[i].item(), "")
                        ftype = classify_failure_type(sample)
                        failure_counts[ftype] = failure_counts.get(ftype, 0) + 1

                # --- Concept Bottleneck Interventions ---
                if intervention != "none" and intervention_samples < 50:
                    try:
                        from pathvqa.utils.concept_intervention import (
                            bottleneck_probe,
                            concept_attribution,
                            counterfactual_explanation,
                        )
                    except ImportError as exc:
                        # Never silently disable C10 metrics: surface the failure
                        # and skip interventions for the rest of this run.
                        self.logger.error(f"Interventions requested but unavailable ({exc}); skipping")
                        intervention = "none"
                    else:
                        # 1. Probe
                        if intervention in ["probe", "all"]:
                            probe_res = bottleneck_probe(
                                outputs,
                                questions,
                                targets,
                                self.region_names,
                                self.answer_vocab,
                                self._compute_symbolic_logits,
                            )
                            intervention_results["probe_baseline_correct"] += probe_res["baseline_correct"]
                            intervention_results["probe_corrected_correct"] += probe_res["corrected_correct"]
                            intervention_results["probe_total"] += probe_res["total"]

                        # 2. Attribution & 3. Counterfactuals (run per sample)
                        if intervention in ["attribution", "counterfactual", "all"]:
                            for b_idx in range(len(questions)):
                                if intervention_samples + b_idx >= 50:
                                    break
                                # Run attribution
                                if intervention in ["attribution", "all"]:
                                    with torch.enable_grad():
                                        attr_res = concept_attribution(
                                            self.model,
                                            images[b_idx : b_idx + 1],
                                            input_ids[b_idx : b_idx + 1],
                                            attn_mask[b_idx : b_idx + 1],
                                            qtype_onehot[b_idx : b_idx + 1] if qtype_onehot is not None else None,
                                            targets[b_idx].item(),
                                        )
                                        max_attr_idx = attr_res["region_attribution"].argmax().item()
                                        reg_name = (
                                            self.region_names[max_attr_idx]
                                            if max_attr_idx < len(self.region_names)
                                            else str(max_attr_idx)
                                        )
                                        intervention_results["attributions"].append(
                                            {"question": questions[b_idx], "top_region_attributed": reg_name}
                                        )

                                # Run counterfactual
                                if intervention in ["counterfactual", "all"]:
                                    single_outputs = self.model(
                                        images[b_idx : b_idx + 1],
                                        input_ids[b_idx : b_idx + 1],
                                        attn_mask[b_idx : b_idx + 1],
                                        qtype_onehot[b_idx : b_idx + 1] if qtype_onehot is not None else None,
                                    )
                                    cf_res = counterfactual_explanation(
                                        single_outputs,
                                        [questions[b_idx]],
                                        self.region_names,
                                        self.idx_to_answer,
                                        self._compute_symbolic_logits,
                                    )
                                    intervention_results["counterfactuals"].extend(cf_res)
                    intervention_samples += len(questions)

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        accuracy = 100.0 * correct / total
        per_type_accuracy = {
            qtype: {"accuracy": 100.0 * c / t, "correct": c, "total": t} for qtype, (c, t) in sorted(per_type.items())
        }
        ece, bin_stats = expected_calibration_error(all_logits, all_targets)
        temp = temperature_scaling(all_logits, all_targets)
        uncertainty = compute_uncertainty(all_logits, method="max_softmax")
        # Out-of-sample conformal metrics: calibration and coverage are computed
        # on disjoint halves (see compute_conformal_metrics).
        conformal_metrics = compute_conformal_metrics(
            all_logits,
            all_targets,
            alphas=[0.05, 0.1, 0.2],
            max_set_size=self.conformal.max_set_size,
        )

        result: Dict[str, Any] = {
            "checkpoint": self.checkpoint_path,
            "ablation": self.ablation,
            "split": split,
            "max_samples": max_samples,
            "seed": self.config.seed,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
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
            "conformal": {str(k): v for k, v in conformal_metrics.items()},
            "failure_taxonomy": {
                "failure_counts": failure_counts,
                "failure_ratios": {k: v / max(1, sum(failure_counts.values())) for k, v in failure_counts.items()},
            },
            "config": _config_to_dict(self.config),
        }
        if qtype_confusion_available:
            result["qtype_confusion"] = {
                "matrix": qtype_confusion,
                "labels": list(QTYPE_LIST),
            }

        if intervention != "none" and intervention_samples > 0:
            result["concept_interventions"] = intervention_results
            if intervention_results["probe_total"] > 0:
                baseline_acc = (
                    100.0 * intervention_results["probe_baseline_correct"] / intervention_results["probe_total"]
                )
                corrected_acc = (
                    100.0 * intervention_results["probe_corrected_correct"] / intervention_results["probe_total"]
                )
                result["concept_interventions"]["probe_accuracy_delta"] = corrected_acc - baseline_acc

        return result


def main():
    """Parse CLI args, load checkpoint, and run evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Neuro-Symbolic PathVQA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--max_samples", type=int)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Results path (default: outputs/eval_<checkpoint>_<split>[_<ablation>].json)",
    )
    parser.add_argument("--ablation", type=str, choices=ABLATION_CHOICES, default=None)
    parser.add_argument(
        "--intervention", type=str, choices=["none", "probe", "attribution", "counterfactual", "all"], default="none"
    )
    args = parser.parse_args()

    # Never clobber a canonical artifact by default: derive a checkpoint+split
    # specific name so repeated evaluations do not overwrite each other.
    if args.output is None:
        stem = Path(args.checkpoint).stem
        suffix = f"_{args.ablation}" if args.ablation else ""
        args.output = str(Path("outputs") / f"eval_{stem}_{args.split}{suffix}.json")

    setup_logging(log_file=Path(args.output).parent / "evaluate.log")
    e = Evaluator(args.checkpoint, args.config, ablation=args.ablation)

    validate_config(e.config)
    set_seed(e.config.seed, deterministic=getattr(e.config.training, "deterministic", False))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    r = e.evaluate(split=args.split, max_samples=args.max_samples, intervention=args.intervention)
    print(f"\nAccuracy: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
    print(f"ECE: {r['calibration']['ece']:.4f}  Temperature: {r['calibration']['temperature']:.3f}")
    print(f"Mean uncertainty: {r['uncertainty']['mean']:.4f} (std: {r['uncertainty']['std']:.4f})")
    for alpha_str, cm in sorted(r.get("conformal", {}).items()):
        print(
            f"  Conformal α={alpha_str}: out-of-sample coverage={cm['coverage']:.2%}, "
            f"avg set size={cm['avg_set_size']:.2f}"
        )
    for qtype in sorted(r.get("per_type_accuracy", {})):
        st = r["per_type_accuracy"][qtype]
        print(f"  {qtype:10s}: {st['accuracy']:.2f}% ({st['correct']}/{st['total']})")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
