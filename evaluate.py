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
import hashlib
import platform
from pathlib import Path
import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np


from pathvqa.data.pathvqa_dataset import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    pathvqa_dataloader,
    prepare_batch,
)
from pathvqa.models.pathvqa_model import build_model
from pathvqa.models.question.question_encoder import get_question_tokenizer
from pathvqa._compat import is_pre_dsl_checkpoint, migrate_legacy_scene_head
from pathvqa.utils.config import Config, backfill_config, validate_config
from pathvqa.utils.logging_utils import setup_logging, get_logger
from pathvqa.utils.seed import set_seed
from pathvqa.utils.metrics import expected_calibration_error, temperature_scaling, compute_uncertainty
from pathvqa.utils.conformal import ConformalPredictor, compute_conformal_metrics, groupwise_split_indices
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
    ("visual", "model_name"),
    ("visual", "lora_rank"),
    ("visual", "lora_alpha"),
    ("visual", "lora_target_modules"),
    ("question", "hidden_dim"),
    ("question", "model_name"),
    ("question", "lora_rank"),
    ("question", "lora_alpha"),
    ("question", "lora_target_modules"),
    ("question", "dropout"),
    # Normalization is preprocessing, not weights, but swapping it at eval time
    # silently changes the input distribution the checkpoint was trained on.
    ("data", "norm"),
    ("symbolic", "enabled"),
    ("symbolic", "num_regions"),
    ("symbolic", "query_parser"),
    ("symbolic", "executor"),
    ("symbolic", "weighting_strategy"),
    ("symbolic", "dsl_enabled"),
]


def _sha256_file(path: str) -> str:
    """Hash a checkpoint in bounded memory for artifact provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_image_derangement(dataset, seed: int) -> list[int]:
    """Return a source-row permutation with no query paired to its own image.

    Rows are grouped by exact decoded RGB content and paired after sorting by
    content hash. A half-split rotation exists when no one image contributes
    more than half of the selected QA rows; it preserves the original image
    marginal exactly while ensuring every replacement is a different image.
    """
    import random

    hashes = []
    for index in range(len(dataset)):
        image = dataset.data[index]["image"]
        if image is None:
            raise ValueError(f"Row {index}: image is missing during image derangement")
        hashes.append(_image_content_fingerprint(image))

    counts: dict[bytes, int] = {}
    for digest in hashes:
        counts[digest] = counts.get(digest, 0) + 1
    n_rows = len(hashes)
    if n_rows < 2 or max(counts.values(), default=0) > n_rows // 2:
        raise ValueError(
            "Cannot build an image-mismatched permutation while preserving the image marginal: "
            "the selected split has too few distinct images."
        )

    rng = random.Random(seed)
    ordered = list(range(n_rows))
    rng.shuffle(ordered)
    ordered.sort(key=lambda index: hashes[index])
    offset = n_rows // 2 if n_rows % 2 == 0 else n_rows // 2 + 1
    sources = [0] * n_rows
    for position, query_row in enumerate(ordered):
        sources[query_row] = ordered[(position + offset) % n_rows]
    return sources


def _image_content_fingerprint(image) -> bytes:
    """Stable identifier for an image, independent of its source filename."""
    rgb = image.convert("RGB")
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(rgb.size).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.digest()


def _dataset_image_cluster_ids(dataset) -> list[str]:
    """Hash the original image for each QA row to support cluster-aware stats."""
    identifiers = []
    for index in range(len(dataset)):
        image = dataset.data[index]["image"]
        if image is None:
            raise ValueError(f"Row {index}: image is missing while deriving image cluster IDs")
        identifiers.append(_image_content_fingerprint(image).hex())
    return identifiers


def _clustered_accuracy_ci(
    correct_flags: Sequence[bool],
    cluster_ids: Sequence[str],
    bootstrap_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Percentile bootstrap interval for accuracy over image-content clusters.

    PathVQA asks several questions about the same image, so resampling QA rows
    as independent overstates precision. Clusters (exact decoded image content)
    are resampled with replacement; every QA row of a sampled image enters at
    once, and the reported interval is for the QA-pair-weighted accuracy.
    """
    if len(correct_flags) != len(cluster_ids):
        raise ValueError("correct_flags and cluster_ids must have the same length")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not correct_flags:
        raise ValueError("Cannot compute a clustered accuracy interval over zero examples")

    grouped: Dict[str, Tuple[int, int]] = {}
    for flag, cluster_id in zip(correct_flags, cluster_ids):
        correct, size = grouped.get(str(cluster_id), (0, 0))
        grouped[str(cluster_id)] = (correct + int(bool(flag)), size + 1)
    cluster_correct = np.asarray([value[0] for value in grouped.values()], dtype=np.float64)
    cluster_size = np.asarray([value[1] for value in grouped.values()], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_clusters = len(cluster_correct)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    chunk_size = min(500, bootstrap_samples)
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        sampled = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        numerators = cluster_correct[sampled].sum(axis=1)
        denominators = cluster_size[sampled].sum(axis=1)
        bootstrap[start:stop] = 100.0 * numerators / denominators
    bootstrap.sort()
    return {
        "method": "percentile bootstrap over exact-image-content clusters",
        "point_estimate": 100.0 * cluster_correct.sum() / cluster_size.sum(),
        "lower": float(bootstrap[int((alpha / 2) * bootstrap_samples)]),
        "upper": float(bootstrap[min(bootstrap_samples - 1, int((1 - alpha / 2) * bootstrap_samples))]),
        "confidence_level": 1.0 - alpha,
        "bootstrap_samples": bootstrap_samples,
        "cluster_count": n_clusters,
        "resampling_unit": "image cluster (all QA pairs of an image together)",
    }


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
        self._checkpoint_sha256 = _sha256_file(self.checkpoint_path)
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

        # Select the evaluation weights now: the scene-head width is the ground
        # truth for num_regions, and a stale config would build a mismatched
        # parser that fails later with an opaque state_dict shape error.
        state_key = "ema_state_dict" if ckpt.get("ema_state_dict") else "model_state_dict"
        eval_state, removed = migrate_legacy_scene_head(ckpt[state_key])
        if removed:
            self.logger.warning(f"Dropping {len(removed)} legacy density-head params from checkpoint")
        region_weight = eval_state.get("scene_parser.region_classifier.weight")
        if region_weight is not None:
            head_regions = int(region_weight.shape[0])
            if self.config.symbolic.num_regions != head_regions:
                self.logger.warning(
                    f"Checkpoint scene head has {head_regions} regions but config says "
                    f"{self.config.symbolic.num_regions}; using the checkpoint value"
                )
                self.config.symbolic.num_regions = head_regions
            if self.region_names and len(self.region_names) != head_regions:
                raise SystemExit(
                    f"Checkpoint region_names has {len(self.region_names)} entries but the scene head "
                    f"predicts {head_regions} regions; refusing to evaluate misaligned region scores"
                )
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

        self.model.load_state_dict(eval_state)
        if state_key == "ema_state_dict":
            # EMA snapshot is the eval checkpoint (Phase 4); best_val_acc was
            # measured with EMA weights, so use them for parity.
            self.logger.info("Using EMA weights from checkpoint (Phase 4)")
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

    def evaluate(
        self,
        split: str = "val",
        max_samples: Optional[int] = None,
        intervention: str = "none",
        image_condition: str = "actual",
    ):
        """Run evaluation on the specified split, including per-question-type accuracy."""
        if image_condition not in {"actual", "blank", "shuffled", "image_only"}:
            raise ValueError("image_condition must be 'actual', 'blank', 'shuffled', or 'image_only'")
        # Build a split-wide permutation with different image content per row.
        image_source_indices = None
        if image_condition == "shuffled":
            from pathvqa.data.pathvqa_dataset import PathVQADataset

            probe = PathVQADataset(
                split=split,
                image_size=self.config.data.image_size,
                max_samples=max_samples,
                answer_to_idx=self.answer_to_idx,
                use_randaugment=False,
                use_cache=False,
                norm=self.config.data.norm,
            )
            image_source_indices = _make_image_derangement(probe, self.config.seed)
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
            image_source_indices=image_source_indices,
        )
        if len(loader.dataset) == 0:
            raise ValueError(f"Evaluation split {split!r} is empty (max_samples={max_samples!r}); nothing to evaluate")
        hf_dataset = loader.dataset.data
        image_cluster_ids = _dataset_image_cluster_ids(loader.dataset)
        dataset_info = hf_dataset.info
        full_split_counts = {name: split_info.num_examples for name, split_info in dataset_info.splits.items()}
        correct, total = 0, 0
        exact_correct = 0
        seen_correct = seen_total = unseen_exact_correct = unseen_total = 0
        per_type = {}  # qtype -> [correct, total]
        all_logits = []
        all_targets = []
        prediction_records = []
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
                model_questions = questions
                if image_condition == "blank":
                    # Zero after normalization represents the channel means,
                    # isolating the question path with a constant image.
                    images = torch.zeros_like(images)
                elif image_condition == "image_only":
                    # Empty text retains tokenizer special tokens, but no
                    # question content reaches the model or symbolic parser.
                    model_questions = [""] * len(questions)
                    empty_enc = self.tokenizer(
                        model_questions,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.question.max_seq_len,
                    )
                    input_ids = empty_enc["input_ids"].to(self.device)
                    attn_mask = empty_enc["attention_mask"].to(self.device)
                qtype_onehot = (
                    encode_question_types(model_questions).to(self.device)
                    if self.config.symbolic.weighting_strategy == "learned"
                    else None
                )
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    outputs = self.model(images, input_ids, attn_mask, qtype_onehot)
                    logits, _ = self._compute_symbolic_logits(outputs, model_questions)
                preds = logits.argmax(dim=1)
                confidences = logits.softmax(dim=1).amax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                for i, question in enumerate(questions):
                    qtype = parse_question(question, self.answer_vocab).qtype
                    row_index = len(prediction_records)
                    target_idx = int(targets[i].item())
                    predicted_idx = int(preds[i].item())
                    raw_target = (
                        batch_data.answers[i] if i < len(batch_data.answers) else self.idx_to_answer.get(target_idx, "")
                    )
                    predicted_answer = self.idx_to_answer.get(predicted_idx, "")
                    target_seen = raw_target in self.answer_to_idx and raw_target != "<UNK>"
                    exact_match = predicted_answer == raw_target and raw_target != "<UNK>"
                    exact_correct += int(exact_match)
                    if target_seen:
                        seen_total += 1
                        seen_correct += int(exact_match)
                    else:
                        unseen_total += 1
                        unseen_exact_correct += int(exact_match)
                    prediction_records.append(
                        {
                            "row_index": row_index,
                            "image_source_row_index": (
                                image_source_indices[row_index] if image_source_indices is not None else row_index
                            ),
                            "image_cluster_id": image_cluster_ids[row_index],
                            "question": question,
                            "qtype": qtype,
                            "target_index": target_idx,
                            "target_answer": self.idx_to_answer.get(target_idx, ""),
                            "gold_answer_raw": raw_target,
                            "target_seen_in_vocab": target_seen,
                            "predicted_index": predicted_idx,
                            "predicted_answer": predicted_answer,
                            "confidence": float(confidences[i].item()),
                            "correct": predicted_idx == target_idx,
                            "exact_match": exact_match,
                        }
                    )
                    stats = per_type.setdefault(qtype, [0, 0])
                    stats[1] += 1
                    is_correct = preds[i] == targets[i]
                    if is_correct:
                        stats[0] += 1

                    # Learned qtype confusion (only when the learned parser ran).
                    if outputs.soft_qtype is not None and image_condition != "image_only":
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
                                model_questions,
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
                                        [model_questions[b_idx]],
                                        self.region_names,
                                        self.idx_to_answer,
                                        self._compute_symbolic_logits,
                                    )
                                    intervention_results["counterfactuals"].extend(cf_res)
                    intervention_samples += len(questions)

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        accuracy = 100.0 * correct / total
        accuracy_ci = _clustered_accuracy_ci(
            [bool(record["correct"]) for record in prediction_records],
            [str(record["image_cluster_id"]) for record in prediction_records],
            bootstrap_samples=2000,
            seed=self.config.seed,
        )
        per_type_accuracy = {
            qtype: {"accuracy": 100.0 * c / t, "correct": c, "total": t} for qtype, (c, t) in sorted(per_type.items())
        }
        ece, bin_stats = expected_calibration_error(all_logits, all_targets)
        calibration_indices, temperature_eval_indices = groupwise_split_indices(image_cluster_ids, self.config.seed)
        calibration_idx = torch.tensor(calibration_indices, dtype=torch.long)
        temperature_eval_idx = torch.tensor(temperature_eval_indices, dtype=torch.long)
        temp = temperature_scaling(all_logits[calibration_idx], all_targets[calibration_idx])
        temperature_scaled_logits = all_logits[temperature_eval_idx] / temp
        temperature_scaled_ece, temperature_scaled_bins = expected_calibration_error(
            temperature_scaled_logits, all_targets[temperature_eval_idx]
        )
        uncertainty = compute_uncertainty(all_logits, method="max_softmax")
        # Out-of-sample conformal metrics: calibration and coverage are computed
        # on disjoint halves (see compute_conformal_metrics).
        conformal_metrics = compute_conformal_metrics(
            all_logits,
            all_targets,
            alphas=[0.05, 0.1, 0.2],
            max_set_size=self.conformal.max_set_size,
            group_ids=image_cluster_ids,
        )

        result: Dict[str, Any] = {
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self._checkpoint_sha256,
            "ablation": self.ablation,
            "split": split,
            "max_samples": max_samples,
            "image_condition": image_condition,
            "image_shuffle_seed": self.config.seed if image_condition == "shuffled" else None,
            "seed": self.config.seed,
            "accuracy": accuracy,
            "accuracy_ci": accuracy_ci,
            "accuracy_semantics": "closed-vocabulary class-index accuracy; unknown answers map to <UNK>",
            "exact_answer_accuracy": 100.0 * exact_correct / total,
            "answer_vocabulary_coverage": seen_total / total,
            "seen_answer_accuracy": 100.0 * seen_correct / seen_total if seen_total else None,
            "unseen_answer_exact_accuracy": 100.0 * unseen_exact_correct / unseen_total if unseen_total else None,
            "seen_answer_count": seen_total,
            "unseen_answer_count": unseen_total,
            "correct": correct,
            "total": total,
            "dataset": {
                "repo_id": DATASET_REPO_ID,
                "hub_revision": DATASET_REVISION,
                "builder": dataset_info.builder_name,
                "config": dataset_info.config_name,
                "version": str(dataset_info.version),
                "fingerprint": hf_dataset._fingerprint,
                "split": split,
                "selected_rows": len(loader.dataset),
                "full_split_counts": full_split_counts,
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": str(self.device),
                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            },
            "predictions": prediction_records,
            "per_type_accuracy": per_type_accuracy,
            "calibration": {
                "ece": ece,
                "temperature": temp,
                "reliability_bins": bin_stats,
                "temperature_scaled_ece_on_heldout_groups": temperature_scaled_ece,
                "temperature_scaled_reliability_bins": temperature_scaled_bins,
                "temperature_fit_unit": "image cluster; all QA pairs from each image kept together",
                "temperature_calibration_groups": len({image_cluster_ids[i] for i in calibration_indices}),
                "temperature_evaluation_groups": len({image_cluster_ids[i] for i in temperature_eval_indices}),
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
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader worker count (use 0 in restricted environments)",
    )
    parser.add_argument(
        "--image_condition",
        choices=["actual", "blank", "shuffled", "image_only"],
        default="actual",
        help=(
            "Actual images, blank images (question-only), seeded wrong-image pairing, "
            "or empty question text (image-only control)"
        ),
    )
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
        condition_suffix = f"_{args.image_condition}" if args.image_condition != "actual" else ""
        args.output = str(Path("outputs") / f"eval_{stem}_{args.split}{suffix}{condition_suffix}.json")

    setup_logging(log_file=Path(args.output).parent / "evaluate.log")
    e = Evaluator(args.checkpoint, args.config, ablation=args.ablation)
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num_workers must be nonnegative")
        e.config.data.num_workers = args.num_workers

    validate_config(e.config)
    set_seed(e.config.seed, deterministic=getattr(e.config.training, "deterministic", False))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    evaluation_kwargs = {
        "split": args.split,
        "max_samples": args.max_samples,
        "intervention": args.intervention,
    }
    if args.image_condition != "actual":
        evaluation_kwargs["image_condition"] = args.image_condition
    r = e.evaluate(**evaluation_kwargs)
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
