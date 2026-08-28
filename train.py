"""Training loop for Neuro-Symbolic PathVQA.

Usage:
    python train.py                         # full neuro-symbolic training
    python train.py --no-symbolic           # neural-only baseline
    python train.py --debug                 # fast debug run (500/100 samples, 5 epochs)
    python train.py --config custom.json    # custom config
    python train.py --experiment upgrade    # custom experiment name (writes outputs/upgrade.json)
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


from pathvqa.data.pathvqa_dataset import (
    PathVQADataset,
    pathvqa_dataloader,
    prepare_batch,
    collate_fn,
    TRANSFORM_VERSION,
)
from pathvqa.models.pathvqa_model import build_model
from pathvqa.models.ema import ModelEMA
from pathvqa.models.question.question_encoder import get_question_tokenizer
from pathvqa.utils.config import Config, get_default_config, get_debug_config, backfill_config, validate_config
from pathvqa.utils.logging_utils import setup_logging, get_logger
from pathvqa.utils.seed import set_seed, seed_worker
from pathvqa.utils.conformal import ConformalPredictor
from pathvqa.symbolic.routing import encode_question_types
from pathvqa.symbolic.query_parser import parse_question
from pathvqa.symbolic.executor import (
    build_region_names,
    build_region_mapping,
    build_attribute_mappings,
)
from pathvqa.symbolic.aux_losses import compute_symbolic_aux_losses
from pathvqa.symbolic.ontology_loss import compute_sibling_regularization
from pathvqa.symbolic.ltn import MedicalLogicTensorNetwork
from pathvqa.symbolic.poe import SymbolicFusionEngine
from pathvqa.data.dataset_adapter import AnatomicalOntology
from pathvqa._compat import is_pre_dsl_checkpoint, migrate_legacy_scene_head
from pathvqa.utils.ablation import apply_ablation, ABLATION_CHOICES


def _collect_provenance() -> Dict[str, Any]:
    """Best-effort software provenance for reproducibility artifacts."""
    import platform
    import subprocess

    def _version(pkg: str) -> Optional[str]:
        try:
            from importlib.metadata import version

            return version(pkg)
        except Exception:
            return None

    def _git_commit() -> Optional[str]:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
                or None
            )
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "pillow": _version("pillow"),
        "numpy": _version("numpy"),
        "git_commit": _git_commit(),
    }


def apply_cli_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    """Apply CLI-level overrides to ``config`` in place.

    Kept as a module function so it can be reapplied *after* a resumed
    checkpoint replaces the config; otherwise ``--experiment``/``--debug``/
    ``--ablation`` would be discarded on resume.
    """
    if overrides.get("experiment_name"):
        config.experiment_name = overrides["experiment_name"]
    if overrides.get("debug"):
        config.debug = True
    if overrides.get("deterministic"):
        config.training.deterministic = True
    if overrides.get("no_symbolic"):
        config.symbolic.enabled = False
    if overrides.get("weighting_strategy"):
        config.symbolic.weighting_strategy = overrides["weighting_strategy"]
    if overrides.get("ablation"):
        apply_ablation(config, overrides["ablation"])
    config.paths.__post_init__()
    return config


class Trainer:
    """Manages the full training lifecycle for PathVQA."""

    def __init__(self, config: Config, resume_path: str = None, overrides: Optional[Dict[str, Any]] = None):
        self.config = config
        self.resume_path = resume_path
        self.overrides = overrides or {}
        self.logger = get_logger("train")
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Resume: load the checkpoint up-front so vocab/config match the saved run
        self.resume_ckpt = None
        if resume_path:
            self.logger.info(f"Loading checkpoint for resume: {resume_path}")
            self.resume_ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            ckpt_config = self.resume_ckpt.get("config")
            if ckpt_config is not None:
                self.logger.info("Using config stored in the checkpoint (resume mode)")
                self.config = backfill_config(ckpt_config)
                # Reapply CLI overrides on top of the checkpoint config.
                apply_cli_overrides(self.config, self.overrides)
            config = self.config

            # A pre-DSL checkpoint must be resumed with the model rebuilt as
            # the architecture that saved it: without dsl_interpreter params
            # strict loading would fail (or random-init would train weights
            # the saved optimizer state never covered).
            legacy_states = [
                s for s in (self.resume_ckpt.get("ema_state_dict"), self.resume_ckpt.get("model_state_dict")) if s
            ]
            if config.symbolic.enabled and legacy_states and is_pre_dsl_checkpoint(*legacy_states):
                config.symbolic.dsl_enabled = False
                self.logger.warning("Pre-DSL checkpoint: resuming without the DSL interpreter")

        # Build answer vocabulary from the training dataset
        self.logger.info("Building vocabulary...")
        train_dataset = PathVQADataset(
            split="train",
            image_size=config.data.image_size,
            max_samples=config.data.max_train_samples,
            use_randaugment=config.data.use_randaugment,
            use_cache=config.data.use_cache,
            cache_dir=config.paths.cache_dir,
            norm=config.data.norm,
            randaugment_num_ops=config.data.randaugment_num_ops,
            randaugment_magnitude=config.data.randaugment_magnitude,
        )
        if self.resume_ckpt is not None and self.resume_ckpt.get("answer_to_idx"):
            self.logger.info("Restoring vocabulary from checkpoint")
            self.answer_vocab = self.resume_ckpt["answer_vocab"]
            self.answer_to_idx = self.resume_ckpt["answer_to_idx"]
            self.idx_to_answer = self.resume_ckpt.get("idx_to_answer", {})
            self.idx_to_answer = {int(k): v for k, v in self.idx_to_answer.items()}
        else:
            self.answer_vocab = train_dataset.answers
            self.answer_to_idx = train_dataset.answer_to_idx
            self.idx_to_answer = train_dataset.idx_to_answer

        # Reserve a class for answers unseen in the training split (~14% of
        # val/test answers). Val/test datasets are built with this shared vocab
        # so their targets align with the classifier head; unseen answers map
        # to <UNK> and are (correctly) counted as errors.
        if "<UNK>" not in self.answer_to_idx:
            unk_idx = len(self.answer_to_idx)
            self.answer_to_idx["<UNK>"] = unk_idx
            self.idx_to_answer[unk_idx] = "<UNK>"
        self.answer_vocab = sorted(self.answer_to_idx)

        # Discover region names and build symbolic mappings if enabled
        if config.symbolic.enabled:
            if self.resume_ckpt is not None and self.resume_ckpt.get("region_names"):
                self.region_names = list(self.resume_ckpt["region_names"])
                self.logger.info(f"Restored {len(self.region_names)} region names from checkpoint")
            else:
                self.region_names = build_region_names(self.answer_vocab)
                self.logger.info(f"Built {len(self.region_names)} region names from answer vocabulary")
            self.region_to_answer_idx = build_region_mapping(self.region_names, self.answer_to_idx).to(self.device)
            self.attribute_mappings = build_attribute_mappings(self.answer_to_idx)
            attr_sizes = {k: len(v) for k, v in self.attribute_mappings.items()}
            self.logger.info(f"Attribute mappings: {attr_sizes}")
            self.logger.info(f"  Regions: {self.region_names[:6]}{'...' if len(self.region_names) > 6 else ''}")
            config.symbolic.num_regions = len(self.region_names)
            config.symbolic.region_names = tuple(self.region_names)
            self.ontology = AnatomicalOntology(config.symbolic.ontology_path)
            self.logger.info(f"Ontology loaded ({len(self.ontology.organ_to_systems)} organ-system mappings)")
            if config.symbolic.ltn_enabled:
                self.ltn_engine = MedicalLogicTensorNetwork(
                    num_regions=config.symbolic.num_regions,
                    region_names=self.region_names,
                ).to(self.device)
                self.logger.info(f"LTN consistency loss enabled (weight={config.symbolic.ltn_weight})")
            else:
                self.ltn_engine = None
        else:
            self.region_names = []
            self.region_to_answer_idx = torch.tensor([], dtype=torch.long)
            self.attribute_mappings = {}
            self.ltn_engine = None

        # HF tokenizer replaces the old word-level question vocabulary (Phase 3)
        self.tokenizer = get_question_tokenizer(config.question.model_name)

        # Save metadata for evaluation
        vocab_info = {
            "answer_vocab_size": len(self.answer_vocab),
            "answer_to_idx": self.answer_to_idx,
            "idx_to_answer": self.idx_to_answer,
            "region_names": self.region_names,
        }
        config.paths.data_dir.mkdir(parents=True, exist_ok=True)
        with open(config.paths.data_dir / "vocab_info.json", "w") as f:
            json.dump(vocab_info, f, indent=2)

        self.logger.info(f"Answer vocab: {len(self.answer_vocab)}")

        # Build dataloaders
        self.logger.info("Building dataloaders...")
        nw = config.data.num_workers
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
            worker_init_fn=seed_worker,
        )
        self.val_loader = pathvqa_dataloader(
            split="val",
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            image_size=config.data.image_size,
            use_cache=config.data.use_cache,
            cache_dir=config.paths.cache_dir,
            norm=config.data.norm,
            max_samples=config.data.max_val_samples,
            answer_to_idx=self.answer_to_idx,
        )

        # Build model
        self.logger.info("Building model...")
        self.model = build_model(config, len(self.answer_vocab), self.attribute_mappings)
        self.model.to(self.device)
        if config.training.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
            self.logger.info("Gradient checkpointing enabled (transformer layers)")
        self.logger.info(
            f"Parameters: {sum(p.numel() for p in self.model.parameters()):,} "
            f"(trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,})"
        )

        self.criterion = nn.NLLLoss()
        self.optimizer = self._build_optimizer()
        self.total_optimizer_steps = (
            math.ceil(len(self.train_loader) / config.training.grad_accum_steps) * config.training.num_epochs
        )
        self.scheduler = self._build_scheduler()
        self.amp_enabled = config.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)
        self.early_stop_counter = 0
        self.best_val_acc = 0.0
        self.best_val_per_type = {}
        self.best_val_epoch = -1
        self.checkpoint_paths = []  # track paths for rolling checkpoint deletion
        self.writer = SummaryWriter(log_dir=config.paths.log_dir / config.experiment_name)
        self.current_epoch = 0
        self.global_step = 0
        self.train_start_time = time.monotonic()
        self.epoch_times = []

        # Conformal predictor (calibrated on val set)
        self.conformal = ConformalPredictor(alpha=config.symbolic.conformal_alpha)

        if getattr(config.symbolic, "executor", "hard") == "differentiable":
            from pathvqa.symbolic.diff_executor import DifferentiableExecutor

            self.diff_executor: Optional[DifferentiableExecutor] = DifferentiableExecutor(
                self.answer_to_idx, self.region_names
            )
        else:
            self.diff_executor = None

        # Shared PoE fusion (single implementation for train/eval/API — see
        # pathvqa.symbolic.poe). No-ops to plain log_softmax when symbolic is
        # disabled, so the call sites stay unconditional.
        self.fusion = SymbolicFusionEngine(
            config_symbolic=config.symbolic,
            answer_to_idx=self.answer_to_idx,
            region_names=self.region_names,
            device=self.device,
            ontology=getattr(self, "ontology", None),
            diff_executor=self.diff_executor,
            dsl_interpreter=getattr(self.model, "dsl_interpreter", None),
            region_to_answer_idx=self.region_to_answer_idx if config.symbolic.enabled else None,
            attribute_mappings=self.attribute_mappings if config.symbolic.enabled else None,
        )

        # EMA shadow of the model weights (used for eval; persisted in checkpoints).
        if config.training.ema_enabled:
            self.ema = ModelEMA(self.model, decay=config.training.ema_decay)
            self.logger.info(f"EMA enabled (decay={config.training.ema_decay})")
        else:
            self.ema = None

        # Resume: restore model/optimizer/scheduler/training-state from checkpoint
        if self.resume_ckpt is not None:
            model_state, removed = migrate_legacy_scene_head(self.resume_ckpt["model_state_dict"])
            if removed:
                self.logger.warning(f"Dropping {len(removed)} legacy density-head params from checkpoint")
            self.model.load_state_dict(model_state)
            if self.ema is not None and self.resume_ckpt.get("ema_state_dict") is not None:
                ema_state, _ = migrate_legacy_scene_head(self.resume_ckpt["ema_state_dict"])
                self.resume_ckpt["ema_state_dict"] = ema_state
            self.optimizer.load_state_dict(self.resume_ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in self.resume_ckpt:
                self.scheduler.load_state_dict(self.resume_ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in self.resume_ckpt:
                self.scaler.load_state_dict(self.resume_ckpt["scaler_state_dict"])
            if self.ema is not None and self.resume_ckpt.get("ema_state_dict") is not None:
                self.ema.load_state_dict(self.resume_ckpt["ema_state_dict"])
                self.logger.info("Restored EMA snapshot from checkpoint")
            self.early_stop_counter = self.resume_ckpt.get("early_stop_counter", 0)
            self.best_val_acc = self.resume_ckpt.get("best_val_acc", 0.0)
            self.best_val_per_type = self.resume_ckpt.get("best_val_per_type", {})
            self.best_val_epoch = self.resume_ckpt.get("best_val_epoch", -1)
            self.current_epoch = self.resume_ckpt.get("epoch", 0)
            self.global_step = self.resume_ckpt.get("global_step", 0)
            self.epoch_times = list(self.resume_ckpt.get("epoch_times", []))

            # Rebuild the rolling checkpoint list from what's on disk (numeric epoch order)
            def _epoch_num(p: Path) -> int:
                return int(p.stem.split("_")[-1])

            self.checkpoint_paths = sorted(
                self.config.paths.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
                key=_epoch_num,
            )
            self.logger.info(
                f"Resumed: epoch={self.current_epoch}, global_step={self.global_step}, "
                f"best_val_acc={self.best_val_acc:.2f}%, "
                f"early_stop_counter={self.early_stop_counter}/{config.training.early_stop_patience}"
            )

    def _build_optimizer(self):
        t = self.config.training
        if t.optimizer == "adam":
            return optim.Adam(self.model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
        elif t.optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
        return optim.SGD(self.model.parameters(), lr=t.learning_rate, momentum=0.9, weight_decay=t.weight_decay)

    def _build_scheduler(self):
        t = self.config.training
        if t.scheduler == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=t.factor, patience=t.patience, min_lr=t.min_lr
            )
        elif t.scheduler == "cosine":
            # Linear warmup then cosine decay to ~0 over the run (Phase 4).
            total_steps = self.total_optimizer_steps
            warmup_steps = int(total_steps * t.warmup_ratio)

            def lr_lambda(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            self.logger.info(f"Cosine schedule: {total_steps} optimizer steps, {warmup_steps} warmup")
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def _compute_symbolic_logits(self, outputs, questions):
        """Combine neural and symbolic logits via Product-of-Experts (PoE).

        Delegates to the shared fusion engine (``pathvqa.symbolic.poe``) so
        training, evaluation, and serving execute byte-identical fusion logic.
        """
        return self.fusion(outputs, questions)

    def train_epoch(self):
        """Run one training epoch.

        Uses gradient accumulation (``grad_accum_steps`` micro-batches per
        optimizer step). AMP GradScaler step/update happens only at
        accumulation boundaries.
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        accum = self.config.training.grad_accum_steps
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        self.optimizer.zero_grad()
        n_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(pbar):
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
            warmup_qtype = getattr(
                self.config.symbolic, "query_parser", "regex"
            ) == "learned" and self.current_epoch < getattr(self.config.symbolic, "qtype_warmup_epochs", 3)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(images, input_ids, attn_mask, qtype_onehot)
                if warmup_qtype:
                    logits = torch.log_softmax(outputs.answer_logits, dim=-1)
                else:
                    logits, _ = self._compute_symbolic_logits(outputs, questions)
                loss = self.criterion(logits, targets) / accum
                if warmup_qtype and outputs.qtype_logits is not None:
                    qtypes = [parse_question(q, self.answer_vocab).qtype for q in questions]
                    qtype_to_idx = {"identity": 0, "location": 1, "yes_no": 2, "attribute": 3, "count": 4}
                    qtype_targets = torch.tensor([qtype_to_idx.get(q, 0) for q in qtypes], device=self.device)
                    qtype_loss = nn.CrossEntropyLoss()(outputs.qtype_logits, qtype_targets)
                    warmup_weight = getattr(self.config.symbolic, "qtype_warmup_weight", 0.1)
                    loss = loss + warmup_weight * qtype_loss / accum
                if self.ltn_engine is not None and outputs.scene_region_logits is not None:
                    ltn_loss, _ = self.ltn_engine.evaluate_clauses(outputs.scene_head_dict(), qtype_onehot)
                    loss = loss + getattr(self.config.symbolic, "ltn_weight", 0.1) * ltn_loss / accum
                if self.config.symbolic.enabled and outputs.scene_region_logits is not None:
                    queries = [parse_question(q, self.answer_vocab) for q in questions]
                    aux = compute_symbolic_aux_losses(
                        scene_logits=outputs.scene_head_dict(),
                        queries=queries,
                        targets=targets,
                        answer_to_idx=self.answer_to_idx,
                        region_to_answer_idx=self.region_to_answer_idx,
                        attribute_mappings=self.attribute_mappings,
                    )
                    cfg = self.config.symbolic
                    loss = loss + (cfg.aux_region_weight * aux["region"]) / accum
                    loss = loss + (cfg.aux_attr_weight * aux["attr"]) / accum
                    loss = loss + (cfg.aux_yn_weight * aux["yn"]) / accum
                if self.config.symbolic.enabled and self.config.symbolic.ontology_sibling_weight > 0:
                    sibling_loss = compute_sibling_regularization(
                        self.model.scene_parser.region_classifier.weight,
                        self.region_names,
                        self.ontology,
                    )
                    loss = loss + (self.config.symbolic.ontology_sibling_weight * sibling_loss) / accum
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * accum
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            is_boundary = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == n_batches
            if is_boundary:
                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.update(self.model)
                if self.config.training.scheduler != "reduce_on_plateau":
                    self.scheduler.step()

            if self.global_step % self.config.training.log_every == 0:
                acc = 100.0 * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%"})
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/acc", acc, self.global_step)
            self.global_step += 1
        return total_loss / n_batches, 100.0 * correct / total

    def validate(self):
        """Run validation, tracking neural + symbolic path accuracy and per-type stats.

        Uses EMA snapshot when available; training weights are restored after.
        """
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        total_loss, correct, total = 0.0, 0, 0
        symbolic_count, symbolic_correct = 0, 0
        per_type = {}  # qtype -> [correct, total]
        gate_values_per_qtype = {}  # qtype -> list of gate values (for learned gate)
        all_logits, all_targets = [], []  # for conformal calibration
        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Validating"):
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
                        logits, trace = self._compute_symbolic_logits(outputs, questions)
                    loss = self.criterion(logits, targets)
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    total_loss += loss.item()
                    all_logits.append(logits.cpu())
                    all_targets.append(targets.cpu())
                    gate_vals = outputs.gate_values.squeeze(-1) if outputs.gate_values is not None else None
                    for i, question in enumerate(questions):
                        qtype = parse_question(question, self.answer_vocab).qtype
                        if gate_vals is not None:
                            gate_values_per_qtype.setdefault(qtype, []).append(gate_vals[i].item())
                        stats = per_type.setdefault(qtype, [0, 0])
                        stats[1] += 1
                        if preds[i] == targets[i]:
                            stats[0] += 1
                    if trace.get("symbolic_used"):
                        for i in range(len(questions)):
                            if trace["symbolic_used"][i]:
                                symbolic_count += 1
                                if preds[i] == targets[i]:
                                    symbolic_correct += 1
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        if symbolic_count > 0:
            sym_acc = 100.0 * symbolic_correct / symbolic_count
            self.logger.info(f"  Symbolic path accuracy: {sym_acc:.2f}% ({symbolic_correct}/{symbolic_count})")
        if gate_values_per_qtype:
            mean_gates = {q: sum(v) / len(v) for q, v in gate_values_per_qtype.items()}
            self.logger.info(f"  Gate values per qtype: {mean_gates}")
            for qtype, vals in gate_values_per_qtype.items():
                mean_g = sum(vals) / len(vals)
                self.writer.add_scalar(f"gate/{qtype}", mean_g, self.current_epoch)
        per_type_accuracy = {
            qtype: {"accuracy": 100.0 * c / t, "correct": c, "total": t} for qtype, (c, t) in sorted(per_type.items())
        }
        # Conformal prediction calibration on validation set
        if all_logits:
            all_logits_t = torch.cat(all_logits)
            all_targets_t = torch.cat(all_targets)
            self.conformal.calibrate(all_logits_t, all_targets_t)
            self.logger.info(
                f"  Conformal (α={self.conformal.alpha}): τ={self.conformal.tau:.4f}, "
                f"coverage={self.conformal.calibration_stats['coverage']:.2%}, "
                f"avg_set_size={self.conformal.calibration_stats['avg_set_size']:.2f}"
            )
            self.writer.add_scalar("conformal/tau", self.conformal.tau, self.current_epoch)
            self.writer.add_scalar(
                "conformal/coverage", self.conformal.calibration_stats["coverage"], self.current_epoch
            )
            self.writer.add_scalar(
                "conformal/avg_set_size", self.conformal.calibration_stats["avg_set_size"], self.current_epoch
            )
        return total_loss / len(self.val_loader), 100.0 * correct / total, per_type_accuracy

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint with all metadata needed for evaluation."""
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            "answer_vocab": self.answer_vocab,
            "answer_to_idx": self.answer_to_idx,
            "idx_to_answer": self.idx_to_answer,
            "region_names": self.region_names,
            "early_stop_counter": self.early_stop_counter,
            "best_val_per_type": self.best_val_per_type,
            "best_val_epoch": self.best_val_epoch,
            "epoch_times": self.epoch_times,
        }
        path = self.config.paths.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(ckpt, path)
        self.logger.info(f"Saved: {path}")
        if path not in self.checkpoint_paths:
            self.checkpoint_paths.append(path)
        max_ckpt = self.config.training.max_checkpoints
        while len(self.checkpoint_paths) > max_ckpt:
            old = self.checkpoint_paths.pop(0)
            if old.exists():
                old.unlink()
        if is_best:
            best_path = self.config.paths.best_model_path
            torch.save(ckpt, best_path)
            self.logger.info(f"Saved best: {best_path}")

    def train(self):
        """Main training loop: iterate epochs, validate, checkpoint, early stop."""
        self.logger.info("=" * 60)
        if self.resume_ckpt is not None:
            self.logger.info(f"RESUMING TRAINING FROM EPOCH {self.current_epoch + 1}")
        else:
            self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 60)
        patience = self.config.training.early_stop_patience
        start_epoch = self.current_epoch + 1 if self.resume_ckpt is not None else 0
        for epoch in range(start_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.monotonic()
            tl, ta = self.train_epoch()
            self.epoch_times.append(time.monotonic() - epoch_start)
            self.logger.info(f"\nEpoch {epoch}: Train Loss={tl:.4f}, Acc={ta:.2f}%")

            should_validate = epoch % self.config.training.validate_every == 0
            if should_validate:
                vl, va, val_per_type = self.validate()
                self.logger.info(f"  Val Loss={vl:.4f}, Acc={va:.2f}%")
                self.writer.add_scalar("val/loss", vl, epoch)
                self.writer.add_scalar("val/acc", va, epoch)
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(va)
                is_best = va > self.best_val_acc
                if is_best:
                    self.best_val_acc = va
                    self.best_val_per_type = val_per_type
                    self.best_val_epoch = epoch
                    self.early_stop_counter = 0
                    self.logger.info(f"  New best val acc: {va:.2f}%")
                else:
                    self.early_stop_counter += 1
                    self.logger.info(f"  No improvement ({self.early_stop_counter}/{patience})")
                if epoch % self.config.training.save_every == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
                if self.early_stop_counter >= patience:
                    self.logger.info(
                        f"\nEarly stopping triggered after {epoch + 1} epochs "
                        f"(no improvement for {patience} validation runs)"
                    )
                    break

        self.save_checkpoint(is_best=False)
        self.logger.info(f"\nBest val acc: {self.best_val_acc:.2f}%")
        if self._should_write_baseline():
            self._write_baseline()

    def _should_write_baseline(self) -> bool:
        """Whether to emit the ``outputs/<experiment>.json`` artifact.

        Debug runs write only under ``checkpoints/debug/`` and must never touch
        production artifacts.
        """
        return not self.config.debug

    def _write_baseline(self):
        """Record run results + resource metrics to ``outputs/``.

        Always writes to ``outputs/<experiment_name>.json`` to avoid overwriting
        canonical artifacts. Includes full symbolic configuration plus software
        provenance (versions, git commit, preprocessing) needed to reproduce the
        run.
        """
        wall_time_s = time.monotonic() - self.train_start_time
        # One training process uses at most one GPU; multiply by *visible* device
        # count (as the old code did) overstated GPU-hours on multi-GPU hosts.
        gpus_used = 1 if self.device.type == "cuda" else 0
        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
        results = {
            "experiment_name": self.config.experiment_name,
            "seed": self.config.seed,
            "config": {
                "image_size": list(self.config.data.image_size),
                "batch_size": self.config.data.batch_size,
                "num_epochs": self.config.training.num_epochs,
                "learning_rate": self.config.training.learning_rate,
                "symbolic_enabled": self.config.symbolic.enabled,
                "symbolic_weight": self.config.symbolic.symbolic_weight,
                "weighting_strategy": self.config.symbolic.weighting_strategy,
                "query_parser": self.config.symbolic.query_parser,
                "executor": self.config.symbolic.executor,
                "num_regions": len(getattr(self, "region_names", []) or self.config.symbolic.num_regions),
                "norm": self.config.data.norm,
                "transform_version": TRANSFORM_VERSION,
                "use_multiscale": self.config.visual.use_multiscale,
                "visual_backbone": self.config.visual.backbone,
                "visual_model_name": self.config.visual.model_name,
                "visual_lora_rank": self.config.visual.lora_rank,
                "question_model_name": self.config.question.model_name,
                "question_lora_rank": self.config.question.lora_rank,
                "deterministic": self.config.training.deterministic,
                "amp": self.amp_enabled,
            },
            "provenance": _collect_provenance(),
            "best_val_acc": round(self.best_val_acc, 4),
            "best_val_epoch": self.best_val_epoch,
            "best_val_per_type_accuracy": self.best_val_per_type,
            "resources": {
                "wall_time_s": round(wall_time_s, 1),
                "wall_time_h": round(wall_time_s / 3600.0, 3),
                "gpu_hours": round(wall_time_s * gpus_used / 3600.0, 3),
                "num_gpus_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpus_used": gpus_used,
                "peak_vram_mb": round(peak_vram_mb, 1),
                "epochs_trained": self.current_epoch + 1,
                "epoch_time_s_mean": round(sum(self.epoch_times) / len(self.epoch_times), 2)
                if self.epoch_times
                else 0.0,
            },
        }
        filename = f"{self.config.experiment_name}.json"
        out = self.config.paths.output_dir / filename
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {out}")


def main():
    """Parse CLI args, build config, and launch training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Neuro-Symbolic PathVQA")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--no-symbolic", action="store_true", help="Disable symbolic reasoning")
    parser.add_argument(
        "--resume",
        type=str,
        metavar="CHECKPOINT",
        help="Resume training from a checkpoint (checkpoints/checkpoint_epoch_N.pt; 'auto' = newest)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (controls outputs/<name>.json artifact + TensorBoard log dir)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=ABLATION_CHOICES,
        default=None,
        help="Run specific ablation config (A=Baseline, B=SOTA, C=Diff, D=NoGate, E=Regex+Diff)",
    )
    parser.add_argument(
        "--weighting-strategy",
        type=str,
        choices=["static", "learned"],
        default=None,
        help="Neuro-symbolic fusion gate: 'static' fixed weight, 'learned' confidence gate",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic cuDNN kernels/algorithms for reproducible runs",
    )
    args = parser.parse_args()
    if args.config:
        config = Config.load(args.config)
    elif args.debug:
        config = get_debug_config()
    else:
        config = get_default_config()

    # Forward CLI flags as config overrides. The A–E matrix itself lives in
    # pathvqa.utils.ablation (shared with evaluate.py and scripts/run_ablations.py);
    # here we only pass the selected key through.
    overrides: Dict[str, Any] = {
        "experiment_name": args.experiment,
        "debug": args.debug,
        "deterministic": args.deterministic,
        "no_symbolic": args.no_symbolic,
        "weighting_strategy": args.weighting_strategy,
        "ablation": args.ablation,
    }
    apply_cli_overrides(config, overrides)

    logger = setup_logging(log_file=config.paths.log_dir / config.experiment_name / "train.log")
    validate_config(config)

    if args.resume == "auto":
        candidates = sorted(
            config.paths.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not candidates:
            raise SystemExit(f"--resume auto: no checkpoint_epoch_*.pt found in {config.paths.checkpoint_dir}")
        args.resume = str(candidates[-1])
        logger.info(f"--resume auto: picked {args.resume}")

    set_seed(config.seed, deterministic=config.training.deterministic)
    if torch.cuda.is_available():
        # benchmark must stay off in deterministic mode (set_seed handles this).
        if not config.training.deterministic:
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    logger.info(
        f"Experiment: {config.experiment_name} | seed: {config.seed} | "
        f"debug: {config.debug} | deterministic: {config.training.deterministic}"
    )
    Trainer(config, resume_path=args.resume, overrides=overrides).train()


if __name__ == "__main__":
    main()
