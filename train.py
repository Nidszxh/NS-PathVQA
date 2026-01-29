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
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, 'src')

from data.pathvqa_dataset import PathVQADataset, pathvqa_dataloader, prepare_batch, collate_fn
from models.pathvqa_model import build_model
from models.ema import ModelEMA
from models.question.question_encoder import get_question_tokenizer
from utils.config import Config, get_default_config, get_debug_config, backfill_config, validate_config
from utils.logging_utils import setup_logging, get_logger
from utils.seed import set_seed, seed_worker
from utils.conformal import ConformalPredictor
from symbolic.routing import encode_question_types
from symbolic.query_parser import parse_question
from symbolic.executor import (
    execute, build_region_names, build_region_mapping,
    build_attribute_mappings,
)
from symbolic.aux_losses import compute_symbolic_aux_losses
from symbolic.ontology_loss import compute_sibling_regularization
from symbolic.ltn import MedicalLogicTensorNetwork
from data.dataset_adapter import AnatomicalOntology


class Trainer:
    """Manages the full training lifecycle for PathVQA."""

    def __init__(self, config: Config, resume_path: str = None):
        self.config = config
        self.resume_path = resume_path
        self.logger = get_logger("train")
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
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
                self.config.paths.__post_init__()
            config = self.config

        # Build answer vocabulary from the training dataset
        self.logger.info("Building vocabulary...")
        train_dataset = PathVQADataset(
            split="train", image_size=config.data.image_size,
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
            self.answer_to_idx["<UNK>"] = len(self.answer_to_idx)
            self.idx_to_answer[len(self.idx_to_answer)] = "<UNK>"
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
        with open(config.paths.data_dir / "vocab_info.json", "w") as f:
            json.dump(vocab_info, f, indent=2)

        self.logger.info(f"Answer vocab: {len(self.answer_vocab)}")

        # Build dataloaders
        self.logger.info("Building dataloaders...")
        nw = config.data.num_workers
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.data.batch_size,
            shuffle=True, num_workers=nw,
            collate_fn=collate_fn, pin_memory=True,
            persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None,
            worker_init_fn=seed_worker,
        )
        self.val_loader = pathvqa_dataloader(
            split="val", batch_size=config.data.batch_size,
            shuffle=False, num_workers=config.data.num_workers,
            image_size=config.data.image_size,
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
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,} "
                         f"(trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,})")

        self.criterion = nn.NLLLoss()
        self.optimizer = self._build_optimizer()
        self.total_optimizer_steps = (
            math.ceil(len(self.train_loader) / config.training.grad_accum_steps)
            * config.training.num_epochs
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

        # EMA shadow of the model weights (used for eval; persisted in checkpoints).
        if config.training.ema_enabled:
            self.ema = ModelEMA(self.model, decay=config.training.ema_decay)
            self.logger.info(f"EMA enabled (decay={config.training.ema_decay})")
        else:
            self.ema = None

        # Resume: restore model/optimizer/scheduler/training-state from checkpoint
        if self.resume_ckpt is not None:
            self.model.load_state_dict(self.resume_ckpt["model_state_dict"])
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
            self.logger.info(f"Resumed: epoch={self.current_epoch}, global_step={self.global_step}, "
                             f"best_val_acc={self.best_val_acc:.2f}%, "
                             f"early_stop_counter={self.early_stop_counter}/{config.training.early_stop_patience}")

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
                self.optimizer, mode="max", factor=t.factor, patience=t.patience, min_lr=t.min_lr)
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

        Fuses in log-probability space: log P_final = (1-g)·log P_neural + g·log P_symbolic
        Uses parse_question to classify each question, then execute() to
        produce symbolic logits. Count queries are routed through the DSL interpreter.
        """
        if not self.config.symbolic.enabled or "scene_region_logits" not in outputs:
            return outputs["answer_logits"], {}
        queries = [parse_question(q, self.answer_vocab) for q in questions]

        # DSL routing for count queries
        dsl_logits = None
        if "patch_features" in outputs:
            dsl_logits = torch.zeros_like(outputs["answer_logits"])
            for i, q in enumerate(queries):
                if q.qtype == "count" and q.program is not None:
                    dsl_out = self.model.dsl_interpreter(
                        q.program,
                        outputs["patch_features"][i:i+1],
                        outputs,
                        self.answer_to_idx,
                    )
                    dsl_logits[i] = dsl_out[0]

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

        # For count queries, prefer DSL logits over executor's zero logits
        if dsl_logits is not None:
            for i, q in enumerate(queries):
                if q.qtype == "count":
                    exec_out["symbolic_logits"][i] = dsl_logits[i]

        # Product-of-Experts fusion in log-probability space
        log_p_neural = torch.log_softmax(outputs["answer_logits"], dim=-1)
        log_p_symbolic = torch.log_softmax(exec_out["symbolic_logits"] + 1e-8, dim=-1)

        if self.config.symbolic.weighting_strategy == "learned" and "gate_values" in outputs:
            g = outputs["gate_values"].squeeze(-1)  # (B,)
            log_p_combined = (1 - g).unsqueeze(1) * log_p_neural + g.unsqueeze(1) * log_p_symbolic
        else:
            w = self.config.symbolic.symbolic_weight
            log_p_combined = (1 - w) * log_p_neural + w * log_p_symbolic

        return log_p_combined, exec_out.get("trace", {})

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
            input_ids, attn_mask, targets, images, questions = prepare_batch(
                batch, self.tokenizer, self.device,
                max_seq_len=self.config.question.max_seq_len)
            qtype_onehot = encode_question_types(questions).to(self.device) if self.config.symbolic.weighting_strategy == "learned" else None
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(images, input_ids, attn_mask, qtype_onehot)
                logits, _ = self._compute_symbolic_logits(outputs, questions)
                loss = self.criterion(logits, targets) / accum
                if self.ltn_engine is not None and "scene_region_logits" in outputs:
                    ltn_loss, _ = self.ltn_engine.evaluate_clauses(outputs, qtype_onehot)
                    loss = loss + (self.config.symbolic.ltn_weight * ltn_loss) / accum
                if self.config.symbolic.enabled and "scene_region_logits" in outputs:
                    queries = [parse_question(q, self.answer_vocab) for q in questions]
                    aux = compute_symbolic_aux_losses(
                        scene_logits=outputs,
                        queries=queries,
                        targets=targets,
                        answer_to_idx=self.answer_to_idx,
                        region_names=self.region_names,
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
                    input_ids, attn_mask, targets, images, questions = prepare_batch(
                        batch, self.tokenizer, self.device,
                        max_seq_len=self.config.question.max_seq_len)
                    qtype_onehot = encode_question_types(questions).to(self.device) if self.config.symbolic.weighting_strategy == "learned" else None
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
                    if "gate_values" in outputs:
                        gate_vals = outputs["gate_values"].squeeze(-1)
                        for i, question in enumerate(questions):
                            qtype = parse_question(question, self.answer_vocab).qtype
                            gate_values_per_qtype.setdefault(qtype, []).append(gate_vals[i].item())
                    for i, question in enumerate(questions):
                        qtype = parse_question(question, self.answer_vocab).qtype
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
            mean_gates = {q: sum(v)/len(v) for q, v in gate_values_per_qtype.items()}
            self.logger.info(f"  Gate values per qtype: {mean_gates}")
            for qtype, vals in gate_values_per_qtype.items():
                mean_g = sum(vals) / len(vals)
                self.writer.add_scalar(f"gate/{qtype}", mean_g, self.current_epoch)
        per_type_accuracy = {
            qtype: {"accuracy": 100.0 * c / t, "correct": c, "total": t}
            for qtype, (c, t) in sorted(per_type.items())
        }
        # Conformal prediction calibration on validation set
        if all_logits:
            all_logits_t = torch.cat(all_logits)
            all_targets_t = torch.cat(all_targets)
            self.conformal.calibrate(all_logits_t, all_targets_t)
            self.logger.info(f"  Conformal (α={self.conformal.alpha}): τ={self.conformal.tau:.4f}, "
                             f"coverage={self.conformal.calibration_stats['coverage']:.2%}, "
                             f"avg_set_size={self.conformal.calibration_stats['avg_set_size']:.2f}")
            self.writer.add_scalar("conformal/tau", self.conformal.tau, self.current_epoch)
            self.writer.add_scalar("conformal/coverage", self.conformal.calibration_stats["coverage"], self.current_epoch)
            self.writer.add_scalar("conformal/avg_set_size", self.conformal.calibration_stats["avg_set_size"], self.current_epoch)
        return total_loss / len(self.val_loader), 100.0 * correct / total, per_type_accuracy

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint with all metadata needed for evaluation."""
        ckpt = {
            "epoch": self.current_epoch, "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "best_val_acc": self.best_val_acc, "config": self.config,
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
                    self.logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs "
                                     f"(no improvement for {patience} validation runs)")
                    break

        self.save_checkpoint(is_best=False)
        self.logger.info(f"\nBest val acc: {self.best_val_acc:.2f}%")
        if not self.config.debug:
            self._write_baseline()

    def _write_baseline(self):
        """Record run results + resource metrics to ``outputs/``.

        Writes to ``outputs/baseline.json`` for the default experiment and
        ``outputs/<experiment_name>.json`` otherwise.
        """
        wall_time_s = time.monotonic() - self.train_start_time
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
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
            },
            "best_val_acc": round(self.best_val_acc, 4),
            "best_val_epoch": self.best_val_epoch,
            "best_val_per_type_accuracy": self.best_val_per_type,
            "resources": {
                "wall_time_s": round(wall_time_s, 1),
                "wall_time_h": round(wall_time_s / 3600.0, 3),
                "gpu_hours": round(wall_time_s * max(num_gpus, 1) / 3600.0, 3),
                "num_gpus": num_gpus,
                "peak_vram_mb": round(peak_vram_mb, 1),
                "epochs_trained": self.current_epoch + 1,
                "epoch_time_s_mean": round(sum(self.epoch_times) / len(self.epoch_times), 2),
            },
        }
        default_name = "pathvqa_neuro_symbolic"
        filename = "baseline.json" if self.config.experiment_name == default_name else f"{self.config.experiment_name}.json"
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
    parser.add_argument("--resume", type=str, metavar="CHECKPOINT",
                        help="Resume training from a checkpoint (checkpoints/checkpoint_epoch_N.pt; 'auto' = newest)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name (controls outputs/<name>.json artifact + TensorBoard log dir)")
    args = parser.parse_args()
    if args.config:
        config = Config.load(args.config)
    elif args.debug:
        config = get_debug_config()
    else:
        config = get_default_config()
    if args.experiment:
        config.experiment_name = args.experiment
    if args.no_symbolic:
        config.symbolic.enabled = False

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

    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    logger.info(f"Experiment: {config.experiment_name} | seed: {config.seed} | debug: {config.debug}")
    Trainer(config, resume_path=args.resume).train()


if __name__ == "__main__":
    main()
