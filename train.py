"""Training loop for Neuro-Symbolic PathVQA.

Usage:
    python train.py                         # full neuro-symbolic training
    python train.py --no-symbolic           # neural-only baseline
    python train.py --debug                 # fast debug run (500/100 samples, 5 epochs)
    python train.py --config custom.json    # custom config

The Trainer handles:
  - Vocabulary building (question vocab cached to JSON)
  - Data loading with dynamic answer vocab discovery
  - Symbolic module setup (region names, attribute mappings)
  - Training loop with gradient clipping and logging
  - Validation with symbolic path accuracy tracking
  - Checkpointing (periodic + best model)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import sys

sys.path.append('src')

from data_loaders.pathvqaDataset import PathVQADataset, pathvqa_dataloader, prepare_batch, collate_fn
from models.pathvqa_model import NeuroSymbolicPathVQA, build_model
from models.text.question_encoder import QuestionVocabulary
from utils.config import Config, get_default_config, get_debug_config
from symbolic.query_parser import parse_question
from symbolic.executor import (
    execute, build_region_names, build_region_mapping,
    build_attribute_mappings,
)


class Trainer:
    """Manages the full training lifecycle for PathVQA."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Build answer vocabulary from the training dataset
        print("Building vocabulary...")
        train_dataset = PathVQADataset(
            split="train", image_size=config.data.image_size,
            max_samples=config.data.max_train_samples,
        )
        self.answer_vocab = train_dataset.answers
        self.answer_to_idx = train_dataset.answer_to_idx
        self.idx_to_answer = train_dataset.idx_to_answer

        # Discover region names and build symbolic mappings if enabled
        if config.symbolic.enabled:
            self.region_names = build_region_names(self.answer_vocab)
            print(f"Built {len(self.region_names)} region names from answer vocabulary")
            self.region_to_answer_idx = build_region_mapping(self.region_names, self.answer_to_idx)
            self.attribute_mappings = build_attribute_mappings(self.answer_to_idx)
            attr_sizes = {k: len(v) for k, v in self.attribute_mappings.items()}
            print(f"Attribute mappings: {attr_sizes}")
            print(f"  Regions: {self.region_names[:6]}{'...' if len(self.region_names) > 6 else ''}")
            config.symbolic.num_regions = len(self.region_names)
            config.symbolic.region_names = tuple(self.region_names)
        else:
            self.region_names = []
            self.region_to_answer_idx = torch.tensor([], dtype=torch.long)
            self.attribute_mappings = {}

        # Build or load question vocabulary
        self.question_vocab = QuestionVocabulary()
        q_path = config.paths.data_dir / "question_vocab.json"
        if q_path.exists():
            self.question_vocab = QuestionVocabulary.load(str(q_path))
        else:
            for item in train_dataset:
                self.question_vocab.build_from_questions([item["question"]])
            config.paths.data_dir.mkdir(parents=True, exist_ok=True)
            self.question_vocab.save(str(q_path))

        # Save metadata for evaluation
        vocab_info = {
            "vocab_size": len(self.question_vocab),
            "answer_vocab_size": len(self.answer_vocab),
            "answer_to_idx": self.answer_to_idx,
            "idx_to_answer": self.idx_to_answer,
            "region_names": self.region_names,
        }
        with open(config.paths.data_dir / "vocab_info.json", "w") as f:
            json.dump(vocab_info, f, indent=2)

        print(f"Question vocab: {len(self.question_vocab)}, Answer vocab: {len(self.answer_vocab)}")

        # Build dataloaders
        print("Building dataloaders...")
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.data.batch_size,
            shuffle=True, num_workers=config.data.num_workers,
            collate_fn=collate_fn, pin_memory=True,
        )
        self.val_loader = pathvqa_dataloader(
            split="val", batch_size=config.data.batch_size,
            shuffle=False, num_workers=config.data.num_workers,
            image_size=config.data.image_size,
            max_samples=config.data.max_val_samples,
        )

        # Build model
        print("Building model...")
        self.model = build_model(config, len(self.question_vocab), len(self.answer_vocab))
        self.model.to(self.device)
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.writer = SummaryWriter(log_dir=config.paths.log_dir / config.experiment_name)
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0

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
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t.num_epochs)
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def _compute_symbolic_logits(self, outputs, questions):
        """Combine neural and symbolic logits via the Executor.

        Uses parse_question to classify each question, then execute() to
        produce symbolic logits that are weighted and added to neural logits.
        """
        if not self.config.symbolic.enabled or "scene_region_logits" not in outputs:
            return outputs["answer_logits"], {}
        queries = [parse_question(q, self.answer_vocab) for q in questions]
        exec_out = execute(
            scene_logits=outputs,
            queries=queries,
            region_names=self.region_names,
            region_to_answer_idx=self.region_to_answer_idx.to(self.device),
            attribute_mappings=self.attribute_mappings,
            answer_to_idx=self.answer_to_idx,
            answer_vocab_size=len(self.answer_vocab),
            neural_logits=outputs["answer_logits"],
        )
        combined = outputs["answer_logits"] + self.config.symbolic.symbolic_weight * exec_out["symbolic_logits"]
        return combined, exec_out.get("trace", {})

    def train_epoch(self):
        """Run one training epoch over the full train loader."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            q_idx, q_len, targets, images = prepare_batch(batch, self.question_vocab, self.device)
            questions = batch["questions"]
            outputs = self.model(images, q_idx, q_len)
            logits, _ = self._compute_symbolic_logits(outputs, questions)
            loss = self.criterion(logits, targets)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.optimizer.step()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item()
            if self.global_step % self.config.training.log_every == 0:
                acc = 100.0 * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%"})
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/acc", acc, self.global_step)
            self.global_step += 1
        return total_loss / len(self.train_loader), 100.0 * correct / total

    def validate(self):
        """Run validation, tracking neural + symbolic path accuracy."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        symbolic_count, symbolic_correct = 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                q_idx, q_len, targets, images = prepare_batch(batch, self.question_vocab, self.device)
                questions = batch["questions"]
                outputs = self.model(images, q_idx, q_len)
                logits, trace = self._compute_symbolic_logits(outputs, questions)
                loss = self.criterion(logits, targets)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item()
                if trace.get("symbolic_used"):
                    for i in range(len(questions)):
                        if trace["symbolic_used"][i]:
                            symbolic_count += 1
                            if preds[i] == targets[i]:
                                symbolic_correct += 1
        if symbolic_count > 0:
            sym_acc = 100.0 * symbolic_correct / symbolic_count
            print(f"  Symbolic path accuracy: {sym_acc:.2f}% ({symbolic_correct}/{symbolic_count})")
        return total_loss / len(self.val_loader), 100.0 * correct / total

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint with all metadata needed for evaluation."""
        ckpt = {
            "epoch": self.current_epoch, "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc, "config": self.config,
            "answer_vocab": self.answer_vocab,
            "answer_to_idx": self.answer_to_idx,
            "idx_to_answer": self.idx_to_answer,
            "region_names": self.region_names,
        }
        path = self.config.paths.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(ckpt, path)
        print(f"Saved: {path}")
        if is_best:
            best_path = self.config.paths.checkpoint_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            print(f"Saved best: {best_path}")

    def train(self):
        """Main training loop: iterate epochs, validate, checkpoint."""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            tl, ta = self.train_epoch()
            print(f"\nEpoch {epoch}: Train Loss={tl:.4f}, Acc={ta:.2f}%")

            should_validate = epoch % self.config.training.validate_every == 0
            if should_validate:
                vl, va = self.validate()
                print(f"  Val Loss={vl:.4f}, Acc={va:.2f}%")
                self.writer.add_scalar("val/loss", vl, epoch)
                self.writer.add_scalar("val/acc", va, epoch)
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(va)
                is_best = va > self.best_val_acc
                if is_best:
                    self.best_val_acc = va
                    print(f"  New best val acc: {va:.2f}%")
                if epoch % self.config.training.save_every == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)

            if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
        print(f"\nBest val acc: {self.best_val_acc:.2f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Neuro-Symbolic PathVQA")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--no-symbolic", action="store_true", help="Disable symbolic reasoning")
    args = parser.parse_args()
    if args.config:
        config = Config.load(args.config)
    elif args.debug:
        config = get_debug_config()
    else:
        config = get_default_config()
    if args.no_symbolic:
        config.symbolic.enabled = False
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    Trainer(config).train()


if __name__ == "__main__":
    main()
