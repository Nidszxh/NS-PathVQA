"""Evaluation script for Neuro-Symbolic PathVQA.

Loads a trained checkpoint and runs inference on the specified split.
Supports both neuro-symbolic and neural-only evaluation, automatically
detecting which mode was used during training from the checkpoint config.

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

sys.path.append("src")

from data_loaders.pathvqaDataset import pathvqa_dataloader, prepare_batch
from models.pathvqa_model import NeuroSymbolicPathVQA, build_model
from models.text.question_encoder import QuestionVocabulary
from utils.config import Config
from symbolic.query_parser import parse_question
from symbolic.executor import (
    execute, build_region_names, build_region_mapping,
    build_attribute_mappings,
)


class Evaluator:
    """Loads a trained checkpoint and evaluates on PathVQA data."""

    def __init__(self, checkpoint_path: str, config_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = Config.load(config_path) if config_path else ckpt["config"]
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
        self.region_to_answer_idx = build_region_mapping(self.region_names, self.answer_to_idx)
        self.attribute_mappings = build_attribute_mappings(self.answer_to_idx)

        # Load question vocabulary and build model
        self.question_vocab = QuestionVocabulary.load(
            str(self.config.paths.data_dir / "question_vocab.json"))
        self.model = build_model(self.config, len(self.question_vocab), len(self.answer_vocab))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Best val acc: {ckpt['best_val_acc']:.2f}%")
        print(f"Symbolic: {'enabled' if self.config.symbolic.enabled else 'disabled'}, "
              f"{len(self.region_names)} regions")
        attr_sizes = {k: len(v) for k, v in self.attribute_mappings.items()}
        print(f"Attribute heads: {attr_sizes}")

    def _compute_symbolic_logits(self, outputs, questions):
        """Combine neural and symbolic logits (mirrors trainer logic)."""
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

    def evaluate(self, split: str = "val", max_samples: int = None):
        """Run evaluation on the specified split."""
        loader = pathvqa_dataloader(split=split, batch_size=self.config.data.batch_size,
            shuffle=False, num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size, max_samples=max_samples)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                q_idx, q_len, targets, images = prepare_batch(batch, self.question_vocab, self.device)
                questions = batch["questions"]
                outputs = self.model(images, q_idx, q_len)
                logits, _ = self._compute_symbolic_logits(outputs, questions)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        accuracy = 100.0 * correct / total
        return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Neuro-Symbolic PathVQA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    e = Evaluator(args.checkpoint, args.config)
    r = e.evaluate(args.split, args.max_samples)
    print(f"\nAccuracy: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
