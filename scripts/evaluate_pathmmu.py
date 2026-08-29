"""Evaluate a saved PathVQA classifier on PathMMU with explicit label coverage.

PathMMU is multiple choice while this checkpoint predicts a fixed PathVQA
answer vocabulary. An item is scored only when every choice and the gold answer
map exactly to a checkpoint class after text normalization. The artifact reports
this coverage; scored accuracy must never be presented without it.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from evaluate import Evaluator
from pathvqa.data.pathvqa_dataset import get_clip_normalize, normalize_text, resize_keep_aspect
from pathvqa.symbolic.routing import encode_question_types


def option_text(option: str) -> str:
    """Remove a leading multiple-choice label, then normalize answer text."""
    value = option.strip()
    if len(value) > 2 and value[0].upper() in "ABCDEFGH" and value[1] in ") .:-":
        value = value[2:].strip()
    return normalize_text(value)


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("data", "questions", "annotations"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Expected a list of PathMMU records (or data/questions/annotations list)")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    evaluator = Evaluator(str(args.checkpoint))
    config = evaluator.config
    records = load_records(args.data_json)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    answer_to_idx = evaluator.answer_to_idx
    image_size = tuple(config.data.image_size)
    normalize = get_clip_normalize() if config.data.norm == "clip" else None
    if normalize is None:
        from torchvision import transforms
        from pathvqa.data.pathvqa_dataset import IMAGENET_MEAN, IMAGENET_STD

        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    rows = []
    skipped = []
    for position, record in enumerate(records):
        try:
            image_candidates = [
                record.get("img"),
                record.get("image"),
                record.get("image_path"),
                record.get("source_img"),
            ]
            image_candidates = [str(candidate) for candidate in image_candidates if candidate]
            image_path = next(
                (
                    args.image_dir / candidate
                    for candidate in image_candidates
                    if (args.image_dir / candidate).is_file()
                ),
                None,
            )
            question = str(record["question"])
            options = record["options"]
            gold = option_text(str(record["answer"]))
            if not image_candidates or not isinstance(options, list) or len(options) < 2:
                raise ValueError("missing image/options")
            option_answers = [option_text(str(option)) for option in options]
            option_indices = [answer_to_idx.get(answer) for answer in option_answers]
            gold_index = answer_to_idx.get(gold)
            if gold not in option_answers:
                raise ValueError("gold answer does not match any normalized option")
            if gold_index is None or any(index is None for index in option_indices):
                skipped.append({"position": position, "reason": "choice_or_gold_not_in_checkpoint_vocabulary"})
                continue
            if image_path is None:
                skipped.append(
                    {
                        "position": position,
                        "reason": "image_file_missing",
                        "image_candidates": image_candidates,
                    }
                )
                continue
            image = Image.open(image_path).convert("RGB")
            image_u8 = resize_keep_aspect(image, image_size)
            image_tensor = pil_to_tensor(image_u8).to(torch.float32) / 255.0
            image_tensor = normalize(image_tensor)
            rows.append(
                {
                    "position": position,
                    "record": record,
                    "question": question,
                    "image": image_tensor,
                    "options": [str(option) for option in options],
                    "option_answers": option_answers,
                    "option_indices": [int(index) for index in option_indices],
                    "gold_answer": gold,
                    "gold_index": int(gold_index),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"position": position, "reason": str(exc)})

    predictions = []
    correct = 0
    evaluator.model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            questions = [row["question"] for row in batch]
            enc = evaluator.tokenizer(
                questions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=config.question.max_seq_len,
            )
            images = torch.stack([row["image"] for row in batch]).to(evaluator.device)
            input_ids = enc["input_ids"].to(evaluator.device)
            attention_mask = enc["attention_mask"].to(evaluator.device)
            qtype_onehot = (
                encode_question_types(questions).to(evaluator.device)
                if config.symbolic.weighting_strategy == "learned"
                else None
            )
            with torch.amp.autocast(
                device_type=evaluator.device.type,
                enabled=evaluator.amp_enabled,
            ):
                outputs = evaluator.model(images, input_ids, attention_mask, qtype_onehot)
                logits, _ = evaluator._compute_symbolic_logits(outputs, questions)
            for local_index, row in enumerate(batch):
                class_scores = logits[local_index, row["option_indices"]]
                selected = int(class_scores.argmax().item())
                predicted_answer = row["option_answers"][selected]
                is_correct = predicted_answer == row["gold_answer"]
                correct += int(is_correct)
                predictions.append(
                    {
                        "position": row["position"],
                        "image": str(
                            row["record"].get("source_img") or row["record"].get("img") or row["record"].get("image")
                        ),
                        "question": row["question"],
                        "options": row["options"],
                        "gold_answer": row["gold_answer"],
                        "predicted_answer": predicted_answer,
                        "correct": is_correct,
                    }
                )

    scored = len(predictions)
    artifact = {
        "benchmark": "PathMMU",
        "checkpoint": str(args.checkpoint),
        "data_json": str(args.data_json),
        "image_dir": str(args.image_dir),
        "protocol": "candidate choice selected by checkpoint class logit; exact normalized vocabulary match required",
        "total_records": len(records),
        "scored_records": scored,
        "coverage": scored / len(records) if records else 0.0,
        "correct": correct,
        "accuracy_on_scored_subset": 100.0 * correct / scored if scored else None,
        "skipped_records": skipped,
        "predictions": predictions,
        "checkpoint_config": asdict(evaluator.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"PathMMU: {correct}/{scored} correct among scored examples; "
        f"coverage={artifact['coverage']:.1%} ({len(skipped)} skipped)."
    )
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
