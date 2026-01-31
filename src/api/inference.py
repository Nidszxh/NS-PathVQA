"""Model loading and inference for the PathVQA FastAPI service.

Loads a trained checkpoint and serves predictions. Handles image
preprocessing (base64 → CLIP-normalized tensor) matching the training pipeline.
"""

import base64
import io
from typing import Dict, Tuple

import PIL.Image
import torch

import sys

sys.path.append("src")

from data.pathvqa_dataset import get_clip_normalize, resize_keep_aspect
from models.pathvqa_model import build_model
from models.question.question_encoder import get_question_tokenizer
from symbolic.query_parser import parse_question
from symbolic.executor import (
    execute, build_region_names, build_region_mapping, build_attribute_mappings,
)
from symbolic.routing import encode_question_types
from data.dataset_adapter import AnatomicalOntology
from utils.config import backfill_config


class PathVQAInference:
    """Load a checkpoint and serve predictions."""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = backfill_config(ckpt["config"])
        self.config.paths.__post_init__()
        self.checkpoint_path = str(checkpoint_path)
        self.best_val_acc = ckpt.get("best_val_acc")

        # Vocab & region mappings
        self.idx_to_answer = ckpt.get("idx_to_answer")
        self.answer_to_idx = ckpt.get("answer_to_idx")
        self.region_names = ckpt.get("region_names", [])
        self.answer_vocab = list(self.answer_to_idx.keys())

        if not self.region_names:
            self.region_names = build_region_names(self.answer_vocab)
        self.region_to_answer_idx = build_region_mapping(
            self.region_names, self.answer_to_idx
        ).to(self.device)
        self.attribute_mappings = build_attribute_mappings(self.answer_to_idx)
        self.ontology = AnatomicalOntology(getattr(self.config.symbolic, "ontology_path", None))

        # Tokenizer & model
        self.tokenizer = get_question_tokenizer(self.config.question.model_name)
        self.model = build_model(self.config, len(self.answer_vocab))
        if ckpt.get("ema_state_dict") is not None:
            self.model.load_state_dict(ckpt["ema_state_dict"])
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.amp_enabled = self.config.training.use_amp and self.device.type == "cuda"

        # Preprocessing
        self._normalize = get_clip_normalize()
        self._resize = lambda img: resize_keep_aspect(img, self.config.data.image_size)

    def _preprocess_image(self, image_b64: str) -> torch.Tensor:
        """Decode base64 image → CLIP-normalized (1, 3, H, W) tensor."""
        img_bytes = base64.b64decode(image_b64)
        pil_image = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = self._resize(pil_image)             # uint8 (3, H, W)
        tensor = tensor.to(torch.float32) / 255.0    # float [0, 1]
        tensor = self._normalize(tensor)              # CLIP-normalized
        return tensor.unsqueeze(0).to(self.device)    # (1, 3, H, W)

    def _tokenize_question(self, question: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a single question → (input_ids, attention_mask) with batch dim."""
        enc = self.tokenizer(
            question, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.config.question.max_seq_len,
        )
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )

    @torch.no_grad()
    def predict(self, image_b64: str, question: str) -> Dict:
        """Return predicted answer, confidence, and question type."""
        images = self._preprocess_image(image_b64)
        input_ids, attn_mask = self._tokenize_question(question)
        qtype = parse_question(question, self.answer_vocab).qtype

        qtype_onehot = None
        if self.config.symbolic.weighting_strategy == "learned":
            qtype_onehot = encode_question_types([question]).to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            outputs = self.model(images, input_ids, attn_mask, qtype_onehot)

        combined, _ = self._compute_symbolic(outputs, [question])
        probs = torch.softmax(combined, dim=-1)
        conf, pred_idx = probs.max(dim=-1)
        answer = self.idx_to_answer[pred_idx.item()]

        return {
            "answer": answer,
            "confidence": round(conf.item(), 4),
            "question_type": qtype,
        }

    @torch.no_grad()
    def explain(self, image_b64: str, question: str) -> Dict:
        """Return prediction with symbolic reasoning trace."""
        images = self._preprocess_image(image_b64)
        input_ids, attn_mask = self._tokenize_question(question)
        query = parse_question(question, self.answer_vocab)

        qtype_onehot = None
        if self.config.symbolic.weighting_strategy == "learned":
            qtype_onehot = encode_question_types([question]).to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            outputs = self.model(images, input_ids, attn_mask, qtype_onehot)

        combined, exec_trace = self._compute_symbolic(outputs, [question])
        probs = torch.softmax(combined, dim=-1)
        conf, pred_idx = probs.max(dim=-1)
        answer = self.idx_to_answer[pred_idx.item()]

        # Build trace
        region_scores = None
        region_names_top = None
        if "scene_region_logits" in outputs:
            region_logits = outputs["scene_region_logits"][0]  # (num_regions,)
            topk = min(5, len(self.region_names))
            values, indices = region_logits.topk(topk)
            region_scores = [round(v.item(), 4) for v in values]
            region_names_top = [self.region_names[i] for i in indices]

        from utils.interpretability import (
            generate_visual_grounding_map,
            generate_counterfactual_explanation,
            generate_hierarchical_reasoning_tree,
        )

        grounding_boxes = None
        patch_indices = None
        if "attended_features" in outputs:
            g_map = generate_visual_grounding_map(outputs["attended_features"][0], grid_size=(7, 7), top_k=3)
            grounding_boxes = g_map["top_boxes"]
            patch_indices = g_map.get("patch_indices")

        cf_explanation = generate_counterfactual_explanation(
            baseline_prediction_str=answer,
            region_scores=region_scores or [],
            region_names=region_names_top or [],
            predicted_changed=False,
            perturbed_region=region_names_top[0] if region_names_top else None,
        )

        tree = generate_hierarchical_reasoning_tree(
            prediction=answer,
            confidence=conf.item(),
            qtype=query.qtype,
            target=query.target,
            region_names=region_names_top or [],
            region_scores=region_scores or [],
            patch_indices=patch_indices,
        )

        return {
            "answer": answer,
            "confidence": round(conf.item(), 4),
            "question_type": query.qtype,
            "trace": {
                "question_type": query.qtype,
                "target": query.target,
                "attribute": query.attribute,
                "symbolic_used": exec_trace.get("symbolic_used", [False])[0],
                "region_scores": region_scores,
                "region_names": region_names_top,
                "grounding_boxes": grounding_boxes,
                "counterfactual_explanation": cf_explanation,
                "reasoning_tree": tree.get("reasoning_trace"),
            },
        }

    def _compute_symbolic(self, outputs, questions):
        """Combine neural and symbolic logits (mirrors evaluate.py logic)."""
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
        if self.config.symbolic.weighting_strategy == "learned" and "gate_values" in outputs:
            gate = outputs["gate_values"].squeeze(-1)
            symbolic_logits = exec_out["symbolic_logits"]
            gate_expanded = gate.unsqueeze(1).expand_as(symbolic_logits)
            combined = (1 - gate_expanded) * outputs["answer_logits"] + gate_expanded * symbolic_logits
            return combined, exec_out.get("trace", {})
        combined = outputs["answer_logits"] + self.config.symbolic.symbolic_weight * exec_out["symbolic_logits"]
        return combined, exec_out.get("trace", {})

    def model_info(self) -> Dict:
        """Return checkpoint metadata."""
        return {
            "checkpoint": self.checkpoint_path,
            "best_val_acc": self.best_val_acc,
            "num_regions": len(self.region_names),
            "num_answers": len(self.answer_vocab),
            "weighting_strategy": self.config.symbolic.weighting_strategy,
            "symbolic_enabled": self.config.symbolic.enabled,
            "device": str(self.device),
        }
