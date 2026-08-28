"""Model loading and inference for the PathVQA FastAPI service.

Loads a trained checkpoint and serves predictions. Handles image
preprocessing (base64 → CLIP-normalized tensor) matching the training pipeline.
"""

import base64
import dataclasses
import io
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import PIL.Image
import torch


from pathvqa.data.pathvqa_dataset import get_clip_normalize, resize_keep_aspect
from pathvqa.models.pathvqa_model import build_model
from pathvqa.models.question.question_encoder import get_question_tokenizer
from pathvqa._compat import is_pre_dsl_checkpoint, migrate_legacy_scene_head
from pathvqa.symbolic.query_parser import parse_question
from pathvqa.symbolic.poe import SymbolicFusionEngine
from pathvqa.symbolic.executor import (
    build_region_names,
    build_region_mapping,
    build_attribute_mappings,
)
from pathvqa.symbolic.routing import encode_question_types
from pathvqa.data.dataset_adapter import AnatomicalOntology
from pathvqa.utils.config import backfill_config

logger = logging.getLogger(__name__)


class PathVQAInference:
    """Load a checkpoint and serve predictions."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """Load a checkpoint and rebuild the model as the architecture that saved it.

        Pre-DSL checkpoints are rebuilt without the DSL interpreter (see
        ``pathvqa._compat``); the shared PoE fusion engine is wired here.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = backfill_config(ckpt["config"])
        self.config.paths.__post_init__()
        # Pre-DSL checkpoints carry no dsl_interpreter.* params: rebuild the
        # model without the interpreter so strict loading succeeds and count
        # queries keep the pre-DSL path (see pathvqa._compat).
        legacy_states = [s for s in (ckpt.get("ema_state_dict"), ckpt.get("model_state_dict")) if s]
        if self.config.symbolic.enabled and legacy_states and is_pre_dsl_checkpoint(*legacy_states):
            self.config.symbolic.dsl_enabled = False
            logger.warning("Pre-DSL checkpoint: building model without the DSL interpreter")
        self.checkpoint_path = str(checkpoint_path)
        self.best_val_acc = ckpt.get("best_val_acc")

        # Vocab & region mappings
        self.idx_to_answer = ckpt.get("idx_to_answer")
        self.answer_to_idx = ckpt.get("answer_to_idx")
        self.region_names = ckpt.get("region_names", [])
        missing = [
            k for k, v in (("idx_to_answer", self.idx_to_answer), ("answer_to_idx", self.answer_to_idx)) if not v
        ]
        if missing:
            raise ValueError(
                f"Checkpoint is missing required metadata: {missing}. "
                "Re-save it from training or provide vocab_info.json."
            )
        self.answer_vocab = list(self.answer_to_idx.keys())

        if not self.region_names:
            self.region_names = build_region_names(self.answer_vocab)
        self.region_to_answer_idx = build_region_mapping(self.region_names, self.answer_to_idx).to(self.device)
        self.attribute_mappings = build_attribute_mappings(self.answer_to_idx)
        self.ontology = AnatomicalOntology(getattr(self.config.symbolic, "ontology_path", None))
        if getattr(self.config.symbolic, "executor", "hard") == "differentiable":
            from pathvqa.symbolic.diff_executor import DifferentiableExecutor

            self.diff_executor: Optional[DifferentiableExecutor] = DifferentiableExecutor(
                self.answer_to_idx, self.region_names
            )
        else:
            self.diff_executor = None

        # Tokenizer & model
        self.tokenizer = get_question_tokenizer(self.config.question.model_name)
        self.model = build_model(self.config, len(self.answer_vocab))
        if ckpt.get("ema_state_dict") is not None:
            ema_state, _ = migrate_legacy_scene_head(ckpt["ema_state_dict"])
            self.model.load_state_dict(ema_state)
        else:
            model_state, _ = migrate_legacy_scene_head(ckpt["model_state_dict"])
            self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()

        self.amp_enabled = self.config.training.use_amp and self.device.type == "cuda"

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

        # Preprocessing
        self._normalize = get_clip_normalize()
        self._resize: Callable[[Any], Any] = lambda img: resize_keep_aspect(img, self.config.data.image_size)

    def _preprocess_image(self, image_b64: str) -> torch.Tensor:
        """Decode base64 image → CLIP-normalized (1, 3, H, W) tensor."""
        import torchvision.transforms.functional as TF

        img_bytes = base64.b64decode(image_b64)
        pil_image = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        resized = self._resize(pil_image)
        # TF comes from a module without type stubs; pin the result explicitly.
        tensor: torch.Tensor = TF.pil_to_tensor(resized).to(self.device)  # uint8 (3, H, W)
        tensor = tensor.to(torch.float32) / 255.0  # float [0, 1]
        normalized: torch.Tensor = self._normalize(tensor)  # CLIP-normalized
        return normalized.unsqueeze(0)  # (1, 3, H, W)

    def _tokenize_question(self, question: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a single question → (input_ids, attention_mask) with batch dim."""
        enc = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.question.max_seq_len,
        )
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )

    @torch.no_grad()
    def predict(self, image_b64: str, question: str) -> Dict[str, Any]:
        """Return predicted answer, confidence, and question type."""
        images = self._preprocess_image(image_b64)
        input_ids, attn_mask = self._tokenize_question(question)
        qtype = parse_question(question, self.answer_vocab).qtype

        qtype_onehot = None
        if self.config.symbolic.weighting_strategy == "learned":
            qtype_onehot = encode_question_types([question]).to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
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
    def explain(self, image_b64: str, question: str) -> Dict[str, Any]:
        """Return prediction with symbolic reasoning trace."""
        images = self._preprocess_image(image_b64)
        input_ids, attn_mask = self._tokenize_question(question)
        query = parse_question(question, self.answer_vocab)

        qtype_onehot = None
        if self.config.symbolic.weighting_strategy == "learned":
            qtype_onehot = encode_question_types([question]).to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            outputs = self.model(images, input_ids, attn_mask, qtype_onehot)

        combined, exec_trace = self._compute_symbolic(outputs, [question])
        probs = torch.softmax(combined, dim=-1)
        conf, pred_idx = probs.max(dim=-1)
        answer = self.idx_to_answer[pred_idx.item()]

        # Build trace
        region_scores = None
        region_names_top = None
        heatmap_2d = None
        if outputs.scene_region_logits is not None:
            region_logits = outputs.scene_region_logits[0]  # (num_regions,)
            topk = min(5, len(self.region_names))
            values, indices = region_logits.topk(topk)
            region_scores = [round(v.item(), 4) for v in values]
            region_names_top = [self.region_names[i] for i in indices]

        from pathvqa.utils.interpretability import (
            generate_visual_grounding_map,
            generate_counterfactual_explanation,
            generate_hierarchical_reasoning_tree,
        )

        grounding_boxes = None
        patch_indices = None
        if outputs.attention_weights is not None:
            g_map = generate_visual_grounding_map(outputs.attention_weights[0], grid_size=(7, 7), top_k=3)
            grounding_boxes = g_map["top_boxes"]
            patch_indices = g_map.get("patch_indices")
            heatmap_2d = g_map.get("grid_heatmap")

        # Counterfactual: zero out the most salient region in the symbolic
        # scene representation and re-decode to see whether the answer flips.
        predicted_changed = False
        perturbed_region = region_names_top[0] if region_names_top else None
        if outputs.scene_region_logits is not None and self.region_names:
            top_r = int(outputs.scene_region_logits[0].argmax().item())
            perturbed_region = self.region_names[top_r]
            sup = outputs.scene_region_logits[0].min() - 1.0
            cf_region = outputs.scene_region_logits.clone()
            cf_region[0, top_r] = sup
            cf_outputs = dataclasses.replace(outputs, scene_region_logits=cf_region)
            if outputs.scene_object_presence is not None:
                cf_presence = outputs.scene_object_presence.clone()
                cf_presence[0, top_r] = 0.0
                cf_outputs = dataclasses.replace(cf_outputs, scene_object_presence=cf_presence)
            cf_combined, _ = self._compute_symbolic(cf_outputs, [question])
            cf_pred_idx = int(torch.softmax(cf_combined, dim=-1).argmax(dim=-1).item())
            predicted_changed = self.idx_to_answer[cf_pred_idx] != answer

        cf_explanation = generate_counterfactual_explanation(
            baseline_prediction_str=answer,
            region_scores=region_scores or [],
            region_names=region_names_top or [],
            predicted_changed=predicted_changed,
            perturbed_region=perturbed_region,
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
                "symbolic_used": bool(exec_trace.get("symbolic_used", [False]) and exec_trace["symbolic_used"][0]),
                "region_scores": region_scores,
                "region_names": region_names_top,
                "grounding_boxes": grounding_boxes,
                "heatmap_2d": heatmap_2d,
                "counterfactual_explanation": cf_explanation,
                "reasoning_tree": tree.get("reasoning_trace"),
            },
        }

    def _compute_symbolic(self, outputs: Any, questions: List[str]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Fuse neural and symbolic logits (PoE, log space).

        Delegates to the shared fusion engine (``pathvqa.symbolic.poe``) so
        training, evaluation, and serving execute byte-identical fusion logic.
        """
        return self.fusion(outputs, questions)

    def model_info(self) -> Dict[str, Any]:
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
