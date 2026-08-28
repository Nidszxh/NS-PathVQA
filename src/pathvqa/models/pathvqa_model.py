"""NeuroSymbolicPathVQA: cross-modal attention with optional symbolic reasoning.

Architecture:
  1. Visual encoder: PLIP (or CLIP ViT-B/32) + LoRA → 49 patch tokens (768-d) → Linear 768→512
  2. Question encoder: DistilBERT + LoRA → [CLS] state (768-d)
  3. Cross-modal fusion: cross-attention transformer (question queries over patches)
  4a. Neural path: MLP over [attended ‖ question_state] → answer logits
  4b. Symbolic path: SceneParser → scene heads + soft routing signals

This module only *emits* the scene heads, soft qtype/target distributions, and
(optional) gate values. Executing the symbolic program and fusing it with the
neural logits happens downstream in ``pathvqa.symbolic.poe.SymbolicFusionEngine``.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from pathvqa.models.visual.visual_encoder import CLIPViTEncoder, MultiScaleVisualEncoder
from pathvqa.models.question.question_encoder import DistilBERTQuestionEncoder
from pathvqa.models.protocols import VisualEncoderProtocol, QuestionEncoderProtocol, ModelOutput
from pathvqa.models.fusion import CrossModalTransformer
from pathvqa.symbolic.scene_parser import SceneParser
from pathvqa.symbolic.routing import LearnedGate
from pathvqa.symbolic.dsl import DifferentiableDSLInterpreter, DSLProgramCompiler
from pathvqa.symbolic.diff_query_classifier import DifferentiableQueryClassifier
from pathvqa.symbolic.diff_target_extractor import DifferentiableTargetExtractor


class NeuroSymbolicPathVQA(nn.Module):
    """Full neuro-symbolic VQA model: neural path + optional symbolic path."""

    def __init__(
        self,
        answer_vocab_size: int,
        num_object_features: int = 512,
        question_hidden_dim: int = 768,
        dropout: float = 0.3,
        symbolic_enabled: bool = True,
        num_regions: int = 50,
        visual_config: Optional[Dict[str, Any]] = None,
        question_config: Optional[Dict[str, Any]] = None,
        weighting_strategy: str = "static",
        attribute_mappings: Optional[Dict[str, torch.Tensor]] = None,
        query_parser: str = "learned",
        use_multiscale: bool = False,
        use_dsl: bool = True,
    ):
        super().__init__()
        self.symbolic_enabled = symbolic_enabled
        self.weighting_strategy = weighting_strategy
        self.query_parser = query_parser
        # Whether the model carries the differentiable DSL interpreter. False
        # for legacy checkpoints that predate it (detected with
        # ``_compat.is_pre_dsl_checkpoint``): those must be rebuilt as the
        # architecture that saved them, or strict state-dict loading fails on
        # the missing dsl_interpreter.* keys.
        self.use_dsl = bool(use_dsl)
        # Whether the encoder supports/uses the multi-scale branch. ``forward``
        # defaults to this value so configured MultiScaleVisualEncoder models
        # actually exercise the crop branch instead of falling back to
        # global-only features.
        self.use_multiscale = bool(use_multiscale)
        visual_config = visual_config or {}
        question_config = question_config or {}

        # Build the right encoder once (MultiScale wraps CLIPViTEncoder, so
        # constructing the plain encoder first would load the HF backbone twice).
        encoder: VisualEncoderProtocol
        if use_multiscale:
            encoder = MultiScaleVisualEncoder(**visual_config)
        else:
            encoder = CLIPViTEncoder(**visual_config)
        self.visual_encoder = encoder
        self.question_encoder: QuestionEncoderProtocol = DistilBERTQuestionEncoder(**question_config)
        self.attention = CrossModalTransformer(
            text_dim=question_hidden_dim,
            visual_dim=num_object_features,
            hidden_dim=num_object_features,
        )
        fusion_dim = num_object_features + question_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, answer_vocab_size),
        )

        if symbolic_enabled:
            self.scene_parser = SceneParser(
                visual_dim=num_object_features,
                num_regions=num_regions,
            )
            if self.use_dsl:
                self.dsl_compiler = DSLProgramCompiler()
                self.dsl_interpreter = DifferentiableDSLInterpreter(
                    visual_dim=num_object_features,
                    attribute_mappings=attribute_mappings,
                )
            if self.query_parser == "learned":
                self.diff_qtype_classifier = DifferentiableQueryClassifier(
                    hidden_dim=question_hidden_dim, num_classes=5
                )
                self.diff_target_extractor = DifferentiableTargetExtractor(
                    hidden_dim=question_hidden_dim, num_regions=num_regions
                )
            if weighting_strategy == "learned":
                self.gate = LearnedGate(
                    visual_dim=num_object_features,
                    num_regions=num_regions,
                    dropout=dropout,
                )

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to trade compute for VRAM."""
        self.visual_encoder.enable_gradient_checkpointing()
        self.question_encoder.enable_gradient_checkpointing()

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        qtype_onehot: Optional[torch.Tensor] = None,
        use_multiscale: Optional[bool] = None,
    ) -> ModelOutput:
        """Forward pass: encode → cross-attention → neural + symbolic signals.

        Emits ``answer_logits`` plus the symbolic heads/gate; fusion happens
        downstream in ``pathvqa.symbolic.poe``. ``use_multiscale`` defaults to the
        value the model was built with, so a multi-scale model exercises its crop
        branch without every caller passing the flag.
        """
        if use_multiscale is None:
            use_multiscale = self.use_multiscale
        visual = self.visual_encoder(images, use_multiscale=use_multiscale)
        q = self.question_encoder(input_ids, attention_mask)
        q_state = q.question_state
        attended, attn_weights = self.attention(q_state, visual.features, visual.mask)
        fused = torch.cat([attended, q_state], dim=-1)  # (B, visual_dim + question_hidden_dim)
        answer_logits = self.classifier(fused)  # (B, answer_vocab_size)

        result = ModelOutput(
            answer_logits=answer_logits,
            patch_features=visual.features,
            attention_weights=attn_weights,
            attended_features=attended,
        )

        if self.symbolic_enabled:
            scene = self.scene_parser(attended)
            result.scene_region_logits = scene.region_logits
            result.scene_object_presence = scene.object_presence
            result.scene_color_logits = scene.color_logits
            result.scene_shape_logits = scene.shape_logits
            result.scene_size_logits = scene.size_logits

            if self.query_parser == "learned":
                # Raw logits for the warm-up CE loss; softmax probabilities for
                # soft routing through the differentiable executor.
                qtype_logits = self.diff_qtype_classifier(q_state, return_probs=False)
                soft_qtype = torch.softmax(qtype_logits, dim=-1)
                result.soft_qtype = soft_qtype
                result.qtype_logits = qtype_logits

                target_logits = self.diff_target_extractor(q_state, return_probs=False)
                soft_target = torch.softmax(target_logits, dim=-1)
                result.soft_target = soft_target
                result.target_logits = target_logits

            if self.weighting_strategy == "learned":
                # Soft routing (learned parser) feeds the gate directly; the
                # regex path falls back to the caller's one-hot qtype.
                gate_qtype = soft_qtype if self.query_parser == "learned" else qtype_onehot
                if gate_qtype is not None:
                    # c_scene: max object-presence over regions → (B, 1).
                    # object_presence is already sigmoid-activated by the scene
                    # parser; a second sigmoid would compress it to [0.5, 0.73].
                    c_scene = scene.object_presence.max(dim=1, keepdim=True).values
                    gate = self.gate(attended, c_scene, gate_qtype, attn_weights)
                    result.gate_values = gate

        return result


def build_model(
    config: Any, answer_vocab_size: int, attribute_mappings: Optional[Dict[str, torch.Tensor]] = None
) -> "NeuroSymbolicPathVQA":
    """Construct an untrained model from a ``Config``, wiring every encoder/symbolic option."""
    visual_kwargs = dict(
        model_name=config.visual.model_name,
        num_object_features=config.visual.num_object_features,
        num_objects=config.visual.num_objects,
        lora_rank=config.visual.lora_rank,
        lora_alpha=config.visual.lora_alpha,
        lora_target_modules=config.visual.lora_target_modules,
    )
    question_kwargs = dict(
        model_name=config.question.model_name,
        lora_rank=config.question.lora_rank,
        lora_alpha=config.question.lora_alpha,
        lora_target_modules=config.question.lora_target_modules,
    )
    model = NeuroSymbolicPathVQA(
        answer_vocab_size=answer_vocab_size,
        num_object_features=config.visual.num_object_features,
        question_hidden_dim=config.question.hidden_dim,
        dropout=config.question.dropout,
        symbolic_enabled=config.symbolic.enabled,
        num_regions=config.symbolic.num_regions,
        visual_config=visual_kwargs,
        question_config=question_kwargs,
        weighting_strategy=config.symbolic.weighting_strategy,
        attribute_mappings=attribute_mappings,
        query_parser=getattr(
            config.symbolic,
            "query_parser",
            "regex" if not getattr(config.symbolic, "differentiable_pipeline", False) else "learned",
        ),
        use_multiscale=getattr(config.visual, "use_multiscale", False),
        use_dsl=getattr(config.symbolic, "dsl_enabled", True),
    )
    return model


if __name__ == "__main__":
    print("Testing NeuroSymbolicPathVQA (neural only)...")
    model = NeuroSymbolicPathVQA(answer_vocab_size=50, symbolic_enabled=False)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (2, 10))
    attn_mask = torch.ones(2, 10, dtype=torch.long)
    outputs = model(images, input_ids, attn_mask)
    print(f"Answer logits shape: {outputs.answer_logits.shape}")

    print("Testing with symbolic module (static fusion)...")
    model2 = NeuroSymbolicPathVQA(answer_vocab_size=50, symbolic_enabled=True, num_regions=10)
    print(f"Parameters with symbolic: {sum(p.numel() for p in model2.parameters()):,}")
    outputs2 = model2(images, input_ids, attn_mask)
    print(f"Region logits shape: {outputs2.scene_region_logits.shape}")
    print(f"Color logits shape: {outputs2.scene_color_logits.shape}")
    print(f"Shape logits shape: {outputs2.scene_shape_logits.shape}")
    print(f"Size logits shape: {outputs2.scene_size_logits.shape}")

    print("Testing with learned gate (entropy wired)...")
    model3 = NeuroSymbolicPathVQA(
        answer_vocab_size=50,
        symbolic_enabled=True,
        num_regions=10,
        weighting_strategy="learned",
    )
    qtype_oh = torch.zeros(2, 5)
    qtype_oh[:, 0] = 1.0  # identity
    outputs3 = model3(images, input_ids, attn_mask, qtype_onehot=qtype_oh)
    print(f"Gate values shape: {outputs3.gate_values.shape}")
    print(f"Gate range: [{outputs3.gate_values.min():.4f}, {outputs3.gate_values.max():.4f}]")
    assert outputs3.gate_values.shape == (2, 1)
    print("PathVQA model test passed!")
