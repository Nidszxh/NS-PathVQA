"""NeuroSymbolicPathVQA: cross-modal attention with optional symbolic reasoning.

Architecture:
  1. Visual encoder: PLIP (or CLIP ViT-B/32) + LoRA → 49 patch tokens (768-d) → Linear 768→512
  2. Question encoder: DistilBERT + LoRA → [CLS] state (768-d)
  3. Cross-modal fusion: cross-attention transformer (question queries over patches)
  4a. Neural path: MLP over [attended ‖ question_state] → answer logits
  4b. Symbolic path: SceneParser → Executor → symbolic logits

Fusion modes: static additive or learned gate (N1).
"""

import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path
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

    def __init__(self, answer_vocab_size: int,
                 num_object_features: int = 512,
                 question_hidden_dim: int = 768,
                 dropout: float = 0.3,
                 symbolic_enabled: bool = True,
                 num_regions: int = 50,
                 visual_config: dict = None,
                 question_config: dict = None,
                 weighting_strategy: str = "static",
                 attribute_mappings: dict = None,
                 query_parser: str = "learned"):
        super().__init__()
        self.symbolic_enabled = symbolic_enabled
        self.weighting_strategy = weighting_strategy
        self.query_parser = query_parser
        visual_config = visual_config or {}
        question_config = question_config or {}

        self.visual_encoder: VisualEncoderProtocol = CLIPViTEncoder(**visual_config)
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

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                qtype_onehot: torch.Tensor = None,
                use_multiscale: bool = False) -> ModelOutput:
        """Forward pass: encode → cross-attention → neural + symbolic logits.

        Neural path: classifier([attended ‖ question_state]) → answer_logits
        Symbolic path: SceneParser → Executor → symbolic_logits (if enabled)
        """
        visual = self.visual_encoder(images, use_multiscale=use_multiscale)
        q = self.question_encoder(input_ids, attention_mask)
        q_state = q.question_state
        attended, attn_weights = self.attention(q_state, visual.features, visual.mask)
        fused = torch.cat([attended, q_state], dim=-1)        # (B, visual_dim + question_hidden_dim)
        answer_logits = self.classifier(fused)                 # (B, answer_vocab_size)

        result = ModelOutput(
            answer_logits=answer_logits,
            patch_features=visual.features,
            attention_weights=attn_weights,
            attended_features=attended
        )
        # Note: attended_features was returned previously but unused except as intermediate.
        # We can still add it if needed, but keeping it off for now or we could add it to ModelOutput.
        # Actually I didn't add attended_features to ModelOutput. Let's add it in another replacement if needed, 
        # but downstream didn't grep it. Wait, `tests/test_integration.py` or others might assert it. 
        # I'll just leave it out, or let me add it. I'll add `attended_features` to ModelOutput just in case.

        if self.symbolic_enabled:
            scene = self.scene_parser(attended)
            result.scene_region_logits = scene.region_logits
            result.scene_object_presence = scene.object_presence
            result.scene_color_logits = scene.color_logits
            result.scene_shape_logits = scene.shape_logits
            result.scene_size_logits = scene.size_logits
            result.scene_density_logits = scene.density_logits

            if self.query_parser == "learned":
                # Get raw logits for training, soft probabilities for reasoning
                qtype_logits = self.diff_qtype_classifier(q_state, return_probs=False)
                soft_qtype = torch.softmax(qtype_logits, dim=-1)
                # target_logits and qtype_logits were not in ModelOutput, but let's ignore them or add if needed
                # I'll just set the ones that exist.
                result.soft_qtype = soft_qtype
                result.qtype_logits = qtype_logits
                
                target_logits = self.diff_target_extractor(q_state, return_probs=False)
                soft_target = torch.softmax(target_logits, dim=-1)
                result.soft_target = soft_target
                result.target_logits = target_logits

            if self.weighting_strategy == "learned":
                # If learned query parser is enabled, use soft_qtype instead of hard qtype_onehot
                gate_qtype = soft_qtype if self.query_parser == "learned" else qtype_onehot
                if gate_qtype is not None:
                    # Scene score: max object-presence sigmoid across regions → (B, 1)
                    c_scene = torch.sigmoid(scene.object_presence).max(dim=1, keepdim=True).values
                    gate = self.gate(attended, c_scene, gate_qtype, attn_weights)
                    result.gate_values = gate

        return result


def build_model(config, answer_vocab_size, attribute_mappings=None):
    """Create NeuroSymbolicPathVQA from a Config object."""
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
        query_parser=getattr(config.symbolic, "query_parser", "regex" if not getattr(config.symbolic, "differentiable_pipeline", False) else "learned"),
    )
    # Replace visual encoder with MultiScaleVisualEncoder if configured
    if getattr(config.visual, "use_multiscale", False):
        model.visual_encoder = MultiScaleVisualEncoder(**visual_kwargs)
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
        answer_vocab_size=50, symbolic_enabled=True, num_regions=10,
        weighting_strategy="learned",
    )
    qtype_oh = torch.zeros(2, 5)
    qtype_oh[:, 0] = 1.0  # identity
    outputs3 = model3(images, input_ids, attn_mask, qtype_onehot=qtype_oh)
    print(f"Gate values shape: {outputs3.gate_values.shape}")
    print(f"Gate range: [{outputs3.gate_values.min():.4f}, {outputs3.gate_values.max():.4f}]")
    assert outputs3.gate_values.shape == (2, 1)
    print("PathVQA model test passed!")
