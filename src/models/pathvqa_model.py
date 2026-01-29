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
import sys

_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.append(_src)

from models.visual.visual_encoder import CLIPViTEncoder, MultiScaleVisualEncoder
from models.question.question_encoder import DistilBERTQuestionEncoder
from models.fusion import CrossModalTransformer
from symbolic.scene_parser import SceneParser
from symbolic.routing import LearnedGate
from symbolic.dsl import DifferentiableDSLInterpreter, DSLProgramCompiler


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
                 attribute_mappings: dict = None):
        super().__init__()
        self.symbolic_enabled = symbolic_enabled
        self.weighting_strategy = weighting_strategy
        visual_config = visual_config or {}
        question_config = question_config or {}

        self.visual_encoder = CLIPViTEncoder(**visual_config)
        self.question_encoder = DistilBERTQuestionEncoder(**question_config)
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
                use_multiscale: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass: encode → cross-attention → neural + symbolic logits.

        Neural path: classifier([attended ‖ question_state]) → answer_logits
        Symbolic path: SceneParser → Executor → symbolic_logits (if enabled)
        """
        visual = self.visual_encoder(images, use_multiscale=use_multiscale)
        q = self.question_encoder(input_ids, attention_mask)
        q_state = q["question_state"]
        attended, attn_weights = self.attention(q_state, visual["features"], visual["mask"])
        fused = torch.cat([attended, q_state], dim=-1)        # (B, visual_dim + question_hidden_dim)
        answer_logits = self.classifier(fused)                 # (B, answer_vocab_size)

        result = {"answer_logits": answer_logits, "attended_features": attended,
                  "patch_features": visual["features"]}

        if self.symbolic_enabled:
            scene = self.scene_parser(attended)
            result["scene_region_logits"] = scene["region_logits"]
            result["scene_object_presence"] = scene["object_presence"]
            result["scene_color_logits"] = scene["color_logits"]
            result["scene_shape_logits"] = scene["shape_logits"]
            result["scene_size_logits"] = scene["size_logits"]
            result["scene_density_logits"] = scene["density_logits"]

            if self.weighting_strategy == "learned" and qtype_onehot is not None:
                # Scene score: max object-presence sigmoid across regions → (B, 1)
                c_scene = torch.sigmoid(scene["object_presence"]).max(dim=1, keepdim=True).values
                gate = self.gate(attended, c_scene, qtype_onehot, attn_weights)
                result["gate_values"] = gate

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
    print(f"Answer logits shape: {outputs['answer_logits'].shape}")

    print("Testing with symbolic module (static fusion)...")
    model2 = NeuroSymbolicPathVQA(answer_vocab_size=50, symbolic_enabled=True, num_regions=10)
    print(f"Parameters with symbolic: {sum(p.numel() for p in model2.parameters()):,}")
    outputs2 = model2(images, input_ids, attn_mask)
    print(f"Region logits shape: {outputs2['scene_region_logits'].shape}")
    print(f"Color logits shape: {outputs2['scene_color_logits'].shape}")
    print(f"Shape logits shape: {outputs2['scene_shape_logits'].shape}")
    print(f"Size logits shape: {outputs2['scene_size_logits'].shape}")

    print("Testing with learned gate (entropy wired)...")
    model3 = NeuroSymbolicPathVQA(
        answer_vocab_size=50, symbolic_enabled=True, num_regions=10,
        weighting_strategy="learned",
    )
    qtype_oh = torch.zeros(2, 5)
    qtype_oh[:, 0] = 1.0  # identity
    outputs3 = model3(images, input_ids, attn_mask, qtype_onehot=qtype_oh)
    print(f"Gate values shape: {outputs3['gate_values'].shape}")
    print(f"Gate range: [{outputs3['gate_values'].min():.4f}, {outputs3['gate_values'].max():.4f}]")
    assert outputs3["attended_features"].shape == (2, 512)
    assert outputs3["gate_values"].shape == (2, 1)
    print("PathVQA model test passed!")
