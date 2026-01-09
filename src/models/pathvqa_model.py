"""Main model definition: NeuroSymbolicPathVQA with cross-modal attention and optional symbolic path.

Architecture:
  1. Visual encoder: ResNet → object proposals → spatial encoding
  2. Question encoder: biLSTM → question state vector
  3. Cross-modal attention: attended visual features conditioned on question
  4a. Neural path: fuse attended features + question state → MLP classifier
  4b. Symbolic path (optional): SceneParser → scene logits → Executor → symbolic logits

The symbolic path is additive: final_logits = neural_logits + weight * symbolic_logits
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path
import sys

_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.append(_src)

from models.visual.visual_encoder import SimpleObjectDetector
from models.text.question_encoder import QuestionEncoder
from symbolic.scene_parser import SceneParser


class CrossModalAttention(nn.Module):
    """Attention over visual features conditioned on the question state.

    Uses concat+MLP scoring: scores = MLP([visual_features, question_state])
    followed by softmax weighted sum over visual features.
    """

    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.attend = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visual_features: torch.Tensor, question_state: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Attend over visual features given a question.

        Args:
            visual_features: (batch, num_objects, visual_dim)
            question_state: (batch, text_dim)
            mask: (batch, num_objects) boolean mask for valid objects

        Returns:
            attended: (batch, visual_dim) question-conditioned visual summary
        """
        q = question_state.unsqueeze(1).expand(-1, visual_features.size(1), -1)
        combined = torch.cat([visual_features, q], dim=-1)
        scores = self.attend(combined).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        attended = (weights.unsqueeze(-1) * visual_features).sum(dim=1)
        return attended


class NeuroSymbolicPathVQA(nn.Module):
    """Full neuro-symbolic VQA model.

    Combines a neural path (visual → question → attention → MLP) with an
    optional symbolic path (scene parser → executor) for interpretable reasoning.
    """

    def __init__(self, vocab_size: int, answer_vocab_size: int,
                 num_object_features: int = 512, max_objects: int = 10,
                 spatial_feat_dim: int = 128, pretrained: bool = True,
                 question_embedding_dim: int = 256,
                 question_hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.3,
                 symbolic_enabled: bool = True,
                 num_regions: int = 50):
        super().__init__()
        self.symbolic_enabled = symbolic_enabled

        self.visual_encoder = SimpleObjectDetector(
            num_object_features=num_object_features, max_objects=max_objects,
            spatial_feat_dim=spatial_feat_dim, pretrained=pretrained,
        )
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size, embedding_dim=question_embedding_dim,
            hidden_dim=question_hidden_dim, num_layers=num_layers,
            dropout=dropout,
        )
        self.attention = CrossModalAttention(
            visual_dim=num_object_features,
            text_dim=question_hidden_dim,
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

    def forward(self, images: torch.Tensor, question_indices: torch.Tensor,
                question_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass: visual + question encoding → attention → neural + symbolic logits.

        Args:
            images: (batch, 3, H, W) normalized image tensors
            question_indices: (batch, seq_len) padded token indices
            question_lengths: (batch,) true lengths for pack_padded_sequence

        Returns:
            Dict with answer_logits and, if symbolic enabled, scene_* logits
        """
        visual = self.visual_encoder(images)
        q = self.question_encoder(question_indices, question_lengths)
        q_state = q["question_state"]
        attended = self.attention(visual["features"], q_state, visual["mask"])
        fused = torch.cat([attended, q_state], dim=-1)
        answer_logits = self.classifier(fused)

        result = {"answer_logits": answer_logits, "attended_features": attended}

        if self.symbolic_enabled:
            scene = self.scene_parser(attended)
            result["scene_region_logits"] = scene["region_logits"]
            result["scene_object_presence"] = scene["object_presence"]
            result["scene_color_logits"] = scene["color_logits"]
            result["scene_shape_logits"] = scene["shape_logits"]
            result["scene_size_logits"] = scene["size_logits"]

        return result


def build_model(config, vocab_size, answer_vocab_size):
    """Convenience factory: create NeuroSymbolicPathVQA from a Config object."""
    return NeuroSymbolicPathVQA(
        vocab_size=vocab_size,
        answer_vocab_size=answer_vocab_size,
        num_object_features=config.visual.num_object_features,
        max_objects=config.visual.max_objects_per_image,
        spatial_feat_dim=config.visual.spatial_feat_dim,
        pretrained=config.visual.pretrained,
        question_embedding_dim=config.question.embedding_dim,
        question_hidden_dim=config.question.hidden_dim,
        num_layers=config.question.num_layers,
        dropout=config.question.dropout,
        symbolic_enabled=config.symbolic.enabled,
        num_regions=config.symbolic.num_regions,
    )


if __name__ == "__main__":
    print("Testing NeuroSymbolicPathVQA (neural only)...")
    model = NeuroSymbolicPathVQA(vocab_size=100, answer_vocab_size=50,
                                 num_object_features=128, max_objects=5,
                                 symbolic_enabled=False)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    images = torch.randn(2, 3, 320, 240)
    q_idx = torch.randint(0, 100, (2, 10))
    q_len = torch.tensor([8, 10])
    outputs = model(images, q_idx, q_len)
    print(f"Answer logits shape: {outputs['answer_logits'].shape}")

    print("Testing with symbolic module...")
    model2 = NeuroSymbolicPathVQA(vocab_size=100, answer_vocab_size=50,
                                  num_object_features=128, max_objects=5,
                                  symbolic_enabled=True, num_regions=10)
    print(f"Parameters with symbolic: {sum(p.numel() for p in model2.parameters()):,}")
    outputs2 = model2(images, q_idx, q_len)
    print(f"Region logits shape: {outputs2['scene_region_logits'].shape}")
    print(f"Color logits shape: {outputs2['scene_color_logits'].shape}")
    print(f"Shape logits shape: {outputs2['scene_shape_logits'].shape}")
    print(f"Size logits shape: {outputs2['scene_size_logits'].shape}")
    print("PathVQA model test passed!")
