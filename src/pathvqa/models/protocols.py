"""Runtime-checkable Protocol definitions for model components."""

import torch
from dataclasses import dataclass
from typing import List, Dict, Protocol, runtime_checkable, Optional, Any

@dataclass
class VisualEncoderOutput:
    features: torch.Tensor
    mask: torch.Tensor

@dataclass
class QuestionEncoderOutput:
    question_state: torch.Tensor

@dataclass
class SceneParserOutput:
    region_logits: torch.Tensor
    object_presence: torch.Tensor
    color_logits: torch.Tensor
    shape_logits: torch.Tensor
    size_logits: torch.Tensor
    density_logits: torch.Tensor

@dataclass
class ExecutorOutput:
    symbolic_logits: torch.Tensor
    region_logits: torch.Tensor
    trace: Dict[str, Any]

@dataclass
class ModelOutput:
    answer_logits: torch.Tensor
    scene_region_logits: Optional[torch.Tensor] = None
    scene_object_presence: Optional[torch.Tensor] = None
    scene_color_logits: Optional[torch.Tensor] = None
    scene_shape_logits: Optional[torch.Tensor] = None
    scene_size_logits: Optional[torch.Tensor] = None
    scene_density_logits: Optional[torch.Tensor] = None
    gate_values: Optional[torch.Tensor] = None
    soft_qtype: Optional[torch.Tensor] = None
    soft_target: Optional[torch.Tensor] = None
    patch_features: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    attended_features: Optional[torch.Tensor] = None
    qtype_logits: Optional[torch.Tensor] = None
    target_logits: Optional[torch.Tensor] = None

@runtime_checkable
class VisualEncoderProtocol(Protocol):
    """Protocol for visual feature extractors."""

    num_object_features: int
    num_objects: int

    def forward(self, images: torch.Tensor, use_multiscale: bool = False) -> VisualEncoderOutput:
        """Return VisualEncoderOutput with 'features' (B, N, D) and 'mask' (B, N)."""
        ...

@runtime_checkable
class QuestionEncoderProtocol(Protocol):
    """Protocol for text question encoders."""

    hidden_dim: int

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> QuestionEncoderOutput:
        """Return QuestionEncoderOutput with 'question_state' (B, hidden_dim)."""
        ...


@runtime_checkable
class FusionProtocol(Protocol):
    """Protocol for cross-modal attention/fusion modules."""

    def forward(
        self,
        question_state: torch.Tensor,
        visual_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return attended visual summary (B, hidden_dim)."""
        ...


@runtime_checkable
class SymbolicExecutorProtocol(Protocol):
    """Protocol for symbolic reasoning execution."""

    def forward(
        self,
        scene_logits: SceneParserOutput,
        queries: List,
        neural_logits: torch.Tensor,
    ) -> ExecutorOutput:
        """Return ExecutorOutput with 'symbolic_logits', 'region_logits', 'trace'."""
        ...
