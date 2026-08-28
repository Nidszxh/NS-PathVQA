"""Runtime-checkable Protocol definitions for model components."""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable, Optional, Tuple


@dataclass
class VisualEncoderOutput:
    """Visual encoder result: patch features and their validity mask."""

    features: torch.Tensor
    mask: torch.Tensor


@dataclass
class QuestionEncoderOutput:
    """Question encoder result: the pooled [CLS] state."""

    question_state: torch.Tensor


@dataclass
class SceneParserOutput:
    """The five scene heads emitted by ``SceneParser``.

    ``object_presence`` is already sigmoid-activated; the rest are raw logits.
    """

    region_logits: torch.Tensor
    object_presence: torch.Tensor
    color_logits: torch.Tensor
    shape_logits: torch.Tensor
    size_logits: torch.Tensor


@dataclass
class ExecutorOutput:
    """Result of symbolic execution: answer-vocab logits plus a trace."""

    symbolic_logits: torch.Tensor
    region_logits: torch.Tensor
    trace: Dict[str, Any]


@dataclass
class ModelOutput:
    """Everything ``NeuroSymbolicPathVQA.forward`` produces.

    ``answer_logits`` is always present; every other field is optional and is
    populated only when the corresponding path (symbolic, learned query parser,
    learned gate) is enabled.
    """

    answer_logits: torch.Tensor
    scene_region_logits: Optional[torch.Tensor] = None
    scene_object_presence: Optional[torch.Tensor] = None
    scene_color_logits: Optional[torch.Tensor] = None
    scene_shape_logits: Optional[torch.Tensor] = None
    scene_size_logits: Optional[torch.Tensor] = None
    gate_values: Optional[torch.Tensor] = None
    soft_qtype: Optional[torch.Tensor] = None
    soft_target: Optional[torch.Tensor] = None
    patch_features: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    attended_features: Optional[torch.Tensor] = None
    qtype_logits: Optional[torch.Tensor] = None
    target_logits: Optional[torch.Tensor] = None

    def scene_head_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return the scene-parser heads as the dict consumed by symbolic modules.

        ``DifferentiableExecutor``, LTN evaluation and the aux-loss functions
        take a ``Dict[str, Tensor]`` of scene heads; this adapts a ModelOutput
        to that contract using the same key names.
        """
        return {
            "scene_region_logits": self.scene_region_logits,
            "scene_object_presence": self.scene_object_presence,
            "scene_color_logits": self.scene_color_logits,
            "scene_shape_logits": self.scene_shape_logits,
            "scene_size_logits": self.scene_size_logits,
        }


@runtime_checkable
class VisualEncoderProtocol(Protocol):
    """Protocol for visual feature extractors."""

    num_object_features: int
    num_objects: int

    def __call__(self, images: torch.Tensor, use_multiscale: bool = False) -> VisualEncoderOutput:
        """Return VisualEncoderOutput with 'features' (B, N, D) and 'mask' (B, N)."""
        ...

    def enable_gradient_checkpointing(self) -> None:
        """Trade compute for VRAM via activation checkpointing."""
        ...


@runtime_checkable
class QuestionEncoderProtocol(Protocol):
    """Protocol for text question encoders."""

    hidden_dim: int

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> QuestionEncoderOutput:
        """Return QuestionEncoderOutput with 'question_state' (B, hidden_dim)."""
        ...

    def enable_gradient_checkpointing(self) -> None:
        """Trade compute for VRAM via activation checkpointing."""
        ...


@runtime_checkable
class FusionProtocol(Protocol):
    """Protocol for cross-modal attention/fusion modules."""

    def __call__(
        self,
        question_state: torch.Tensor,
        visual_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Return (attended summary (B, hidden_dim), attention weights)."""
        ...


@runtime_checkable
class SymbolicExecutorProtocol(Protocol):
    """Protocol for symbolic reasoning execution."""

    def __call__(
        self,
        scene_logits: SceneParserOutput,
        queries: List[Any],
        neural_logits: torch.Tensor,
    ) -> ExecutorOutput:
        """Return ExecutorOutput with 'symbolic_logits', 'region_logits', 'trace'."""
        ...
