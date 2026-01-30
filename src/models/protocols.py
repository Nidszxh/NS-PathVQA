"""Runtime-checkable Protocol definitions for model components."""

from typing import Dict, List, Protocol, runtime_checkable
import torch


@runtime_checkable
class VisualEncoderProtocol(Protocol):
    """Protocol for visual feature extractors."""

    num_object_features: int
    num_objects: int

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return dict with 'features' (B, N, D) and 'mask' (B, N)."""
        ...


@runtime_checkable
class QuestionEncoderProtocol(Protocol):
    """Protocol for text question encoders."""

    hidden_dim: int

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return dict with 'question_state' (B, hidden_dim)."""
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
        scene_logits: Dict[str, torch.Tensor],
        queries: List,
        neural_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return dict with 'symbolic_logits', 'region_logits', 'trace'."""
        ...
