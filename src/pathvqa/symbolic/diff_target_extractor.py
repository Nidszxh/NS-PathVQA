"""Differentiable target-region extractor backing the learned query parser.

The regex parser names the anatomical target from the question text; this
module instead attends over the region axis so the executor can weight every
region softly.
"""

import torch
import torch.nn as nn


class DifferentiableTargetExtractor(nn.Module):
    """Learned, differentiable target-region extractor.

    Replaces regex-based target extraction. Predicts which of the ``num_regions``
    is the primary target of the question (e.g. 'lung', 'glomerulus').
    """

    def __init__(self, hidden_dim: int = 768, num_regions: int = 50, dropout: float = 0.1):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_regions)
        )

    def forward(self, question_state: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        """Score every region as the question's target.

        Emits raw logits by default so callers can feed them straight to
        ``nn.CrossEntropyLoss`` during the query-type warm-up, and softmaxes
        only when the soft attention weights are requested.

        Args:
            question_state: (B, hidden_dim) DistilBERT [CLS] state.
            return_probs: If True, return softmax probabilities; else raw logits.

        Returns:
            (B, num_regions) soft target weights when ``return_probs`` is True,
            otherwise the unnormalized logits.
        """
        logits: torch.Tensor = self.extractor(question_state)
        if return_probs:
            return torch.softmax(logits, dim=-1)
        return logits
