"""Differentiable question-type classifier backing the learned query parser.

The regex parser (``pathvqa.symbolic.query_parser``) hard-dispatches on
question text; this module learns the same five-way split end-to-end so the
symbolic path receives gradients.
"""

import torch
import torch.nn as nn


class DifferentiableQueryClassifier(nn.Module):
    """Learned, differentiable question-type classifier.

    Replaces regex-based ``parse_question()`` for determining the question type.
    """

    def __init__(self, hidden_dim: int = 768, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, question_state: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        """Classify question type from the encoder state.

        Emits raw logits by default so callers can feed them straight to
        ``nn.CrossEntropyLoss`` during the query-type warm-up, and softmaxes
        only when the soft routing weights are requested.

        Args:
            question_state: (B, hidden_dim) DistilBERT [CLS] state.
            return_probs: If True, return softmax probabilities; else raw logits.

        Returns:
            (B, num_classes) soft qtype weights when ``return_probs`` is True,
            otherwise the unnormalized logits.
        """
        logits: torch.Tensor = self.classifier(question_state)
        if return_probs:
            return torch.softmax(logits, dim=-1)
        return logits
