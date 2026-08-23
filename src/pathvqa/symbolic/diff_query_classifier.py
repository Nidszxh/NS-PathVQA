import torch
import torch.nn as nn

class DifferentiableQueryClassifier(nn.Module):
    """
    Learned, differentiable question type classifier.
    Replaces regex-based parse_question() for determining the question type.
    """
    def __init__(self, hidden_dim: int = 768, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        # Using log_softmax or raw logits is better for cross_entropy loss during warm-up.
        # We output raw logits and optionally softmax them for the soft routing,
        # but nn.CrossEntropyLoss expects raw logits.

    def forward(self, question_state: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        """
        Input: question_state (B, hidden_dim) from DistilBERT [CLS]
        Output: soft qtype weights (B, num_classes) if return_probs is True, else logits
        """
        logits = self.classifier(question_state)
        if return_probs:
            return torch.softmax(logits, dim=-1)
        return logits
