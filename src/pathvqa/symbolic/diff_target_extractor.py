import torch
import torch.nn as nn

class DifferentiableTargetExtractor(nn.Module):
    """
    Learned, differentiable target region extractor.
    Replaces regex-based target extraction. Predicts which of the `num_regions` 
    is the primary target of the question (e.g. 'lung', 'glomerulus').
    """
    def __init__(self, hidden_dim: int = 768, num_regions: int = 50, dropout: float = 0.1):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_regions)
        )

    def forward(self, question_state: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        """
        Input: question_state (B, hidden_dim) from DistilBERT [CLS]
        Output: soft target weights (B, num_regions) if return_probs is True, else logits
        """
        logits = self.extractor(question_state)
        if return_probs:
            return torch.softmax(logits, dim=-1)
        return logits
