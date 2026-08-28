"""Learned confidence gate for neuro-symbolic fusion.

Computes g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)])). The gate
values weight the Product-of-Experts fusion in *log-probability* space:

    log p = (1 - g) · log p_neural + g · log p_symbolic

(see ``pathvqa.symbolic.poe``). When weighting_strategy="static", a fixed
weight is used instead of the gate.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

QTYPE_LIST = ["identity", "location", "yes_no", "attribute", "count"]
QTYPE_TO_IDX = {q: i for i, q in enumerate(QTYPE_LIST)}
NUM_QTYPES = len(QTYPE_LIST)


def encode_question_types(questions: List[Any]) -> torch.Tensor:
    """Encode questions into one-hot qtype tensors (B, NUM_QTYPES).

    Accepts raw question strings or pre-parsed ``Query`` objects (the PoE
    fusion engine passes already-parsed queries; re-parsing a ``Query`` would
    crash on ``Query.lower``).
    """
    from pathvqa.symbolic.query_parser import Query, parse_question

    batch_size = len(questions)
    onehot = torch.zeros(batch_size, NUM_QTYPES)
    for i, q in enumerate(questions):
        parsed = q if isinstance(q, Query) else parse_question(q)
        idx = QTYPE_TO_IDX.get(parsed.qtype, 0)
        onehot[i, idx] = 1.0
    return onehot


def attn_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention distribution: ent(attn) = -Σ p log p.

    Accepts (B, H, 1, N), (B, H, N), or (B, N); heads are averaged down to
    (B, N) before normalizing. Returns (B, 1).
    """
    if weights.dim() == 4:
        # (B, H, 1, N) → mean over heads → squeeze the singleton query dim
        p = weights.mean(dim=1).squeeze(1)
    elif weights.dim() == 3:
        # (B, H, N) → mean over heads
        p = weights.mean(dim=1)
    else:
        p = weights
    # Normalize along token dimension to ensure valid distribution
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum(dim=-1, keepdim=True)
    return entropy


class LearnedGate(nn.Module):
    """Learned confidence gate for neuro-symbolic fusion.

    Produces g ∈ [0,1] per sample from attended features, scene scores, qtype, and attention entropy:
      g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)]))
    """

    def __init__(
        self, visual_dim: int = 512, num_regions: int = 50, num_qtypes: int = NUM_QTYPES, dropout: float = 0.1
    ):
        """Build the gate MLP over the concatenated [h_attn ‖ c_scene ‖ qtype ‖ entropy] vector.

        ``num_regions`` is accepted for signature compatibility with the other
        symbolic components but is not an input to the gate, which scores a
        single global ``c_scene`` value rather than per-region evidence.
        """
        super().__init__()
        # c_scene is always (B, 1) — max over regions, so it adds 1 to the width
        gate_input_dim = visual_dim + 1 + num_qtypes + 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        h_attn: torch.Tensor,
        c_scene: torch.Tensor,
        qtype_onehot: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gate values. Returns (batch, 1) in [0, 1]."""
        batch_size = h_attn.size(0)
        device = h_attn.device
        dtype = h_attn.dtype
        if attn_weights is not None:
            ent = attn_entropy(attn_weights).to(device=device, dtype=dtype)
        else:
            ent = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        gate_input = torch.cat([h_attn, c_scene, qtype_onehot, ent], dim=-1)
        return torch.sigmoid(self.gate_mlp(gate_input))

    def get_mean_gate_per_qtype(
        self,
        h_attn: torch.Tensor,
        c_scene: torch.Tensor,
        qtype_onehot: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Mean gate value per question type (for logging)."""
        g = self.forward(h_attn, c_scene, qtype_onehot, attn_weights)  # (batch, 1)
        result = {}
        for i, qtype in enumerate(QTYPE_LIST):
            mask = qtype_onehot[:, i] > 0.5
            if mask.any():
                result[qtype] = g[mask].mean().item()
            else:
                result[qtype] = 0.0
        return result
