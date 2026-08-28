"""Cross-modal fusion via single-block cross-attention transformer."""

import torch
import torch.nn as nn


class CrossModalTransformer(nn.Module):
    """Single-block cross-attention: question queries attend over patch tokens.

    Output is (batch, hidden_dim), consumed by the neural classifier and SceneParser.
    """

    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        """Build the Q/K/V projections, output projection, and feed-forward block."""
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(text_dim, hidden_dim)
        self.k_proj = nn.Linear(visual_dim, hidden_dim)
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, question_state: torch.Tensor, visual_features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend over visual features given a question state. Returns (attended, weights)."""
        batch = question_state.size(0)
        num_objects = visual_features.size(1)

        # Project + reshape to (batch, num_heads, seq, head_dim)
        q = self.q_proj(question_state).view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(visual_features).view(batch, num_objects, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual_features).view(batch, num_objects, self.num_heads, self.head_dim).transpose(1, 2)

        # QK^T / sqrt(d_k), then mask padded patches with dtype-min so the
        # softmax below yields exactly zero there under fp16.
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2)  # (B, 1, H, head_dim)
        attended = attended.reshape(batch, self.head_dim * self.num_heads)
        attended = self.out_proj(attended)
        attended = self.norm(self.dropout(attended))
        out = attended + self.ffn(attended)

        # Downstream heads scatter-add over exactly these dimensions.
        assert out.shape == (batch, self.head_dim * self.num_heads)
        assert weights.shape == (batch, self.num_heads, 1, num_objects)

        return out, weights


if __name__ == "__main__":
    print("Testing CrossModalTransformer...")
    model = CrossModalTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    q_state = torch.randn(2, 768)
    feats = torch.randn(2, 49, 512)
    mask = torch.ones(2, 49, dtype=torch.bool)
    out, w = model(q_state, feats, mask)
    print(f"Attended shape: {out.shape}, weights shape: {w.shape}")
    print("Fusion test passed!")
