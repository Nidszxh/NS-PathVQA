"""Exponential Moving Average (EMA) of model weights.

Keeps a shadow copy updated as ``shadow = decay * shadow + (1-decay) * weights``
after each optimizer step. The EMA snapshot is persisted in checkpoints.
"""

from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains an exponential moving average of a model's state_dict."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: nn.Module) -> None:
        """Snapshot current weights as initial EMA state. Tracks all params + buffers
        so the EMA state_dict matches model.state_dict() key-for-key."""
        for name, param in model.named_parameters():
            if param.is_floating_point():
                self.shadow[name] = param.detach().clone()
        for name, buffer in model.named_buffers():
            if buffer.is_floating_point():
                self.shadow[name] = buffer.detach().clone()

    def update(self, model: nn.Module) -> None:
        """In-place EMA update: shadow = decay * shadow + (1-decay) * weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    # EMA: θ_ema ← β·θ_ema + (1-β)·θ
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)
            for name, buffer in model.named_buffers():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(buffer.detach(), alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Load EMA weights into model (for eval). Training weights are backed up."""
        self._backup = {name: p.detach().clone() for name, p in model.named_parameters()}
        for name, buf in model.named_buffers():
            self._backup[name] = buf.detach().clone()
        self._load_into(model, self.shadow)

    def restore(self, model: nn.Module) -> None:
        """Restore the model's training weights saved by ``apply_shadow``."""
        self._load_into(model, self._backup)
        self._backup = {}

    @staticmethod
    def _load_into(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state:
                    param.copy_(state[name])
            for name, buf in model.named_buffers():
                if name in state:
                    buf.copy_(state[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


if __name__ == "__main__":
    model = nn.Linear(4, 2)
    ema = ModelEMA(model, decay=0.9)
    old = {k: v.clone() for k, v in model.named_parameters()}
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    expected = 0.9 * old["weight"] + 0.1 * model.weight.detach()
    torch.testing.assert_close(ema.shadow["weight"], expected, atol=1e-6, rtol=1e-6)
    trained = {k: v.clone() for k, v in model.named_parameters()}
    ema.apply_shadow(model)
    torch.testing.assert_close(model.weight, ema.shadow["weight"])
    ema.restore(model)
    torch.testing.assert_close(model.weight, trained["weight"])
    print("EMA test passed!")