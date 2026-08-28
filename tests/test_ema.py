"""Exponential moving average of model weights."""

import pytest
import torch
from pathvqa.models.ema import ModelEMA


def test_ema_update_math():
    model = torch.nn.Linear(4, 2)
    ema = ModelEMA(model, decay=0.9)
    old_w = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    expected = 0.9 * old_w + 0.1 * model.weight.detach()
    torch.testing.assert_close(ema.shadow["weight"], expected, atol=1e-6, rtol=1e-6)


def test_ema_apply_and_restore():
    model = torch.nn.Linear(4, 2)
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.add_(1.0)
    trained_w = model.weight.detach().clone()
    ema.apply_shadow(model)
    assert not torch.allclose(model.weight, trained_w)
    ema.restore(model)
    torch.testing.assert_close(model.weight, trained_w)


def test_ema_state_dict_roundtrip():
    model = torch.nn.Linear(4, 2)
    ema = ModelEMA(model, decay=0.9)
    ema.update(model)
    state = ema.state_dict()
    ema2 = ModelEMA(model, decay=0.9)
    ema2.load_state_dict(state)
    for k in state:
        torch.testing.assert_close(ema2.shadow[k], state[k])


def test_ema_shadow_covers_full_state_dict():
    model = torch.nn.Linear(4, 2)
    ema = ModelEMA(model, decay=0.9)
    assert set(ema.shadow.keys()) == set(dict(model.state_dict()).keys())


def test_ema_rejects_invalid_decay():
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError):
        ModelEMA(model, decay=1.0)


def test_ema_tracks_buffers():
    """Floating-point buffers (e.g. BatchNorm running stats) are EMA-tracked too."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.BatchNorm1d(2))
    ema = ModelEMA(model, decay=0.5)
    assert "1.running_mean" in ema.shadow
    old_stats = ema.shadow["1.running_mean"].clone()
    with torch.no_grad():
        # Simulate a training step updating the buffer
        torch.nn.init.normal_(model[1].running_mean)
    ema.update(model)
    expected = 0.5 * old_stats + 0.5 * model[1].running_mean.detach()
    torch.testing.assert_close(ema.shadow["1.running_mean"], expected)

    trained_stat = model[1].running_mean.detach().clone()
    ema.apply_shadow(model)
    torch.testing.assert_close(model[1].running_mean, ema.shadow["1.running_mean"])
    ema.restore(model)
    torch.testing.assert_close(model[1].running_mean, trained_stat)
