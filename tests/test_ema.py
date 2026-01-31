import torch
from models.ema import ModelEMA


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