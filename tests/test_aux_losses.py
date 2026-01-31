"""Tests for symbolic auxiliary losses."""

import torch
from symbolic.aux_losses import compute_symbolic_aux_losses
from symbolic.scene_parser import COLOR_VALUES


def _make_dummy_data():
    region_names = ["bone", "lung", "liver"]
    answer_vocab = ["yes", "no", "bone", "lung", "liver", "red", "blue", "large"]
    answer_to_idx = {a: i for i, a in enumerate(answer_vocab)}
    region_to_answer_idx = torch.tensor([
        answer_to_idx["bone"],
        answer_to_idx["lung"],
        answer_to_idx["liver"],
    ])
    attribute_mappings = {
        "color": torch.tensor([
            answer_to_idx.get("red", -1),
            answer_to_idx.get("blue", -1),
        ] + [-1] * (len(COLOR_VALUES) - 2)),
    }
    return region_names, answer_vocab, answer_to_idx, region_to_answer_idx, attribute_mappings


def test_region_aux_loss():
    region_names, answer_vocab, answer_to_idx, region_to_answer_idx, attr_maps = _make_dummy_data()
    B = 4
    scene_logits = {
        "scene_region_logits": torch.randn(B, len(region_names), requires_grad=True),
        "scene_object_presence": torch.sigmoid(torch.randn(B, len(region_names))),
        "scene_color_logits": torch.randn(B, len(COLOR_VALUES)),
    }

    class FakeQuery:
        def __init__(self, qtype, target=None, attribute=None):
            self.qtype = qtype
            self.target = target
            self.attribute = attribute

    queries = [
        FakeQuery("identity", target="bone"),
        FakeQuery("location", target="lung"),
        FakeQuery("yes_no", target="lung"),
        FakeQuery("attribute", attribute="color"),
    ]
    targets = torch.tensor([
        answer_to_idx["bone"],
        answer_to_idx["lung"],
        answer_to_idx["yes"],
        answer_to_idx["red"],
    ])

    losses = compute_symbolic_aux_losses(
        scene_logits, queries, targets, answer_to_idx,
        region_names, region_to_answer_idx, attr_maps,
    )

    assert "region" in losses
    assert "attr" in losses
    assert "yn" in losses
    assert losses["region"].ndim == 0
    assert losses["yn"].ndim == 0

    # Gradient flows
    total = losses["region"] + losses["attr"] + losses["yn"]
    total.backward()
    assert scene_logits["scene_region_logits"].grad is not None


def test_aux_loss_all_zero_when_no_scene():
    scene_logits = {}
    queries = []
    targets = torch.tensor([0, 1])
    losses = compute_symbolic_aux_losses(
        scene_logits, queries, targets, {}, [], torch.tensor([]), {},
    )
    assert losses["region"].item() == 0.0
    assert losses["attr"].item() == 0.0
    assert losses["yn"].item() == 0.0
