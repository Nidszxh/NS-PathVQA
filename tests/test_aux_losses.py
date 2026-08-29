"""Tests for symbolic auxiliary losses."""

import torch
from pathvqa.symbolic.aux_losses import compute_symbolic_aux_losses, compute_target_region_loss
from pathvqa.symbolic.scene_parser import COLOR_VALUES


def _make_dummy_data():
    region_names = ["bone", "lung", "liver"]
    answer_vocab = ["yes", "no", "bone", "lung", "liver", "red", "blue", "large"]
    answer_to_idx = {a: i for i, a in enumerate(answer_vocab)}
    region_to_answer_idx = torch.tensor(
        [
            answer_to_idx["bone"],
            answer_to_idx["lung"],
            answer_to_idx["liver"],
        ]
    )
    attribute_mappings = {
        "color": torch.tensor(
            [
                answer_to_idx.get("red", -1),
                answer_to_idx.get("blue", -1),
            ]
            + [-1] * (len(COLOR_VALUES) - 2)
        ),
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
    targets = torch.tensor(
        [
            answer_to_idx["bone"],
            answer_to_idx["lung"],
            answer_to_idx["yes"],
            answer_to_idx["red"],
        ]
    )

    losses = compute_symbolic_aux_losses(
        scene_logits,
        queries,
        targets,
        answer_to_idx,
        region_to_answer_idx,
        attr_maps,
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
        scene_logits,
        queries,
        targets,
        {},
        torch.tensor([]),
        {},
    )
    assert losses["region"].item() == 0.0
    assert losses["attr"].item() == 0.0
    assert losses["yn"].item() == 0.0


def test_yn_aux_loss_only_supervises_the_queried_region():
    """A yes/no label must not be spread over every region's presence head."""
    region_names, answer_vocab, answer_to_idx, region_to_answer_idx, attr_maps = _make_dummy_data()

    class FakeQuery:
        qtype = "yes_no"
        target = "lung"
        attribute = None

    presence = torch.tensor([[0.2, 0.5, 0.8]])
    scene = {"scene_region_logits": torch.randn(1, len(region_names)), "scene_object_presence": presence}
    baseline = compute_symbolic_aux_losses(
        scene, [FakeQuery()], torch.tensor([answer_to_idx["yes"]]), answer_to_idx, region_to_answer_idx, attr_maps
    )["yn"]

    moved_other = presence.clone()
    moved_other[0, 0] = 0.99  # "bone" is not the query target
    scene_other = {"scene_region_logits": scene["scene_region_logits"], "scene_object_presence": moved_other}
    shifted = compute_symbolic_aux_losses(
        scene_other,
        [FakeQuery()],
        torch.tensor([answer_to_idx["yes"]]),
        answer_to_idx,
        region_to_answer_idx,
        attr_maps,
    )["yn"]
    assert torch.allclose(baseline, shifted)

    moved_target = presence.clone()
    moved_target[0, 1] = 0.01  # "lung" is the query target
    scene_target = {"scene_region_logits": scene["scene_region_logits"], "scene_object_presence": moved_target}
    changed = compute_symbolic_aux_losses(
        scene_target,
        [FakeQuery()],
        torch.tensor([answer_to_idx["yes"]]),
        answer_to_idx,
        region_to_answer_idx,
        attr_maps,
    )["yn"]
    assert changed.item() > baseline.item()


def test_yn_aux_loss_skips_non_region_targets():
    """No presence head exists for findings like "tumor"; contribute no loss."""
    region_names, answer_vocab, answer_to_idx, region_to_answer_idx, attr_maps = _make_dummy_data()
    answer_to_idx["tumor"] = len(answer_to_idx)

    class FakeQuery:
        qtype = "yes_no"
        target = "tumor"
        attribute = None

    scene = {
        "scene_region_logits": torch.randn(1, len(region_names)),
        "scene_object_presence": torch.rand(1, len(region_names)),
    }
    losses = compute_symbolic_aux_losses(
        scene,
        [FakeQuery()],
        torch.tensor([answer_to_idx["yes"]]),
        answer_to_idx,
        region_to_answer_idx,
        attr_maps,
    )
    assert losses["yn"].item() == 0.0


def test_yn_aux_loss_negative_answer():
    """A gold 'no' answer drives the yes/no BCE toward zero presence."""
    region_names, answer_vocab, answer_to_idx, region_to_answer_idx, attr_maps = _make_dummy_data()
    B = 1
    scene_logits = {
        "scene_region_logits": torch.randn(B, len(region_names)),
        "scene_object_presence": torch.sigmoid(torch.randn(B, len(region_names))),
        "scene_color_logits": torch.randn(B, len(COLOR_VALUES)),
    }

    class FakeQuery:
        qtype = "yes_no"
        target = "lung"
        attribute = None

    losses = compute_symbolic_aux_losses(
        scene_logits,
        [FakeQuery()],
        torch.tensor([answer_to_idx["no"]]),
        answer_to_idx,
        region_to_answer_idx,
        attr_maps,
    )
    assert losses["yn"].item() >= 0.0


def test_yn_aux_loss_is_finite_for_saturated_fp16_presence():
    """Autocast sigmoid outputs can round to exact 0/1 in fp16."""
    region_names, answer_vocab, answer_to_idx, region_to_answer_idx, attr_maps = _make_dummy_data()

    class FakeQuery:
        qtype = "yes_no"
        target = "lung"
        attribute = None

    scene = {
        "scene_region_logits": torch.zeros(1, len(region_names), dtype=torch.float16),
        "scene_object_presence": torch.tensor([[0.5, 1.0, 0.0]], dtype=torch.float16),
    }
    losses = compute_symbolic_aux_losses(
        scene,
        [FakeQuery()],
        torch.tensor([answer_to_idx["no"]]),
        answer_to_idx,
        region_to_answer_idx,
        attr_maps,
    )

    assert torch.isfinite(losses["yn"])
    assert losses["yn"].item() > 0.0


def test_target_region_loss_supervises_known_question_targets():
    class FakeQuery:
        def __init__(self, target):
            self.target = target

    logits = torch.tensor([[0.0, 2.0], [9.0, -9.0], [1.0, 1.0]], requires_grad=True)
    queries = [FakeQuery("LUNG"), FakeQuery("unknown concept"), FakeQuery("")]
    loss = compute_target_region_loss(logits, queries, ["lung", "liver"])
    expected = torch.nn.functional.cross_entropy(logits[:1], torch.tensor([0]))

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[1:]) == 0


def test_target_region_loss_without_mappable_targets_is_graph_connected_zero():
    class FakeQuery:
        target = "not a region"

    logits = torch.randn(1, 2, requires_grad=True)
    loss = compute_target_region_loss(logits, [FakeQuery()], ["lung", "liver"])

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_target_region_loss_uses_explicit_no_region_class():
    class FakeQuery:
        def __init__(self, target):
            self.target = target

    logits = torch.zeros(2, 3, requires_grad=True)
    loss = compute_target_region_loss(logits, [FakeQuery("lung"), FakeQuery("")], ["lung", "liver"])
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))

    torch.testing.assert_close(loss, expected)
