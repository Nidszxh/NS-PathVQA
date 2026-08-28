"""Learned confidence gate and question-type encoding."""

import torch
from pathvqa.symbolic.routing import LearnedGate, encode_question_types, QTYPE_LIST, NUM_QTYPES


def test_encode_question_types():
    questions = [
        "what is the diagnosis?",
        "where is the lesion?",
        "is this normal?",
        "what color is the tissue?",
        "how many cells are there?",
    ]
    onehot = encode_question_types(questions)
    assert onehot.shape == (5, NUM_QTYPES)
    assert onehot.sum(dim=1).allclose(torch.ones(5))


def test_encode_question_types_priority():
    questions = ["what color is the tissue?", "what is the diagnosis?"]
    onehot = encode_question_types(questions)
    assert onehot[0, QTYPE_LIST.index("attribute")] == 1.0
    assert onehot[1, QTYPE_LIST.index("identity")] == 1.0


def test_learned_gate_output_shape():
    gate = LearnedGate(visual_dim=512, num_regions=10, num_qtypes=NUM_QTYPES)
    h_attn = torch.randn(4, 512)
    c_scene = torch.randn(4, 1)  # max over regions → (B, 1)
    qtype_onehot = torch.zeros(4, NUM_QTYPES)
    qtype_onehot[:, 0] = 1.0
    g = gate(h_attn, c_scene, qtype_onehot)
    assert g.shape == (4, 1)
    assert (g >= 0).all() and (g <= 1).all()


def test_learned_gate_different_qtypes():
    """Different question types must drive different gate outputs.

    The qtype one-hot is concatenated into the gate MLP input, so two samples
    that differ only in qtype must produce different gate values (dropout is
    disabled to isolate the effect).
    """
    from pathvqa.symbolic.routing import QTYPE_TO_IDX

    torch.manual_seed(0)
    gate = LearnedGate(visual_dim=512, num_regions=10, num_qtypes=NUM_QTYPES)
    gate.eval()

    h_attn = torch.randn(2, 512)
    c_scene = torch.randn(2, 1)
    q_identity = torch.zeros(2, NUM_QTYPES)
    q_identity[:, QTYPE_TO_IDX["identity"]] = 1.0
    q_yes_no = torch.zeros(2, NUM_QTYPES)
    q_yes_no[:, QTYPE_TO_IDX["yes_no"]] = 1.0

    g1 = gate(h_attn, c_scene, q_identity)
    g2 = gate(h_attn, c_scene, q_yes_no)
    assert g1.shape == g2.shape == (2, 1)
    assert not torch.allclose(g1, g2), "gate output must depend on question type"

    # Identical inputs must reproduce identical outputs (no hidden randomness).
    g1_again = gate(h_attn, c_scene, q_identity)
    assert torch.allclose(g1, g1_again)


def test_learned_gate_mean_per_qtype():
    gate = LearnedGate(visual_dim=512, num_regions=10, num_qtypes=NUM_QTYPES)
    h_attn = torch.randn(4, 512)
    c_scene = torch.randn(4, 1)
    qtype_onehot = torch.zeros(4, NUM_QTYPES)
    qtype_onehot[0:2, 0] = 1.0  # identity
    qtype_onehot[2:4, 2] = 1.0  # yes_no
    means = gate.get_mean_gate_per_qtype(h_attn, c_scene, qtype_onehot)
    assert "identity" in means
    assert "yes_no" in means
    assert "attribute" in means
    assert isinstance(means["identity"], float)
    assert means["identity"] >= 0 and means["identity"] <= 1


def test_gate_gradient_flows():
    gate = LearnedGate(visual_dim=512, num_regions=10, num_qtypes=NUM_QTYPES)
    h_attn = torch.randn(2, 512, requires_grad=True)
    c_scene = torch.randn(2, 1)
    qtype_onehot = torch.zeros(2, NUM_QTYPES)
    qtype_onehot[:, 0] = 1.0
    g = gate(h_attn, c_scene, qtype_onehot)
    loss = g.sum()
    loss.backward()
    assert h_attn.grad is not None


def test_attn_entropy_handles_ranked_shapes():
    """attn_entropy accepts (B, H, 1, N), (B, H, N) and (B, N) attention shapes."""
    from pathvqa.symbolic.routing import attn_entropy

    torch.manual_seed(0)
    b, h, n = 2, 3, 7

    p4 = torch.softmax(torch.randn(b, h, 1, n), dim=-1)
    e4 = attn_entropy(p4)
    assert e4.shape == (b, 1)
    assert (e4 >= 0).all()

    p3 = torch.softmax(torch.randn(b, h, n), dim=-1)
    e3 = attn_entropy(p3)
    assert e3.shape == (b, 1)

    # Uniform distributions have the max entropy log(n); identical across heads.
    p_uniform = torch.full((b, h, 1, n), 1.0 / n)
    e_uniform = attn_entropy(p_uniform)
    assert torch.allclose(e_uniform, torch.log(torch.tensor(float(n))).expand(b, 1), atol=1e-5)
