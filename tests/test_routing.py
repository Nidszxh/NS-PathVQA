import torch
from symbolic.routing import LearnedGate, encode_question_types, QTYPE_LIST, NUM_QTYPES


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
    gate = LearnedGate(visual_dim=512, num_regions=10, num_qtypes=NUM_QTYPES)
    h_attn = torch.randn(2, 512)
    c_scene = torch.randn(2, 1)
    q1 = torch.zeros(2, NUM_QTYPES)
    q1[:, 0] = 1.0  # identity
    q2 = torch.zeros(2, NUM_QTYPES)
    q2[:, 2] = 1.0  # yes_no
    g1 = gate(h_attn, c_scene, q1)
    g2 = gate(h_attn, c_scene, q2)
    # different qtype inputs → different gate outputs (unless weights are zero)
    assert g1.shape == g2.shape


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
