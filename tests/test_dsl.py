"""Unit tests for the DSL program compiler and differentiable interpreter."""

import torch
from pathvqa.symbolic.dsl import DSLProgramCompiler, DifferentiableDSLInterpreter, DSLNode
from pathvqa.symbolic.query_parser import parse_question


def test_dsl_program_compiler():
    compiler = DSLProgramCompiler()

    # Count
    p_count = compiler.compile("How many nuclei are seen?", target="nuclei", qtype="count")
    assert p_count.op == "Count"
    assert p_count.args["target"] == "nuclei"
    assert p_count.child is not None
    assert p_count.child.op == "Filter"

    # Verify / Yes-No
    p_verify = compiler.compile("Is this a benign tumor?", target="tumor", qtype="yes_no")
    assert p_verify.op == "Verify"

    # QueryAttr
    p_attr = compiler.compile("What color is the cytoplasm?", target="cytoplasm", attribute="color", qtype="attribute")
    assert p_attr.op == "QueryAttr"
    assert p_attr.args["attribute"] == "color"

    # Relate
    p_relate = compiler.compile("Is there a nucleus inside the cell?", target="nucleus", qtype="relate")
    assert p_relate.op == "Relate"
    assert p_relate.args["relation"] == "inside"
    assert compiler.compile("Which structure is left of the lesion?", qtype="relate").args["relation"] == "left_of"
    assert compiler.compile("Where is the lesion located?", qtype="location").args["relation"] == "located_in"

    # Exist
    p_exist = compiler.compile("Is a lesion present?", target="lesion", qtype="exist")
    assert p_exist.op == "Exist"


def test_differentiable_dsl_interpreter():
    interpreter = DifferentiableDSLInterpreter(visual_dim=512)
    dummy_patches = torch.randn(4, 49, 512, requires_grad=True)
    vocab = {"yes": 0, "no": 1, "1": 2, "2": 3, "few": 4, "many": 5}

    prog_count = DSLNode(op="Count", args={"target": "nuclei"}, child=DSLNode(op="Filter", args={"concept": "nuclei"}))
    out = interpreter(prog_count, dummy_patches, {}, vocab)
    assert out.shape == (4, len(vocab))

    # Gradient flow check
    loss = out.sum()
    loss.backward()
    assert dummy_patches.grad is not None


def test_query_parser_dsl_integration():
    q = parse_question("How many glomeruli are in this kidney biopsy?")
    assert q.qtype == "count"
    assert q.program is not None
    assert q.program.op == "Count"


def test_query_attr_interpreter():
    from pathvqa.symbolic.executor import build_attribute_mappings, COLOR_VALUES

    vocab = {"yes": 0, "no": 1, "red": 2, "blue": 3, "yellow": 4, "large": 5}
    attr_maps = build_attribute_mappings(vocab)

    interpreter = DifferentiableDSLInterpreter(visual_dim=512, attribute_mappings=attr_maps)
    dummy_patches = torch.randn(4, 49, 512, requires_grad=True)
    scene_logits = {"scene_color_logits": torch.randn(4, len(COLOR_VALUES), requires_grad=True)}

    prog = DSLNode(
        op="QueryAttr",
        args={"attribute": "color", "target": "lesion"},
        child=DSLNode(op="Filter", args={"concept": "lesion"}),
    )
    out = interpreter(prog, dummy_patches, scene_logits, vocab)
    assert out.shape == (4, len(vocab))

    # Color vocab indices (red=2, blue=3, yellow=4) carry log-space evidence
    # (they map to COLOR_VALUES positions via attribute_mappings)
    assert (out[:, 2] != 0).any() or (out[:, 3] != 0).any() or (out[:, 4] != 0).any()
    # Non-color vocab entries are masked rather than left at zero: zero would
    # outrank negative log-probabilities after the fusion log_softmax.
    assert (out[:, 0] <= -30.0).all()  # "yes"
    assert (out[:, 5] <= -30.0).all()  # "large"

    # Gradient flows through scene_color_logits (not through patch_features)
    loss = out[:, 2:5].sum()
    loss.backward()
    assert scene_logits["scene_color_logits"].grad is not None


def test_count_evidence_survives_log_softmax():
    """Count logits must be log-probabilities, not raw similarities in [0, 1].

    Regression: writing raw similarities left every other answer at zero, so
    ``log_softmax`` ranked the zero classes above the predicted count and the
    fused prediction never followed the DSL.
    """
    interpreter = DifferentiableDSLInterpreter(visual_dim=16, hidden_dim=8)
    interpreter.eval()
    patches = torch.randn(2, 5, 16)
    vocab = {"yes": 0, "no": 1, "1": 2, "2": 3, "3": 4, "10": 5}
    prog = DSLNode(op="Count", args={"target": "nuclei"}, child=DSLNode(op="Filter", args={"concept": "nuclei"}))

    with torch.no_grad():
        logits = interpreter(prog, patches, {}, vocab)
        probs = torch.softmax(logits, dim=-1)

    count_indices = [vocab[k] for k in ("1", "2", "3", "10")]
    # All probability mass must sit on the count answers.
    assert torch.allclose(probs[:, count_indices].sum(dim=1), torch.ones(2), atol=1e-5)
    assert (probs[:, [vocab["yes"], vocab["no"]]] < 1e-6).all()
    assert torch.isfinite(logits).all()
    # The DSL must still be interpretable as a ranking over count answers.
    assert (logits[:, count_indices].max(dim=1).values > -1e8).all()


def test_count_answer_values_covers_vocabulary_forms():
    """Count candidates must follow the vocabulary, not a fixed 1..5 list."""
    from pathvqa.symbolic.dsl import count_answer_values

    vocab = {
        "yes": 0,
        "no": 1,
        "1": 2,
        "9": 3,
        "one": 4,
        "four": 5,
        "six": 6,
        "few": 7,
        "numerous": 8,
        "lung": 9,
    }
    values = count_answer_values(vocab)
    assert values["1"] == 1.0
    assert values["9"] == 9.0
    assert values["four"] == 4.0
    assert values["six"] == 6.0
    assert values["few"] == 3.0
    assert values["numerous"] == 10.0
    assert "yes" not in values and "lung" not in values


def test_unmatched_program_returns_all_zero_logits():
    """A program whose target has no answer slot must yield zeros.

    The fusion engine treats an all-zero row as "no DSL evidence" and keeps the
    neural distribution, so this contract is load-bearing.
    """
    interpreter = DifferentiableDSLInterpreter(visual_dim=16, hidden_dim=8)
    interpreter.eval()
    patches = torch.randn(1, 5, 16)
    vocab = {"yes": 0, "no": 1, "1": 2}
    prog = DSLNode(op="Exist", args={"target": "not-in-vocab"}, child=DSLNode(op="Filter", args={"concept": "x"}))
    with torch.no_grad():
        logits = interpreter(prog, patches, {}, vocab)
    assert torch.count_nonzero(logits).item() == 0
