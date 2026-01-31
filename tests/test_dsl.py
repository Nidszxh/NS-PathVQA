"""Unit tests for DSL Program Compiler and Differentiable Interpreter (N4)."""

import torch
from symbolic.dsl import DSLProgramCompiler, DifferentiableDSLInterpreter, DSLNode
from symbolic.query_parser import parse_question


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
    from symbolic.executor import build_attribute_mappings, COLOR_VALUES
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

    # Color vocab indices (red=2, blue=3, yellow=4) should have non-zero logits
    # (they map to COLOR_VALUES positions via attribute_mappings)
    assert (out[:, 2] != 0).any() or (out[:, 3] != 0).any() or (out[:, 4] != 0).any()
    # Non-color vocab entries (yes, no, large) should remain zero
    assert (out[:, 0] == 0).all()  # "yes"
    assert (out[:, 5] == 0).all()  # "large"

    # Gradient flows through scene_color_logits (not through patch_features)
    loss = out.sum()
    loss.backward()
    assert scene_logits["scene_color_logits"].grad is not None
