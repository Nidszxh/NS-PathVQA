"""Differentiable DSL: program AST, compiler, and interpreter for compositional reasoning."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


@dataclass
class DSLNode:
    """Base AST node for neuro-symbolic program representation."""
    op: str
    args: Dict[str, Any]
    child: Optional["DSLNode"] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"op": self.op, "args": self.args}
        if self.child is not None:
            result["child"] = self.child.to_dict()
        return result


class DSLProgramCompiler:
    """Compiles natural language questions into structured DSL program trees."""

    @staticmethod
    def compile(question: str, target: str = "", attribute: str = "", qtype: str = "identity") -> DSLNode:
        q = question.lower().strip()

        if qtype == "count" or "how many" in q:
            target_entity = target or "nuclei"
            return DSLNode(
                op="Count",
                args={"target": target_entity},
                child=DSLNode(op="Filter", args={"concept": target_entity}),
            )
        elif qtype == "attribute":
            attr = attribute or "color"
            return DSLNode(
                op="QueryAttr",
                args={"attribute": attr, "target": target},
                child=DSLNode(op="Filter", args={"concept": target or "lesion"}),
            )
        elif qtype == "yes_no":
            return DSLNode(
                op="Verify",
                args={"condition": target or "abnormality"},
                child=DSLNode(op="Filter", args={"concept": target or "lesion"}),
            )
        elif qtype == "location":
            return DSLNode(
                op="Relate",
                args={"relation": "located_in", "target": target},
                child=DSLNode(op="Filter", args={"concept": target or "tissue"}),
            )
        else:  # identity
            return DSLNode(
                op="Exist",
                args={"target": target},
                child=DSLNode(op="Filter", args={"concept": target or "organ"}),
            )


class DifferentiableDSLInterpreter(nn.Module):
    """Executes compiled DSL program trees differentiably over visual patch tokens."""

    def __init__(self, visual_dim: int = 512, hidden_dim: int = 128,
                 attribute_mappings: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.visual_dim = visual_dim
        self.attribute_mappings = attribute_mappings or {}
        self.concept_filter = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Soft counting projection: predicts continuous count estimation from attended patches
        self.count_head = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        program: DSLNode,
        patch_features: torch.Tensor,
        scene_logits: Dict[str, torch.Tensor],
        answer_to_idx: Dict[str, int],
    ) -> torch.Tensor:
        """Execute a DSL program over patch tokens. Returns (B, vocab_size) logits."""
        batch_size = patch_features.size(0)
        device = patch_features.device
        vocab_size = len(answer_to_idx)
        logits = torch.zeros(batch_size, vocab_size, device=device)

        # Spatial concept filtering: each patch → scalar in [0,1]
        # Concept filter: patch_features → (B, N_patches, 1)
        patch_weights = self.concept_filter(patch_features)

        if program.op == "Count":
            # Soft counting: Σ(patch_weights) gives continuous instance count
            soft_count = patch_weights.sum(dim=1).squeeze(-1)  # (B,)
            
            # Map soft count to common discrete count answers in medical VQA
            count_map = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0,
                         "one": 1.0, "two": 2.0, "three": 3.0, "few": 3.0, "multiple": 8.0, "many": 10.0}
            for ans_str, val in count_map.items():
                if ans_str in answer_to_idx:
                    aidx = answer_to_idx[ans_str]
                    # Gaussian kernel: exp(-0.5 * (soft_count - target)^2)
                    dist = torch.exp(-0.5 * torch.square(soft_count - val))
                    logits[:, aidx] = dist

        elif program.op == "Verify":
            # Verification: max patch weight → yes/no confidence
            conf = patch_weights.max(dim=1).values.squeeze(-1)  # (B,)
            yes_idx = answer_to_idx.get("yes")
            no_idx = answer_to_idx.get("no")
            if yes_idx is not None and no_idx is not None:
                logits[:, yes_idx] = conf * 2.0
                logits[:, no_idx] = (1.0 - conf) * 2.0

        elif program.op == "QueryAttr":
            attr = program.args.get("attribute", "color")
            attr_logits = scene_logits.get(f"scene_{attr}_logits")
            if attr_logits is not None and attr in self.attribute_mappings:
                mapping = self.attribute_mappings[attr].to(device)
                valid_mask = mapping >= 0
                if valid_mask.any():
                    target_indices = mapping[valid_mask]
                    logits[:, target_indices] = attr_logits[:, valid_mask]

        return logits


if __name__ == "__main__":
    print("Testing DSL Program Compiler & Interpreter...")
    compiler = DSLProgramCompiler()
    prog_count = compiler.compile("How many nuclei are present?", target="nuclei", qtype="count")
    print(f"Compiled Count Program: {prog_count.to_dict()}")

    prog_verify = compiler.compile("Is there a malignant tumor?", target="tumor", qtype="yes_no")
    print(f"Compiled Verify Program: {prog_verify.to_dict()}")

    prog_attr = compiler.compile("What color is the lesion?", target="lesion", attribute="color", qtype="attribute")
    print(f"Compiled QueryAttr Program: {prog_attr.to_dict()}")

    # Build attribute mappings: color value index → answer vocab index
    from executor import build_attribute_mappings, COLOR_VALUES
    dummy_vocab = {"yes": 0, "no": 1, "red": 2, "blue": 3, "yellow": 4, "large": 5, "small": 6}
    attr_maps = build_attribute_mappings(dummy_vocab)

    interpreter = DifferentiableDSLInterpreter(visual_dim=512, attribute_mappings=attr_maps)
    dummy_patches = torch.randn(2, 49, 512)

    # Test Count
    out_count = interpreter(prog_count, dummy_patches, {}, dummy_vocab)
    print(f"Output count shape: {out_count.shape}")

    # Test QueryAttr
    scene_logits = {
        "scene_color_logits": torch.randn(2, len(COLOR_VALUES)),
    }
    out_attr = interpreter(prog_attr, dummy_patches, scene_logits, dummy_vocab)
    print(f"Output QueryAttr shape: {out_attr.shape}")
    # "red" (idx 2) and "blue" (idx 3) and "yellow" (idx 4) should have non-zero logits
    assert (out_attr[:, 2] != 0).any() or (out_attr[:, 3] != 0).any() or (out_attr[:, 4] != 0).any()
    print("DSL Module test passed!")
