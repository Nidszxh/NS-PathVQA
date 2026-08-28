"""Differentiable DSL: program AST, compiler, and interpreter for compositional reasoning."""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


# Coarse directional priors over a normalized patch grid, expressed as offsets
# in [x, y] (right, down). Used to give ``Relate`` a genuine (if simple) spatial
# signal instead of discarding its relation argument.
RELATION_DIRECTION_OFFSETS: Dict[str, Tuple[float, float]] = {
    "located_in": (0.0, 0.0),
    "inside": (0.0, 0.0),
    "within": (0.0, 0.0),
    "near": (0.0, 0.0),
    "left_of": (-1.0, 0.0),
    "right_of": (1.0, 0.0),
    "above": (0.0, -1.0),
    "below": (0.0, 1.0),
}


def _relation_patch_weights(relation: str, num_patches: int, device: torch.device) -> torch.Tensor:
    """Directional weight vector over row-major patches for a spatial relation.

    Patches lying toward the relation's direction receive larger weight. The
    neutral relations (``located_in`` and friends) yield a uniform distribution,
    reducing to attention-weighted presence.
    """
    if num_patches <= 0:
        return torch.ones(0, device=device)
    if relation not in RELATION_DIRECTION_OFFSETS:
        return torch.full((num_patches,), 1.0 / num_patches, device=device)

    side = max(1, math.ceil(math.sqrt(num_patches)))
    indices = torch.arange(num_patches, device=device)
    rows = (indices // side).to(torch.float32)
    cols = (indices % side).to(torch.float32)
    xn = cols / max(side - 1, 1)
    yn = rows / max(side - 1, 1)
    dx, dy = RELATION_DIRECTION_OFFSETS[relation]
    alignment = 1.0 + dx * (xn - 0.5) * 2.0 + dy * (yn - 0.5) * 2.0
    alignment = alignment.clamp(min=1e-3)
    normalized: torch.Tensor = alignment / alignment.sum()
    return normalized


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
        """Build the AST for one question, dispatching on the parsed qtype."""
        q = question.lower().strip()

        if qtype == "count" or re.search(r"^how many ", q):
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
        elif qtype in ("location", "relate"):
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

    def __init__(
        self, visual_dim: int = 512, hidden_dim: int = 128, attribute_mappings: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Build the concept filter, soft count head, and attribute maps."""
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

        # Concept filter: (B, N_patches, visual_dim) → per-patch match score in [0, 1]
        patch_weights = self.concept_filter(patch_features)

        if program.op == "Count":
            # Soft counting: the count head predicts a per-patch count
            # contribution, gated by how strongly each patch matches the
            # concept filter. Σ gives a continuous instance count estimate.
            count_scores = torch.sigmoid(self.count_head(patch_features))  # (B, N, 1)
            per_patch = count_scores * patch_weights
            soft_count = per_patch.sum(dim=1).squeeze(-1)  # (B,)

            # Map soft count to common discrete count answers in medical VQA
            count_map = {
                "1": 1.0,
                "2": 2.0,
                "3": 3.0,
                "4": 4.0,
                "5": 5.0,
                "one": 1.0,
                "two": 2.0,
                "three": 3.0,
                "few": 3.0,
                "multiple": 8.0,
                "many": 10.0,
            }
            for ans_str, val in count_map.items():
                if ans_str in answer_to_idx:
                    aidx = answer_to_idx[ans_str]
                    # Gaussian kernel: exp(-0.5 * (soft_count - target)^2)
                    dist = torch.exp(-0.5 * torch.square(soft_count - val))
                    logits[:, aidx] = dist

        elif program.op == "Exist":
            # Identity: strongest patch matching the target concept scattered to
            # the target answer slot (falls back to max over all patches).
            target = program.args.get("target", "")
            conf = patch_weights.max(dim=1).values.squeeze(-1)  # (B,)
            t_aidx: Optional[int] = answer_to_idx.get(target) if target else None
            if t_aidx is None:
                # Unseen target has no slot to write to; spread a uniform,
                # deliberately low signal across the vocab instead.
                logits[:, :] = conf.unsqueeze(1) / max(vocab_size, 1)
            else:
                logits[:, t_aidx] = conf

        elif program.op == "Relate":
            # Location: attention-weighted presence of the target concept,
            # modulated by a relation-conditioned spatial prior over patches.
            target = program.args.get("target", "")
            relation = program.args.get("relation", "located_in")
            relation_weights = _relation_patch_weights(relation, patch_weights.size(1), patch_weights.device)
            conf = (patch_weights.squeeze(-1) * relation_weights.unsqueeze(0)).sum(dim=1)  # (B,)
            r_aidx: Optional[int] = answer_to_idx.get(target) if target else None
            if r_aidx is not None:
                logits[:, r_aidx] = conf
            else:
                logits[:, :] = conf.unsqueeze(1) / max(vocab_size, 1)

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
    from pathvqa.symbolic.scene_parser import COLOR_VALUES
    from pathvqa.symbolic.executor import build_attribute_mappings

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
