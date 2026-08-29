"""Symbolic auxiliary losses for training the scene parser heads.

Provides direct supervision signals to each SceneParser head:
  - Region CE loss for identity/location questions
  - Attribute CE loss for attribute questions (color/shape/size)
  - BCE loss for yes/no object-presence supervision
"""

from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from pathvqa.symbolic.executor import QTYPE_TO_INT


def compute_target_region_loss(
    target_logits: torch.Tensor, queries: List[Any], region_names: List[str]
) -> torch.Tensor:
    """Supervise the learned target extractor from parsed question targets.

    The regex parser supplies weak text-derived labels, not visual ground truth.
    When the head has an explicit no-target class, unmatched questions train
    that class; legacy heads without it train only on matched targets.
    This direct loss is needed for learned-parser + hard-executor runs, where
    argmax target routing has no gradient path from answer loss.
    """
    if target_logits.ndim != 2 or target_logits.size(0) != len(queries):
        raise ValueError("target_logits must have shape (len(queries), num_regions)")
    has_no_target_class = target_logits.size(1) == len(region_names) + 1
    if target_logits.size(1) != len(region_names) and not has_no_target_class:
        raise ValueError("target_logits width must match region_names")
    region_to_idx = {name.lower().strip(): index for index, name in enumerate(region_names)}
    rows: List[int] = []
    targets: List[int] = []
    for row, query in enumerate(queries):
        target = (query.target or "").lower().strip()
        if target in region_to_idx:
            rows.append(row)
            targets.append(region_to_idx[target])
        elif has_no_target_class:
            rows.append(row)
            targets.append(len(region_names))
    if not rows:
        return target_logits.sum() * 0.0
    row_tensor = torch.tensor(rows, dtype=torch.long, device=target_logits.device)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=target_logits.device)
    return F.cross_entropy(target_logits.index_select(0, row_tensor), target_tensor)


def compute_symbolic_aux_losses(
    scene_logits: Dict[str, torch.Tensor],
    queries: List[Any],
    targets: torch.Tensor,
    answer_to_idx: Dict[str, int],
    region_to_answer_idx: torch.Tensor,
    attribute_mappings: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute auxiliary losses for scene parser heads.

    Args:
        scene_logits: dict from model forward with scene_region_logits, etc.
        queries: list of Query objects
        targets: answer target indices (B,)
        answer_to_idx: answer string → vocab index
        region_to_answer_idx: (N_reg,) tensor mapping region idx → answer vocab idx
        attribute_mappings: dict of attr_name → mapping tensor

    Returns:
        dict with 'region', 'attr', 'yn' loss tensors (0-dim).
    """
    region_logits = scene_logits.get("scene_region_logits")
    object_presence = scene_logits.get("scene_object_presence")
    device = region_logits.device if region_logits is not None else torch.device("cpu")

    zero = torch.tensor(0.0, device=device)
    result = {"region": zero.clone(), "attr": zero.clone(), "yn": zero.clone()}

    if region_logits is None:
        return result

    # Build reverse lookup: answer vocab idx → answer string
    idx_to_answer = {v: k for k, v in answer_to_idx.items()}

    # Build: answer vocab idx → region index (for identity/location)
    answer_to_region_idx = {}
    for r_idx in range(region_to_answer_idx.size(0)):
        a_idx = region_to_answer_idx[r_idx].item()
        if a_idx >= 0:
            answer_to_region_idx[a_idx] = r_idx

    batch_size = region_logits.size(0)
    region_losses, attr_losses, yn_losses = [], [], []

    for i in range(batch_size):
        q = queries[i]
        target_a_idx = int(targets[i].item())
        qtype_id = QTYPE_TO_INT.get(q.qtype, 0)

        # Region CE for identity/location
        if qtype_id in (0, 1) and target_a_idx in answer_to_region_idx:
            r_idx = answer_to_region_idx[target_a_idx]
            region_losses.append(F.cross_entropy(region_logits[i : i + 1], torch.tensor([r_idx], device=device)))

        # BCE for yes/no: supervise only the queried region's presence head.
        # Spreading a global yes/no label over every region teaches "all organs
        # present/absent", which is not the question and corrupts the presence
        # head the executor consumes. Non-region targets have no head here.
        elif qtype_id == 2 and object_presence is not None:
            target_answer = idx_to_answer.get(target_a_idx, "")
            target_region_str = q.target.lower().strip() if q.target else ""
            target_a_for_region = answer_to_idx.get(target_region_str) if target_region_str else None
            region_idx = answer_to_region_idx.get(target_a_for_region) if target_a_for_region is not None else None
            if region_idx is None:
                continue
            if target_answer == "yes":
                yn_target = torch.ones((), device=device)
            elif target_answer == "no":
                yn_target = torch.zeros((), device=device)
            else:
                continue
            # object_presence is already sigmoid-activated. Under autocast a
            # saturated fp16 sigmoid can equal exactly 0 or 1; clamping in
            # fp16 with eps=1e-7 is ineffective because 1-eps rounds to 1.
            # Promote before clamping and evaluating BCE so both the target
            # boundary and logarithms remain finite.
            eps = 1e-6
            p = object_presence[i, region_idx].float().clamp(eps, 1.0 - eps)
            yn_losses.append(-(yn_target.float() * p.log() + (1.0 - yn_target.float()) * (1.0 - p).log()))

        # Attribute CE
        elif qtype_id == 3 and q.attribute:
            attr_logits = scene_logits.get(f"scene_{q.attribute}_logits")
            if attr_logits is not None and q.attribute in attribute_mappings:
                mapping = attribute_mappings[q.attribute].to(device)
                target_answer = idx_to_answer.get(target_a_idx, "")
                # Find attribute value index matching the target answer
                for a_val_idx in range(mapping.size(0)):
                    if mapping[a_val_idx].item() == target_a_idx:
                        attr_losses.append(
                            F.cross_entropy(attr_logits[i : i + 1], torch.tensor([a_val_idx], device=device))
                        )
                        break

    if region_losses:
        result["region"] = torch.stack(region_losses).mean()
    if yn_losses:
        result["yn"] = torch.stack(yn_losses).mean()
    if attr_losses:
        result["attr"] = torch.stack(attr_losses).mean()

    return result
