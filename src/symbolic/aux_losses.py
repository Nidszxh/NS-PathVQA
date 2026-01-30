"""Symbolic auxiliary losses for training the scene parser heads.

Provides direct supervision signals to each SceneParser head:
  - Region CE loss for identity/location questions
  - Attribute CE loss for attribute questions (color/shape/size)
  - BCE loss for yes/no object-presence supervision
"""

from typing import Dict, List
import torch
import torch.nn.functional as F
from symbolic.executor import QTYPE_TO_INT


def compute_symbolic_aux_losses(
    scene_logits: Dict[str, torch.Tensor],
    queries: list,
    targets: torch.Tensor,
    answer_to_idx: Dict[str, int],
    region_names: List[str],
    region_to_answer_idx: torch.Tensor,
    attribute_mappings: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute auxiliary losses for scene parser heads.

    Args:
        scene_logits: dict from model forward with scene_region_logits, etc.
        queries: list of Query objects
        targets: answer target indices (B,)
        answer_to_idx: answer string → vocab index
        region_names: list of region name strings
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
        target_a_idx = targets[i].item()
        qtype_id = QTYPE_TO_INT.get(q.qtype, 0)

        # Region CE for identity/location
        if qtype_id in (0, 1) and target_a_idx in answer_to_region_idx:
            r_idx = answer_to_region_idx[target_a_idx]
            region_losses.append(F.cross_entropy(
                region_logits[i:i+1], torch.tensor([r_idx], device=device)
            ))

        # BCE for yes/no
        elif qtype_id == 2 and object_presence is not None:
            target_answer = idx_to_answer.get(target_a_idx, "")
            if target_answer == "yes":
                yn_target = torch.ones_like(object_presence[i])
            elif target_answer == "no":
                yn_target = torch.zeros_like(object_presence[i])
            else:
                continue
            # object_presence is sigmoid'd; use BCE (safe since we're in no_grad-like context)
            # Workaround: compute BCE manually to avoid autocast issue
            eps = 1e-7
            p = object_presence[i].clamp(eps, 1 - eps)
            yn_losses.append((-(yn_target * p.log() + (1 - yn_target) * (1 - p).log())).mean())

        # Attribute CE
        elif qtype_id == 3 and q.attribute:
            attr_logits = scene_logits.get(f"scene_{q.attribute}_logits")
            if attr_logits is not None and q.attribute in attribute_mappings:
                mapping = attribute_mappings[q.attribute].to(device)
                target_answer = idx_to_answer.get(target_a_idx, "")
                # Find attribute value index matching the target answer
                for a_val_idx in range(mapping.size(0)):
                    if mapping[a_val_idx].item() == target_a_idx:
                        attr_losses.append(F.cross_entropy(
                            attr_logits[i:i+1], torch.tensor([a_val_idx], device=device)
                        ))
                        break

    if region_losses:
        result["region"] = torch.stack(region_losses).mean()
    if yn_losses:
        result["yn"] = torch.stack(yn_losses).mean()
    if attr_losses:
        result["attr"] = torch.stack(attr_losses).mean()

    return result
