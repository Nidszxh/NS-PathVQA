"""Differentiable executor: mixes per-qtype branch logits with soft routing weights.

The hard executor (``pathvqa.symbolic.executor``) dispatches on a discrete
qtype; this module computes *every* branch logit and blends them by
``soft_qtype`` so gradients flow back to the query classifier and target
extractor.
"""

import torch
from typing import Dict, List, Optional
from pathvqa.symbolic.executor import build_region_mapping, build_attribute_mappings


class DifferentiableExecutor:
    """Differentiable execution engine.

    Instead of hard-branching on qtype, computes logits for all primitive
    operations and takes a ``soft_qtype``-weighted sum. ``count`` is always
    emitted as zeros so those queries fall through to the neural path in the
    PoE fusion.
    """

    def __init__(self, answer_to_idx: Dict[str, int], region_names: List[str]):
        self.answer_to_idx = answer_to_idx
        self.vocab_size = len(answer_to_idx)
        self.region_names = region_names
        self.region_mapping = build_region_mapping(region_names, answer_to_idx)
        self.attr_mappings = build_attribute_mappings(answer_to_idx)

        # -1 sentinel marks "absent from answer vocab"; the yes/no branch
        # skips itself when either index is missing.
        self.yes_idx = answer_to_idx.get("yes", -1)
        self.no_idx = answer_to_idx.get("no", -1)

    def execute(
        self,
        scene_logits: Dict[str, torch.Tensor],
        soft_qtype: torch.Tensor,
        soft_target: torch.Tensor,
        yes_no_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blend the qtype branches into symbolic answer logits.

        ``soft_qtype``'s column order must match
        ``pathvqa.symbolic.routing.QTYPE_LIST`` (identity, location, yes_no,
        attribute, count); the stack below reuses the identity branch for
        location and leaves count at zero.

        Args:
            scene_logits: Scene-head outputs keyed by tensor name.
            soft_qtype: (B, 5) probabilities over the five qtypes.
            soft_target: (B, num_regions) target-region probabilities.
            yes_no_valid: Optional (B,) boolean mask marking rows whose target
                is an actual region. Rows marked False emit no yes/no evidence,
                mirroring the hard executor. When omitted, validity is inferred
                from target concentration (a uniform distribution carries no
                region-specific presence evidence); a one-region vocabulary is
                treated as valid because every distribution is trivially sharp.

        Returns:
            (B, vocab_size) symbolic logits.
        """
        device = soft_qtype.device
        batch_size = soft_qtype.size(0)

        region_logits = scene_logits.get("scene_region_logits")
        if region_logits is None:
            return torch.zeros(batch_size, self.vocab_size, device=device)

        reg_map = self.region_mapping.to(device)

        # 1. Identity / Location branch (Indices 0 and 1)
        # Scatter add region logits to vocab
        ident_logits = torch.zeros(batch_size, self.vocab_size, device=device)
        valid_mask = reg_map >= 0
        if valid_mask.any():
            valid_indices = reg_map[valid_mask]
            gathered = region_logits[:, valid_mask].to(ident_logits.dtype)
            idx_expanded = valid_indices.unsqueeze(0).expand(batch_size, -1)
            ident_logits.scatter_add_(1, idx_expanded, gathered)

        # 2. Yes/No branch (Index 2)
        # Use the BCE-trained sigmoid presence score rather than the sign of a
        # multi-class region logit. The score is not post-hoc calibrated.
        yn_logits = torch.zeros(batch_size, self.vocab_size, device=device)
        if self.yes_idx >= 0 and self.no_idx >= 0:
            presence = scene_logits.get("scene_object_presence")
            target = soft_target.to(region_logits.dtype)
            if presence is not None and presence is not region_logits:
                p = presence.to(torch.float32).clamp(min=1e-6, max=1.0 - 1e-6)
                log_odds = torch.log(p) - torch.log1p(-p)
                yes_score = (target.float() * log_odds).sum(dim=1)
            else:
                # Fallback: no presence head available.
                yes_score = (target * region_logits).sum(dim=1)

            # A yes/no question about a non-region target has no presence head
            # (the hard executor emits no evidence for it). Without this mask
            # the soft target's uniform fallback would manufacture a
            # "is any region present?" proxy answer.
            if yes_no_valid is not None:
                valid = yes_no_valid.to(device=device, dtype=torch.bool).reshape(-1)
                if valid.numel() != batch_size:
                    raise ValueError(f"yes_no_valid has {valid.numel()} entries but batch size is {batch_size}")
            elif soft_target.size(1) <= 1:
                # Every distribution over a single region is "uniform"; there is
                # no ambiguity to detect, so keep the evidence.
                valid = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                chance = 1.0 / soft_target.size(1)
                valid = soft_target.max(dim=1).values > chance
            yes_score = yes_score * valid.to(yes_score.dtype)

            yn_logits[:, self.yes_idx] = yes_score.to(yn_logits.dtype)
            yn_logits[:, self.no_idx] = -yes_score.to(yn_logits.dtype)

        # 3. Attribute branch (Index 3)
        # Combine color, shape, size mappings
        attr_logits = torch.zeros(batch_size, self.vocab_size, device=device)
        for attr_key in ["color", "shape", "size"]:
            key = f"scene_{attr_key}_logits"
            if key in scene_logits:
                attr_val_logits = scene_logits[key]
                amap = self.attr_mappings[attr_key].to(device)
                vmask = amap >= 0
                if vmask.any():
                    v_indices = amap[vmask]
                    gath = attr_val_logits[:, vmask].to(attr_logits.dtype)
                    idx_exp = v_indices.unsqueeze(0).expand(batch_size, -1)
                    attr_logits.scatter_add_(1, idx_exp, gath)

        # 4. Count branch (Index 4)
        # Uses 0s (will fall back to neural path in PoE fusion)
        count_logits = torch.zeros(batch_size, self.vocab_size, device=device)

        # Stack and multiply by soft_qtype
        # Branches: 0=ident, 1=loc (uses ident), 2=yn, 3=attr, 4=count
        stacked_logits = torch.stack(
            [
                ident_logits,  # identity
                ident_logits,  # location
                yn_logits,  # yes_no
                attr_logits,  # attribute
                count_logits,  # count
            ],
            dim=1,
        )  # (B, 5, vocab_size)

        # Weighted sum: (B, 5, 1) * (B, 5, V) -> sum dim 1 -> (B, V)
        soft_qtype_expanded = soft_qtype.unsqueeze(-1)
        final_logits = (soft_qtype_expanded * stacked_logits).sum(dim=1)

        return final_logits
