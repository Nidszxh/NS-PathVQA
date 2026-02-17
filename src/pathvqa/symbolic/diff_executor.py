import torch
from typing import Dict, List, Optional
from pathvqa.symbolic.executor import build_region_mapping, build_attribute_mappings

class DifferentiableExecutor:
    """
    Differentiable execution engine.
    Instead of hard-branching on qtype, computes logits for all primitive operations
    and computes a weighted sum using `soft_qtype`.
    """
    def __init__(self, answer_to_idx: Dict[str, int], region_names: List[str]):
        self.answer_to_idx = answer_to_idx
        self.vocab_size = len(answer_to_idx)
        self.region_names = region_names
        self.region_mapping = build_region_mapping(region_names, answer_to_idx)
        self.attr_mappings = build_attribute_mappings(answer_to_idx)
        
        self.yes_idx = answer_to_idx.get("yes", -1)
        self.no_idx = answer_to_idx.get("no", -1)

    def execute(
        self, 
        scene_logits: Dict[str, torch.Tensor],
        soft_qtype: torch.Tensor,
        soft_target: torch.Tensor
    ) -> torch.Tensor:
        """
        soft_qtype: (B, 5) probabilities for [identity, location, yes_no, attribute, count]
        soft_target: (B, num_regions) probabilities
        Returns: (B, vocab_size) symbolic logits
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
            gathered = region_logits[:, valid_mask]
            idx_expanded = valid_indices.unsqueeze(0).expand(batch_size, -1)
            ident_logits.scatter_add_(1, idx_expanded, gathered)
            
        # 2. Yes/No branch (Index 2)
        yn_logits = torch.zeros(batch_size, self.vocab_size, device=device)
        if self.yes_idx >= 0 and self.no_idx >= 0:
            # Score = dot product of soft_target attention and region_logits confidence
            yes_score = (soft_target * region_logits).sum(dim=1)
            # Normalize or scale for no_score (inverse)
            no_score = -yes_score
            yn_logits[:, self.yes_idx] = yes_score
            yn_logits[:, self.no_idx] = no_score

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
                    gath = attr_val_logits[:, vmask]
                    idx_exp = v_indices.unsqueeze(0).expand(batch_size, -1)
                    attr_logits.scatter_add_(1, idx_exp, gath)
                    
        # 4. Count branch (Index 4)
        # Uses 0s (will fall back to neural path in PoE fusion)
        count_logits = torch.zeros(batch_size, self.vocab_size, device=device)
        
        # Stack and multiply by soft_qtype
        # Branches: 0=ident, 1=loc (uses ident), 2=yn, 3=attr, 4=count
        stacked_logits = torch.stack([
            ident_logits,   # identity
            ident_logits,   # location
            yn_logits,      # yes_no
            attr_logits,    # attribute
            count_logits    # count
        ], dim=1) # (B, 5, vocab_size)
        
        # Weighted sum: (B, 5, 1) * (B, 5, V) -> sum dim 1 -> (B, V)
        soft_qtype_expanded = soft_qtype.unsqueeze(-1)
        final_logits = (soft_qtype_expanded * stacked_logits).sum(dim=1)
        
        return final_logits
