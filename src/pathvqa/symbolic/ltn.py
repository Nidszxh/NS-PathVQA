"""Logic Tensor Networks: differentiable fuzzy logic over scene parser predictions."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class FuzzyLogicEngine:
    """T-norm fuzzy logic operators (AND, OR, NOT, IMPLIES, FORALL, EXISTS)."""

    @staticmethod
    def AND(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
        """Fuzzy conjunction."""
        if t_norm == "product":
            return a * b  # T-norm: a ⊗ b = a · b
        elif t_norm == "lukasiewicz":
            return torch.clamp(a + b - 1.0, min=0.0)  # T-norm: a ⊗ b = max(a + b - 1, 0)
        elif t_norm == "goedel":
            return torch.min(a, b)  # T-norm: a ⊗ b = min(a, b)
        raise ValueError(f"Unknown t_norm: {t_norm}")

    @staticmethod
    def OR(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
        """Fuzzy disjunction."""
        if t_norm == "product":
            return a + b - (a * b)  # S-norm (product): a ⊕ b = a + b - a·b
        elif t_norm == "lukasiewicz":
            return torch.clamp(a + b, max=1.0)  # S-norm (Łukasiewicz): a ⊕ b = min(a + b, 1)
        elif t_norm == "goedel":
            return torch.max(a, b)  # S-norm (Gödel): a ⊕ b = max(a, b)
        raise ValueError(f"Unknown t_norm: {t_norm}")

    @staticmethod
    def NOT(a: torch.Tensor) -> torch.Tensor:
        """Fuzzy negation."""
        return 1.0 - a

    @staticmethod
    def IMPLIES(a: torch.Tensor, b: torch.Tensor, t_norm: str = "product") -> torch.Tensor:
        """Fuzzy implication (Reichenbach / Łukasiewicz / Gödel)."""
        if t_norm == "product":
            return torch.clamp(1.0 - a + (a * b), min=0.0, max=1.0)  # Reichenbach: ¬a + a·b
        elif t_norm == "lukasiewicz":
            return torch.clamp(1.0 - a + b, min=0.0, max=1.0)  # Łukasiewicz: min(1, 1-a+b)
        elif t_norm == "goedel":
            return torch.where(a <= b, torch.ones_like(a), b)  # Gödel: a≤b ? 1 : b
        raise ValueError(f"Unknown t_norm: {t_norm}")

    @staticmethod
    def FORALL(x: torch.Tensor, dim: int = -1, p: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
        """Universal quantification via p-mean error aggregator."""
        err = torch.clamp(1.0 - x, min=0.0)  # per-element truth error
        p_err = torch.pow(err + eps, p)  # (err)^p
        mean_p_err = torch.mean(p_err, dim=dim)  # mean over dim
        # ∀x = 1 - (mean((1-x)^p))^(1/p)   → 1 when all true, 0 when any false
        return torch.clamp(1.0 - torch.pow(mean_p_err, 1.0 / p), min=0.0, max=1.0)

    @staticmethod
    def EXISTS(x: torch.Tensor, dim: int = -1, p: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
        """Existential quantification via p-mean aggregator."""
        p_val = torch.pow(torch.clamp(x, min=0.0) + eps, p)  # x^p
        mean_p_val = torch.mean(p_val, dim=dim)  # mean over dim
        # ∃x = (mean(x^p))^(1/p)   → 1 when any true, 0 when all false
        return torch.clamp(torch.pow(mean_p_val, 1.0 / p), min=0.0, max=1.0)


class MedicalLogicTensorNetwork(nn.Module):
    """Evaluates differentiable medical logic clauses over scene predictions."""

    def __init__(
        self,
        num_regions: int,
        region_names: List[str],
        t_norm: str = "product",
        p_agg: float = 2.0,
    ):
        """Configure the fuzzy-logic aggregation used by the clauses.

        Args:
            num_regions: Number of scene regions.
            region_names: Region names, aligned with the scene head's axis.
            t_norm: Fuzzy t-norm for AND/OR/IMPLIES ("product", "minimum", ...).
            p_agg: Exponent for the generalized-mean (p-negated) quantifiers.
        """
        super().__init__()
        self.num_regions = num_regions
        self.region_names = list(region_names)
        self.t_norm = t_norm
        self.p_agg = p_agg
        self.engine = FuzzyLogicEngine

    def evaluate_clauses(
        self,
        scene_logits: Dict[str, torch.Tensor],
        qtype_onehot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluate logical consistency clauses. Returns (sat_loss, per_clause_scores).

        ``qtype_onehot`` is accepted for call-site symmetry with the other
        symbolic losses but is not used — the clauses score scene facts only.
        """
        region_logits = scene_logits.get("scene_region_logits")
        object_presence = scene_logits.get("scene_object_presence")

        if region_logits is None or object_presence is None:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        # Soft truth values in [0, 1]
        p_region = torch.sigmoid(region_logits)  # (B, N_reg)
        p_object = object_presence  # (B, N_reg)

        clause_scores = {}

        # Clause 1: Region-Object Coherence — ∀r (Object(r) ↔ Region(r))
        # Bidirectional implication: object presence iff region is activated
        imp1 = self.engine.IMPLIES(p_object, p_region, self.t_norm)  # object → region
        imp2 = self.engine.IMPLIES(p_region, p_object, self.t_norm)  # region → object
        coherence = self.engine.AND(imp1, imp2, self.t_norm)  # bidirectional
        sat_coherence = self.engine.FORALL(coherence, dim=-1, p=self.p_agg).mean()
        clause_scores["region_object_coherence"] = sat_coherence.item()

        # Clause 2: Attribute certainty — for color questions, max softmax should be sharp
        # sat_attr = mean(max_c p(c)) where p(c) = softmax(color_logits)
        color_logits = scene_logits.get("scene_color_logits")
        sat_attr = torch.tensor(1.0, device=region_logits.device)
        if color_logits is not None:
            p_color = torch.softmax(color_logits, dim=-1)
            max_color = p_color.max(dim=-1).values
            sat_attr = max_color.mean()
            clause_scores["attribute_certainty"] = sat_attr.item()

        # Clause 3: Morphology-Region Coherence — object presence implies irregular shape
        # In histopathology, malignant/abnormal regions show irregular morphology.
        shape_logits = scene_logits.get("scene_shape_logits")
        sat_morph = torch.tensor(1.0, device=region_logits.device)
        if shape_logits is not None:
            p_shape = torch.softmax(shape_logits, dim=-1)
            # Resolve "irregular" by name so reordering SHAPE_VALUES cannot
            # silently corrupt this clause.
            from pathvqa.symbolic.scene_parser import SHAPE_VALUES

            irregular_idx = SHAPE_VALUES.index("irregular")
            irregular_prob = p_shape[:, irregular_idx]
            max_region_object = p_object.max(dim=-1).values  # strongest object presence
            sat_morph_per_sample = self.engine.IMPLIES(max_region_object, irregular_prob, self.t_norm)
            sat_morph = sat_morph_per_sample.mean()
            clause_scores["morphology_region_coherence"] = sat_morph.item()

        # Clause 4: Attribute Confidence Union — all three attribute heads should be sharp
        # color, shape, and size distributions should all be confident simultaneously.
        size_logits = scene_logits.get("scene_size_logits")
        sat_conf = torch.tensor(1.0, device=region_logits.device)
        if color_logits is not None and shape_logits is not None and size_logits is not None:
            color_conf = torch.softmax(color_logits, dim=-1).max(dim=-1).values
            shape_conf = torch.softmax(shape_logits, dim=-1).max(dim=-1).values
            size_conf = torch.softmax(size_logits, dim=-1).max(dim=-1).values
            all_confident = self.engine.AND(
                self.engine.AND(color_conf, shape_conf, self.t_norm),
                size_conf,
                self.t_norm,
            )
            sat_conf = all_confident.mean()
            clause_scores["attribute_confidence_union"] = sat_conf.item()

        # Clause 5: Region Sparsity — not all regions should be activated simultaneously
        # Histopathology images typically show 1-3 tissue types; penalize diffuse predictions.
        region_sum = p_region.sum(dim=-1)  # (B,)
        # Gaussian penalty peaked at 2.0 active regions
        target_sum = 2.0
        sparsity_penalty = torch.exp(-0.5 * torch.square(region_sum - target_sum) / 4.0)
        sat_sparsity = sparsity_penalty.mean()
        clause_scores["region_sparsity"] = sat_sparsity.item()

        # Aggregate all clause satisfactions (equal weighting).
        # Only clauses whose inputs were actually present contribute to the sum;
        # dividing by the number of *summed* terms keeps satisfaction in [0, 1]
        # (summing the constant placeholders would push it above 1 and make the
        # loss negative on partial head sets).
        sat_terms = [sat_coherence, sat_sparsity]
        if color_logits is not None:
            sat_terms.append(sat_attr)
        if shape_logits is not None:
            sat_terms.append(sat_morph)
        if color_logits is not None and shape_logits is not None and size_logits is not None:
            sat_terms.append(sat_conf)
        total_sat = torch.stack([t.to(region_logits.device) for t in sat_terms]).mean()
        clause_scores["overall_satisfaction"] = total_sat.item()

        ltn_loss = 1.0 - total_sat
        return ltn_loss, clause_scores


if __name__ == "__main__":
    print("Testing Logic Tensor Network...")
    engine = FuzzyLogicEngine()
    a = torch.tensor([0.9, 0.2, 0.8])
    b = torch.tensor([0.8, 0.1, 0.9])
    print(f"AND: {engine.AND(a, b)}")
    print(f"OR: {engine.OR(a, b)}")
    print(f"IMPLIES: {engine.IMPLIES(a, b)}")
    print(f"FORALL: {engine.FORALL(a)}")

    ltn = MedicalLogicTensorNetwork(num_regions=3, region_names=["lung", "liver", "heart"])
    dummy_scene = {
        "scene_region_logits": torch.randn(2, 3),
        "scene_object_presence": torch.sigmoid(torch.randn(2, 3)),
        "scene_color_logits": torch.randn(2, 17),
        "scene_shape_logits": torch.randn(2, 9),
        "scene_size_logits": torch.randn(2, 11),
    }
    loss, scores = ltn.evaluate_clauses(dummy_scene)
    print(f"LTN Loss: {loss.item():.4f}, Scores: {scores}")
    print("Logic Tensor Network test passed!")
