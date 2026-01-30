"""Ontology-aware training regularizer for the region classifier.

Encourages region classifier weight vectors for organs in the same anatomical
system to be close in embedding space, using the SNOMED-CT hierarchy.
"""

from typing import List, Set, Tuple
import torch
import torch.nn.functional as F
from data.dataset_adapter import AnatomicalOntology


def compute_sibling_regularization(
    region_classifier_weight: torch.Tensor,
    region_names: List[str],
    ontology: AnatomicalOntology,
    margin: float = 0.5,
) -> torch.Tensor:
    """Regularize region classifier weights so same-system organs are close.

    L_sibling = Σ_{(r1, r2) ∈ siblings} max(0, δ - cos(W[r1], W[r2]))

    Args:
        region_classifier_weight: (num_regions, visual_dim) weight matrix
        region_names: list of region name strings
        ontology: AnatomicalOntology instance
        margin: margin δ for the contrastive loss

    Returns:
        scalar regularization loss
    """
    device = region_classifier_weight.device
    num_regions = region_classifier_weight.size(0)
    if num_regions < 2:
        return torch.tensor(0.0, device=device)

    # Find sibling pairs: organs that share a parent system
    sibling_pairs: Set[Tuple[int, int]] = set()
    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            name_i = region_names[i].lower()
            name_j = region_names[j].lower()
            if ontology.are_related(name_i, name_j) and name_i != name_j:
                sibling_pairs.add((i, j))

    if not sibling_pairs:
        return torch.tensor(0.0, device=device)

    # Compute cosine similarity for each sibling pair
    loss = torch.tensor(0.0, device=device)
    for i, j in sibling_pairs:
        w_i = F.normalize(region_classifier_weight[i], dim=0)
        w_j = F.normalize(region_classifier_weight[j], dim=0)
        cos_sim = (w_i * w_j).sum()
        # Contrastive: push siblings together (cos_sim should be high)
        loss = loss + F.relu(margin - cos_sim)

    return loss / len(sibling_pairs)
