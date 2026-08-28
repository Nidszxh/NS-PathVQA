"""Symbolic executor: maps scene logits and parsed queries to answer-vocab logits.

For each query type:
  - identity/location: scatter_add region logits to matching answer indices
  - yes_no: convert the calibrated object-presence probability to log-odds and
    write it to the yes/no answers
  - attribute: scatter_add attribute logits to matching answers
  - count: no-op (neural handles counting)

The scatter branches are vectorized over the batch; the per-query branches
(ontology fallback, yes/no, attribute, count) loop over samples in Python.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathvqa.symbolic.query_parser import Query
from pathvqa.symbolic.scene_parser import COLOR_VALUES, SHAPE_VALUES, SIZE_VALUES
from pathvqa.data.dataset_adapter import AnatomicalOntology
from pathvqa.models.protocols import SceneParserOutput, ExecutorOutput


# Anatomical region prefixes and simple organ names in PathVQA.
REGION_ANSWER_PREFIXES = [
    "gastrointestinal",
    "cardiovascular",
    "hematologic",
    "endocrine",
    "female reproductive",
    "nervous",
    "respiratory",
    "urinary",
    "hepatobiliary",
]
SIMPLE_REGIONS = [
    "oral",
    "lung",
    "liver",
    "spleen",
    "heart",
    "brain",
    "skin",
    "extremities",
    "joints",
    "kidney",
    "abdomen",
    "pancreas",
    "breast",
    "stomach",
    "colon",
    "esophagus",
    "appendix",
    "intestine",
    "face",
    "eyes",
    "uterus",
    "ovary",
    "nose",
    "skull",
    "petrous",
]


def _is_region_answer(answer: str) -> bool:
    """Check if a given answer string refers to an anatomical region."""
    a = answer.lower().strip()
    for prefix in REGION_ANSWER_PREFIXES:
        if a.startswith(prefix):
            return True
    if a in SIMPLE_REGIONS:
        return True
    for region in SIMPLE_REGIONS:
        if a.startswith(region + ","):
            return True
    return False


def _build_index_mapping(source_names: List[str], target_to_idx: Dict[str, int]) -> torch.Tensor:
    """Map source name indices to target vocab indices. -1 if not found."""
    mapping = torch.full((len(source_names),), -1, dtype=torch.long)
    for i, name in enumerate(source_names):
        if name in target_to_idx:
            mapping[i] = target_to_idx[name]
    return mapping


def build_attribute_mappings(
    answer_to_idx: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """Build attribute value → answer-vocab index mappings for color/shape/size."""
    return {
        "color": _build_index_mapping(COLOR_VALUES, answer_to_idx),
        "shape": _build_index_mapping(SHAPE_VALUES, answer_to_idx),
        "size": _build_index_mapping(SIZE_VALUES, answer_to_idx),
    }


def build_region_mapping(region_names: List[str], answer_to_idx: Dict[str, int]) -> torch.Tensor:
    """Build region name → answer-vocab index mapping."""
    return _build_index_mapping(region_names, answer_to_idx)


def build_region_names(answer_vocab: List[str]) -> List[str]:
    """Discover anatomical region names from the answer vocabulary."""
    region_names = []
    for a in answer_vocab:
        a_lower = a.lower().strip()
        if _is_region_answer(a_lower) and a_lower not in ("yes", "no"):
            region_names.append(a)
    return region_names


def presence_log_odds(presence: torch.Tensor) -> torch.Tensor:
    """Convert calibrated presence probabilities to log-odds.

    ``scene_object_presence`` is a sigmoid output, so it is a calibrated
    presence probability (unlike multi-class region-classification logits,
    whose sign carries no presence meaning). Clamping keeps the transform
    finite at the saturated endpoints.
    """
    p = presence.clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def ontology_fallback(
    target: str,
    ontology: AnatomicalOntology,
    answer_to_idx: Dict[str, int],
) -> Optional[int]:
    """Ontology-aware fallback for identity/location queries.

    If exact target not in answer_to_idx, try synonym expansion and parent traversal.
    """
    if not target:
        return None
    norm = ontology.normalize_term(target)
    # Direct match after normalization
    if norm in answer_to_idx:
        return answer_to_idx[norm]
    # Check if any answer contains the normalized term
    for ans, idx in answer_to_idx.items():
        if norm in ans.lower():
            return idx
    # Parent system traversal
    for parent in ontology.get_parent_systems(norm):
        if parent in answer_to_idx:
            return answer_to_idx[parent]
        # Check partial matches for parent
        for ans, idx in answer_to_idx.items():
            if parent in ans.lower():
                return idx
    return None


class VectorizedSymbolicExecutor(nn.Module):
    """Vectorized PyTorch executor for symbolic reasoning over scenes."""

    def __init__(
        self,
        region_names: List[str],
        answer_to_idx: Dict[str, int],
        attribute_mappings: Optional[Dict[str, torch.Tensor]] = None,
        ontology: Optional[AnatomicalOntology] = None,
    ):
        """Register the region and attribute index maps as persistent buffers.

        ``attribute_mappings`` is derived from the answer vocabulary when not
        supplied. Every map uses ``-1`` to mark "no matching answer entry".
        """
        super().__init__()
        self.region_names = list(region_names)
        self.answer_to_idx = dict(answer_to_idx)
        self.vocab_size = len(answer_to_idx)
        self.ontology = ontology

        self.region_to_vocab: torch.Tensor
        region_map = build_region_mapping(self.region_names, self.answer_to_idx)
        self.register_buffer("region_to_vocab", region_map)

        if attribute_mappings is None:
            attribute_mappings = build_attribute_mappings(self.answer_to_idx)
        for k, v in attribute_mappings.items():
            self.register_buffer(f"attr_map_{k}", v)

    def forward(
        self,
        scene_logits: SceneParserOutput,
        queries: List[Query],
        neural_logits: torch.Tensor,
    ) -> ExecutorOutput:
        """Dispatch the batch through the executor, one ``Query`` per row."""
        return execute(
            scene_logits=scene_logits,
            queries=queries,
            region_names=self.region_names,
            region_to_answer_idx=self.region_to_vocab,
            attribute_mappings={
                "color": getattr(self, "attr_map_color"),
                "shape": getattr(self, "attr_map_shape"),
                "size": getattr(self, "attr_map_size"),
            },
            answer_to_idx=self.answer_to_idx,
            answer_vocab_size=self.vocab_size,
            neural_logits=neural_logits,
            ontology=self.ontology,
        )


QTYPE_TO_INT = {"identity": 0, "location": 1, "yes_no": 2, "attribute": 3, "count": 4}


def execute(
    scene_logits: SceneParserOutput,
    queries: List[Query],
    region_names: List[str],
    region_to_answer_idx: torch.Tensor,
    attribute_mappings: Dict[str, torch.Tensor],
    answer_to_idx: Dict[str, int],
    answer_vocab_size: int,
    neural_logits: torch.Tensor,
    ontology: Optional[AnatomicalOntology] = None,
) -> ExecutorOutput:
    """Execute symbolic reasoning: map scene logits to answer vocabulary.

    Selects the identity/location, yes/no, attribute and count branches per query
    and scatters them into the answer vocabulary, preserving the batch dimension.

    Raises:
        ValueError: If ``len(queries)`` does not match the batch size.
    """
    device = neural_logits.device
    batch_size = neural_logits.size(0)
    if len(queries) != batch_size:
        raise ValueError(f"Number of queries ({len(queries)}) must match batch size ({batch_size})")

    symbolic_logits = torch.zeros(batch_size, answer_vocab_size, device=device)

    # Accept both SceneParserOutput (.region_logits) and ModelOutput
    # (.scene_region_logits) naming conventions.
    region_logits = getattr(scene_logits, "region_logits", None)
    if region_logits is None:
        region_logits = getattr(scene_logits, "scene_region_logits", None)
    if region_logits is None:
        return ExecutorOutput(
            symbolic_logits=symbolic_logits,
            region_logits=torch.zeros(0, device=device),
            trace={"symbolic_used": [False] * batch_size},
        )

    reg_map = region_to_answer_idx.to(device)  # (N_reg,) → maps region idx → answer vocab idx
    yes_idx = answer_to_idx.get("yes")
    no_idx = answer_to_idx.get("no")

    # Calibrated object-presence probability, accepted under either the
    # SceneParserOutput (``object_presence``) or ModelOutput
    # (``scene_object_presence``) naming convention.
    presence = getattr(scene_logits, "object_presence", None)
    if presence is None:
        presence = getattr(scene_logits, "scene_object_presence", None)

    # Convert queries to qtype integer tensor
    qtype_ids = torch.tensor([QTYPE_TO_INT.get(q.qtype, 0) for q in queries], device=device, dtype=torch.long)

    # Masks for each qtype (B,)
    is_ident_loc = (qtype_ids == 0) | (qtype_ids == 1)
    is_yes_no = qtype_ids == 2
    is_attr = qtype_ids == 3

    # Identity / Location: scatter_add region logits to matching answer indices
    # symbolic_logits[b, answer_idx[r]] += region_logits[b, r]  for valid regions r
    if is_ident_loc.any():
        valid_mask = reg_map >= 0  # (N_reg,) — True where region maps to a vocab entry
        if valid_mask.any():
            valid_indices = reg_map[valid_mask]  # (K,)
            mask_2d = is_ident_loc.unsqueeze(1).to(symbolic_logits.dtype)  # (B, 1)
            gathered = region_logits[:, valid_mask].to(symbolic_logits.dtype) * mask_2d  # (B, K)
            idx_expanded = valid_indices.unsqueeze(0).expand(batch_size, -1)  # (B, K)
            symbolic_logits.scatter_add_(1, idx_expanded, gathered)

        # Ontology fallback: if exact target not in region_names, try synonym/parent lookup
        if ontology is not None:
            ident_loc_indices = is_ident_loc.nonzero(as_tuple=True)[0]
            for idx in ident_loc_indices:
                i = idx.item()
                query = queries[i]
                if query.target:
                    # Check if target was already matched via region scatter
                    already_matched = False
                    for r_idx in range(len(region_names)):
                        if region_to_answer_idx[r_idx].item() >= 0:
                            if region_names[r_idx].lower() == query.target.lower():
                                already_matched = True
                                break
                    if not already_matched:
                        fallback_idx = ontology_fallback(query.target, ontology, answer_to_idx)
                        if fallback_idx is not None:
                            # Boost the ontology-matched answer with a positive
                            # score. softplus keeps a negative region logit from
                            # *suppressing* a correct ontology match.
                            max_conf = region_logits[i, :].max()
                            symbolic_logits[i, fallback_idx] = F.softplus(max_conf)

    # Yes/No: use the calibrated object-presence probability as evidence.
    # A multi-class region logit's sign does not encode presence, so we convert
    # the sigmoid presence score to log-odds before routing it to yes/no.
    if is_yes_no.any():
        yn_indices = is_yes_no.nonzero(as_tuple=True)[0]
        for idx in yn_indices:
            i = idx.item()
            query = queries[i]
            r_conf: torch.Tensor

            target_l = query.target.lower() if query.target else ""
            matches = [rn for rn in region_names if rn.lower() == target_l]
            if target_l and matches:
                r_idx = region_names.index(matches[0])
                r_conf = presence[i, r_idx] if presence is not None else region_logits[i, r_idx]
            elif query.target and query.target in answer_to_idx:
                r_conf = presence[i, :].max() if presence is not None else region_logits[i, :].max()
            else:
                # No specific target — boost all region answers
                for j, name in enumerate(region_names):
                    if name in answer_to_idx:
                        aidx = answer_to_idx[name]
                        symbolic_logits[i, aidx] = symbolic_logits[i, aidx] + region_logits[i, j]
                r_conf = presence[i, :].max() if presence is not None else region_logits[i, :].max()

            # Yes/no: positive log-odds -> yes, negative -> no. Both classes are
            # written so the softmax over symbolic logits is calibrated.
            if presence is not None:
                evidence = presence_log_odds(r_conf)
            else:
                evidence = r_conf
            if yes_idx is not None and no_idx is not None:
                symbolic_logits[i, yes_idx] = evidence
                symbolic_logits[i, no_idx] = -evidence

    # Attribute: symbolic_logits[b, answer_idx[attr_val]] = attr_logits[b, attr_val_idx]
    # where attr_val matches the answer vocab entry
    if is_attr.any():
        attr_indices = is_attr.nonzero(as_tuple=True)[0]
        for idx in attr_indices:
            i = idx.item()
            query = queries[i]
            attr_logits = getattr(scene_logits, f"{query.attribute}_logits", None)
            if attr_logits is None:
                attr_logits = getattr(scene_logits, f"scene_{query.attribute}_logits", None)
            if attr_logits is not None and query.attribute in attribute_mappings:
                mapping = attribute_mappings[query.attribute].to(device)
                valid_attr_mask = mapping >= 0
                if valid_attr_mask.any():
                    target_indices = mapping[valid_attr_mask]
                    symbolic_logits[i, target_indices] = attr_logits[i, valid_attr_mask]

    # Build trace: count (4) is the only non-symbolic type
    symbolic_used = (qtype_ids != 4).tolist()

    return ExecutorOutput(
        symbolic_logits=symbolic_logits,
        region_logits=region_logits,
        trace={"symbolic_used": symbolic_used},
    )


if __name__ == "__main__":
    print("Testing Executor...")
    region_names = ["bone", "gastrointestinal", "lung"]
    answer_vocab = [
        "yes",
        "no",
        "bone",
        "bone, calvarium",
        "gastrointestinal",
        "gastrointestinal system",
        "red",
        "blue",
        "large",
        "small",
    ]
    answer_to_idx = {a: i for i, a in enumerate(answer_vocab)}
    region_to_answer_idx = build_region_mapping(region_names, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)

    batch_size = 2
    scene_logits = SceneParserOutput(
        region_logits=torch.randn(batch_size, len(region_names)),
        object_presence=torch.sigmoid(torch.randn(batch_size, len(region_names))),
        color_logits=torch.randn(batch_size, len(COLOR_VALUES)),
        shape_logits=torch.randn(batch_size, len(SHAPE_VALUES)),
        size_logits=torch.randn(batch_size, len(SIZE_VALUES)),
    )
    queries = [
        Query(qtype="identity", target="bone"),
        Query(qtype="location", target="gastrointestinal"),
    ]
    neural_logits = torch.randn(batch_size, len(answer_vocab))

    result = execute(
        scene_logits,
        queries,
        region_names,
        region_to_answer_idx,
        attr_mappings,
        answer_to_idx,
        len(answer_vocab),
        neural_logits,
    )
    print(f"Symbolic logits shape: {result.symbolic_logits.shape}")
    print(f"Region logits shape: {result.region_logits.shape}")
    print(f"Symbolic used: {result.trace['symbolic_used']}")

    # Test VectorizedSymbolicExecutor wrapper
    vec_executor = VectorizedSymbolicExecutor(region_names, answer_to_idx, attr_mappings)
    vec_out = vec_executor(scene_logits, queries, neural_logits)
    assert torch.allclose(result.symbolic_logits, vec_out.symbolic_logits)
    print("Vectorized executor test passed!")

    # Yes/no queries must write evidence to the yes/no answer indices, not to
    # the target word's index.
    queries_yn = [Query(qtype="yes_no", target="lung")]
    scene_yn = SceneParserOutput(
        region_logits=scene_logits.region_logits[:1],
        object_presence=scene_logits.object_presence[:1],
        color_logits=scene_logits.color_logits[:1],
        shape_logits=scene_logits.shape_logits[:1],
        size_logits=scene_logits.size_logits[:1],
    )
    result_yn = execute(
        scene_yn,
        queries_yn,
        region_names,
        region_to_answer_idx,
        attr_mappings,
        answer_to_idx,
        len(answer_vocab),
        neural_logits[:1],
    )
    yn_val = result_yn.symbolic_logits[0, answer_to_idx["yes"]]
    no_val = result_yn.symbolic_logits[0, answer_to_idx["no"]]
    print(f"  yes={yn_val:.4f}, no={no_val:.4f}")
    # Lung is a region name, so it maps to yes/no
    assert yn_val != 0.0 or no_val != 0.0, "yes/no should boost yes or no index"
    print("Yes/no fix verified!")

    # Test mixed qtypes
    queries_mixed = [
        Query(qtype="identity", target="bone"),
        Query(qtype="yes_no", target="lung"),
        Query(qtype="attribute", attribute="color"),
        Query(qtype="count", target="nuclei"),
    ]
    scene_mixed = SceneParserOutput(
        region_logits=torch.randn(4, len(region_names)),
        object_presence=torch.sigmoid(torch.randn(4, len(region_names))),
        color_logits=torch.randn(4, len(COLOR_VALUES)),
        shape_logits=torch.randn(4, len(SHAPE_VALUES)),
        size_logits=torch.randn(4, len(SIZE_VALUES)),
    )
    neural_mixed = torch.randn(4, len(answer_vocab))
    result_mixed = execute(
        scene_mixed,
        queries_mixed,
        region_names,
        region_to_answer_idx,
        attr_mappings,
        answer_to_idx,
        len(answer_vocab),
        neural_mixed,
    )
    assert result_mixed.trace["symbolic_used"] == [True, True, True, False]
    print("Mixed qtype test passed!")

    print("All executor tests passed!")
