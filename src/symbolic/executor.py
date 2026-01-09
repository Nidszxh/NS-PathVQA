"""Symbolic executor: maps scene logits and query type to answer vocabulary entries.

The Executor is the core of the symbolic reasoning path. It takes:
  1. Scene logits (region, attribute, object presence from SceneParser)
  2. Parsed queries (Query objects from QueryParser)
  3. Answer vocabulary mappings (region_name → answer_idx, attribute_value → answer_idx)

For each query type:
  - identity/location: adds region logits to answer entries matching region names
  - yes_no: adds max region confidence to the target region answer entry
  - attribute: adds attribute logits to matching color/shape/size answer entries
  - count: uses object presence sigmoid sum as a uniform boost

All mappings are built dynamically from the dataset answer vocabulary at training time.

Region names are discovered from the answer vocabulary using prefix/suffix matching
on known anatomical terms (e.g. "gastrointestinal system", "lung").
"""

import torch
from typing import Dict, List, Tuple
from pathlib import Path
import sys

_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.append(_src)

from symbolic.query_parser import Query
from symbolic.scene_parser import COLOR_VALUES, SHAPE_VALUES, SIZE_VALUES


# Known prefixes for anatomical region answers in PathVQA
REGION_ANSWER_PREFIXES = [
    "gastrointestinal", "cardiovascular", "hematologic", "endocrine",
    "female reproductive", "nervous", "respiratory",
    "urinary", "hepatobiliary",
]
# Simple organ names that appear as answers
SIMPLE_REGIONS = [
    "oral", "lung", "liver", "spleen", "heart", "brain", "skin",
    "extremities", "joints", "kidney", "abdomen", "pancreas",
    "breast", "stomach", "colon", "esophagus", "appendix",
    "intestine", "face", "eyes", "uterus", "ovary", "nose",
    "skull", "petrous",
]
NUM_COLORS = len(COLOR_VALUES)
NUM_SHAPES = len(SHAPE_VALUES)
NUM_SIZES = len(SIZE_VALUES)


def _is_region_answer(answer: str) -> bool:
    """Check if a given answer string refers to an anatomical region."""
    a = answer.lower().strip()
    for prefix in REGION_ANSWER_PREFIXES:
        if a.startswith(prefix):
            return True
    if a in SIMPLE_REGIONS:
        return True
    # Catch answers like "bone, calvarium" where a region name is followed by a comma
    for region in SIMPLE_REGIONS:
        if a.startswith(region + ","):
            return True
    return False


def _build_index_mapping(
    source_names: List[str], target_to_idx: Dict[str, int]
) -> torch.Tensor:
    """Build a tensor mapping each source name index to its target vocab index.

    Returns a LongTensor of shape (len(source_names),) where each entry is
    the target index or -1 if the source name is not in the target vocabulary.
    """
    mapping = torch.full((len(source_names),), -1, dtype=torch.long)
    for i, name in enumerate(source_names):
        if name in target_to_idx:
            mapping[i] = target_to_idx[name]
    return mapping


def build_attribute_mappings(
    answer_to_idx: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """Build index mappings from attribute values (color/shape/size) to answer vocabulary."""
    return {
        "color": _build_index_mapping(COLOR_VALUES, answer_to_idx),
        "shape": _build_index_mapping(SHAPE_VALUES, answer_to_idx),
        "size": _build_index_mapping(SIZE_VALUES, answer_to_idx),
    }


def build_region_mapping(
    region_names: List[str], answer_to_idx: Dict[str, int]
) -> torch.Tensor:
    """Build index mapping from region names to answer vocabulary entries."""
    return _build_index_mapping(region_names, answer_to_idx)


def build_region_names(answer_vocab: List[str]) -> List[str]:
    """Discover region names from the answer vocabulary.

    Filters the full answer list to entries that match known anatomical
    region patterns (prefix-matched or in the simple regions list).
    """
    region_names = []
    for a in answer_vocab:
        a_lower = a.lower().strip()
        if _is_region_answer(a_lower) and a_lower not in ("yes", "no"):
            region_names.append(a)
    return region_names


def execute(
    scene_logits: Dict[str, torch.Tensor],
    queries: List[Query],
    region_names: List[str],
    region_to_answer_idx: torch.Tensor,
    attribute_mappings: Dict[str, torch.Tensor],
    answer_to_idx: Dict[str, int],
    answer_vocab_size: int,
    neural_logits: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Execute symbolic reasoning: map scene logits to answer vocabulary.

    For each query in the batch:
      - identity/location: add each region's logit to the matching answer index
      - yes_no: boost the target region answer with max region confidence
      - attribute: add color/shape/size logits to the matching answer indices
      - count: use sum of object presences as a uniform boost

    Args:
        scene_logits: Dict from SceneParser forward (region_logits, object_presence, etc.)
        queries: Parsed Query objects for each sample in the batch
        region_names: List of anatomical region names
        region_to_answer_idx: Tensor mapping each region index → answer vocab index
        attribute_mappings: Dict of attribute name → tensor mapping value index → answer idx
        answer_to_idx: Dict mapping answer string → vocabulary index
        answer_vocab_size: Total number of answer classes
        neural_logits: (batch, answer_vocab_size) neural path logits (used for device/device only)

    Returns:
        Dict with symbolic_logits, region_logits (raw), and trace (symbolic_used flags)
    """
    device = neural_logits.device
    batch_size = neural_logits.size(0)

    symbolic_logits = torch.zeros(batch_size, answer_vocab_size, device=device)
    trace = {"symbolic_used": [False] * batch_size}

    region_logits = scene_logits.get("scene_region_logits")
    if region_logits is None:
        return {"symbolic_logits": symbolic_logits, "region_logits": None, "trace": trace}

    for i, query in enumerate(queries):
        if query.qtype in ("identity", "location"):
            # Boost each region's answer entry with its corresponding logit
            for j in range(len(region_names)):
                aidx = region_to_answer_idx[j]
                if aidx >= 0:
                    symbolic_logits[i, aidx] = symbolic_logits[i, aidx] + region_logits[i, j]
            trace["symbolic_used"][i] = True

        elif query.qtype == "yes_no":
            # For yes/no questions about a region, boost that region's answer
            if query.target and query.target in answer_to_idx:
                aidx = answer_to_idx[query.target]
                symbolic_logits[i, aidx] = region_logits[i, :].max()
                trace["symbolic_used"][i] = True
            else:
                for j, name in enumerate(region_names):
                    if name in answer_to_idx:
                        aidx = answer_to_idx[name]
                        symbolic_logits[i, aidx] = symbolic_logits[i, aidx] + region_logits[i, j]

        elif query.qtype == "attribute":
            # Boost attribute answer entries (color/shape/size)
            attr_logits_key = f"scene_{query.attribute}_logits"
            attr_logits = scene_logits.get(attr_logits_key)
            if attr_logits is not None and query.attribute in attribute_mappings:
                mapping = attribute_mappings[query.attribute].to(device)
                for j in range(len(mapping)):
                    aidx = mapping[j]
                    if aidx >= 0:
                        symbolic_logits[i, aidx] = attr_logits[i, j]
                trace["symbolic_used"][i] = True

        elif query.qtype == "count":
            # Count questions: no meaningful symbolic signal from current setup,
            # skip to avoid adding uniform noise
            pass

    return {
        "symbolic_logits": symbolic_logits,
        "region_logits": region_logits,
        "trace": trace,
    }


if __name__ == "__main__":
    print("Testing Executor...")
    region_names = ["bone", "gastrointestinal", "lung"]
    answer_vocab = ["yes", "no", "bone", "bone, calvarium", "gastrointestinal",
                     "gastrointestinal system", "red", "blue", "large", "small"]
    answer_to_idx = {a: i for i, a in enumerate(answer_vocab)}
    region_to_answer_idx = build_region_mapping(region_names, answer_to_idx)
    attr_mappings = build_attribute_mappings(answer_to_idx)

    batch_size = 2
    scene_logits = {
        "scene_region_logits": torch.randn(batch_size, len(region_names)),
        "scene_object_presence": torch.sigmoid(torch.randn(batch_size, len(region_names))),
        "scene_color_logits": torch.randn(batch_size, len(COLOR_VALUES)),
        "scene_shape_logits": torch.randn(batch_size, len(SHAPE_VALUES)),
        "scene_size_logits": torch.randn(batch_size, len(SIZE_VALUES)),
    }
    queries = [
        Query(qtype="identity", target="bone"),
        Query(qtype="location", target="gastrointestinal"),
    ]
    neural_logits = torch.randn(batch_size, len(answer_vocab))

    result = execute(scene_logits, queries, region_names, region_to_answer_idx,
                     attr_mappings, answer_to_idx, len(answer_vocab), neural_logits)
    print(f"Symbolic logits shape: {result['symbolic_logits'].shape}")
    print(f"Region logits shape: {result['region_logits'].shape}")
    print(f"Symbolic used: {result['trace']['symbolic_used']}")
    print("Executor test passed!")
