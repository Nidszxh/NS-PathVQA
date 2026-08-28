"""Dataset adapters and anatomical ontology grounding for region discovery."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class AnatomicalOntology:
    """Bidirectional organ ↔ system lookup over an anatomy vocabulary.

    Built from ``SYSTEM_TO_ORGANS`` (plus an optional custom JSON ontology),
    inverted into ``organ_to_systems`` so queries work in either direction.
    """

    # Standard SNOMED-CT / RadLex organ-system hierarchy
    SYSTEM_TO_ORGANS: Dict[str, List[str]] = {
        "gastrointestinal system": [
            "stomach", "gastric", "colon", "esophagus", "appendix", "intestine",
            "bowel", "rectum", "duodenum", "jejunum", "ileum", "oral", "tongue",
            "pharynx", "salivary gland", "cecum", "sigmoid",
        ],
        "hepatobiliary system": [
            "liver", "hepatic", "gallbladder", "bile duct", "biliary", "pancreas",
            "pancreatic", "ampulla of vater",
        ],
        "respiratory system": [
            "lung", "pulmonary", "bronchus", "bronchial", "trachea", "larynx",
            "pleura", "mediastinum", "alveoli", "nasal", "nose",
        ],
        "cardiovascular system": [
            "heart", "cardiac", "myocardium", "pericardium", "endocardium",
            "aorta", "artery", "arteriole", "vein", "venule", "blood vessel", "vascular",
        ],
        "urinary system": [
            "kidney", "renal", "renal cortex", "renal medulla", "glomerulus",
            "ureter", "urinary bladder", "bladder", "urethra",
        ],
        "nervous system": [
            "brain", "cerebral", "cerebellum", "spinal cord", "meninges",
            "dura", "cortex", "neuron", "glial", "astrocyte", "petrous", "skull",
        ],
        "female reproductive system": [
            "uterus", "endometrium", "myometrium", "cervix", "ovary", "ovarian",
            "fallopian tube", "vagina", "vulva", "placenta", "breast", "mammary",
        ],
        "endocrine system": [
            "thyroid", "parathyroid", "adrenal", "adrenal cortex", "pituitary",
            "islets of langerhans", "pineal gland",
        ],
        "hematologic system": [
            "spleen", "splenic", "bone marrow", "lymph node", "lymphatic",
            "thymus", "tonsil", "blood", "leukocyte",
        ],
        "integumentary and musculoskeletal system": [
            "skin", "dermal", "epidermis", "bone", "joints", "cartilage",
            "skeletal muscle", "extremities", "soft tissue", "calvarium",
        ],
    }

    SYNONYM_MAP: Dict[str, str] = {
        "gastric": "stomach",
        "hepatic": "liver",
        "pulmonary": "lung",
        "renal": "kidney",
        "cardiac": "heart",
        "cerebral": "brain",
        "dermal": "skin",
        "splenic": "spleen",
        "ovarian": "ovary",
        "mammary": "breast",
        "colorectal": "colon",
        "pancreatic": "pancreas",
        "bronchial": "bronchus",
        "calvarium": "skull",
    }

    def __init__(self, custom_ontology_path: Optional[str] = None):
        self.organ_to_systems: Dict[str, Set[str]] = defaultdict(set)
        for system, organs in self.SYSTEM_TO_ORGANS.items():
            for organ in organs:
                self.organ_to_systems[organ].add(system)

        if custom_ontology_path and not Path(custom_ontology_path).exists():
            # A configured-but-missing ontology is almost certainly a mistake
            # (typo'd path); surface it instead of silently using the default.
            import logging

            logging.getLogger("pathvqa.data").warning(
                f"Custom ontology path does not exist, using built-in ontology: {custom_ontology_path}"
            )
        if custom_ontology_path and Path(custom_ontology_path).exists():
            with open(custom_ontology_path) as f:
                custom_data = json.load(f)
                for system, organs in custom_data.get("systems", {}).items():
                    for organ in organs:
                        self.organ_to_systems[organ].add(system)

    def normalize_term(self, term: str) -> str:
        """Normalize an anatomical term via synonym lookup."""
        t = term.lower().strip()
        return self.SYNONYM_MAP.get(t, t)

    def get_parent_systems(self, organ_or_structure: str) -> List[str]:
        """Return parent anatomical system(s) for an organ."""
        norm = self.normalize_term(organ_or_structure)
        if norm in self.SYSTEM_TO_ORGANS:
            return [norm]
        return sorted(self.organ_to_systems.get(norm, set()))

    def are_related(self, entity1: str, entity2: str) -> bool:
        """Check if two entities share an organ or system relationship."""
        e1, e2 = self.normalize_term(entity1), self.normalize_term(entity2)
        if e1 == e2:
            return True
        sys1 = set(self.get_parent_systems(e1))
        sys2 = set(self.get_parent_systems(e2))
        if sys1 & sys2:
            return True
        # Direct system ↔ member-organ relation (e.g. "lung" ∈ "respiratory system").
        if e1 in self.SYSTEM_TO_ORGANS and e2 in self.SYSTEM_TO_ORGANS[e1]:
            return True
        if e2 in self.SYSTEM_TO_ORGANS and e1 in self.SYSTEM_TO_ORGANS[e2]:
            return True
        return False


def _ordered_unique(items: List[str]) -> List[str]:
    """Deduplicate while preserving first-seen order.

    Region indices must follow the answer-vocabulary insertion order used by
    ``build_region_names`` at training time; ``sorted()`` would permute them
    and misalign adapter-generated checkpoints.
    """
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class DatasetAdapter:
    """Base class for dataset-specific region discovery."""

    def __init__(self, ontology: Optional[AnatomicalOntology] = None):
        self.ontology = ontology or AnatomicalOntology()

    def discover_regions(self, answer_vocab: List[str]) -> List[str]:
        """Discover anatomical region names from the answer vocabulary."""
        raise NotImplementedError

    def get_coverage_report(self, answer_vocab: List[str], discovered_regions: List[str]) -> Dict[str, Any]:
        """Return coverage stats and unlinked answer samples."""
        linked = set()
        unlinked = []
        for answer in answer_vocab:
            a_lower = answer.lower().strip()
            if answer in discovered_regions or self.ontology.get_parent_systems(a_lower):
                linked.add(answer)
            else:
                unlinked.append(answer)
        return {
            "total_answers": len(answer_vocab),
            "linked_count": len(linked),
            "unlinked_count": len(unlinked),
            "coverage": len(linked) / max(1, len(answer_vocab)),
            "unlinked_samples": unlinked[:20],
        }


class PathVQAAdapter(DatasetAdapter):
    """PathVQA adapter with ontology-grounded region discovery."""

    REGION_PREFIXES = [
        "gastrointestinal", "cardiovascular", "hematologic", "endocrine",
        "female reproductive", "nervous", "respiratory",
        "urinary", "hepatobiliary",
    ]

    @staticmethod
    def _simple_regions() -> List[str]:
        """Production region list (kept identical to the executor's source of truth).

        Training derives region names from
        ``pathvqa.symbolic.executor.SIMPLE_REGIONS``; drifting here would permute
        region indices relative to trained checkpoints. Imported lazily because
        the executor imports this module for ``AnatomicalOntology``.
        """
        from pathvqa.symbolic.executor import SIMPLE_REGIONS as _prod

        return list(_prod)

    def __init__(self, umls_mapping_path: Optional[str] = None, ontology: Optional[AnatomicalOntology] = None):
        super().__init__(ontology)
        self.umls_mapping: Dict[str, str] = {}
        if umls_mapping_path and not Path(umls_mapping_path).exists():
            import logging

            logging.getLogger("pathvqa.data").warning(
                f"UMLS mapping path does not exist, continuing without UMLS aliases: {umls_mapping_path}"
            )
        if umls_mapping_path and Path(umls_mapping_path).exists():
            with open(umls_mapping_path) as f:
                self.umls_mapping = json.load(f)

    def _is_region_answer(self, answer: str) -> bool:
        a = answer.lower().strip()
        for prefix in self.REGION_PREFIXES:
            if a.startswith(prefix):
                return True
        simple_regions = self._simple_regions()
        if a in simple_regions:
            return True
        for region in simple_regions:
            if a.startswith(region + ","):
                return True
        if bool(self.ontology.get_parent_systems(a)):
            return True
        return False

    def discover_regions(self, answer_vocab: List[str]) -> List[str]:
        regions = []
        for answer in answer_vocab:
            a = answer.lower().strip()
            if self._is_region_answer(a) and a not in ("yes", "no"):
                regions.append(answer)
        return _ordered_unique(regions)


class VQARADAdapter(DatasetAdapter):
    """VQA-RAD radiology dataset adapter."""

    RAD_REGIONS = [
        "chest", "head", "neck", "abdomen", "pelvis", "spine", "lung",
        "brain", "heart", "liver", "kidney", "bone", "pleura", "mediastinum",
    ]

    def discover_regions(self, answer_vocab: List[str]) -> List[str]:
        regions = []
        for answer in answer_vocab:
            a = answer.lower().strip()
            if a in self.RAD_REGIONS or any(a.startswith(r) for r in self.RAD_REGIONS):
                regions.append(answer)
        return _ordered_unique(regions)


class SLAKEAdapter(DatasetAdapter):
    """SLAKE bilingual medical VQA adapter."""

    SLAKE_REGIONS = [
        "brain", "lung", "heart", "liver", "kidney", "spleen", "stomach",
        "colon", "pancreas", "spine", "rib", "trachea", "mediastinum",
    ]

    def discover_regions(self, answer_vocab: List[str]) -> List[str]:
        regions = []
        for answer in answer_vocab:
            a = answer.lower().strip()
            if a in self.SLAKE_REGIONS or any(a.startswith(r) for r in self.SLAKE_REGIONS):
                regions.append(answer)
        return _ordered_unique(regions)


if __name__ == "__main__":
    ontology = AnatomicalOntology()
    print("Testing AnatomicalOntology...")
    print(f"Parents for 'renal cortex': {ontology.get_parent_systems('renal cortex')}")
    print(f"Parents for 'gastric': {ontology.get_parent_systems('gastric')}")
    print(f"Are 'gastric' and 'colon' related? {ontology.are_related('gastric', 'colon')}")

    adapter = PathVQAAdapter(ontology=ontology)
    dummy_vocab = ["gastrointestinal system", "lung", "renal cortex", "red", "yes", "no", "malignant"]
    discovered = adapter.discover_regions(dummy_vocab)
    print(f"Discovered regions: {discovered}")
    report = adapter.get_coverage_report(dummy_vocab, discovered)
    print(f"Coverage: {report['coverage']:.2%}")
    print("Dataset adapter test passed!")
