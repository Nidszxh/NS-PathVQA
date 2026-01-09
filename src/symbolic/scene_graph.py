"""Scene graph dataclass: symbolic fact representation for the visual scene.

Each SceneGraph holds the predicted region, detected objects, and
attribute-value pairs, each with associated confidence scores.
These are produced by the SceneParser and consumed by the Executor
to answer questions symbolically.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SceneGraph:
    """Symbolic representation of a visual scene.

    Attributes:
        region: Predicted anatomical region (e.g. "lung", "cardiovascular system")
        region_confidence: Softmax confidence for the predicted region
        objects: List of detected object names in the scene
        object_confidences: Per-object presence confidence (sigmoid scores)
        attributes: Dict of (attr_type, attr_value) pairs discovered
        attribute_confidences: Confidence per extracted attribute
    """
    region: str
    region_confidence: float
    objects: List[str] = field(default_factory=list)
    object_confidences: Dict[str, float] = field(default_factory=dict)
    attributes: Dict[Tuple[str, str], str] = field(default_factory=dict)
    attribute_confidences: Dict[Tuple[str, str], float] = field(default_factory=dict)
