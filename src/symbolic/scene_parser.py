"""Neural bridge from attended visual features to symbolic scene logits.

Predicts: region logits, object presence per region, and attribute logits
(color 17, shape 9, size 11).
"""

import torch
import torch.nn as nn
from typing import Dict


COLOR_VALUES = ["yellow", "red", "grey", "white", "blue", "pale", "black",
                "tan", "pink", "dark", "green", "gray", "purple", "clear",
                "brown", "whitish", "yellowish"]
SHAPE_VALUES = ["irregular", "tubular", "round", "spindle", "oval",
                "elongated", "stellate", "polygonal", "spherical"]
SIZE_VALUES = ["large", "small", "enlarged", "marked", "gross", "massive",
               "microscopic", "moderate", "minimal", "tiny", "medium"]
DENSITY_VALUES = ["sparse", "moderate", "dense", "crowded"]


class SceneParser(nn.Module):
    """Projects attended features to region, object-presence, and attribute logits."""

    def __init__(self, visual_dim: int, num_regions: int):
        super().__init__()
        self.num_regions = num_regions

        # Region classifier: which anatomical region is depicted
        self.region_classifier = nn.Linear(visual_dim, num_regions)
        # Object presence: sigmoid scores per region
        self.object_classifier = nn.Linear(visual_dim, num_regions)

        # Attribute heads
        self.color_classifier = nn.Linear(visual_dim, len(COLOR_VALUES))
        self.shape_classifier = nn.Linear(visual_dim, len(SHAPE_VALUES))
        self.size_classifier = nn.Linear(visual_dim, len(SIZE_VALUES))
        # Nuclei density head (domain-specific for histopathology)
        self.density_classifier = nn.Linear(visual_dim, len(DENSITY_VALUES))

    def forward(self, attended_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict scene attributes from attended features."""
        region_logits = self.region_classifier(attended_features)
        object_logits = torch.sigmoid(self.object_classifier(attended_features))
        color_logits = self.color_classifier(attended_features)
        shape_logits = self.shape_classifier(attended_features)
        size_logits = self.size_classifier(attended_features)
        density_logits = self.density_classifier(attended_features)
        return {
            "region_logits": region_logits,
            "object_presence": object_logits,
            "color_logits": color_logits,
            "shape_logits": shape_logits,
            "size_logits": size_logits,
            "density_logits": density_logits,
        }


if __name__ == "__main__":
    print("Testing SceneParser...")
    parser = SceneParser(visual_dim=512, num_regions=10)
    dummy = torch.randn(4, 512)
    out = parser(dummy)
    print(f"Region logits: {out['region_logits'].shape}")
    print(f"Object presence: {out['object_presence'].shape}")
    print(f"Color logits: {out['color_logits'].shape}")
    print(f"Shape logits: {out['shape_logits'].shape}")
    print(f"Size logits: {out['size_logits'].shape}")
    print(f"Num colors={len(COLOR_VALUES)}, shapes={len(SHAPE_VALUES)}, sizes={len(SIZE_VALUES)}")
    print("SceneParser test passed!")
