"""Visual perception module: ResNet backbone + object proposal + spatial encoding."""

from typing import Dict
import torch
import torch.nn as nn
import torchvision


class SimpleObjectDetector(nn.Module):
    """Extracts top-k object features from images using a CNN backbone.

    Uses a pretrained ResNet as backbone, then predicts an objectness score
    per spatial location to select the top-k most salient regions. Each region
    is encoded with visual features + spatial coordinates (cx, cy, w, h).
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 num_object_features: int = 512, max_objects: int = 10,
                 spatial_feat_dim: int = 128):
        super().__init__()
        self.num_object_features = num_object_features
        self.max_objects = max_objects

        # Load ResNet backbone, removing the final pooling + FC layers
        if backbone == "resnet50":
            resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        elif backbone == "resnet101":
            resnet = torchvision.models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.feature_dim = 2048
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Proposal network: predicts which spatial locations are most salient
        self.proposal_network = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.objectness = nn.Conv2d(256, 1, kernel_size=1)

        # Project backbone features per proposal
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, num_object_features),
            nn.ReLU(),
            nn.LayerNorm(num_object_features),
        )
        # Encode spatial box coordinates (cx, cy, w, h) for each proposal
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, spatial_feat_dim),
            nn.ReLU(),
            nn.Linear(spatial_feat_dim, spatial_feat_dim),
        )
        # Fuse visual features with spatial encoding
        self.spatial_projector = nn.Linear(num_object_features + spatial_feat_dim, num_object_features)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract top-k object features and masks.

        Returns:
            features: (batch, max_objects, num_object_features) tensor
            mask: (batch, max_objects) boolean mask of valid objects
        """
        batch_size = images.size(0)
        device = images.device
        feature_maps = self.backbone(images)
        proposals = self.proposal_network(feature_maps)
        objectness = self.objectness(proposals)

        b, _, h, w = objectness.shape
        objectness_flat = objectness.view(batch_size, -1)
        k = min(self.max_objects, h * w)
        _, top_k_indices = torch.topk(objectness_flat, k, dim=1)
        top_k_y = top_k_indices // w
        top_k_x = top_k_indices % w

        all_features = torch.zeros(batch_size, self.max_objects, self.num_object_features, device=device)
        all_masks = torch.zeros(batch_size, self.max_objects, dtype=torch.bool, device=device)

        # Vectorized gather: extract features at top-k locations for all samples at once
        b_idx = torch.arange(batch_size, device=device)[:, None, None]
        c_idx = torch.arange(self.feature_dim, device=device)[None, :, None]
        obj_feats = feature_maps[b_idx, c_idx, top_k_y[:, None, :], top_k_x[:, None, :]]
        obj_feats = obj_feats.permute(0, 2, 1).contiguous()
        obj_feats = self.feature_projection(obj_feats)

        boxes = torch.stack([
            top_k_x.float() / w, top_k_y.float() / h,
            torch.ones_like(top_k_x, dtype=torch.float) / w,
            torch.ones_like(top_k_y, dtype=torch.float) / h,
        ], dim=-1)
        spatial = self.spatial_encoder(boxes)
        combined = torch.cat([obj_feats, spatial], dim=-1)
        obj_feats = self.spatial_projector(combined)
        all_features[:, :k] = obj_feats
        all_masks[:, :k] = True

        return {"features": all_features, "mask": all_masks}


if __name__ == "__main__":
    print("Testing SimpleObjectDetector...")
    model = SimpleObjectDetector(num_object_features=128, max_objects=5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randn(2, 3, 320, 240)
    outputs = model(dummy)
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Mask shape: {outputs['mask'].shape}")
    print("Visual encoder test passed!")
