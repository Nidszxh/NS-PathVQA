import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Tuple
import numpy as np


class VisualEncoder(nn.Module):
    """
    Visual encoder that combines object detection and feature extraction
    
    Architecture:
    1. Backbone CNN (ResNet) for feature extraction
    2. Object detector (Faster R-CNN) for object localization
    3. Feature aggregation for each detected object
    """
    
    def __init__(
        self,
        backbone: str = "resnet101",
        pretrained: bool = True,
        num_object_features: int = 2048,
        max_objects: int = 10,
        spatial_feat_dim: int = 128,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_object_features = num_object_features
        self.max_objects = max_objects
        self.spatial_feat_dim = spatial_feat_dim
        
        # Load pretrained Faster R-CNN
        print(f"Loading Faster R-CNN with {backbone} backbone (pretrained={pretrained})...")
        self.detector = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Get backbone feature dimension
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        
        # We don't need classification head for CLEVR (all objects are similar)
        # Just use the detector to get regions and features
        
        # Spatial encoding network
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 64),  # [x, y, w, h]
            nn.ReLU(),
            nn.Linear(64, spatial_feat_dim),
            nn.ReLU()
        )
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(in_features, num_object_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combine spatial and semantic features
        self.feature_fusion = nn.Sequential(
            nn.Linear(num_object_features + spatial_feat_dim, num_object_features),
            nn.ReLU(),
            nn.LayerNorm(num_object_features)
        )
        
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for param in self.detector.backbone.parameters():
                param.requires_grad = False
    
    def extract_features(
        self, 
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract object features from images
        
        Args:
            images: [B, 3, H, W] tensor of images
        
        Returns:
            features: [B, max_objects, num_object_features] object features
            boxes: [B, max_objects, 4] bounding boxes
            valid_mask: [B, max_objects] mask indicating valid objects
        """
        batch_size = images.size(0)
        device = images.device
        
        # Set detector to eval mode for inference
        was_training = self.detector.training
        self.detector.eval()
        
        with torch.no_grad():
            # Get detections from Faster R-CNN
            detections = self.detector(images)
        
        # Restore training mode if needed
        if was_training:
            self.detector.train()
        
        # Process detections for each image
        all_features = []
        all_boxes = []
        all_masks = []
        
        for detection in detections:
            boxes = detection['boxes']  # [N, 4]
            scores = detection['scores']  # [N]
            
            # Sort by confidence and take top-k
            if len(boxes) > self.max_objects:
                top_k_indices = torch.argsort(scores, descending=True)[:self.max_objects]
                boxes = boxes[top_k_indices]
                scores = scores[top_k_indices]
            
            num_objects = len(boxes)
            
            # Extract RoI features (we'll use a simpler approach for now)
            # In production, you'd extract features from RoI pooling
            # For simplicity, we'll use the box coordinates as features
            
            # Normalize boxes to [0, 1]
            h, w = images.size(2), images.size(3)
            normalized_boxes = boxes.clone()
            normalized_boxes[:, [0, 2]] /= w
            normalized_boxes[:, [1, 3]] /= h
            
            # Convert to [x, y, w, h] format
            box_features = torch.zeros(num_objects, 4, device=device)
            if num_objects > 0:
                box_features[:, 0] = (normalized_boxes[:, 0] + normalized_boxes[:, 2]) / 2  # center x
                box_features[:, 1] = (normalized_boxes[:, 1] + normalized_boxes[:, 3]) / 2  # center y
                box_features[:, 2] = normalized_boxes[:, 2] - normalized_boxes[:, 0]  # width
                box_features[:, 3] = normalized_boxes[:, 3] - normalized_boxes[:, 1]  # height
            
            # Encode spatial features
            spatial_features = self.spatial_encoder(box_features)  # [N, spatial_feat_dim]
            
            # Create dummy semantic features (in practice, extract from RoI pooling)
            # For now, use a simple encoding based on box position and size
            semantic_features = self._encode_box_appearance(box_features)  # [N, num_object_features]
            
            # Fuse spatial and semantic features
            combined_features = torch.cat([semantic_features, spatial_features], dim=-1)
            object_features = self.feature_fusion(combined_features)  # [N, num_object_features]
            
            # Pad to max_objects
            padded_features = torch.zeros(
                self.max_objects, self.num_object_features, device=device
            )
            padded_boxes = torch.zeros(self.max_objects, 4, device=device)
            valid_mask = torch.zeros(self.max_objects, device=device, dtype=torch.bool)
            
            if num_objects > 0:
                padded_features[:num_objects] = object_features
                padded_boxes[:num_objects] = box_features
                valid_mask[:num_objects] = True
            
            all_features.append(padded_features)
            all_boxes.append(padded_boxes)
            all_masks.append(valid_mask)
        
        # Stack into batch
        features = torch.stack(all_features)  # [B, max_objects, num_object_features]
        boxes = torch.stack(all_boxes)  # [B, max_objects, 4]
        valid_mask = torch.stack(all_masks)  # [B, max_objects]
        
        return features, boxes, valid_mask
    
    def _encode_box_appearance(self, box_features: torch.Tensor) -> torch.Tensor:
        """
        Simple encoding of box appearance based on spatial properties
        This is a placeholder - in practice, use RoI pooling from backbone
        """
        # Create a simple encoding based on position and size
        # In a real implementation, this would be CNN features from RoI pooling
        
        batch_size = box_features.size(0)
        device = box_features.device
        
        # Simple MLP encoding of box features
        # This simulates appearance features that would come from RoI pooling
        appearance = torch.cat([
            box_features,
            box_features ** 2,  # Add non-linearity
            torch.sin(box_features * np.pi),
            torch.cos(box_features * np.pi)
        ], dim=-1)  # [N, 16]
        
        # Project to feature dimension
        appearance_features = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_object_features),
            nn.ReLU()
        ).to(device)(appearance)
        
        return appearance_features
    
    def forward(
        self, 
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through visual encoder
        
        Args:
            images: [B, 3, H, W] tensor of images
        
        Returns:
            Dictionary containing:
                - features: [B, max_objects, num_object_features]
                - boxes: [B, max_objects, 4]
                - valid_mask: [B, max_objects]
        """
        features, boxes, valid_mask = self.extract_features(images)
        
        return {
            'features': features,
            'boxes': boxes,
            'valid_mask': valid_mask
        }


class SimpleObjectDetector(nn.Module):
    """
    Simplified object detector for CLEVR
    Uses a CNN backbone + Region Proposal approach
    
    This is a lighter alternative to Faster R-CNN for initial testing
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_object_features: int = 2048,
        max_objects: int = 10
    ):
        super().__init__()
        
        self.num_object_features = num_object_features
        self.max_objects = max_objects
        
        # Load pretrained ResNet as backbone
        if backbone == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "resnet101":
            resnet = torchvision.models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Object proposal network
        # Uses a sliding window approach on feature maps
        self.proposal_network = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Object confidence score
        self.objectness = nn.Conv2d(256, 1, kernel_size=1)
        
        # Feature projection for each object
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, num_object_features),
            nn.ReLU(),
            nn.LayerNorm(num_object_features)
        )
        
        # Spatial encoding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W]
        
        Returns:
            Dictionary with features, boxes, and masks
        """
        batch_size = images.size(0)
        device = images.device
        
        # Extract features
        feature_maps = self.backbone(images)  # [B, feature_dim, H', W']
        
        # Get object proposals
        proposals = self.proposal_network(feature_maps)  # [B, 256, H', W']
        objectness = self.objectness(proposals)  # [B, 1, H', W']
        
        # Get top-k object locations
        b, c, h, w = objectness.shape
        objectness_flat = objectness.view(batch_size, -1)  # [B, H'*W']
        
        # Get top-k objects
        k = min(self.max_objects, h * w)
        top_k_scores, top_k_indices = torch.topk(objectness_flat, k, dim=1)
        
        # Convert indices to spatial locations
        top_k_y = top_k_indices // w
        top_k_x = top_k_indices % w
        
        # Extract features at object locations
        all_features = []
        all_boxes = []
        all_masks = []
        
        for i in range(batch_size):
            # Get features at object locations
            y_coords = top_k_y[i]
            x_coords = top_k_x[i]
            
            # Extract features from feature map
            obj_features = feature_maps[i, :, y_coords, x_coords].T  # [k, feature_dim]
            
            # Project to object feature space
            obj_features = self.feature_projection(obj_features)  # [k, num_object_features]
            
            # Create bounding boxes (normalized coordinates)
            boxes = torch.zeros(k, 4, device=device)
            boxes[:, 0] = x_coords.float() / w  # center x
            boxes[:, 1] = y_coords.float() / h  # center y
            boxes[:, 2] = 1.0 / w  # width (uniform for simplicity)
            boxes[:, 3] = 1.0 / h  # height (uniform for simplicity)
            
            # Add spatial encoding to features
            spatial_features = self.spatial_encoder(boxes)
            combined_features = torch.cat([obj_features, spatial_features], dim=-1)
            obj_features = nn.Linear(combined_features.size(-1), obj_features.size(-1)).to(device)(combined_features)

            # Pad to max_objects
            padded_features = torch.zeros(
                self.max_objects, self.num_object_features, device=device
            )
            padded_boxes = torch.zeros(self.max_objects, 4, device=device)
            valid_mask = torch.zeros(self.max_objects, device=device, dtype=torch.bool)
            
            padded_features[:k] = obj_features
            padded_boxes[:k] = boxes
            valid_mask[:k] = True
            
            all_features.append(padded_features)
            all_boxes.append(padded_boxes)
            all_masks.append(valid_mask)
        
        features = torch.stack(all_features)  # [B, max_objects, num_object_features]
        boxes = torch.stack(all_boxes)  # [B, max_objects, 4]
        valid_mask = torch.stack(all_masks)  # [B, max_objects]
        
        return {
            'features': features,
            'boxes': boxes,
            'valid_mask': valid_mask
        }


def build_visual_encoder(config) -> nn.Module:
    """
    Build visual encoder from configuration
    
    Args:
        config: Configuration object
    
    Returns:
        Visual encoder module
    """
    if config.visual.detector_type == "faster_rcnn":
        return VisualEncoder(
            backbone=config.visual.backbone,
            pretrained=config.visual.pretrained,
            num_object_features=config.visual.num_object_features,
            max_objects=config.visual.max_objects_per_image,
            spatial_feat_dim=config.visual.spatial_feat_dim,
            freeze_backbone=config.visual.freeze_backbone
        )
    elif config.visual.detector_type == "simple":
        return SimpleObjectDetector(
            backbone=config.visual.backbone,
            pretrained=config.visual.pretrained,
            num_object_features=config.visual.num_object_features,
            max_objects=config.visual.max_objects_per_image
        )
    else:
        raise ValueError(f"Unknown detector type: {config.visual.detector_type}")


# Testing
if __name__ == "__main__":
    print("Testing Visual Encoder...")
    print("=" * 60)
    
    # Create dummy config
    class DummyConfig:
        class visual:
            backbone = "resnet50"
            pretrained = True
            detector_type = "simple"  # Use simple for faster testing
            num_object_features = 512
            max_objects_per_image = 10
            spatial_feat_dim = 128
            freeze_backbone = False
    
    config = DummyConfig()
    
    # Build model
    model = build_visual_encoder(config)
    model.eval()
    
    print(f"Model built successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy images
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 320, 240)
    
    print(f"\nTesting with batch size {batch_size}...")
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"\nOutput shapes:")
    print(f"  Features: {outputs['features'].shape}")
    print(f"  Boxes: {outputs['boxes'].shape}")
    print(f"  Valid mask: {outputs['valid_mask'].shape}")
    
    print(f"\nNumber of valid objects per image:")
    for i in range(batch_size):
        num_valid = outputs['valid_mask'][i].sum().item()
        print(f"  Image {i}: {num_valid} objects")
    
    print("\n✓ Visual encoder test complete!")