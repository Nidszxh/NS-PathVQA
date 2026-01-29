"""Visual encoder: frozen CLIP/PLIP ViT-B/32 + LoRA → patch tokens → projection."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from transformers import CLIPVisionModel


class CLIPViTEncoder(nn.Module):
    """Frozen CLIP/PLIP ViT-B/32 + LoRA encoder. Only LoRA adapters and projection are trainable."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_object_features: int = 512,
        num_objects: int = 49,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Tuple[str, ...] = (
            "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"
        ),
    ):
        super().__init__()
        self.num_object_features = num_object_features
        self.num_objects = num_objects
        self.model_name = model_name

        # Frozen vision tower: images → (B, 1+num_patches, hidden_size)
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # LoRA adapters
        self.backbone = inject_adapter_in_model(
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                bias="none",
            ),
            self.backbone,
        )

        # Project hidden_size (768) → num_object_features (512)
        hidden = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden, num_object_features)

    def enable_gradient_checkpointing(self) -> None:
        """Enable HF gradient checkpointing on the vision tower."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(self, images: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Encode images into patch-token features. Returns 'features' (B, N, D) and 'mask' (B, N)."""
        batch_size = images.size(0)
        device = images.device
        out = self.backbone(pixel_values=images)
        # Drop CLS token → (B, num_patches, hidden) → take first num_objects patches → project to 512-d
        patches = out.last_hidden_state[:, 1:, :]
        patches = patches[:, : self.num_objects, :]
        features = self.projection(patches)
        mask = torch.ones(batch_size, features.size(1), dtype=torch.bool, device=device)
        return {"features": features, "mask": mask}


class MultiScaleVisualEncoder(CLIPViTEncoder):
    """Dual-resolution encoder: global 224x224 + high-res quadrant crops, fused via learned gate."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_object_features: int = 512,
        num_objects: int = 49,
        num_crops: int = 4,
        **kwargs,
    ):
        super().__init__(model_name=model_name, num_object_features=num_object_features, num_objects=num_objects, **kwargs)
        self.num_crops = num_crops
        # Fusion gate between global and local features
        self.scale_fusion = nn.Sequential(
            nn.Linear(num_object_features * 2, num_object_features),
            nn.LayerNorm(num_object_features),
            nn.GELU(),
            nn.Linear(num_object_features, num_object_features),
        )

    def _extract_crops(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 4 quadrant crops, each resized to (H, W)."""
        b, c, h, w = images.shape
        h_half, w_half = h // 2, w // 2
        q1 = images[:, :, :h_half, :w_half]
        q2 = images[:, :, :h_half, w_half:]
        q3 = images[:, :, h_half:, :w_half]
        q4 = images[:, :, h_half:, w_half:]
        crops = [F.interpolate(q, size=(h, w), mode="bilinear", align_corners=False) for q in [q1, q2, q3, q4]]
        return torch.stack(crops, dim=1)  # (B, 4, C, H, W)

    def forward(self, images: torch.Tensor, use_multiscale: bool = False) -> Dict[str, torch.Tensor]:
        global_out = super().forward(images)
        if not use_multiscale:
            return global_out

        # Multi-scale crop encoding
        crops = self._extract_crops(images)  # (B, 4, 3, 224, 224)
        b, n_crops, c, h, w = crops.shape
        crops_flat = crops.view(b * n_crops, c, h, w)
        crop_feats = super().forward(crops_flat)["features"]  # (B*4, 49, 512)
        crop_feats = crop_feats.view(b, n_crops, self.num_objects, self.num_object_features)
        local_summary = crop_feats.mean(dim=1)  # (B, 49, 512)

        # Gate: sigmoid(MLP([global ‖ local])) per token → blended features
        fused = self.scale_fusion(torch.cat([global_out["features"], local_summary], dim=-1))
        return {"features": fused, "mask": global_out["mask"]}


if __name__ == "__main__":
    print("Testing CLIPViTEncoder and MultiScaleVisualEncoder...")
    model = CLIPViTEncoder()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"(trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Global features shape: {out['features'].shape}")

    ms_model = MultiScaleVisualEncoder()
    ms_out = ms_model(dummy, use_multiscale=True)
    print(f"Multi-scale features shape: {ms_out['features'].shape}")
    print("Visual encoder test passed!")

    print("\nTesting PLIP encoder...")
    plip_model = CLIPViTEncoder(model_name="vinid/plip")
    print(f"PLIP parameters: {sum(p.numel() for p in plip_model.parameters()):,} "
          f"(trainable: {sum(p.numel() for p in plip_model.parameters() if p.requires_grad):,})")
    plip_out = plip_model(dummy)
    print(f"PLIP features shape: {plip_out['features'].shape}")
    assert plip_out["features"].shape == (2, 49, 512)
    print("PLIP encoder test passed!")
