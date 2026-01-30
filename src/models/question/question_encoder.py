"""Question encoding: frozen DistilBERT + LoRA. Output is [CLS] state (768-d)."""

from typing import Dict
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModel, AutoTokenizer


class DistilBERTQuestionEncoder(nn.Module):
    """Frozen DistilBERT + LoRA question encoder. Only LoRA adapters are trainable."""

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 lora_rank: int = 16, lora_alpha: int = 32,
                 lora_target_modules=("q_lin", "k_lin", "v_lin", "out_lin")):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone = inject_adapter_in_model(
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                bias="none",
            ),
            self.backbone,
        )
        self.hidden_dim = self.backbone.config.hidden_size

    def enable_gradient_checkpointing(self) -> None:
        """Enable HF gradient checkpointing on the DistilBERT backbone."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode questions into fixed-size state vectors. Returns 'question_state' (B, 768)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token (index 0) as the question state representation
        question_state = out.last_hidden_state[:, 0, :]
        return {"question_state": question_state}


def get_question_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Return the HF tokenizer for the question encoder."""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    print("Testing DistilBERTQuestionEncoder...")
    tokenizer = get_question_tokenizer()
    model = DistilBERTQuestionEncoder()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"(trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")
    dummy_ids = torch.randint(0, tokenizer.vocab_size, (4, 10))
    dummy_mask = torch.ones(4, 10, dtype=torch.long)
    outputs = model(dummy_ids, dummy_mask)
    print(f"State shape: {outputs['question_state'].shape}")
    print("Question encoder test passed!")
