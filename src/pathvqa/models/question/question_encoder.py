"""Question encoding: frozen DistilBERT + LoRA. Output is [CLS] state (768-d)."""

from typing import Any, Tuple

import torch
import torch.nn as nn
from pathvqa.models.protocols import QuestionEncoderOutput
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModel, AutoTokenizer


class DistilBERTQuestionEncoder(nn.Module):
    """Frozen DistilBERT + LoRA question encoder. Only LoRA adapters are trainable."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Tuple[str, ...] = ("q_lin", "k_lin", "v_lin", "out_lin"),
    ):
        """Load the frozen DistilBERT tower and inject LoRA adapters."""
        super().__init__()
        try:
            self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            # Preserve first-run behavior when the model is not cached yet.
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

    def train(self, mode: bool = True) -> "DistilBERTQuestionEncoder":
        """Keep the frozen DistilBERT backbone in eval mode while training LoRA.

        Otherwise DistilBERT's dropout stays active during training but is
        disabled at evaluation, producing inconsistent question features even
        though the backbone weights are frozen.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> QuestionEncoderOutput:
        """Encode questions into fixed-size state vectors."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] is index 0 in DistilBERT's output sequence.
        question_state = out.last_hidden_state[:, 0, :]

        # Downstream fusion expects exactly (B, hidden_dim).
        batch_size = input_ids.size(0)
        assert question_state.shape == (batch_size, self.hidden_dim), (
            f"Expected question_state (B, {self.hidden_dim}), got {question_state.shape}"
        )

        return QuestionEncoderOutput(question_state=question_state)


def get_question_tokenizer(model_name: str = "distilbert-base-uncased") -> Any:
    """Return the HF tokenizer for the question encoder."""
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        # Preserve first-run behavior when tokenizer assets are not cached.
        return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    print("Testing DistilBERTQuestionEncoder...")
    tokenizer = get_question_tokenizer()
    model = DistilBERTQuestionEncoder()
    print(
        f"Parameters: {sum(p.numel() for p in model.parameters()):,} "
        f"(trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,})"
    )
    dummy_ids = torch.randint(0, tokenizer.vocab_size, (4, 10))
    dummy_mask = torch.ones(4, 10, dtype=torch.long)
    outputs = model(dummy_ids, dummy_mask)
    print(f"State shape: {outputs.question_state.shape}")
    print("Question encoder test passed!")
