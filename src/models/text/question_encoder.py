"""Question encoding module: bidirectional LSTM + vocabulary management."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import json
from collections import Counter


class QuestionVocabulary:
    """Token-level vocabulary for question text.

    Manages word-to-index mapping with special tokens for
    padding, start, end, and unknown words. Supports building
    from a list of question strings and save/load to JSON.
    """

    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.word_count = Counter()

    def build_from_questions(self, questions: List[str], min_count: int = 1):
        """Build vocabulary from a list of question strings."""
        for question in questions:
            for word in question.lower().split():
                self.word_count[word] += 1
        for word, count in self.word_count.items():
            if count >= min_count and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence: str) -> List[int]:
        """Convert question string to token index list."""
        return [self.word2idx.get(word.lower(), self.word2idx['<UNK>'])
                for word in sentence.split()]

    def decode(self, indices: List[int]) -> str:
        """Convert token index list back to question string."""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        """Serialize vocabulary to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {str(k): v for k, v in self.idx2word.items()},
                'word_count': dict(self.word_count),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Deserialize vocabulary from JSON."""
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.word_count = Counter(data['word_count'])
        return vocab


class QuestionEncoder(nn.Module):
    """Bidirectional LSTM that encodes questions into fixed-size vectors."""

    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        # Projects bidirectional output (hidden_dim*2) to hidden_dim
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, question_indices: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode questions into fixed-size state vectors.

        Args:
            question_indices: (batch, seq_len) padded token indices
            lengths: (batch,) true lengths for pack_padded_sequence

        Returns:
            question_state: (batch, hidden_dim) final question encoding
        """
        embedded = self.dropout(self.embedding(question_indices))

        if lengths is not None:
            # Pack to skip padding for efficiency
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        outputs, (hidden, cell) = self.rnn(embedded)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Concatenate last-layer forward and backward hidden states
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        final_state = self.projection(final_hidden)

        return {
            'question_state': final_state,
        }


if __name__ == "__main__":
    print("Testing QuestionEncoder...")
    model = QuestionEncoder(vocab_size=100, embedding_dim=128, hidden_dim=256)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randint(0, 100, (4, 10))
    lengths = torch.tensor([8, 10, 6, 9])
    outputs = model(dummy, lengths)
    print(f"State shape: {outputs['question_state'].shape}")
    print("Question encoder test passed!")
