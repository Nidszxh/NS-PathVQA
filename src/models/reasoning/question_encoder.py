"""
Question Encoder Module
Encodes natural language questions and generates programs
Save as: src/models/reasoning/question_encoder.py
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter


class QuestionVocabulary:
    """Vocabulary for questions and programs"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.word_count = Counter()
        
    def add_sentence(self, sentence: str):
        """Add words from sentence to vocabulary"""
        for word in sentence.lower().split():
            self.word_count[word] += 1
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def build_from_questions(self, questions: List[str], min_count: int = 1):
        """Build vocabulary from list of questions"""
        # Count words
        for question in questions:
            for word in question.lower().split():
                self.word_count[word] += 1
        
        # Add words that meet threshold
        for word, count in self.word_count.items():
            if count >= min_count and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, sentence: str) -> List[int]:
        """Convert sentence to indices"""
        return [self.word2idx.get(word.lower(), self.word2idx['<UNK>']) 
                for word in sentence.split()]
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices to sentence"""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {str(k): v for k, v in self.idx2word.items()},
                'word_count': dict(self.word_count)
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file"""
        vocab = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.word_count = Counter(data['word_count'])
        return vocab


class ProgramVocabulary:
    """Vocabulary for program operations"""
    
    def __init__(self):
        self.op2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2}
        self.idx2op = {0: '<PAD>', 1: '<START>', 2: '<END>'}
        
        # Add CLEVR operations
        operations = [
            # Filtering
            'filter_color', 'filter_shape', 'filter_material', 'filter_size',
            # Query
            'query_color', 'query_shape', 'query_material', 'query_size',
            # Comparison
            'same_color', 'same_shape', 'same_material', 'same_size',
            'equal_integer', 'less_than', 'greater_than',
            # Counting
            'count', 'exist', 'unique',
            # Spatial
            'relate', 'intersect', 'union',
            # Scene
            'scene'
        ]
        
        # Add attributes as operations
        attributes = [
            # Colors
            'gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow',
            # Shapes
            'cube', 'sphere', 'cylinder',
            # Materials
            'rubber', 'metal',
            # Sizes
            'small', 'large',
            # Relations
            'left', 'right', 'behind', 'front'
        ]
        
        for op in operations + attributes:
            if op not in self.op2idx:
                idx = len(self.op2idx)
                self.op2idx[op] = idx
                self.idx2op[idx] = op
    
    def encode_program(self, program: List[Dict]) -> List[int]:
        """Encode program to indices"""
        indices = [self.op2idx['<START>']]
        
        for step in program:
            op_type = step['type']
            if op_type in self.op2idx:
                indices.append(self.op2idx[op_type])
            
            # Add value inputs if present
            if 'value_inputs' in step and step['value_inputs']:
                for val in step['value_inputs']:
                    if val in self.op2idx:
                        indices.append(self.op2idx[val])
        
        indices.append(self.op2idx['<END>'])
        return indices
    
    def __len__(self):
        return len(self.op2idx)


class QuestionEncoder(nn.Module):
    """
    Encodes questions using LSTM/GRU
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        encoder_type: str = 'lstm'
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN encoder
        if encoder_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        elif encoder_type == 'gru':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Project bidirectional output
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        question_indices: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode questions
        
        Args:
            question_indices: [B, max_len] tensor of word indices
            lengths: [B] tensor of actual sequence lengths
        
        Returns:
            encoded: [B, max_len, hidden_dim] encoded sequence
            final_state: [B, hidden_dim] final hidden state
        """
        batch_size = question_indices.size(0)
        
        # Embed words
        embedded = self.embedding(question_indices)  # [B, max_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Pack sequence if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Encode with RNN
        if self.encoder_type == 'lstm':
            outputs, (hidden, cell) = self.rnn(embedded)
        else:
            outputs, hidden = self.rnn(embedded)
        
        # Unpack if packed
        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Project bidirectional outputs
        outputs = self.projection(outputs)  # [B, max_len, hidden_dim]
        
        # Get final state (concatenate forward and backward)
        if self.encoder_type == 'lstm':
            # hidden is [num_layers*2, B, hidden_dim]
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # [B, hidden_dim*2]
        else:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        final_state = self.projection(final_hidden)  # [B, hidden_dim]
        
        return outputs, final_state


class ProgramGenerator(nn.Module):
    """
    Generates programs from question encodings
    Uses sequence-to-sequence architecture
    """
    
    def __init__(
        self,
        question_dim: int,
        program_vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.program_vocab_size = program_vocab_size
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTM(
            program_vocab_size + question_dim,  # Input: embedded program + question context
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, program_vocab_size)
        
        # Attention over question
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        question_encoding: torch.Tensor,
        question_state: torch.Tensor,
        program_indices: Optional[torch.Tensor] = None,
        max_length: int = 27
    ) -> torch.Tensor:
        """
        Generate program
        
        Args:
            question_encoding: [B, question_len, hidden_dim]
            question_state: [B, hidden_dim]
            program_indices: [B, program_len] (for teacher forcing)
            max_length: Maximum program length
        
        Returns:
            logits: [B, program_len, program_vocab_size]
        """
        batch_size = question_encoding.size(0)
        device = question_encoding.device
        
        # Initialize hidden state from question
        hidden = question_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        
        # Start token
        current_input = torch.zeros(batch_size, 1, self.program_vocab_size, device=device)
        current_input[:, 0, 1] = 1  # <START> token (index 1)
        
        outputs = []
        
        if program_indices is not None:
            # Teacher forcing mode
            seq_len = program_indices.size(1)
            for t in range(seq_len - 1):  # -1 because we don't predict after <END>
                # One-hot encode current token
                current_token_onehot = torch.zeros(
                    batch_size, self.program_vocab_size, device=device
                )
                current_token_onehot.scatter_(1, program_indices[:, t:t+1], 1)
                
                # Concatenate with question context
                decoder_input = torch.cat([
                    current_token_onehot.unsqueeze(1),
                    question_state.unsqueeze(1)
                ], dim=-1)
                
                # Decode step
                output, (hidden, cell) = self.decoder_rnn(decoder_input, (hidden, cell))
                
                # Attention over question
                attended, _ = self.attention(output, question_encoding, question_encoding)
                output = output + attended
                
                # Project to vocabulary
                logits = self.output_projection(output)  # [B, 1, vocab_size]
                outputs.append(logits)
        else:
            # Inference mode (autoregressive)
            for t in range(max_length):
                # Concatenate with question context
                decoder_input = torch.cat([
                    current_input,
                    question_state.unsqueeze(1)
                ], dim=-1)
                
                # Decode step
                output, (hidden, cell) = self.decoder_rnn(decoder_input, (hidden, cell))
                
                # Attention over question
                attended, _ = self.attention(output, question_encoding, question_encoding)
                output = output + attended
                
                # Project to vocabulary
                logits = self.output_projection(output)
                outputs.append(logits)
                
                # Get next input (greedy for now)
                next_token = logits.argmax(dim=-1)  # [B, 1]
                current_input = torch.zeros(batch_size, 1, self.program_vocab_size, device=device)
                current_input.scatter_(2, next_token.unsqueeze(-1), 1)
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # [B, seq_len, vocab_size]
        
        return outputs


class QuestionProgramEncoder(nn.Module):
    """
    Complete question processing module
    Combines question encoding and program generation
    """
    
    def __init__(
        self,
        question_vocab_size: int,
        program_vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        encoder_type: str = 'lstm'
    ):
        super().__init__()
        
        self.question_encoder = QuestionEncoder(
            vocab_size=question_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            encoder_type=encoder_type
        )
        
        self.program_generator = ProgramGenerator(
            question_dim=hidden_dim,
            program_vocab_size=program_vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(
        self,
        question_indices: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        program_indices: Optional[torch.Tensor] = None,
        max_program_length: int = 27
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            question_indices: [B, max_len] question token indices
            lengths: [B] actual lengths
            program_indices: [B, program_len] for teacher forcing
            max_program_length: max length for generation
        
        Returns:
            Dictionary with:
                - question_encoding: [B, max_len, hidden_dim]
                - question_state: [B, hidden_dim]
                - program_logits: [B, program_len, program_vocab_size]
        """
        # Encode question
        question_encoding, question_state = self.question_encoder(
            question_indices, lengths
        )
        
        # Generate program
        program_logits = self.program_generator(
            question_encoding,
            question_state,
            program_indices,
            max_program_length
        )
        
        return {
            'question_encoding': question_encoding,
            'question_state': question_state,
            'program_logits': program_logits
        }


# Testing
if __name__ == "__main__":
    print("Testing Question Encoder...")
    print("=" * 60)
    
    # Create vocabularies
    print("Creating vocabularies...")
    question_vocab = QuestionVocabulary()
    questions = [
        "What is the color of the cube?",
        "How many red spheres are there?",
        "Is there a large metal cylinder?"
    ]
    question_vocab.build_from_questions(questions)
    
    program_vocab = ProgramVocabulary()
    
    print(f"Question vocab size: {len(question_vocab)}")
    print(f"Program vocab size: {len(program_vocab)}")
    
    # Build model
    model = QuestionProgramEncoder(
        question_vocab_size=len(question_vocab),
        program_vocab_size=len(program_vocab),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data
    batch_size = 4
    max_len = 10
    dummy_questions = torch.randint(0, len(question_vocab), (batch_size, max_len))
    dummy_lengths = torch.tensor([8, 10, 6, 9])
    
    print(f"\nTesting forward pass...")
    outputs = model(dummy_questions, dummy_lengths)
    
    print(f"\nOutput shapes:")
    print(f"  Question encoding: {outputs['question_encoding'].shape}")
    print(f"  Question state: {outputs['question_state'].shape}")
    print(f"  Program logits: {outputs['program_logits'].shape}")
    
    print("\n✓ Question encoder test complete!")