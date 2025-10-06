"""
Integrated Neuro-Symbolic Visual Reasoning Model
Combines visual perception, question encoding, and symbolic reasoning
Save as: src/models/neuro_symbolic_vqa.py
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.visual.visual_encoder import SimpleObjectDetector
from models.reasoning.question_encoder import QuestionProgramEncoder, QuestionVocabulary, ProgramVocabulary
from modules.executor import AttributePredictor, ProgramExecutor, SceneRepresentation


class NeuroSymbolicVQA(nn.Module):
    """
    Complete Neuro-Symbolic Visual Question Answering System
    
    Architecture:
    1. Visual Encoder: Extracts object features from images
    2. Attribute Predictor: Predicts object attributes
    3. Question Encoder: Encodes questions
    4. Program Generator: Generates executable programs
    5. Program Executor: Executes programs on scene representations
    """
    
    def __init__(
        self,
        question_vocab_size: int,
        program_vocab_size: int,
        answer_vocab_size: int,
        visual_feature_dim: int = 512,
        question_embedding_dim: int = 256,
        question_hidden_dim: int = 512,
        max_objects: int = 10,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.device = device
        self.max_objects = max_objects
        self.answer_vocab_size = answer_vocab_size
        
        # 1. Visual Encoder
        self.visual_encoder = SimpleObjectDetector(
            backbone='resnet50',
            pretrained=True,
            num_object_features=visual_feature_dim,
            max_objects=max_objects
        )
        
        # 2. Attribute Predictor
        self.attribute_predictor = AttributePredictor(
            feature_dim=visual_feature_dim,
            num_colors=8,
            num_shapes=3,
            num_materials=2,
            num_sizes=2
        )
        
        # 3. Question Encoder + Program Generator
        self.question_program_encoder = QuestionProgramEncoder(
            question_vocab_size=question_vocab_size,
            program_vocab_size=program_vocab_size,
            embedding_dim=question_embedding_dim,
            hidden_dim=question_hidden_dim,
            num_layers=2,
            dropout=0.3
        )
        
        # 4. Answer Predictor (for end-to-end training)
        self.answer_predictor = nn.Sequential(
            nn.Linear(question_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, answer_vocab_size)
        )
        
        # 5. Program Executor (not a nn.Module, created separately)
        self.executor = None  # Will be initialized after moving to device
        
        self.to(device)
    
    def initialize_executor(self):
        """Initialize program executor after model is on device"""
        self.executor = ProgramExecutor(
            module_dim=512,
            attribute_predictor=self.attribute_predictor,
            device=self.device
        )
    
    def forward(
        self,
        images: torch.Tensor,
        question_indices: torch.Tensor,
        question_lengths: Optional[torch.Tensor] = None,
        program_indices: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            images: [B, 3, H, W] input images
            question_indices: [B, max_len] question token indices
            question_lengths: [B] actual question lengths
            program_indices: [B, program_len] ground truth programs (for training)
            mode: 'train' or 'eval'
        
        Returns:
            Dictionary containing:
                - visual_features: Object features
                - attribute_logits: Predicted attributes
                - question_encoding: Encoded questions
                - program_logits: Generated program logits
                - answer_logits: Predicted answers
        """
        batch_size = images.size(0)
        
        # 1. Extract visual features
        visual_outputs = self.visual_encoder(images)
        object_features = visual_outputs['features']  # [B, max_objects, feature_dim]
        object_boxes = visual_outputs['boxes']  # [B, max_objects, 4]
        valid_masks = visual_outputs['valid_mask']  # [B, max_objects]
        
        # 2. Predict attributes
        attribute_logits = self.attribute_predictor(object_features)
        
        # 3. Encode question and generate program
        question_outputs = self.question_program_encoder(
            question_indices=question_indices,
            lengths=question_lengths,
            program_indices=program_indices,
            max_program_length=27
        )
        
        question_encoding = question_outputs['question_encoding']
        question_state = question_outputs['question_state']
        program_logits = question_outputs['program_logits']
        
        # 4. Predict answer (direct prediction for training)
        answer_logits = self.answer_predictor(question_state)
        
        return {
            'visual_features': object_features,
            'visual_boxes': object_boxes,
            'visual_masks': valid_masks,
            'attribute_logits': attribute_logits,
            'question_encoding': question_encoding,
            'question_state': question_state,
            'program_logits': program_logits,
            'answer_logits': answer_logits
        }
    
    def answer_question(
        self,
        image: torch.Tensor,
        question_indices: torch.Tensor,
        question_length: Optional[torch.Tensor] = None,
        use_program: bool = True
    ) -> str:
        """
        Answer a single question about an image
        
        Args:
            image: [3, H, W] single image
            question_indices: [max_len] question tokens
            question_length: actual length
            use_program: whether to use program execution or direct prediction
        
        Returns:
            answer: predicted answer string
        """
        self.eval()
        
        with torch.no_grad():
            # Add batch dimension
            image = image.unsqueeze(0)
            question_indices = question_indices.unsqueeze(0)
            if question_length is not None:
                question_length = question_length.unsqueeze(0)
            
            # Forward pass
            outputs = self.forward(
                images=image,
                question_indices=question_indices,
                question_lengths=question_length,
                mode='eval'
            )
            
            if use_program and self.executor is not None:
                # Use program execution
                # Convert program logits to program
                program_tokens = outputs['program_logits'][0].argmax(dim=-1)
                
                # Create scene representation
                scene = SceneRepresentation(
                    features=outputs['visual_features'][0],
                    boxes=outputs['visual_boxes'][0],
                    valid_mask=outputs['visual_masks'][0]
                )
                
                # Execute program (simplified - need proper program structure)
                # For now, use direct prediction
                answer_idx = outputs['answer_logits'][0].argmax().item()
                return f"answer_{answer_idx}"  # Need answer vocabulary mapping
            else:
                # Direct answer prediction
                answer_idx = outputs['answer_logits'][0].argmax().item()
                return f"answer_{answer_idx}"  # Need answer vocabulary mapping


class VQALoss(nn.Module):
    """
    Combined loss for VQA training
    """
    
    def __init__(
        self,
        attribute_weight: float = 1.0,
        program_weight: float = 1.0,
        answer_weight: float = 1.0
    ):
        super().__init__()
        
        self.attribute_weight = attribute_weight
        self.program_weight = program_weight
        self.answer_weight = answer_weight
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Attribute prediction losses (if available)
        if 'color_targets' in targets:
            attr_logits = outputs['attribute_logits']
            batch_size, num_objects = attr_logits['color_logits'].shape[:2]
            
            # Reshape for loss computation
            color_logits = attr_logits['color_logits'].view(-1, attr_logits['color_logits'].size(-1))
            shape_logits = attr_logits['shape_logits'].view(-1, attr_logits['shape_logits'].size(-1))
            material_logits = attr_logits['material_logits'].view(-1, attr_logits['material_logits'].size(-1))
            size_logits = attr_logits['size_logits'].view(-1, attr_logits['size_logits'].size(-1))
            
            color_targets = targets['color_targets'].view(-1)
            shape_targets = targets['shape_targets'].view(-1)
            material_targets = targets['material_targets'].view(-1)
            size_targets = targets['size_targets'].view(-1)
            
            losses['color_loss'] = self.ce_loss(color_logits, color_targets)
            losses['shape_loss'] = self.ce_loss(shape_logits, shape_targets)
            losses['material_loss'] = self.ce_loss(material_logits, material_targets)
            losses['size_loss'] = self.ce_loss(size_logits, size_targets)
            
            losses['attribute_loss'] = (
                losses['color_loss'] + 
                losses['shape_loss'] + 
                losses['material_loss'] + 
                losses['size_loss']
            ) / 4
        else:
            losses['attribute_loss'] = torch.tensor(0.0, device=outputs['answer_logits'].device)
        
        # Program generation loss (if available)
        if 'program_targets' in targets:
            program_logits = outputs['program_logits']
            program_targets = targets['program_targets']
            
            # Reshape
            prog_logits = program_logits.view(-1, program_logits.size(-1))
            prog_targets = program_targets.view(-1)
            
            losses['program_loss'] = self.ce_loss(prog_logits, prog_targets)
        else:
            losses['program_loss'] = torch.tensor(0.0, device=outputs['answer_logits'].device)
        
        # Answer prediction loss
        answer_logits = outputs['answer_logits']
        answer_targets = targets['answer_targets']
        
        losses['answer_loss'] = self.ce_loss(answer_logits, answer_targets)
        
        # Total loss
        losses['total_loss'] = (
            self.attribute_weight * losses['attribute_loss'] +
            self.program_weight * losses['program_loss'] +
            self.answer_weight * losses['answer_loss']
        )
        
        return losses


def build_model(config, question_vocab, program_vocab, answer_vocab):
    """
    Build model from configuration
    
    Args:
        config: Configuration object
        question_vocab: Question vocabulary
        program_vocab: Program vocabulary
        answer_vocab: Answer vocabulary
    
    Returns:
        model: Initialized model
    """
    model = NeuroSymbolicVQA(
        question_vocab_size=len(question_vocab),
        program_vocab_size=len(program_vocab),
        answer_vocab_size=len(answer_vocab),
        visual_feature_dim=config.visual.num_object_features,
        question_embedding_dim=config.question.embedding_dim,
        question_hidden_dim=config.question.hidden_dim,
        max_objects=config.visual.max_objects_per_image,
        device=config.training.device
    )
    
    # Initialize executor
    model.initialize_executor()
    
    return model


# Testing
if __name__ == "__main__":
    print("Testing Neuro-Symbolic VQA Model...")
    print("=" * 60)
    
    # Create dummy vocabularies
    from models.reasoning.question_encoder import QuestionVocabulary, ProgramVocabulary
    
    question_vocab = QuestionVocabulary()
    questions = ["What is the color of the cube?", "How many red spheres?"]
    question_vocab.build_from_questions(questions)
    
    program_vocab = ProgramVocabulary()
    
    # Simple answer vocab
    answer_vocab = {'yes': 0, 'no': 1, 'red': 2, 'blue': 3, 'cube': 4, '0': 5, '1': 6, '2': 7}
    
    # Build model
    print("\nBuilding model...")
    model = NeuroSymbolicVQA(
        question_vocab_size=len(question_vocab),
        program_vocab_size=len(program_vocab),
        answer_vocab_size=len(answer_vocab),
        visual_feature_dim=512,
        question_embedding_dim=256,
        question_hidden_dim=512,
        max_objects=10,
        device='cpu'
    )
    
    model.initialize_executor()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    max_program_len = 27
    dummy_images = torch.randn(batch_size, 3, 320, 240)
    dummy_questions = torch.randint(0, len(question_vocab), (batch_size, 10))
    dummy_lengths = torch.tensor([8, 10])
    
    outputs = model(
        images=dummy_images,
        question_indices=dummy_questions,
        question_lengths=dummy_lengths
    )
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v.shape}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    loss_fn = VQALoss()
    
    # Prepare targets
    # 1. Program targets (padded to max_program_len)
    program_targets = torch.zeros(batch_size, max_program_len, dtype=torch.long)
    program_targets[:, :10] = torch.randint(1, len(program_vocab), (batch_size, 10))
    
    # 2. Answer targets
    answer_targets = torch.randint(0, len(answer_vocab), (batch_size,))
    
    # 3. Dummy attribute targets (for attribute loss)
    color_targets = torch.randint(0, 8, (batch_size, model.max_objects))
    shape_targets = torch.randint(0, 3, (batch_size, model.max_objects))
    material_targets = torch.randint(0, 2, (batch_size, model.max_objects))
    size_targets = torch.randint(0, 2, (batch_size, model.max_objects))
    
    targets = {
        'program_targets': program_targets,
        'answer_targets': answer_targets,
        'color_targets': color_targets,
        'shape_targets': shape_targets,
        'material_targets': material_targets,
        'size_targets': size_targets
    }
    
    losses = loss_fn(outputs, targets)
    
    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✓ Model test complete!")