import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
import numpy as np


class SceneRepresentation:
    """
    Represents a scene with objects and their attributes
    """
    
    def __init__(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        valid_mask: torch.Tensor
    ):
        """
        Args:
            features: [num_objects, feature_dim]
            boxes: [num_objects, 4] (x, y, w, h normalized)
            valid_mask: [num_objects] boolean mask
        """
        self.features = features
        self.boxes = boxes
        self.valid_mask = valid_mask
        self.num_objects = valid_mask.sum().item()
        
        # Predicted attributes (will be set by attribute predictor)
        self.colors = None
        self.shapes = None
        self.materials = None
        self.sizes = None
    
    def get_valid_objects(self):
        """Get only valid objects"""
        return self.features[self.valid_mask]
    
    def get_valid_boxes(self):
        """Get only valid boxes"""
        return self.boxes[self.valid_mask]


class AttributePredictor(nn.Module):
    """
    Predicts object attributes from features
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_colors: int = 8,
        num_shapes: int = 3,
        num_materials: int = 2,
        num_sizes: int = 2
    ):
        super().__init__()
        
        self.color_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_colors)
        )
        
        self.shape_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_shapes)
        )
        
        self.material_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_materials)
        )
        
        self.size_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_sizes)
        )
        
        # Attribute vocabularies
        self.color_vocab = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
        self.shape_vocab = ['cube', 'sphere', 'cylinder']
        self.material_vocab = ['rubber', 'metal']
        self.size_vocab = ['small', 'large']
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict attributes for objects
        
        Args:
            features: [B, num_objects, feature_dim]
        
        Returns:
            Dictionary with attribute logits
        """
        return {
            'color_logits': self.color_classifier(features),
            'shape_logits': self.shape_classifier(features),
            'material_logits': self.material_classifier(features),
            'size_logits': self.size_classifier(features)
        }
    
    def predict_attributes(self, features: torch.Tensor) -> Dict[str, List[str]]:
        """
        Predict attribute labels
        
        Args:
            features: [num_objects, feature_dim]
        
        Returns:
            Dictionary with predicted attribute strings
        """
        with torch.no_grad():
            features_batch = features.unsqueeze(0)  # [1, num_objects, feature_dim]
            logits = self.forward(features_batch)
            
            colors = [self.color_vocab[i] for i in logits['color_logits'][0].argmax(dim=-1)]
            shapes = [self.shape_vocab[i] for i in logits['shape_logits'][0].argmax(dim=-1)]
            materials = [self.material_vocab[i] for i in logits['material_logits'][0].argmax(dim=-1)]
            sizes = [self.size_vocab[i] for i in logits['size_logits'][0].argmax(dim=-1)]
            
            return {
                'colors': colors,
                'shapes': shapes,
                'materials': materials,
                'sizes': sizes
            }


class ReasoningModule(nn.Module):
    """Base class for reasoning modules"""
    
    def __init__(self, module_dim: int):
        super().__init__()
        self.module_dim = module_dim
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class FilterModule(ReasoningModule):
    """Filters objects by attribute"""
    
    def __init__(self, module_dim: int, attribute_type: str, attribute_vocab: List[str]):
        super().__init__(module_dim)
        self.attribute_type = attribute_type
        self.attribute_vocab = attribute_vocab
        self.num_values = len(attribute_vocab)
        
        # Learnable filter
        self.filter_net = nn.Sequential(
            nn.Linear(module_dim + self.num_values, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        object_features: torch.Tensor,
        attribute_logits: torch.Tensor,
        value: str
    ) -> torch.Tensor:
        """
        Filter objects by attribute value
        
        Args:
            object_features: [num_objects, module_dim]
            attribute_logits: [num_objects, num_values]
            value: attribute value to filter by
        
        Returns:
            mask: [num_objects] binary mask for filtered objects
        """
        # Create one-hot encoding for target value
        value_idx = self.attribute_vocab.index(value) if value in self.attribute_vocab else 0
        value_encoding = torch.zeros(
            object_features.size(0), self.num_values, device=object_features.device
        )
        value_encoding[:, value_idx] = 1.0
        
        # Concatenate features with target value
        combined = torch.cat([object_features, value_encoding], dim=-1)
        
        # Compute filter scores
        scores = self.filter_net(combined).squeeze(-1)  # [num_objects]
        
        # Binarize (threshold at 0.5)
        mask = (scores > 0.5).float()
        
        return mask


class QueryModule(ReasoningModule):
    """Queries attribute of objects"""
    
    def __init__(self, module_dim: int, attribute_type: str, num_values: int):
        super().__init__(module_dim)
        self.attribute_type = attribute_type
        self.num_values = num_values
        
        # Attention over objects
        self.attention = nn.Sequential(
            nn.Linear(module_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        object_features: torch.Tensor,
        attribute_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Query attribute from objects
        
        Args:
            object_features: [num_objects, module_dim]
            attribute_logits: [num_objects, num_values]
            mask: [num_objects] mask for valid/selected objects
        
        Returns:
            prediction: [num_values] attribute distribution
        """
        # Compute attention weights
        attention_logits = self.attention(object_features).squeeze(-1)  # [num_objects]
        
        # Apply mask if provided
        if mask is not None:
            attention_logits = attention_logits.masked_fill(mask == 0, -1e9)
        
        # Softmax attention
        attention_weights = torch.softmax(attention_logits, dim=0)  # [num_objects]
        
        # Weighted sum of attribute predictions
        prediction = torch.sum(
            attention_weights.unsqueeze(-1) * attribute_logits, dim=0
        )  # [num_values]
        
        return prediction


class CountModule(ReasoningModule):
    """Counts objects"""
    
    def __init__(self, module_dim: int, max_count: int = 10):
        super().__init__(module_dim)
        self.max_count = max_count
        
        # MLP for count prediction
        self.count_net = nn.Sequential(
            nn.Linear(module_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_count + 1)  # 0 to max_count
        )
    
    def forward(
        self,
        object_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> int:
        """
        Count objects
        
        Args:
            object_features: [num_objects, module_dim]
            mask: [num_objects] mask for objects to count
        
        Returns:
            count: integer count
        """
        if mask is not None:
            # Sum features of masked objects
            masked_features = object_features * mask.unsqueeze(-1)
            aggregated = masked_features.sum(dim=0)  # [module_dim]
        else:
            aggregated = object_features.mean(dim=0)  # [module_dim]
        
        # Predict count
        count_logits = self.count_net(aggregated)  # [max_count + 1]
        count = count_logits.argmax().item()
        
        return count


class ExistModule(ReasoningModule):
    """Checks if objects exist"""
    
    def __init__(self, module_dim: int):
        super().__init__(module_dim)
        
        self.exist_net = nn.Sequential(
            nn.Linear(module_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # yes/no
        )
    
    def forward(
        self,
        object_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Check if any objects exist
        
        Args:
            object_features: [num_objects, module_dim]
            mask: [num_objects] mask for objects to check
        
        Returns:
            exists: boolean
        """
        if mask is not None:
            aggregated = (object_features * mask.unsqueeze(-1)).sum(dim=0)
        else:
            aggregated = object_features.mean(dim=0)
        
        exist_logits = self.exist_net(aggregated)  # [2]
        exists = exist_logits.argmax().item() == 1
        
        return exists


class CompareModule(ReasoningModule):
    """Compares attributes or counts"""
    
    def __init__(self, module_dim: int, comparison_type: str = "greater_than"):
        super().__init__(module_dim)
        self.comparison_type = comparison_type
        
        self.compare_net = nn.Sequential(
            nn.Linear(module_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # yes/no
        )
    
    def forward(self, value1: Any, value2: Any) -> bool:
        """
        Compare two values
        
        Args:
            value1, value2: Values to compare (int, float, or tensor)
        
        Returns:
            result: boolean comparison result
        """
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # Direct numerical comparison
            if self.comparison_type == "greater_than":
                return value1 > value2
            elif self.comparison_type == "less_than":
                return value1 < value2
            elif self.comparison_type == "equal_integer":
                return value1 == value2
        
        # For tensor comparison, use neural network
        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            combined = torch.cat([value1, value2], dim=-1)
            logits = self.compare_net(combined)
            return logits.argmax().item() == 1
        
        return False


class SpatialRelationModule(ReasoningModule):
    """Filters objects by spatial relation"""
    
    def __init__(self, module_dim: int):
        super().__init__(module_dim)
        
        # Relation predictor
        self.relation_net = nn.Sequential(
            nn.Linear(module_dim * 2 + 8, 256),  # +8 for relative box coordinates
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        object_features: torch.Tensor,
        boxes: torch.Tensor,
        reference_mask: torch.Tensor,
        relation: str
    ) -> torch.Tensor:
        """
        Filter objects by spatial relation to reference objects
        
        Args:
            object_features: [num_objects, module_dim]
            boxes: [num_objects, 4] (x, y, w, h)
            reference_mask: [num_objects] mask for reference objects
            relation: 'left', 'right', 'front', 'behind'
        
        Returns:
            mask: [num_objects] mask for objects satisfying relation
        """
        num_objects = object_features.size(0)
        device = object_features.device
        
        # Get reference object (use first one for simplicity)
        ref_indices = torch.where(reference_mask > 0)[0]
        if len(ref_indices) == 0:
            return torch.zeros(num_objects, device=device)
        
        ref_idx = ref_indices[0]
        ref_features = object_features[ref_idx].unsqueeze(0).repeat(num_objects, 1)
        ref_box = boxes[ref_idx]
        
        # Compute relative positions
        relative_boxes = boxes - ref_box.unsqueeze(0)
        
        # Determine relation using heuristics
        relation_scores = torch.zeros(num_objects, device=device)
        
        if relation == "left":
            relation_scores = (relative_boxes[:, 0] < -0.1).float()
        elif relation == "right":
            relation_scores = (relative_boxes[:, 0] > 0.1).float()
        elif relation == "behind":
            relation_scores = (relative_boxes[:, 1] < -0.1).float()
        elif relation == "front":
            relation_scores = (relative_boxes[:, 1] > 0.1).float()
        
        # Exclude reference object itself
        relation_scores[ref_idx] = 0
        
        return relation_scores


class ProgramExecutor:
    """
    Executes symbolic programs on scene representations
    """
    
    def __init__(
        self,
        module_dim: int,
        attribute_predictor: AttributePredictor,
        device: str = 'cpu'
    ):
        self.module_dim = module_dim
        self.attribute_predictor = attribute_predictor
        self.device = device
        
        # Initialize modules
        self.filters = {
            'color': FilterModule(module_dim, 'color', attribute_predictor.color_vocab),
            'shape': FilterModule(module_dim, 'shape', attribute_predictor.shape_vocab),
            'material': FilterModule(module_dim, 'material', attribute_predictor.material_vocab),
            'size': FilterModule(module_dim, 'size', attribute_predictor.size_vocab)
        }
        
        self.queries = {
            'color': QueryModule(module_dim, 'color', len(attribute_predictor.color_vocab)),
            'shape': QueryModule(module_dim, 'shape', len(attribute_predictor.shape_vocab)),
            'material': QueryModule(module_dim, 'material', len(attribute_predictor.material_vocab)),
            'size': QueryModule(module_dim, 'size', len(attribute_predictor.size_vocab))
        }
        
        self.count_module = CountModule(module_dim)
        self.exist_module = ExistModule(module_dim)
        self.compare_module = CompareModule(module_dim)
        self.spatial_module = SpatialRelationModule(module_dim)
        
        # Move to device
        for module in self.filters.values():
            module.to(device)
        for module in self.queries.values():
            module.to(device)
        self.count_module.to(device)
        self.exist_module.to(device)
        self.compare_module.to(device)
        self.spatial_module.to(device)
    
    def execute(
        self,
        program: List[Dict],
        scene: SceneRepresentation
    ) -> Any:
        """
        Execute a program on a scene
        
        Args:
            program: List of program operations
            scene: Scene representation
        
        Returns:
            answer: Program execution result
        """
        # Get object features and boxes
        features = scene.get_valid_objects()
        boxes = scene.get_valid_boxes()
        
        # Predict attributes
        attributes = self.attribute_predictor.forward(features.unsqueeze(0))
        color_logits = attributes['color_logits'][0]
        shape_logits = attributes['shape_logits'][0]
        material_logits = attributes['material_logits'][0]
        size_logits = attributes['size_logits'][0]
        
        # Initialize with all objects
        current_mask = torch.ones(features.size(0), device=self.device)
        memory = []  # Stack for intermediate results
        
        # Execute each operation
        for op in program:
            op_type = op['type']
            
            if op_type == 'scene':
                # Start with all objects
                memory.append(current_mask)
            
            elif op_type.startswith('filter_'):
                attr_type = op_type.replace('filter_', '')
                value = op['value_inputs'][0] if 'value_inputs' in op else None
                
                if attr_type == 'color':
                    mask = self.filters['color'](features, color_logits, value)
                elif attr_type == 'shape':
                    mask = self.filters['shape'](features, shape_logits, value)
                elif attr_type == 'material':
                    mask = self.filters['material'](features, material_logits, value)
                elif attr_type == 'size':
                    mask = self.filters['size'](features, size_logits, value)
                else:
                    mask = current_mask
                
                current_mask = current_mask * mask
                memory.append(current_mask)
            
            elif op_type.startswith('query_'):
                attr_type = op_type.replace('query_', '')
                
                if attr_type == 'color':
                    result = self.queries['color'](features, color_logits, current_mask)
                    answer_idx = result.argmax().item()
                    return self.attribute_predictor.color_vocab[answer_idx]
                elif attr_type == 'shape':
                    result = self.queries['shape'](features, shape_logits, current_mask)
                    answer_idx = result.argmax().item()
                    return self.attribute_predictor.shape_vocab[answer_idx]
                elif attr_type == 'material':
                    result = self.queries['material'](features, material_logits, current_mask)
                    answer_idx = result.argmax().item()
                    return self.attribute_predictor.material_vocab[answer_idx]
                elif attr_type == 'size':
                    result = self.queries['size'](features, size_logits, current_mask)
                    answer_idx = result.argmax().item()
                    return self.attribute_predictor.size_vocab[answer_idx]
            
            elif op_type == 'count':
                count = self.count_module(features, current_mask)
                return str(count)
            
            elif op_type == 'exist':
                exists = self.exist_module(features, current_mask)
                return 'yes' if exists else 'no'
            
            elif op_type == 'relate':
                relation = op['value_inputs'][0] if 'value_inputs' in op else 'left'
                mask = self.spatial_module(features, boxes, current_mask, relation)
                current_mask = mask
                memory.append(current_mask)
            
            elif op_type in ['greater_than', 'less_than', 'equal_integer']:
                # Pop two values from memory
                if len(memory) >= 2:
                    val2 = memory.pop()
                    val1 = memory.pop()
                    result = self.compare_module(val1, val2)
                    return 'yes' if result else 'no'
        
        # Default return
        return 'unknown'


# Testing
if __name__ == "__main__":
    print("Testing Reasoning Modules...")
    print("=" * 60)
    
    # Test attribute predictor
    print("\nTesting Attribute Predictor...")
    feature_dim = 512
    num_objects = 5
    
    attr_predictor = AttributePredictor(feature_dim)
    dummy_features = torch.randn(1, num_objects, feature_dim)
    
    attr_logits = attr_predictor(dummy_features)
    print(f"Color logits shape: {attr_logits['color_logits'].shape}")
    print(f"Shape logits shape: {attr_logits['shape_logits'].shape}")
    
    # Test executor
    print("\nTesting Program Executor...")
    executor = ProgramExecutor(feature_dim, attr_predictor)
    
    # Create dummy scene
    scene = SceneRepresentation(
        features=torch.randn(num_objects, feature_dim),
        boxes=torch.rand(num_objects, 4),
        valid_mask=torch.ones(num_objects, dtype=torch.bool)
    )
    
    # Test simple program
    program = [
        {'type': 'scene'},
        {'type': 'filter_color', 'value_inputs': ['red']},
        {'type': 'count'}
    ]
    
    result = executor.execute(program, scene)
    print(f"Execution result: {result}")
    
    print("\n✓ Reasoning modules test complete!")