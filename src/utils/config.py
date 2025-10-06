from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration"""
    data_dir: str = "data/CLEVR_v1.0"
    image_size: Tuple[int, int] = (320, 240)
    batch_size: int = 64
    num_workers: int = 4
    
    # Dataset splits
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    
    # For quick testing
    max_train_samples: int = None  # None for all
    max_val_samples: int = None


@dataclass
class VisualConfig:
    """Visual perception module configuration"""
    # Model architecture
    backbone: str = "resnet101"  # Options: resnet50, resnet101, vit_base
    pretrained: bool = True
    
    # Object detection
    detector_type: str = "faster_rcnn"  # Options: faster_rcnn, detr
    num_object_features: int = 2048
    max_objects_per_image: int = 10
    
    # Feature extraction
    spatial_feat_dim: int = 128
    semantic_feat_dim: int = 512
    
    # Training
    freeze_backbone: bool = False
    learning_rate: float = 1e-4


@dataclass
class QuestionConfig:
    """Question processing configuration"""
    # Vocabulary
    vocab_size: int = 1000
    max_question_length: int = 46  # Max length in CLEVR
    
    # Model architecture
    encoder_type: str = "lstm"  # Options: lstm, gru, transformer
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    
    # Program generation
    max_program_length: int = 27  # Max in CLEVR
    program_vocab_size: int = 100
    
    # Training
    learning_rate: float = 1e-3


@dataclass
class ReasoningConfig:
    """Symbolic reasoning module configuration"""
    # Module types
    module_dim: int = 512
    
    # Available modules
    modules: List[str] = None
    
    def __post_init__(self):
        if self.modules is None:
            self.modules = [
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
            ]
    
    # Execution
    max_execution_steps: int = 50
    
    # Attributes
    colors: List[str] = None
    shapes: List[str] = None
    materials: List[str] = None
    sizes: List[str] = None
    relations: List[str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = ['gray', 'red', 'blue', 'green', 'brown', 
                          'purple', 'cyan', 'yellow']
        if self.shapes is None:
            self.shapes = ['cube', 'sphere', 'cylinder']
        if self.materials is None:
            self.materials = ['rubber', 'metal']
        if self.sizes is None:
            self.sizes = ['small', 'large']
        if self.relations is None:
            self.relations = ['left', 'right', 'behind', 'front']


@dataclass
class TrainingConfig:
    """Training configuration"""
    # General
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimization
    optimizer: str = "adam"  # Options: adam, adamw, sgd
    scheduler: str = "reduce_on_plateau"  # Options: reduce_on_plateau, cosine, step
    
    # Scheduler params
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6
    
    # Gradient
    grad_clip: float = 5.0
    
    # Training phases
    pretrain_visual_epochs: int = 5
    pretrain_question_epochs: int = 10
    joint_training_epochs: int = 35
    
    # Loss weights
    visual_loss_weight: float = 1.0
    program_loss_weight: float = 1.0
    answer_loss_weight: float = 1.0
    
    # Checkpointing
    save_every: int = 5  # Save every N epochs
    keep_best: int = 3   # Keep top N checkpoints
    
    # Logging
    log_every: int = 100  # Log every N batches
    validate_every: int = 1  # Validate every N epochs
    
    # Device
    device: str = "cuda"  # Options: cuda, cpu
    mixed_precision: bool = True


@dataclass
class PathConfig:
    """Path configuration"""
    # Base paths
    root_dir: Path = Path(".")
    data_dir: Path = Path("data/CLEVR_v1.0")
    
    # Output paths
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    output_dir: Path = Path("outputs")
    
    # Specific paths
    best_model_path: Path = Path("checkpoints/best_model.pt")
    vocab_path: Path = Path("data/vocab.json")
    
    def __post_init__(self):
        # Create directories
        for path in [self.checkpoint_dir, self.log_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration class"""
    # Sub-configurations
    data: DataConfig = None
    visual: VisualConfig = None
    question: QuestionConfig = None
    reasoning: ReasoningConfig = None
    training: TrainingConfig = None
    paths: PathConfig = None
    
    # Experiment
    experiment_name: str = "clevr_baseline"
    seed: int = 42
    debug: bool = False
    
    def __post_init__(self):
        # Initialize sub-configs if not provided
        if self.data is None:
            self.data = DataConfig()
        if self.visual is None:
            self.visual = VisualConfig()
        if self.question is None:
            self.question = QuestionConfig()
        if self.reasoning is None:
            self.reasoning = ReasoningConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.paths is None:
            self.paths = PathConfig()
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct Config object
        # This is a simplified version; for production, use a proper deserialization
        config = cls()
        
        # Update values from dict
        for key, value in config_dict.items():
            if hasattr(config, key) and isinstance(value, dict):
                sub_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
        
        print(f"✓ Configuration loaded from {path}")
        return config
    
    def print_config(self):
        """Print configuration in a readable format"""
        print("\n" + "=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        
        print(f"\nExperiment: {self.experiment_name}")
        print(f"Seed: {self.seed}")
        print(f"Debug: {self.debug}")
        
        print("\n" + "-" * 80)
        print("DATA CONFIG")
        print("-" * 80)
        print(f"  Data directory: {self.data.data_dir}")
        print(f"  Image size: {self.data.image_size}")
        print(f"  Batch size: {self.data.batch_size}")
        print(f"  Num workers: {self.data.num_workers}")
        
        print("\n" + "-" * 80)
        print("VISUAL CONFIG")
        print("-" * 80)
        print(f"  Backbone: {self.visual.backbone}")
        print(f"  Pretrained: {self.visual.pretrained}")
        print(f"  Detector: {self.visual.detector_type}")
        print(f"  Max objects: {self.visual.max_objects_per_image}")
        
        print("\n" + "-" * 80)
        print("QUESTION CONFIG")
        print("-" * 80)
        print(f"  Encoder: {self.question.encoder_type}")
        print(f"  Embedding dim: {self.question.embedding_dim}")
        print(f"  Hidden dim: {self.question.hidden_dim}")
        print(f"  Max question length: {self.question.max_question_length}")
        
        print("\n" + "-" * 80)
        print("REASONING CONFIG")
        print("-" * 80)
        print(f"  Module dim: {self.reasoning.module_dim}")
        print(f"  Number of modules: {len(self.reasoning.modules)}")
        print(f"  Max execution steps: {self.reasoning.max_execution_steps}")
        
        print("\n" + "-" * 80)
        print("TRAINING CONFIG")
        print("-" * 80)
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Optimizer: {self.training.optimizer}")
        print(f"  Scheduler: {self.training.scheduler}")
        print(f"  Mixed precision: {self.training.mixed_precision}")
        print(f"  Device: {self.training.device}")
        
        print("\n" + "=" * 80)


# Default configuration instance
def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


# Quick config for testing
def get_debug_config() -> Config:
    """Get debug configuration with small dataset"""
    config = Config()
    config.debug = True
    config.experiment_name = "debug_run"
    
    # Small dataset
    config.data.max_train_samples = 1000
    config.data.max_val_samples = 200
    config.data.batch_size = 16
    
    # Fewer epochs
    config.training.num_epochs = 5
    config.training.pretrain_visual_epochs = 1
    config.training.pretrain_question_epochs = 2
    config.training.joint_training_epochs = 2
    
    return config


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration system...")
    
    # Create default config
    config = get_default_config()
    config.print_config()
    
    # Save config
    config.save("outputs/config_default.json")
    
    # Test debug config
    print("\n\n")
    debug_config = get_debug_config()
    debug_config.print_config()
    debug_config.save("outputs/config_debug.json")
    
    print("\n✓ Configuration test complete!")