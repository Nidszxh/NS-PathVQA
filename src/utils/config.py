"""Configuration management: dataclass hierarchy with JSON save/load."""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading parameters."""
    image_size: Tuple[int, int] = (320, 240)
    batch_size: int = 64
    num_workers: int = 4
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None


@dataclass
class VisualConfig:
    """Visual encoder (SimpleObjectDetector) parameters."""
    backbone: str = "resnet50"
    pretrained: bool = True
    num_object_features: int = 512
    max_objects_per_image: int = 10
    spatial_feat_dim: int = 128


@dataclass
class QuestionConfig:
    """Question encoder (biLSTM) parameters."""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Optimization and training loop parameters."""
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6
    grad_clip: float = 5.0
    save_every: int = 5
    log_every: int = 100
    validate_every: int = 1
    device: str = "cuda"


@dataclass
class SymbolicConfig:
    """Symbolic reasoning module parameters.

    The SceneParser predicts anatomical regions, object presence,
    and attribute logits (color/shape/size). The Executor maps these
    to answer vocabulary logits based on the parsed question type.
    """
    enabled: bool = True
    num_regions: int = 50
    region_names: tuple = ()
    symbolic_weight: float = 0.3

    def __post_init__(self):
        if isinstance(self.region_names, list):
            self.region_names = tuple(self.region_names)


@dataclass
class PathConfig:
    """Filesystem paths for data, checkpoints, logs, and outputs."""
    root_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("data/pathvqa"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    best_model_path: Path = field(default_factory=lambda: Path("checkpoints/best_model.pt"))

    def __post_init__(self):
        for path in [self.checkpoint_dir, self.log_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Root configuration aggregating all sub-configs."""
    experiment_name: str = "pathvqa_neuro_symbolic"
    seed: int = 42
    debug: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    question: QuestionConfig = field(default_factory=QuestionConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def save(self, path: str):
        """Serialize config to JSON (Path objects converted to strings)."""
        import json
        from dataclasses import asdict

        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        with open(path, 'w') as f:
            json.dump(convert(asdict(self)), f, indent=2)
        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Deserialize config from JSON, restoring Path fields."""
        import json
        with open(path) as f:
            d = json.load(f)
        config = cls()
        for key, value in d.items():
            if key in ['data', 'visual', 'question', 'symbolic', 'training', 'paths']:
                sub = getattr(config, key)
                for sk, sv in value.items():
                    if hasattr(sub, sk):
                        if 'path' in sk.lower() or 'dir' in sk.lower():
                            sv = Path(sv)
                        setattr(sub, sk, sv)
            else:
                if hasattr(config, key):
                    setattr(config, key, value)
        print(f"Configuration loaded from {path}")
        return config


def get_default_config() -> Config:
    return Config()


def get_debug_config() -> Config:
    """Create a debug config with reduced data and epochs for fast testing."""
    config = Config()
    config.debug = True
    config.experiment_name = "debug_run"
    config.data.max_train_samples = 500
    config.data.max_val_samples = 100
    config.data.batch_size = 16
    config.training.num_epochs = 5
    return config


if __name__ == "__main__":
    import json
    from dataclasses import asdict
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    cfg = get_default_config()
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Default config: {json.dumps(convert(asdict(cfg)), indent=2)}")
    print(f"Symbolic enabled: {cfg.symbolic.enabled}, regions: {cfg.symbolic.num_regions}")
    dbg = get_debug_config()
    print(f"Debug config: max_train={dbg.data.max_train_samples}, "
          f"batch={dbg.data.batch_size}, epochs={dbg.training.num_epochs}")
    print("Config test passed!")
