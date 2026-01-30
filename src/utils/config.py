"""Hierarchical dataclass configuration with validation for NS-PathVQA."""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading parameters."""
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 6
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    use_randaugment: bool = True
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    use_cache: bool = True
    norm: str = "clip"


@dataclass
class VisualConfig:
    """Visual encoder (CLIP ViT-B/32 or PLIP frozen + LoRA) parameters."""
    backbone: str = "plip"
    model_name: str = "vinid/plip"
    num_object_features: int = 512
    num_objects: int = 49
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    use_multiscale: bool = False

    def __post_init__(self):
        if isinstance(self.lora_target_modules, list):
            self.lora_target_modules = tuple(self.lora_target_modules)
        if self.backbone == "clip_vit_b_32" and self.model_name == "vinid/plip":
            self.model_name = "openai/clip-vit-base-patch32"
        elif self.backbone == "plip" and self.model_name == "openai/clip-vit-base-patch32":
            self.model_name = "vinid/plip"


@dataclass
class QuestionConfig:
    """Question encoder (DistilBERT frozen + LoRA) parameters."""
    encoder: str = "distilbert"
    model_name: str = "distilbert-base-uncased"
    hidden_dim: int = 768
    max_seq_len: int = 64
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_lin", "k_lin", "v_lin", "out_lin")
    dropout: float = 0.3

    def __post_init__(self):
        if isinstance(self.lora_target_modules, list):
            self.lora_target_modules = tuple(self.lora_target_modules)


@dataclass
class TrainingConfig:
    """Optimization and training loop parameters."""
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6
    grad_clip: float = 5.0
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = False
    ema_enabled: bool = True
    ema_decay: float = 0.999
    save_every: int = 5
    log_every: int = 100
    validate_every: int = 1
    early_stop_patience: int = 10
    max_checkpoints: int = 3
    device: str = "cuda"
    use_amp: bool = True


@dataclass
class SymbolicConfig:
    """Symbolic reasoning module parameters.

    SceneParser predicts regions, object presence, and attribute logits.
    Executor maps these to answer vocab logits per parsed question type.
    """
    enabled: bool = True
    num_regions: int = 50
    region_names: tuple = ()
    symbolic_weight: float = 0.3
    weighting_strategy: str = "static"  # "static" or "learned"
    ltn_enabled: bool = True
    ltn_weight: float = 0.1
    aux_region_weight: float = 0.5
    aux_attr_weight: float = 0.3
    aux_yn_weight: float = 0.2
    ontology_sibling_weight: float = 0.1
    conformal_alpha: float = 0.1
    conformal_max_set_size: int = 1
    ontology_path: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.region_names, list):
            self.region_names = tuple(self.region_names)
        if self.weighting_strategy not in ("static", "learned"):
            raise ValueError(f"weighting_strategy must be 'static' or 'learned', got '{self.weighting_strategy}'")


@dataclass
class PathConfig:
    """Filesystem paths for data, checkpoints, logs, and outputs."""
    root_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("data/pathvqa"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    best_model_path: Path = field(default_factory=lambda: Path("checkpoints/best_model.pt"))

    def __post_init__(self):
        for path in [self.checkpoint_dir, self.log_dir, self.output_dir, self.cache_dir]:
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
                        if ('path' in sk.lower() or 'dir' in sk.lower()) and sv is not None:
                            sv = Path(sv)
                        setattr(sub, sk, sv)
                if hasattr(sub, '__post_init__'):
                    sub.__post_init__()
            else:
                if hasattr(config, key):
                    setattr(config, key, value)
        print(f"Configuration loaded from {path}")
        return config


def get_default_config() -> Config:
    return Config()


def validate_config(cfg: Config) -> None:
    """Reject unknown/invalid config values early, before training starts."""
    if cfg.visual.backbone not in ("clip_vit_b_32", "plip"):
        raise ValueError(f"Unknown visual backbone: {cfg.visual.backbone!r}")
    if cfg.question.encoder not in ("distilbert",):
        raise ValueError(f"Unknown question encoder: {cfg.question.encoder!r}")
    if cfg.training.optimizer not in ("adam", "adamw", "sgd"):
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer!r}")
    if cfg.training.scheduler not in ("reduce_on_plateau", "cosine", "step"):
        raise ValueError(f"Unknown scheduler: {cfg.training.scheduler!r}")
    if cfg.training.grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {cfg.training.grad_accum_steps}")
    if not (0.0 <= cfg.training.ema_decay < 1.0):
        raise ValueError(f"ema_decay must be in [0, 1), got {cfg.training.ema_decay}")
    if cfg.symbolic.enabled and cfg.symbolic.symbolic_weight < 0:
        raise ValueError(f"symbolic_weight must be >= 0, got {cfg.symbolic.symbolic_weight}")


def backfill_config(cfg: Config) -> Config:
    """Fill fields missing from an older serialized Config.

    Checkpoints pickle their config at save time. When new fields are added,
    old checkpoints lack them. Backfill from a fresh default so old checkpoints
    stay loadable.
    """
    from dataclasses import fields
    for sub_name in ("data", "visual", "question", "symbolic", "training", "paths"):
        sub = getattr(cfg, sub_name, None)
        default_sub = getattr(Config(), sub_name)
        if sub is None:
            setattr(cfg, sub_name, default_sub)
            continue
        missing = [f.name for f in fields(default_sub) if not hasattr(sub, f.name)]
        for name in missing:
            setattr(sub, name, getattr(default_sub, name))
        if hasattr(sub, "__post_init__"):
            sub.__post_init__()
    return cfg


def get_debug_config() -> Config:
    """Debug config with reduced data and epochs.

    Checkpoints write to ``checkpoints/debug/`` to avoid overwriting production.
    """
    config = Config()
    config.debug = True
    config.experiment_name = "debug_run"
    config.data.max_train_samples = 500
    config.data.max_val_samples = 100
    config.data.batch_size = 16
    config.training.num_epochs = 5
    config.paths.checkpoint_dir = Path("checkpoints/debug")
    config.paths.best_model_path = Path("checkpoints/debug/best_model.pt")
    config.paths.__post_init__()
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
