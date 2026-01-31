from pathlib import Path
import pytest
from utils.config import (
    Config, SymbolicConfig, get_debug_config, get_default_config, PathConfig,
    backfill_config, validate_config,
)


def test_defaults_match_agents_docs():
    cfg = get_default_config()
    assert cfg.data.batch_size == 32
    assert cfg.data.num_workers == 6
    assert cfg.data.image_size == (224, 224)
    assert cfg.training.learning_rate == 1e-4
    assert cfg.training.num_epochs == 50
    assert cfg.training.early_stop_patience == 10
    assert cfg.training.max_checkpoints == 3
    assert cfg.training.use_amp is True
    assert cfg.training.grad_clip == 5.0
    assert cfg.symbolic.enabled is True
    assert cfg.symbolic.symbolic_weight == 0.3
    assert cfg.seed == 42


def test_phase3_model_defaults():
    cfg = get_default_config()
    assert cfg.data.norm == "clip"
    assert cfg.visual.backbone == "plip"
    assert cfg.visual.model_name == "vinid/plip"
    assert cfg.visual.num_object_features == 512
    assert cfg.visual.num_objects == 49
    assert cfg.visual.lora_rank == 16
    assert cfg.visual.lora_alpha == 32
    assert cfg.question.encoder == "distilbert"
    assert cfg.question.model_name == "distilbert-base-uncased"
    assert cfg.question.hidden_dim == 768
    assert cfg.question.max_seq_len == 64


def test_phase4_training_defaults():
    cfg = get_default_config()
    assert cfg.training.optimizer == "adamw"
    assert cfg.training.scheduler == "cosine"
    assert cfg.training.warmup_ratio == 0.05
    assert cfg.training.grad_accum_steps == 1
    assert cfg.training.gradient_checkpointing is False
    assert cfg.training.ema_enabled is True
    assert cfg.training.ema_decay == 0.999


def test_validate_config_accepts_defaults():
    validate_config(get_default_config())


@pytest.mark.parametrize("mutate, match", [
    (lambda c: setattr(c.visual, "backbone", "resnet50"), "Unknown visual backbone"),
    (lambda c: setattr(c.question, "encoder", "bloom"), "Unknown question encoder"),
    (lambda c: setattr(c.training, "optimizer", "rmsprop"), "Unknown optimizer"),
    (lambda c: setattr(c.training, "scheduler", "warmup"), "Unknown scheduler"),
    (lambda c: setattr(c.training, "grad_accum_steps", 0), "grad_accum_steps"),
    (lambda c: setattr(c.training, "ema_decay", 1.0), "ema_decay"),
])
def test_validate_config_rejects_invalid(mutate, match):
    cfg = get_default_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_debug_overrides():
    cfg = get_debug_config()
    assert cfg.debug is True
    assert cfg.experiment_name == "debug_run"
    assert cfg.data.max_train_samples == 500
    assert cfg.data.max_val_samples == 100
    assert cfg.data.batch_size == 16
    assert cfg.training.num_epochs == 5


def test_debug_never_touches_real_checkpoints():
    """A --debug run must never write into the production checkpoint dir."""
    cfg = get_debug_config()
    real = PathConfig()
    assert cfg.paths.checkpoint_dir == Path("checkpoints/debug")
    assert cfg.paths.best_model_path == Path("checkpoints/debug/best_model.pt")
    assert cfg.paths.checkpoint_dir != real.checkpoint_dir
    assert cfg.paths.best_model_path != real.best_model_path


def test_debug_skips_baseline_artifact():
    cfg = get_debug_config()
    assert cfg.debug is True


def test_backfill_config_adds_missing_fields():
    """Simulate an old checkpoint config (pre-Phase-2: no use_cache/norm)."""
    cfg = get_default_config()
    delattr(cfg.data, "use_cache")
    delattr(cfg.data, "norm")
    delattr(cfg.data, "use_randaugment")
    backfill_config(cfg)
    assert cfg.data.use_cache is True
    assert cfg.data.norm == "clip"
    assert cfg.data.use_randaugment is True
    assert cfg.paths.cache_dir == Path("data/cache")


def test_lora_target_modules_normalized_from_list():
    cfg = get_default_config()
    cfg.visual.lora_target_modules = ["q_proj", "v_proj"]
    cfg.visual.__post_init__()
    assert isinstance(cfg.visual.lora_target_modules, tuple)
    cfg.question.lora_target_modules = ["q_lin", "v_lin"]
    cfg.question.__post_init__()
    assert isinstance(cfg.question.lora_target_modules, tuple)


def test_symbolic_region_names_normalized():
    cfg = SymbolicConfig(region_names=["lung", "liver"])
    assert isinstance(cfg.region_names, tuple)


def test_config_save_load_roundtrip(tmp_path):
    cfg = get_default_config()
    cfg.data.batch_size = 8
    cfg.symbolic = SymbolicConfig(region_names=["lung"])
    path = str(tmp_path / "config.json")
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.data.batch_size == 8
    assert loaded.symbolic.region_names == ("lung",)
    assert isinstance(loaded.paths.checkpoint_dir, type(cfg.paths.checkpoint_dir))


def test_config_ignores_unknown_fields(tmp_path):
    path = str(tmp_path / "config.json")
    get_default_config().save(path)
    with open(path) as f:
        content = f.read()
    with open(path, "w") as f:
        f.write(content.replace("}", ", \"unknown_field\": 42}", 1))
    loaded = Config.load(path)
    assert not hasattr(loaded, "unknown_field")
