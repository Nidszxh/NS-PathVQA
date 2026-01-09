# PathVQA — Neuro-Symbolic Visual Question Answering

## Run

```bash
source <your-venv>/bin/activate                   # activate venv
uv pip install -r requirements.txt                 # install deps
python train.py                                    # train with symbolic reasoning (default)
python train.py --no-symbolic                      # train neural-only baseline
python train.py --debug                            # 500 train + 100 val, batch=16, 5 epochs
python train.py --debug --no-symbolic              # debug neural-only
python train.py --config path/to/config.json       # custom JSON config
python evaluate.py --checkpoint checkpoints/best_model.pt
python evaluate.py --checkpoint checkpoints/best_model.pt --split test
python src/models/text/question_encoder.py          # smoke test
python src/models/visual/visual_encoder.py           # smoke test
python src/models/pathvqa_model.py                   # smoke test
python src/utils/config.py                            # smoke test
python src/data_loaders/pathvqaDataset.py              # smoke test
python src/symbolic/scene_parser.py                    # smoke test
python src/symbolic/query_parser.py                    # smoke test
python src/symbolic/executor.py                        # smoke test
```

## Setup

- **Venv**: `/home/nidszxh/.ichigo/` — always activate first. Use `uv pip install <pkg>`.
- **Always run scripts from repo root** — `train.py` and `evaluate.py` hardcode `sys.path.append('src')`.
- **`requirements.txt`**: `torch`, `torchvision`, `Pillow`, `tqdm`, `tensorboard`, `datasets` — all six used.
- **PathVQA data** auto-downloads from HuggingFace on first `pathvqa_dataloader()` call.
- **Output dirs** (`checkpoints/`, `logs/`, `outputs/`, `data/`) are gitignored and auto-created by `PathConfig`.
- **Smoke tests** exist as `__main__` blocks in all modules — no test framework, no CI, no linter/formatter.

## Architecture

```
ResNet backbone → object proposals (top-k) + spatial encoding
                                                        │
LSTM(question) → question_state                         │
                        │                               │
                        └──── CrossModalAttention ──────┘
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                    SceneParser   attended       │
                    (neural →      features      │
                     symbolic)        │          │
                          │          └──── MLP ──┤
                          │               │      │
                    SymbolicExecutor   neural    │
                    (query + facts →    logits   │
                     answer logits)       │      │
                          │               │      │
                          └────── + ──────┘      │
                                     │            │
                               combined logits    │
                                     │            │
                               answer_logits ─────┘
```

| Layer | File | Key class |
|---|---|---|
| Visual perception | `src/models/visual/visual_encoder.py` | `SimpleObjectDetector` (ResNet → proposals → top-k features + spatial, vectorized) |
| Question encoding | `src/models/text/question_encoder.py` | `QuestionEncoder` (biLSTM) + `QuestionVocabulary` |
| Cross-modal | `src/models/pathvqa_model.py` | `CrossModalAttention` (concat+MLP attention, FP16-safe mask) |
| Scene parser | `src/symbolic/scene_parser.py` | `SceneParser` (predicts region logits from attended features) |
| Query parser | `src/symbolic/query_parser.py` | `parse_question()` (rule-based regex → structured Query) |
| Executor | `src/symbolic/executor.py` | `execute()` (query + region logits → symbolic answer logits, 10 regions discovered) |
| Classifier | `src/models/pathvqa_model.py` | `NeuroSymbolicPathVQA` (neural MLP + symbolic path, combined) |
| Config | `src/utils/config.py` | `Config` → sub-dataclasses (+ `SymbolicConfig`) |
| Data | `src/data_loaders/pathvqaDataset.py` | `PathVQADataset` wraps `load_dataset("flaviagiammarino/path-vqa")` |

**Symbolic path**: `SceneParser` predicts anatomical region logits + attribute logits (color/shape/size) + object presence from attended visual features → `QueryParser` classifies question type (yes_no/identity/location/count/attribute) → `Executor` adds symbolic logits to neural logits for the corresponding answer vocabulary entries. Combined via `answer_logits + symbolic_weight * symbolic_logits`. Attribute heads predict over 17 color values, 9 shape values, and 11 size values extracted from the dataset.

Entry points: `train.py` (`Trainer`) and `evaluate.py` (`Evaluator`). `pathvqa.py` is a no-op.

## Conventions

- **Answer vocabulary** built from training set at runtime via `PathVQADataset.__init__`.
- **Question vocabulary** cached to `data/question_vocab.json` after first build.
- **Checkpoints** (`.pt` files): `epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_acc`, `config`, `answer_vocab`, `answer_to_idx`, `idx_to_answer`, `region_names`.
- **Rolling checkpoint management**: Only the last 3 periodic checkpoints are kept; old ones are auto-deleted. The best model is always saved separately as `best_model.pt`.
- **Early stopping**: Training halts after 10 consecutive validation runs without accuracy improvement. Configured via `early_stop_patience` in `TrainingConfig`.
- **Mixed precision**: Automatic mixed precision (AMP) via `torch.amp.GradScaler` + `autocast`. Enabled by default when CUDA is available. Provides ~2x training speedup on RTX 4060 and reduces VRAM usage by ~40%.
- **cuDNN benchmark** and **TF32 matmul precision** (`torch.set_float32_matmul_precision('high')`) enabled at startup.
- **TensorBoard** logs at `logs/<experiment_name>/`.
- **`__main__` smoke tests** exist in all 12 modules.
- **`--debug`**: `max_train_samples=500`, `max_val_samples=100`, `batch_size=16`, `num_epochs=5`.

## Config

### Default values (`src/utils/config.py:Config`)

| Field | Default | Description |
|---|---|---|
| `data.batch_size` | `32` | Samples per GPU step (RTX 4060 8GB safe) |
| `data.num_workers` | `6` | DataLoader worker processes |
| `data.image_size` | `(320, 240)` | Resized input dimensions |
| `training.learning_rate` | `1e-4` | Adam optimizer LR |
| `training.num_epochs` | `50` | Max training epochs |
| `training.early_stop_patience` | `10` | Val runs without improvement before stop |
| `training.max_checkpoints` | `3` | Number of periodic checkpoints to retain |
| `training.use_amp` | `True` | Enable automatic mixed precision |
| `training.grad_clip` | `5.0` | Max gradient norm |
| `symbolic.symbolic_weight` | `0.3` | Weight for symbolic logits in combination: `neural + weight * symbolic` |

### Debug config overrides

| Field | Debug value |
|---|---|
| `max_train_samples` | `500` |
| `max_val_samples` | `100` |
| `batch_size` | `16` |
| `num_epochs` | `5` |
