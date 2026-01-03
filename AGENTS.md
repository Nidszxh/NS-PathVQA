# PathVQA — Neuro-Symbolic Visual Question Answering

## Run

```bash
source /home/nidszxh/.ichigo/bin/activate        # activate venv
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
python -c "import sys; sys.path.insert(0,'src'); from symbolic.executor import execute, build_region_names"  # symbolic smoke test
```

## Setup

- **Venv**: `/home/nidszxh/.ichigo/` — always activate first. Use `uv pip install <pkg>`.
- **Always run scripts from repo root** — `train.py` and `evaluate.py` hardcode `sys.path.append('src')`.
- **`requirements.txt` was originally incomplete**: had `kagglehub` (unused) instead of `datasets` (used by code) — fixed.
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
| Visual perception | `src/models/visual/visual_encoder.py` | `SimpleObjectDetector` (ResNet → proposals → top-k features + spatial) |
| Question encoding | `src/models/text/question_encoder.py` | `QuestionEncoder` (biLSTM) + `QuestionVocabulary` |
| Cross-modal | `src/models/pathvqa_model.py` | `CrossModalAttention` (concat+MLP attention) |
| Scene parser | `src/symbolic/scene_parser.py` | `SceneParser` (predicts region logits from attended features) |
| Query parser | `src/symbolic/query_parser.py` | `parse_question()` (rule-based regex → structured Query) |
| Executor | `src/symbolic/executor.py` | `execute()` (query + region logits → symbolic answer logits) |
| Scene graph | `src/symbolic/scene_graph.py` | `SceneGraph` dataclass (symbolic fact representation) |
| Classifier | `src/models/pathvqa_model.py` | `NeuroSymbolicPathVQA` (neural MLP + symbolic path, combined) |
| Config | `src/utils/config.py` | `Config` → sub-dataclasses (+ `SymbolicConfig`) |
| Data | `src/data_loaders/pathvqaDataset.py` | `PathVQADataset` wraps `load_dataset("flaviagiammarino/path-vqa")` |

**Symbolic path**: `SceneParser` predicts anatomical region logits + attribute logits (color/shape/size) + object presence from attended visual features → `QueryParser` classifies question type (yes_no/identity/location/count/attribute) → `Executor` adds symbolic logits to neural logits for the corresponding answer vocabulary entries. Combined via `answer_logits + symbolic_weight * symbolic_logits`. Attribute heads predict over 17 color values, 9 shape values, and 11 size values extracted from the dataset.

Entry points: `train.py` (`Trainer`) and `evaluate.py` (`Evaluator`). `pathvqa.py` is a no-op.

## Conventions

- **Answer vocabulary** built from training set at runtime via `PathVQADataset.__init__`.
- **Question vocabulary** cached to `data/question_vocab.json` after first build.
- **Checkpoints** (`.pt` files): `epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_acc`, `config`, `answer_vocab`, `answer_to_idx`, `idx_to_answer`, `region_names`.
- **TensorBoard** logs at `logs/<experiment_name>/`.
- **No tests, CI, linter, or formatter** — `__main__` smoke test blocks in every module.
- **`--debug`**: `max_train_samples=500`, `max_val_samples=100`, `batch_size=16`, `num_epochs=5`.
