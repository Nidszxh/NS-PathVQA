# Neuro-Symbolic Visual Reasoning System

A hybrid neuro-symbolic framework for visual question answering on the PathVQA medical dataset. Combines a ResNet/LSTM neural backbone with a symbolic reasoning module (scene parser → query parser → executor) for explainable predictions.

## Quick Start

```bash
# 0. Activate virtual environment
source <your-venv>/bin/activate

# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Train with neuro-symbolic reasoning (default)
python train.py

# 3. Train neural-only baseline
python train.py --no-symbolic

# 4. Debug mode (500 train, 100 val, 5 epochs)
python train.py --debug
python train.py --debug --no-symbolic

# 5. Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt
python evaluate.py --checkpoint checkpoints/best_model.pt --split test

# 6. Smoke tests (all modules)
python src/models/text/question_encoder.py
python src/models/visual/visual_encoder.py
python src/models/pathvqa_model.py
python src/utils/config.py
python src/data_loaders/pathvqaDataset.py
python src/symbolic/scene_parser.py
python src/symbolic/query_parser.py
python src/symbolic/executor.py
```

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

## Components

| Module | File | Purpose |
|---|---|---|
| Visual perception | `src/models/visual/visual_encoder.py` | ResNet → top-k object proposals + spatial encoding (vectorized) |
| Question encoding | `src/models/text/question_encoder.py` | biLSTM question encoder + vocabulary management |
| Cross-modal attention | `src/models/pathvqa_model.py` | Concat+MLP attention over visual features conditioned on question |
| Scene parser | `src/symbolic/scene_parser.py` | Neural → symbolic: region/attribute/object logits from attended features |
| Query parser | `src/symbolic/query_parser.py` | Rule-based regex → structured Query (yes_no/identity/location/attribute/count) |
| Executor | `src/symbolic/executor.py` | Maps scene logits + query type → answer vocabulary logits + reasoning trace |
| Classifier | `src/models/pathvqa_model.py` | Neural MLP classifier (fusion of attended features + question state) |
| Config | `src/utils/config.py` | Dataclass hierarchy with JSON save/load |

## CLI Flags

- `--debug`: 500 train / 100 val samples, batch_size=16, 5 epochs
- `--no-symbolic`: Disable scene parser and executor (neural-only baseline)
- `--config path/to/config.json`: Custom JSON configuration
- Evaluate: `--checkpoint`, `--split {val,test}`, `--max_samples`, `--output`

## Dataset

PathVQA (flaviagiammarino/path-vqa on HuggingFace): 19,654 train / 6,259 val / 6,719 test. Medical pathology images with diverse QA pairs.

## Training Optimizations

| Feature | Benefit |
|---|---|
| **AMP (automatic mixed precision)** | ~2x training speedup, ~40% lower VRAM via FP16 Tensor Cores. Enabled by default on CUDA GPUs. |
| **cuDNN benchmark** | Auto-tunes convolution algorithms at startup for optimal throughput. |
| **TF32 matmul precision** | Uses Tensor Cores for FP32 matrix multiplications. Enabled by default. |
| **Vectorized visual encoder** | Batched gather replaces Python for-loop over batch items — ~33% faster validation. |
| **Persistent DataLoader workers** | Processes stay alive across epochs — no recreation or image re-decode overhead between epochs. |
| **Prefetch factor (2)** | Each worker prefetches 2 batches ahead to keep GPU fed. |
| **Rolling checkpoints** | Keeps only the last 3 checkpoints + best model. Auto-deletes old ones to save disk space. |
| **Early stopping** | Stops training after 10 validation runs without improvement. Prevents overfitting and wasted compute. |

## Neuro-Symbolic Design

The neural path (visual encoder → question encoder → cross-modal attention → MLP) learns rich representations end-to-end. The symbolic path (scene parser → executor) provides complementary structured reasoning:

- **SceneParser** predicts anatomical region logits, object presence scores, and attribute logits (color/shape/size) from attended visual features.
- **QueryParser** classifies each question into one of 5 types using rule-based regex patterns.
- **Executor** maps the symbolic predictions to answer vocabulary entries based on the query type, adding interpretable logits to the neural output.

Combined via: `answer_logits + symbolic_weight * symbolic_logits`

See `AGENTS.md` for detailed architecture diagram, conventions, and component reference.
