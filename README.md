# Neuro-Symbolic Visual Reasoning System

A hybrid neuro-symbolic framework for visual question answering on the PathVQA medical dataset. Combines a ResNet/LSTM neural backbone with a symbolic reasoning module (scene parser → query parser → executor) for explainable predictions.

## Quick Start

```bash
# 0. Activate virtual environment
source /home/nidszxh/.ichigo/bin/activate

# 1. Train with neuro-symbolic reasoning (default)
python train.py

# 2. Train neural-only baseline
python train.py --no-symbolic

# 3. Debug mode (500 train, 100 val, 5 epochs)
python train.py --debug
python train.py --debug --no-symbolic

# 4. Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt
python evaluate.py --checkpoint checkpoints/best_model.pt --split test

# 5. Smoke tests
python src/models/text/question_encoder.py
python src/models/visual/visual_encoder.py
python src/models/pathvqa_model.py
python src/utils/config.py
python -c "import sys; sys.path.insert(0,'src'); from symbolic.executor import execute, build_region_names"
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
| Visual perception | `src/models/visual/visual_encoder.py` | ResNet → top-k object proposals + spatial encoding |
| Question encoding | `src/models/text/question_encoder.py` | biLSTM question encoder + vocabulary management |
| Cross-modal attention | `src/models/pathvqa_model.py` | Concat+MLP attention over visual features conditioned on question |
| Scene parser | `src/symbolic/scene_parser.py` | Neural → symbolic: region/attribute/object logits from attended features |
| Query parser | `src/symbolic/query_parser.py` | Rule-based regex → structured Query (yes_no/identity/location/attribute/count) |
| Executor | `src/symbolic/executor.py` | Maps scene logits + query type → answer vocabulary logits + reasoning trace |
| Scene graph | `src/symbolic/scene_graph.py` | SceneGraph dataclass for symbolic fact representation |
| Classifier | `src/models/pathvqa_model.py` | Neural MLP classifier (fusion of attended features + question state) |
| Config | `src/utils/config.py` | Dataclass hierarchy with JSON save/load |

## CLI Flags

- `--debug`: 500 train / 100 val samples, batch_size=16, 5 epochs
- `--no-symbolic`: Disable scene parser and executor (neural-only baseline)
- `--config path/to/config.json`: Custom JSON configuration
- Evaluate: `--checkpoint`, `--split {val,test}`, `--max_samples`, `--output`

## Dataset

PathVQA (flaviagiammarino/path-vqa on HuggingFace): 19,654 train / 6,259 val / 6,719 test. Medical pathology images with diverse QA pairs.

Query type distribution: ~49% yes_no, ~40% identity, ~7% location, ~2% other, <1% attribute/count. Symbolic path fires on ~68% of validation questions.

See `AGENTS.md` for detailed architecture diagram, conventions, and component reference.
