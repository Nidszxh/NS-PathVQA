# PathVQA — Agent Guide

> Working rules for this repo. Roadmap is `TODO.md`; system reference is `PROJECT.md`; novelty claims are `NOVELTIES.md`. If any conflict with `TODO.md`, that file wins.

## Rules

1. **Always activate the venv first**: `source ~/.zangestu/bin/activate` before any Python command.
2. **Never `git add`, `commit`, or `push`** without explicit user instruction.
3. **No new dependencies** without justification — keep the footprint minimal.
4. **Run from repo root** — `train.py`, `evaluate.py`, `serve.py` use `sys.path.insert(0, 'src')`.
5. **Use `--debug`** for quick iteration; never run full training without asking.
6. **Preserve neuro-symbolic fusion** — don't silently replace it with a neural-only path.
7. **Ruff E402 trap**: new modules with `sys.path.append` must be added to `pyproject.toml` per-file-ignores or ruff fails.

## Run

```bash
source ~/.zangestu/bin/activate
uv pip install -r requirements.txt

# Training
python train.py                                    # neuro-symbolic (default)
python train.py --no-symbolic                      # neural-only
python train.py --debug                            # 500/100 samples, batch=16, 5 epochs
python train.py --resume auto                      # resume from newest checkpoint

# Eval / API
python evaluate.py --checkpoint checkpoints/best_model.pt
python serve.py --checkpoint checkpoints/best_model.pt              # FastAPI server on :8000

# Verification (run after every code change)
pytest && ruff check .

# Smoke tests (14 modules)
python src/models/pathvqa_model.py
python src/symbolic/executor.py
```

## Setup

- **Venv**: `/home/nidszxh/.zangestu/` (Python ≤3.12 — torch requires it; system python is 3.14). If missing: `uv venv ~/.zangestu && source ~/.zangestu/bin/activate && uv pip install -r requirements.txt`.
- **`requirements.txt`** pinned to installed versions (torch 2.12.1+cu130, transformers 5.12.1, peft 0.20.0, fastapi 0.139.0, uvicorn 0.49.0).
- **Data** auto-downloads from `flaviagiammarino/path-vqa` on HF; cache at `data/hf-cache/`.
- **Image modes mixed** (CMYK/RGB/L) — `.convert("RGB")` handles all; don't remove.
- **Output dirs** (`checkpoints/`, `logs/`, `outputs/`, `data/`) are gitignored, auto-created.
- **Verification**: 105 pytest tests + `ruff check .` + 14 smoke tests.

## Working principles

- **Neuro-symbolic**: preserve the fusion and its explainability. Additive baseline must stay as ablation (`weighting_strategy=static`).
- **Medical domain**: research system, not clinical. Keep terminology accurate.
- **Reproducibility**: fixed seeds, `--debug` fast, smoke tests as gate.
- **Minimal footprint**: no new deps without justification. Symbolic pipeline passes tensors directly.

## Roadmap status

Phases 1–10 shipped. Production P1 (FastAPI) + P2 (Docker) shipped. P3 (packaging/CI) parked.

## Architecture

```
PLIP (frozen+LoRA) → 49 patches (768-d) → Linear 768→512
                                                          │
DistilBERT (frozen+LoRA) → [CLS] → question (768-d)      │
                          │                               │
                          └── CrossModalTransformer ──────┘
                                        │
                          ┌─────────────┼─────────────┐
                      SceneParser    attended           │
                      (region/         features        │
                       attr logits)      │         MLP ──┤
                          │             └──── neural   │
                       Executor            logits      │
                          │                             │
                          └─── gated fusion ────────────┘
```

| Component | File |
|---|---|
| Visual encoder | `src/models/visual/visual_encoder.py` — PLIP + LoRA → 768→512 projection |
| Question encoder | `src/models/question/question_encoder.py` — DistilBERT + LoRA + HF tokenizer |
| Cross-modal | `src/models/fusion.py` — cross-attention transformer block |
| Scene parser | `src/symbolic/scene_parser.py` — region/attribute/object logits |
| Query parser | `src/symbolic/query_parser.py` — rule-based regex classification |
| Executor | `src/symbolic/executor.py` — scene logits → answer-vocab logits |
| Routing gate | `src/symbolic/routing.py` — `g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype)]))` |
| Main model | `src/models/pathvqa_model.py` — `NeuroSymbolicPathVQA`, `build_model()` |
| Config | `src/utils/config.py` — dataclass hierarchy + validation |
| Dataset | `src/data/pathvqa_dataset.py` — HF dataset + CLIP prep + cache |
| API | `src/api/inference.py` + `schemas.py` — FastAPI prediction endpoints |

Entry points: `train.py`, `evaluate.py`, `serve.py`.

## Conventions

- **Answer vocab** built from training set at runtime. `train.py` reserves `<UNK>` for unseen answers (~14% of val/test).
- **Region names** derived from answer vocab via `build_region_names()` — model is dataset-dependent.
- **HF tokenizer** (`get_question_tokenizer()`).
- **Checkpoints** store: model/optimizer/scheduler/scaler/ema state, config, vocab, region names, early-stop state. `backfill_config()` fills missing fields from older configs.
- **Resume**: `--resume <path>` or `--resume auto`. Config in checkpoint wins. RNG state not saved.
- **Debug runs** write to `checkpoints/debug/`, never touch production artifacts.
- **AMP** enabled by default on CUDA. cuDNN benchmark + TF32 matmul set at startup.
- **RandAugment** train-only on uint8 tensor, before float/normalize.
- **uint8 tensor cache** keyed by `(image_size, norm)` at `data/cache/`. Atomic writes.
- **Smoke tests** (`python src/<file>`) in all 14 modules with `__main__` guards.
- **Ruff E402**: `pyproject.toml` per-file-ignores E402 for modules using `sys.path.append` (`pathvqa_model.py`, `executor.py`). New modules with similar patterns must be added there or ruff fails.

## Config defaults (`src/utils/config.py`)

| Field | Default |
|---|---|
| `data.batch_size` | `32` |
| `data.image_size` | `(224, 224)` |
| `data.use_randaugment` | `True` |
| `data.use_cache` | `True` |
| `data.norm` | `"clip"` |
| `visual.backbone` | `plip` |
| `visual.model_name` | `vinid/plip` |
| `visual.num_object_features` | `512` |
| `visual.num_objects` | `49` |
| `visual.lora_rank/alpha` | `16`/`32` |
| `question.encoder` | `distilbert` |
| `question.hidden_dim` | `768` |
| `question.lora_rank/alpha` | `16`/`32` |
| `training.learning_rate` | `1e-4` |
| `training.num_epochs` | `50` |
| `training.early_stop_patience` | `10` |
| `training.use_amp` | `True` |
| `training.grad_clip` | `5.0` |
| `symbolic.symbolic_weight` | `0.3` |
| `symbolic.ltn_enabled` | `True` |

**Debug overrides**: batch=16, epochs=5, 500 train / 100 val.

## Docker

```bash
docker compose up --build          # builds + runs API on :8000, mounts checkpoints read-only
```

Requires NVIDIA runtime for GPU passthrough. `CHECKPOINT_PATH` env var selects the model file.

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + device info |
| `/model/info` | GET | Checkpoint metadata |
| `/predict` | POST | image_base64 + question → answer + confidence |
| `/predict/explain` | POST | same, plus symbolic execution trace |
