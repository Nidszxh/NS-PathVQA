# PathVQA — Agent Guide

> Working rules and verified reference for this repo. System reference, novelty register and audit history are consolidated in `DOCS.md`; if any document conflicts with `DOCS.md`, that file wins.

## Hard rules

1. **Always activate the venv first**: `source ~/.zangestu/bin/activate` (Python 3.12; system python is 3.14 and has no torch).
2. **No commits or adds until the user says so** — never `git add`, `commit`, or `push` unless explicitly instructed in that message.
3. **No new dependencies** without justification.
4. **Run everything from repo root** — `train.py`, `evaluate.py`, `serve.py` assume it.
5. **Use `--debug` for iteration**; never start a full training run without asking.
6. **Preserve neuro-symbolic fusion** — don't replace it with a neural-only path. The additive baseline must stay available as ablation (`symbolic.weighting_strategy="static"`).

## Commands

```bash
source ~/.zangestu/bin/activate
uv pip install -r requirements.txt   # pinned runtime deps
uv pip install -e .                  # once per env; provides `pathvqa.*` imports

# Training
python train.py                          # neuro-symbolic (default)
python train.py --no-symbolic            # neural-only ablation
python train.py --debug                  # 500 train/100 val, batch=16, 5 epochs -> checkpoints/debug/
python train.py --resume auto            # newest checkpoints/checkpoint_epoch_N.pt ('--resume <path>' also works)
python train.py --experiment NAME        # names outputs/<name>.json artifact
python train.py --ablation A|B|C|D|E     # A=Baseline B=SOTA C=Diff D=NoGate E=Regex+Diff
python train.py --weighting-strategy static|learned   # fusion gate (overrides config)
python train.py --deterministic          # bit-reproducible cuDNN kernels (slower)

# Evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt \
    [--split val|test] [--max_samples N] [--output FILE] [--ablation A-E] \
    [--intervention none|probe|attribution|counterfactual|all]

# API server (:8000)
python serve.py --checkpoint checkpoints/best_model.pt [--host HOST] [--port PORT]
```

## Verification (run after every code change)

```bash
ruff format --check .     # formatting gate (CI)
ruff check .              # lint gate (CI)
mypy src/                 # strict type gate (CI); 40 files, currently clean
pytest                    # 260 tests, ~90 s; addopts enforces >=80% coverage
pytest --no-cov -q PATH   # fast subset during iteration
```

Always pass `--no-cov` when running a subset (`pytest tests/test_config.py` alone
fails the 80% coverage gate).

Smoke tests — 16 modules have `__main__` guards; run them as modules:

```bash
python -m pathvqa.models.pathvqa_model
python -m pathvqa.symbolic.executor
# all 16: models/{pathvqa_model,fusion,ema}, models/visual/visual_encoder,
# models/question/question_encoder, symbolic/{scene_parser,query_parser,executor,dsl,ltn},
# utils/{config,interpretability,seed,logging_utils}, data/{dataset_adapter,pathvqa_dataset}
```

(`symbolic/routing.py` and `symbolic/aux_losses.py` have no `__main__` guards.)

Direct-file invocation (`python src/pathvqa/...py`) fails — always use `-m`.

## Layout

```
src/pathvqa/
├── _compat.py      legacy-checkpoint unpickle shim (utils.config alias)
├── api/            inference.py, schemas.py        FastAPI service layer
├── data/           pathvqa_dataset.py, dataset_adapter.py
├── models/         pathvqa_model.py (NeuroSymbolicPathVQA, build_model), fusion.py,
│                   ema.py, protocols.py, visual/visual_encoder.py,
│                   question/question_encoder.py
├── symbolic/       scene_parser.py, query_parser.py, executor.py, dsl.py,
│                   routing.py, ltn.py, aux_losses.py, ontology_loss.py,
│                   poe.py,
│                   diff_query_classifier.py, diff_target_extractor.py, diff_executor.py
└── utils/          config.py, metrics.py, conformal.py, concept_intervention.py,
                    interpretability.py, visualization.py, seed.py, logging_utils.py

Entry points: train.py, evaluate.py, serve.py;
scripts/run_ablations.py (A–E matrix), scripts/generate_paper_figures.py
(publication figures from outputs/*.json)
CI: .github/workflows/ci.yml runs the four gates above.
```

## Gotchas

- **Untracked files**: `src/pathvqa/data/*.py` are untracked but NOT ignored — `.gitignore` now uses anchored `/data/`, so `git ls-files src/pathvqa/data/` is empty. Add them explicitly when committing.
- **HF cache**: `pathvqa_dataset.py` sets `HF_HOME` to `data/hf-cache/`; auto-downloads `flaviagiammarino/path-vqa`.
- **Image modes mixed** (CMYK/RGB/L) — keep `.convert("RGB")`.
- **Answer vocab** built from training set at runtime; `<UNK>` reserved for unseen answers (~14% of val/test).
- **Region names** derived from answer vocab via `build_region_names()` — model is dataset-dependent.
- **Checkpoints** store model/optimizer/scheduler/scaler/EMA state, config, vocab, region names, early-stop state. `backfill_config()` fills fields missing from older configs (detects missing fields from the *instance* `__dict__`, not `hasattr` — dataclass class defaults would otherwise hide them); checkpoint config wins on resume. RNG state not saved.
- **Legacy checkpoints**: the never-consumed density head was removed, so old state dicts carry `scene_parser.density_classifier.*`. `migrate_legacy_scene_head()` (`src/pathvqa/_compat.py`) strips them in train/evaluate/inference so `strict=True` still loads. Pre-restructure checkpoints (pickled as `utils.config.*`) are handled by the module-alias shim in `_compat.py`.
- **Pre-DSL checkpoints**: the Aug-18 `best_model.pt` (521 keys) predates the differentiable DSL interpreter and carries no `dsl_interpreter.*` params. `is_pre_dsl_checkpoint()` (`src/pathvqa/_compat.py`) detects them from the state dict; train/evaluate/inference then flip `symbolic.dsl_enabled=False` so the model rebuilds as the architecture that saved them — random-initialized interpreter weights would otherwise load silently and mis-evaluate. Val: 59.55% / test: 59.73% under current code (Aug-era headline 60.33/59.87 — same drift direction on both splits; `train.py --debug` and `evaluate.py` agree exactly on identical weights, so the rulers are self-consistent).
- **Multi-scale**: `NeuroSymbolicPathVQA.forward(use_multiscale=None)` defaults to the value the model was built with (`config.visual.use_multiscale`); previously every caller silently fell back to global-only, leaving the crop branch untrained.
- **Debug runs** write only to `checkpoints/debug/` — never touch production artifacts; `Trainer._should_write_baseline()` also suppresses `outputs/<name>.json` in debug.
- **uint8 tensor cache** at `data/cache/`, keyed by `(image_size, norm, TRANSFORM_VERSION)` (e.g. `224x224_clip_v1`); bump `TRANSFORM_VERSION` in `pathvqa_dataset.py` whenever preprocessing changes. Writes are atomic.
- **RandAugment** applies train-only on uint8 tensors, before float conversion/normalize.
- **AMP** enabled by default on CUDA; TF32 matmul + cuDNN benchmark set at startup. Scene-head logits are fp16/bf16 under autocast — ops requiring matching dtypes (e.g. `scatter_add_`) must cast sources to the buffer dtype (regression-tested in `tests/test_audit_regressions.py::TestDiffExecutorAutocast`).
- **`evaluate.py --ablation`** applies overrides before the model is built; evaluating a checkpoint that lacks the required modules fails at `load_state_dict` (by design — no silent mis-evaluation).
- **Fusion is PoE in log space**: `log p = (1-g)·log p_neural + g·log p_symbolic`, NLLLoss on fused log-probs — not an arithmetic logits blend. The single implementation is `SymbolicFusionEngine` (`src/pathvqa/symbolic/poe.py`); train.py, evaluate.py and api/inference.py all delegate to it — never re-implement fusion locally. `soft_target` must be sized from `len(region_names)`, not `config.symbolic.num_regions` (stale checkpoint configs would crash).

## Config defaults (`src/pathvqa/utils/config.py`)

| Field | Default |
|---|---|
| `data.batch_size` | 32 |
| `data.image_size` | (224, 224) |
| `data.use_randaugment` / `use_cache` / `norm` | True / True / "clip" |
| `visual.backbone` / `model_name` | plip / vinid/plip |
| `visual.num_object_features` / `num_objects` | 512 / 49 |
| `visual.lora_rank` / `lora_alpha` | 16 / 32 |
| `visual.use_multiscale` | False (global-only unless enabled) |
| `question.encoder` / `hidden_dim` | distilbert / 768 |
| `question.lora_rank` / `lora_alpha` | 16 / 32 |
| `training.learning_rate` | 1e-4 |
| `training.num_epochs` / `early_stop_patience` | 50 / 10 |
| `training.use_amp` / `grad_clip` | True / 5.0 |
| `training.deterministic` | False |
| `symbolic.query_parser` / `executor` | learned / differentiable |
| `symbolic.weighting_strategy` / `symbolic_weight` | static / 0.3 |
| `symbolic.qtype_warmup_epochs` / `qtype_warmup_weight` | 3 / 0.1 |
| `symbolic.ltn_enabled` / `ltn_weight` | True / 0.1 |
| `symbolic.conformal_alpha` / `conformal_max_set_size` | 0.1 / 1 |

Debug overrides: batch=16, epochs=5, 500 train / 100 val samples.

## Docker & API

```bash
docker compose up --build    # builds + serves API on :8000; mounts checkpoints read-only
```

Requires NVIDIA runtime for GPU passthrough. `CHECKPOINT_PATH` env var selects the model file (default `checkpoints/best_model.pt`).

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + device info |
| `/model/info` | GET | Checkpoint metadata |
| `/predict` | POST | image_base64 + question → answer + confidence |
| `/predict/explain` | POST | Same, plus symbolic execution trace |
