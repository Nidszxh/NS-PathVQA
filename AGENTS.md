# PathVQA — Agent Guide

> Working rules and verified reference. System reference, novelty register and
> history: `CHANGELOG.md`. Evidence status and implementation ledger: `AUDIT.md`.
> The master-run procedure is `RUN.md`. If documents disagree, `CHANGELOG.md` wins;
> if docs disagree with code, trust the code.

## Hard rules

1. **Activate the venv first**: `source ~/.zangestu/bin/activate` (Python 3.12; the system interpreter has no torch).
2. **No commits or adds until the user says so** — never `git add`, `commit`, or `push` unless explicitly instructed in that message.
3. **No new dependencies** without justification.
4. **Run from repo root** — `train.py`, `evaluate.py`, `serve.py` resolve `data/`, `checkpoints/`, `logs/`, `outputs/` relative to cwd.
5. **Use `--debug` for iteration**; never start a full training run without asking.
6. **Preserve neuro-symbolic fusion** — don't replace it with a neural-only path. The neural-only baseline must stay available as an ablation (`--ablation I`, or `--no-symbolic`).

## Setup

```bash
source ~/.zangestu/bin/activate
uv pip install -r requirements.txt   # pinned runtime versions (local truth)
uv pip install -e .                  # once per env; provides `pathvqa.*`
```

CI installs differently (`uv pip install --system -e ".[dev]"`), so a local
`.[dev]` install can drift from the pins in `requirements.txt`.

## Commands

```bash
# Training
python train.py                          # neuro-symbolic (default)
python train.py --no-symbolic            # neural-only ablation
python train.py --debug                  # 500 train/100 val, batch=16, 5 epochs -> checkpoints/debug/
python train.py --resume auto            # newest checkpoints/checkpoint_epoch_N.pt ('--resume <path>' also works)
python train.py --experiment NAME        # names outputs/<name>.json + logs/<name>/train.log
python train.py --ablation A-K           # A-H parser×executor×gate factorial, I-K controls; see utils/ablation.py
python train.py --weighting-strategy static|learned   # fusion gate (overrides config)
python train.py --deterministic          # reproducible cuDNN kernels (slower)

# Evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt \
    [--split val|test] [--max_samples N] [--output FILE] [--ablation A-K] \
    [--image_condition actual|blank|shuffled|image_only] [--num_workers 0] \
    [--intervention none|probe|attribution|counterfactual|all]

# API server (:8000)
python serve.py --checkpoint checkpoints/best_model.pt [--host HOST] [--port PORT]
```

## Verification (run after every code change)

```bash
ruff format --check .     # CI gate
ruff check .              # CI gate
mypy src/                 # strict; 40 files
pytest                    # full suite; addopts enforces >=80% coverage (latest audit: 86.35%)
pytest --no-cov -q tests/test_ablation.py   # fast subset during iteration
```

Always pass `--no-cov` for subsets — `pytest tests/test_config.py` alone fails
the 80% coverage gate.

- **Checkpoint-sized temp files**: integration tests save real checkpoints
  (~1.4 GB each) under pytest's basetemp (default `/tmp/pytest-of-$USER`). On
  tmpfs/quota-constrained boxes this surfaces as `OSError: [Errno 28]`; pass
  `--basetemp=scratch/pytest` (gitignored) and delete it afterwards.
- **HF downloads**: tests instantiate real CLIP/DistilBERT encoders, so the first
  run needs network or a populated HF cache — on a fresh offline box the suite
  fails before exercising the code.
- **16 modules have `__main__` smoke guards**; run them with `-m`:
  `python -m pathvqa.symbolic.executor`, `python -m pathvqa.models.pathvqa_model`,
  plus `models/{fusion,ema}`, `models/visual/visual_encoder`,
  `models/question/question_encoder`, `symbolic/{scene_parser,query_parser,dsl,ltn}`,
  `utils/{config,interpretability,seed,logging_utils}`,
  `data/{dataset_adapter,pathvqa_dataset}`.

## Layout

```
src/pathvqa/
├── _compat.py      legacy-checkpoint unpickle shim + pre-DSL/density-head migrations
├── api/            inference.py, schemas.py          FastAPI service layer
├── data/           pathvqa_dataset.py (HF loader, cache, preprocessing),
│                   dataset_adapter.py (ontology-grounded regions)
├── models/         pathvqa_model.py (NeuroSymbolicPathVQA, build_model), fusion.py,
│                   ema.py, protocols.py, visual/visual_encoder.py,
│                   question/question_encoder.py
├── symbolic/       scene_parser.py, query_parser.py, diff_*.py, executor.py, dsl.py,
│                   routing.py, poe.py (shared fusion), ltn.py, aux_losses.py,
│                   ontology_loss.py
└── utils/          config.py, ablation.py (A–K matrix), metrics.py, conformal.py,
                    concept_intervention.py, interpretability.py, visualization.py,
                    seed.py, logging_utils.py
```

Entry points: `train.py`, `evaluate.py`, `serve.py`.
Scripts: `run_ablations.py` (A–K × repeated seeds), `run_image_dependence.py` +
`analyze_image_dependence.py` (paired image controls), `export_expert_review.py`,
`evaluate_pathmmu.py`, `generate_paper_figures.py` (real artifacts only; `--mock`
is watermarked and not evidence). CI: `.github/workflows/ci.yml` runs the four gates.

## Gotchas

- **Checkpoint format**: `torch.load(..., weights_only=False)` is required — `.pt` files pickle the `Config` dataclass. Stores epoch/step, model/EMA/optimizer/scheduler/scaler state, config, vocab mappings, region names, early-stop state; no RNG state. Checkpoint config wins on resume, then explicit CLI overrides are re-applied.
- **Legacy checkpoint migrations**: `backfill_config()` detects missing fields from the instance `__dict__` (not `hasattr`, since dataclass class defaults would hide them). `migrate_legacy_scene_head()` strips `scene_parser.density_classifier.*`. `is_pre_dsl_checkpoint()` rebuilds without `dsl_interpreter.*` — never "fix" that warning; random-init weights would silently mis-evaluate counts.
- **Pre-canonical retained checkpoint**: the legacy regex parser + hard executor + **static** PoE checkpoint (`g=0.3`, 40 regions) was saved as `checkpoints/best_model.pre_canonical.pt` before canonical training. Its config predates explicit parser/executor fields; backfill maps it to regex/hard, and missing DSL state disables the DSL interpreter. Its reproduction artifacts are `outputs/final_repro_{val,test}.json` (59.9457% val / 59.8154% test). In the completed run workspace, `checkpoints/best_model.pt` is the canonical seed-42 checkpoint (best epoch 22, stopped after 33 epochs): 60.28% val / 60.01% test. These ignored checkpoint/output files are not expected in a fresh clone. See `RUN.md` and `AUDIT.md`.
- **Evaluation weights**: `evaluate.py` and the API load the EMA snapshot when present (parity with the `best_val_acc` that selected the checkpoint). `--config` may override data/training fields, but `_ARCHITECTURE_FIELDS` (model names, dims, parser/executor/gate) are pinned back to the checkpoint's values with a warning.
- **`evaluate.py --ablation`** applies overrides before the model is built, so evaluating a checkpoint that lacks the required modules fails at `load_state_dict` — intentional, no silent mis-evaluation.
- **Fusion is PoE in log space**: `log p = (1-g)·log p_neural + g·log p_symbolic`, NLLLoss on fused log-probs. Single implementation: `SymbolicFusionEngine` (`symbolic/poe.py`); train/evaluate/API delegate to it — never re-implement locally. Size `soft_target` from `len(region_names)`, never `config.symbolic.num_regions`.
- **Fusion is per row**: identity/location rows receive the scene parser's region-answer distribution; count rows without a DSL interpreter, yes/no rows whose target is not a region, and attribute rows without a mapping write no evidence, keep `log p_neural` unchanged, and report `trace["symbolic_used"][i] == False`. Never blend with an all-zero/uniform symbolic distribution.
- **Yes/no evidence needs a region head**: the executor writes yes/no log-odds only when the parsed target matches a discovered region. A non-region target (e.g. "malignant") has no concept head; the max over organ presence answers a different question, so the row stays neural-only.
- **DSL output is log evidence**: `DifferentiableDSLInterpreter` writes `log(...)` on candidate answers and uses a finite `-30` floor for masked answers; the fusion DSL buffer is float32 so fp16 autocast cannot turn evidence floors into `-inf`. Count candidates come from the answer vocabulary (`count_answer_values`), not a fixed 1–5 list. Returning raw `[0,1]` scores would be inverted by the fusion `log_softmax`.
- **Learned parser must drive the selected executor**: with `symbolic.query_parser="learned"`, the hard executor consumes the argmax qtype and the most probable region target (above uniform chance), and the DSL program is recompiled for that qtype. Do not silently execute regex queries in that configuration.
- **`PathConfig.best_model_path` is independent of `checkpoint_dir`** — tests/scripts that redirect checkpoints must set both, or they overwrite the real `checkpoints/best_model.pt`.
- **Debug runs** write only under `checkpoints/debug/`; `Trainer._should_write_baseline()` also suppresses `outputs/<name>.json` when `config.debug`.
- **Dataset**: pinned HF revision `1685832883334b5bb5beaf4e4b333fdeecaa4ad9` of `flaviagiammarino/path-vqa`; `HF_HOME` is forced to `data/hf-cache/`; first run downloads. Answer vocab comes from the train split (`<UNK>` ≈ 14% of val/test). `checkpoints/`, `data/`, `logs/`, `outputs/` are gitignored, so a fresh clone has no weights or artifacts.
- **Images/cache**: image modes are mixed (CMYK/RGB/L) — keep `.convert("RGB")`. uint8 tensor cache at `data/cache/`, keyed by `(image_size, norm, TRANSFORM_VERSION)`; bump `TRANSFORM_VERSION` in `pathvqa_dataset.py` when preprocessing changes. RandAugment is train-only, applied on uint8 before float/normalize.
- **Regions**: `build_region_names()` derives region names from the answer vocab, so checkpoints are dataset-dependent.
- **Multi-scale**: `NeuroSymbolicPathVQA.forward(use_multiscale=None)` inherits the construction setting (`config.visual.use_multiscale`); don't pass an explicit `False` or the crop branch silently stops training.
- **AMP on CUDA**: scene-head logits are fp16/bf16 under autocast; ops needing matching dtypes (e.g. `scatter_add_`) must cast sources to the buffer dtype (regression-tested in `tests/test_audit_regressions.py::TestDiffExecutorAutocast`). The yes/no auxiliary BCE promotes presence probabilities to float32 before clamping and loss calculation to avoid fp16 rounding `1 - eps` to 1.0.

## Config defaults (`src/pathvqa/utils/config.py`)

| Field | Default |
|---|---|
| `data.batch_size` / `data.num_workers` | 32 / 6 |
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
`data.num_workers=6` can fail in locked-down containers — `evaluate.py` has
`--num_workers 0`; for training, override in a config JSON.

## Docker & API

```bash
docker compose up --build    # serves API on :8000; mounts checkpoints read-only
```

Requires the NVIDIA runtime for GPU passthrough. `CHECKPOINT_PATH` selects the
checkpoint (default `checkpoints/best_model.pt`).

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + checkpoint/device info |
| `/model/info` | GET | Checkpoint metadata |
| `/predict` | POST | image_base64 + question → answer + confidence |
| `/predict/explain` | POST | Same, plus symbolic execution trace |
