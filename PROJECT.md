# Neuro-Symbolic PathVQA — Project Reference

> **What this file owns:** the reference description of the **current system** — architecture, components, dataset, usage, setup, working conventions, config defaults, and file structure. It does **not** hold the execution plan (see `TODO.md`, which records phases and locked decisions), novelty claims (`NOVELTIES.md`), or the architecture critique / research roadmap (`NS-PATHVQA-AUDIT.md`). If this file conflicts with `TODO.md`, **`TODO.md` wins**.

A hybrid **neuro-symbolic Visual Question Answering (VQA)** system for medical pathology images (PathVQA). A neural path (PLIP + DistilBERT with LoRA, cross-attention fusion) is fused with an interpretable symbolic path (scene parser → query parser → executor) via a learned confidence gate. The system provides interpretable reasoning traces while achieving competitive accuracy on the PathVQA benchmark at a fraction of the compute of LLM-based systems.

## Document map (one owner per fact, 4 files total)

| File | Owns | Do NOT look here for |
|---|---|---|
| `PROJECT.md` | **Current-system reference** — architecture, components, dataset, usage, setup, conventions, config defaults, optimizations, file structure | Future plans, novelty claims, critique |
| `TODO.md` | **Execution roadmap** — phases, locked design decisions, technical gotchas, production track. *The authoritative plan.* | Current-code facts, novelty claims |
| `NOVELTIES.md` | **Novelty register** (C1–C10) — claims, status, verification, anti-claims, planned-contribution implementation specs | How to run the code |
| `NS-PATHVQA-AUDIT.md` | **Architecture critique & research roadmap** — strengths/limitations, ranked technical proposals, design alternatives, publication strategy | Execution order (superseded by `TODO.md` for anything already scheduled) |

If `NOVELTIES.md` or `NS-PATHVQA-AUDIT.md` conflicts with `TODO.md`, **`TODO.md` wins**.

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The Dataset](#2-the-dataset)
3. [Overall Architecture](#3-overall-architecture)
4. [Components](#4-components)
5. [How the Model Answers a Question](#5-how-the-model-answers-a-question)
6. [Project History](#6-project-history)
7. [Setup](#7-setup)
8. [Usage](#8-usage)
9. [Configuration](#9-configuration)
10. [Training & Evaluation Pipeline](#10-training--evaluation-pipeline)
11. [Optimizations](#11-optimizations)
12. [File Structure](#12-file-structure)
13. [Working Conventions & Gotchas](#13-working-conventions--gotchas)

---

## 1. What is this project?

This project is a **hybrid neuro-symbolic framework for Visual Question Answering (VQA)** on medical pathology images. It answers natural-language questions about medical images by combining:

- **A neural path** — a 2025-level deep learning stack (frozen PLIP + DistilBERT with LoRA adapters, cross-attention, MLP head) that learns rich visual-question representations end-to-end.
- **A symbolic path** — an interpretable, rule-based reasoning pipeline (scene parser → query parser → executor) that produces structured, explainable predictions on top of the neural features, grounded in a medical ontology (UMLS/RadLex).

The two paths are combined by a **learned, per-sample confidence gate** that decides how much to trust the symbolic path for each question, so the symbolic module *complements* the neural network with structured medical-domain reasoning (e.g. "which organ is affected", "what color is the lesion", "is the tumor malignant") rather than replacing it. A fixed additive fusion mode is retained behind a config flag as the ablation baseline.

The project trains end-to-end on an **RTX 4060 (8GB VRAM)** consumer GPU and is fully reproducible through a single training command.

### 1.1 The end-to-end system pipeline

The repo is an end-to-end ML pipeline. All stages are shipped except CI/CD (planned, see `TODO.md` Production track).

```
Dataset → Preprocessing → PyTorch model → Experiment tracking → Evaluation → FastAPI inference → Docker → CI/CD
```

| # | Stage | Where | Status |
|---|---|---|---|
| 1 | Dataset | `src/data/pathvqa_dataset.py` + `data/hf-cache/` | Shipped |
| 2 | Preprocessing | same file — `normalize_text`, aspect-preserving resize + center crop to 224×224, CLIP norm, RandAugment (train only), uint8 tensor cache, shared answer vocab + `<UNK>` | Shipped |
| 3 | PyTorch model | `src/models/*` + `src/symbolic/*` | Shipped |
| 4 | Experiment tracking | TensorBoard + structured logs → `logs/<experiment>/` | Shipped |
| 5 | Evaluation | `evaluate.py` → structured report in `outputs/` | Shipped |
| 6 | FastAPI inference | `serve.py` + `src/api/` | Shipped |
| 7 | Docker | `Dockerfile` + `docker-compose.yml` | Shipped |
| 8 | CI/CD | — | Planned (`TODO.md` Production track P3) |

### 1.2 Status at a glance

- **Best result:** 60.33% val / 59.87% test accuracy (CLIP+DistilBERT+LoRA + learned gate), **0.38 GPU-h**, ~4.1GB peak VRAM, single RTX 4060.
- **Phase 1 baseline:** 53.79% val accuracy (pre-modernization preprocessing) → `outputs/baseline.json`.
- **Phase 2:** data pipeline throughput +1.8× via uint8 tensor cache → `outputs/cache_benchmark.json`.
- **Phases 1–10:** shipped (see `TODO.md`). Production P1 (FastAPI) + P2 (Docker) shipped; P3 (CI/CD + packaging) parked.
- **Paper draft:** `outputs/paper_draft.md` (structure complete; final numbers pending final training runs).

---

## 2. The Dataset

**PathVQA** (`flaviagiammarino/path-vqa` on HuggingFace) — a visual question answering dataset of medical pathology images. Canonical HF mirror of the official UCSD-AI4H dataset (author split, deduplicated 32,632 QA pairs).

| Split | Samples |
|---|---|
| Train | 19,654 |
| Validation | 6,259 |
| Test | 6,719 |

Key properties:

- **Question types**: yes/no, identity ("what is shown?"), location ("where is it?"), attribute ("what color/shape/size?"), and count ("how many...").
- **Answers** are free-form strings (e.g. `"gastrointestinal system"`, `"malignant"`, `"red"`, `"yes"`).
- Images are **truncated TIFF files** — the benign `Truncated File Read` warnings are silenced in the dataloader. Modes are mixed (CMYK/RGB/L JPEGs) — handled by `.convert("RGB")`. Most images are 4:3 landscape, but **~5% are portrait** and the shortest side can be as small as ~180px.
- The dataset **auto-downloads** from HuggingFace on first load; it is cached locally (`data/hf-cache/`) and never committed.

### How the dataset is used

- Images are **aspect-preserving resized + center-cropped** to **224×224** (`resize_keep_aspect` — no distortion) and normalized with **CLIP mean/std** `(0.48145, 0.45783, 0.40821)` / `(0.26863, 0.26130, 0.27578)`, read at runtime from the `CLIPImageProcessor` config rather than hardcoded. ImageNet norm is not used — it is incompatible with CLIP features.
- **RandAugment** is applied on the training split only (val/test stay augmentation-free); magnitudes are kept moderate because pathology images are low-texture.
- A **preprocessed tensor cache** stores resized, center-cropped uint8 tensors keyed by `(image_size, norm)` — it is written atomically (temp file + rename) for worker concurrency, and rebuilds cleanly if the preprocessing config changes.
- The **answer vocabulary** is built at runtime from the training split (sorted unique answers).
- **Question tokenization** uses the **HF DistilBERT tokenizer** (`input_ids` + `attention_mask`); evaluation has no dependency on a prior training run.
- **Anatomical region names** used by the symbolic path are **ontology-grounded**: they are discovered from the training answer vocabulary and linked to standard medical ontologies (UMLS/RadLex, local mapping), giving a transferability story across medical VQA datasets. PathVQA-specific prefix matching remains as the fallback for unlinked answers, and linkage coverage is reported at train time.

---

## 3. Overall Architecture

```
PLIP (frozen + LoRA) → 49 patch tokens (768-d) → Linear 768→512 → object features (B, 49, 512) + mask
                                                                                              │
DistilBERT (frozen + LoRA) → [CLS] → question_state (768-d)                                    │
                                        │                                                    │
                                        └──── cross-attention transformer block ─────────────┘
                                                      │
                                          ┌───────────┼───────────┐
                                          │           │           │
                                    SceneParser   attended       │
                                    (neural →     features      │
                                     symbolic)       │          │
                                          │          └──── MLP ──┤
                                          │               │      │
                                    SymbolicExecutor   neural    │
                                    (query + facts →   logits   │
                                     answer logits)      │      │
                                          │               │      │
                                          └── routing gate ──┘    │
                                              │                  │
                                        gated fusion              │
                                              │                  │
                                        answer_logits ────────────┘
```

**The core fusion equation:**

```
g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)]))
final_logits = (1 - g) · neural_logits + g · symbolic_logits
```

where `h_attn` is the attended visual summary (512-d), `c_scene` is the scene-parser confidence (max over object-presence sigmoids), `onehot(qtype)` encodes the parsed question type, and `ent(attn)` is the attention entropy. The gate makes symbolic *trust* explicit and question-type-conditional, per sample.

The **fixed additive mode** `final = neural + symbolic_weight * symbolic` (default `symbolic_weight = 0.3`) remains available behind `symbolic.weighting_strategy ∈ {static, learned}` — it is the ablation baseline against which the gate is measured, and is never silently dropped.

> For the detailed forward-pass data flow (tensor shapes at every step) and a critique of this fusion (gradient dead-end at the executor boundary, etc.), see `NS-PATHVQA-AUDIT.md` §2.

---

## 4. Components

| # | Component | File | What it does |
|---|---|---|---|
| 1 | **Visual encoder** | `src/models/visual/visual_encoder.py` | **PLIP, frozen + LoRA** (rank 16, alpha 32, targets visual projection/MLPs). Feeds **all 49 patch tokens** (7×7 @ 224px, 768-d) through a **Linear 768→512 projection** → `(batch, num_objects=49, dim=512)` + all-ones mask. Positional embeddings are reused from PLIP (no box/spatial encoder). The projection keeps `visual.num_object_features` = 512, so the SceneParser/CrossModalAttention input contract is unchanged. Also hosts `MultiScaleVisualEncoder` (global 224×224 + 4 high-res quadrant crops, learned fusion gate) behind `use_multiscale` (currently off by default). |
| 2 | **Question encoder** | `src/models/question/question_encoder.py` | **DistilBERT, frozen + LoRA** (rank 16, alpha 32, targets attention q/k/v); hidden dim **768**. HF tokenizer (`input_ids` + `attention_mask`). Classifier `fusion_dim = 512 + 768 = 1280`. |
| 3 | **Cross-modal fusion** | `src/models/fusion.py` | **Cross-attention transformer block**: question queries attend over the 49 patch tokens, keeping the FP16-safe mask. Replaces the earlier concat+MLP attention. |
| 4 | **Neural classifier** | `src/models/pathvqa_model.py` | MLP over `[attended features ‖ question state]` → answer vocabulary logits. |
| 5 | **Scene parser** | `src/symbolic/scene_parser.py` | `SceneParser` — the *neural → symbolic bridge*. From attended features it predicts anatomical **region logits** (num_regions), **object presence** (sigmoid per region), and **attribute logits** for **17 colors, 9 shapes, 11 sizes**. Interface unchanged from the pre-modernization system. |
| 6 | **Query parser** | `src/symbolic/query_parser.py` | `parse_question()` — **rule-based regex** classification into `identity / location / yes_no / attribute / count`. Priority order: **attribute > count > yes_no > identity > location**, falling back to identity. Extracts the target word/phrase by matching question words against the answer vocabulary. |
| 7 | **Executor** | `src/symbolic/executor.py` | `execute()` — the *symbolic reasoning core*, vectorized via `torch.scatter_add_` (`VectorizedSymbolicExecutor`). Maps scene logits + parsed query to answer-vocabulary logits: identity/location → boost each region's answer entry; yes/no → boost binary `"yes"`/`"no"` classes with max region confidence; attribute → boost matching color/shape/size answers; count → skipped (no meaningful symbolic signal, neural-only). |
| 8 | **Routing gate** | `src/symbolic/routing.py` | The learned per-sample gate `g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)]))`; fuses via `(1-g)·neural + g·symbolic`. Static additive fusion remains behind `symbolic.weighting_strategy ∈ {static, learned}`. Mean `g` per question type is logged to TensorBoard. |
| 9 | **Main model** | `src/models/pathvqa_model.py` | `NeuroSymbolicPathVQA` — glues everything together; conditionally instantiates the scene parser and routing gate. Factory: `build_model(config, vocab_size, answer_vocab_size)`. |
| 10 | **Dataset adapter** | `src/data/dataset_adapter.py` | `DatasetAdapter.discover_regions(answer_vocab)` — ontology-grounded region discovery (UMLS/RadLex local linkage with PathVQA prefix matching as fallback); reports linkage coverage %. `PathVQAAdapter` (shipped), `VQARADAdapter` + `SLAKEAdapter` (implemented, not trained/evaluated — see `TODO.md`/`NS-PATHVQA-AUDIT.md` P11). |
| 11 | **LTN loss** | `src/symbolic/ltn.py` | `MedicalLogicTensorNetwork` — Product/Łukasiewicz T-norm fuzzy-logic clause satisfaction as an auxiliary loss (region-object coherence, attribute-certainty clauses). Disabled by default (`ltn_enabled=False`); adds `L_sat = 1 - satisfaction` to the CE loss when enabled. |
| 12 | **DSL interpreter** | `src/symbolic/dsl.py` | `DSLProgramCompiler` + `DifferentiableDSLInterpreter` — compiles questions into typed AST trees (`Filter`, `Count`, `Verify`, `QueryAttr`, `Relate`, `Exist`) and executes them differentiably over patch tokens with soft counting (`Σ σ(W v_i)`, Gaussian-kernel discretization). Implemented but **not wired into the main training pipeline** — standalone / future use. |
| 13 | **Config** | `src/utils/config.py` | Dataclass hierarchy `Config → {Data, Visual, Question, Training, Symbolic, Path}Config` with JSON save/load and **validation on load** (rejects unknown backbone/encoder/invalid combos early). Defaults are in §9 below. |
| 14 | **Data loader** | `src/data/pathvqa_dataset.py` | `PathVQADataset` wraps `load_dataset("flaviagiammarino/path-vqa")`. HF cache pinned to gitignored `data/hf-cache/`. Applies CLIP preprocessing + RandAugment (train only) and the uint8 tensor cache. Exposes `collate_fn`, `prepare_batch` (returns `input_ids`, `attention_mask`, `targets`, `images`), `normalize_text`, and `pathvqa_dataloader()` (accepts an injected `answer_to_idx` so val/test targets align with the train-vocab classifier head; unseen answers map to `<UNK>`). |
| 15 | **Interpretability utilities** | `src/utils/interpretability.py` | Faithfulness test, attention–anatomy Spearman correspondence, perception/parsing/execution failure taxonomy classifiers, counterfactual explanation generation. |
| 16 | **API** | `src/api/inference.py`, `src/api/schemas.py` | `PathVQAInference` model loading/caching layer + Pydantic request/response schemas backing `serve.py`. |

### Symbolic design details

- **Region names** are **ontology-grounded**: discovered from the answer vocabulary and linked to UMLS/RadLex via a local mapping, with the PathVQA prefix/organ-name matcher (`gastrointestinal`, `cardiovascular`, `hematologic`, `endocrine`, `female reproductive`, `nervous`, `respiratory`, `urinary`, `hepatobiliary` systems plus simple organ names like `lung`, `liver`, `heart`, `brain`, `skin`, `kidney`, ...) as the fallback. Linkage coverage % is reported at train time.
- The number of regions (`config.symbolic.num_regions`, default 50) is **overridden at runtime** by the discovered count — the scene parser is built with the actual discovered region count.
- Attribute vocabularies (color/shape/size) are extracted from the dataset's attribute answers.

---

## 5. How the Model Answers a Question

1. **Encode the image** — PLIP (frozen + LoRA) produces **49 patch tokens** (768-d) with PLIP positional embeddings; a **Linear 768→512 projection** maps them to fixed-size object features `(batch, 49, 512)`.
2. **Encode the question** — the question is tokenized by the **HF DistilBERT tokenizer** (`input_ids` + `attention_mask`) and passed through DistilBERT (frozen + LoRA) → question state (768-d).
3. **Attend** — a **cross-attention transformer block** makes the question queries attend over the 49 patch tokens → attended visual summary.
4. **Neural prediction** — `[attended ‖ question]` → MLP → neural logits over all answers.
5. **Symbolic prediction**:
   - `SceneParser` predicts region/attribute/object-presence logits from the same attended features.
   - `parse_question()` classifies the question type and extracts the target.
   - `Executor` converts scene logits into symbolic logits over the answer vocabulary according to the question type.
6. **Gate** — the routing gate computes `g` from the attended features, scene-parser confidence, question type, and attention entropy; `final = (1-g)·neural + g·symbolic` → `argmax` is the predicted answer. (In `static` mode, `final = neural + 0.3 × symbolic`.)

---

## 6. Project History

Milestone commits (oldest → newest):

| Milestone | Work done |
|---|---|
| **Initial setup** | Project skeleton, CLEVR-era experiments, dataset acquisition. |
| **Overall project completed** | First complete training pipeline for PathVQA. |
| **Bug fixes & refactors** | Dead code cleanup, bug fixes, documentation. |
| **Symbolic reasoning module** | Full neuro-symbolic integration: `SceneParser`, `QueryParser`, `Executor`, region discovery, attribute mappings, symbolic logits fusion. |
| **Optimization & finalization** | Vectorized visual encoder, persistent workers + prefetch, rolling checkpoints, early stopping, mixed-precision tuning, symbolic bug fixes, docs finalized. |
| **Full-system modernization** | Executed `TODO.md` Phases 1–10: unit-test safety net + recorded baseline; data pipeline (RandAugment, uint8 tensor cache); model upgrade to CLIP ViT-B/32 + DistilBERT + LoRA with cross-attention and HF tokenization; training machinery (AdamW + cosine + warmup, gradient accumulation/checkpointing, EMA, `set_seed`, structured logging); **learned confidence gate (C1)**; evaluation modernization (per-question-type breakdown, ECE calibration, uncertainty, BLEU/ROUGE, entity precision/recall, paired significance, structured report); external baselines table (C3 evidence); **ontology-grounded region discovery (C4)**; interpretability & failure taxonomy (faithfulness, attention–anatomy correspondence, perception/parsing/execution error buckets — C7); paper assembly. |
| **Neuro-symbolic 2026 core additions** | Differentiable LTN auxiliary loss (C2), DSL program synthesis + soft counting (C5, standalone), multi-scale morphological visual encoding (C6), medical ontology DAG + cross-dataset adapters, hierarchical reasoning traces + counterfactual explanations, vectorized symbolic executor. |

### Current status

- **Working**: full training (`train.py`), evaluation (`evaluate.py`), neural-only baseline (`--no-symbolic`), learned-gate ablation (`--weighting-strategy static|learned`), debug mode, custom configs, 14 `__main__` smoke tests + unit suite (105 pytest tests).
- **Completed features**: learned gated fusion (C1), differentiable LTN auxiliary loss (C2), resource-efficiency narrative (C3), ontology-grounded region discovery (C4), DSL soft-counting (C5, standalone), multi-scale visual encoding (C6, off by default), architecture-grounded failure taxonomy (C7), calibrated evaluation with statistical significance (C8), AMP training, EMA, gradient accumulation + checkpointing, uint8 tensor cache, RandAugment, checkpoint management, early stopping, TensorBoard + structured logging.
- **Planned** (`TODO.md` Production track): CI/CD (ruff → mypy → pytest → integration), packaging (`pip install -e .`, drop `sys.path.append`), fully differentiable neuro-symbolic pipeline (C9), concept bottleneck intervention testing (C10) — see `NOVELTIES.md` for specs.
- **Deliberately parked** (out of critical path, recorded in `TODO.md`): DSL program parsing wired into training, real count/spatial/comparative reasoning (count questions remain neural-only), scene-graph rewrite, SLAKE/RAD-VQA training runs, probabilistic logic fusion, testing/CI expansion, package restructuring.
- **Known simplification**: the executor skips `count` questions (no meaningful symbolic signal).

---

## 7. Setup

```bash
# 1. Create & activate the venv (torch needs Python ≤ 3.12; system Python 3.14 has no torch)
uv venv ~/.zangestu && source ~/.zangestu/bin/activate

# 2. Install dependencies (pinned with == for reproducibility)
uv pip install -r requirements.txt    # torch 2.12.1, torchvision, transformers 5.12.1, peft 0.20.0, datasets, Pillow, tqdm, tensorboard, accelerate, fastapi, uvicorn

# 3. Add new deps with
uv pip install <pkg>
```

- The venv lives at `/home/nidszxh/.zangestu/` (gitignored).
- **Always run scripts from repo root** — `train.py`, `evaluate.py`, and `serve.py` hardcode `sys.path.insert(0, 'src')`.
- Data and output directories (`checkpoints/`, `logs/`, `outputs/`, `data/`) are gitignored and auto-created.
- Data auto-downloads from `flaviagiammarino/path-vqa` on HF; `HF_HOME` is pinned to gitignored `data/hf-cache/` inside `pathvqa_dataset.py`, so caches aren't lost.
- Image modes are mixed (CMYK/RGB/L) — `.convert("RGB")` handles all; don't remove.

---

## 8. Usage

### Train

```bash
python train.py                                    # neuro-symbolic, learned gate (default)
python train.py --weighting-strategy static        # fixed additive fusion (ablation baseline)
python train.py --no-symbolic                      # neural-only baseline
python train.py --debug                            # small train/val sample (500/100), batch=16, 5 epochs
python train.py --debug --no-symbolic               # debug neural-only
python train.py --config path/to/config.json       # custom JSON config
python train.py --resume checkpoints/checkpoint_epoch_N.pt   # resume from checkpoint
python train.py --resume auto                      # resume from newest checkpoint_epoch_*.pt
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt                      # defaults to --split val
python evaluate.py --checkpoint checkpoints/best_model.pt --split test
python evaluate.py --checkpoint checkpoints/best_model.pt --max_samples 100 --output outputs/foo.json
```

Evaluation writes a structured report (`outputs/eval_results.json`) with per-question-type accuracy, calibration (ECE + reliability), uncertainty, BLEU/ROUGE, entity precision/recall, and paired significance tests against configured baselines.

### Serve

```bash
python serve.py --checkpoint checkpoints/best_model.pt              # FastAPI server (uvicorn)
```

Exposes `POST /predict`, `POST /predict/explain` (symbolic reasoning trace), `GET /health`, `GET /model/info`. Checkpoint is loaded once and cached. `Dockerfile` + `docker-compose.yml` provide GPU serving.

### Lint & test

```bash
pytest && ruff check .                             # 105 unit tests + lint
```

### Smoke tests

```bash
python src/models/question/question_encoder.py     # smoke test (14 modules total)
python src/models/visual/visual_encoder.py
python src/models/pathvqa_model.py
python src/models/fusion.py
python src/models/ema.py
python src/utils/config.py
python src/utils/seed.py
python src/utils/logging_utils.py
python src/utils/interpretability.py
python src/symbolic/scene_parser.py
python src/symbolic/query_parser.py
python src/symbolic/executor.py
python src/symbolic/dsl.py
python src/symbolic/ltn.py
```

---

## 9. Configuration

All settings live in the dataclass hierarchy in `src/utils/config.py` and can be overridden via a JSON config file. Configuration is **validated on load** — unknown backbones/encoders and invalid combinations are rejected before training starts. `config.py` supports data, visual, question, training, symbolic, and paths sub-configs; `Config.load()`/`save()` for JSON round-trips.

### Defaults

| Field | Default |
|---|---|
| `data.batch_size` | `32` |
| `data.image_size` | `(224, 224)` |
| `data.use_randaugment` | `True` |
| `data.use_cache` | `True` |
| `data.norm` | `"clip"` |
| `visual.backbone` | `plip` |
| `visual.num_object_features` | `512` |
| `visual.num_objects` | `49` |
| `visual.lora_rank` / `lora_alpha` | `16` / `32` |
| `question.encoder` | `distilbert` |
| `question.hidden_dim` | `768` |
| `question.lora_rank` / `lora_alpha` | `16` / `32` |
| `training.learning_rate` | `1e-4` |
| `training.num_epochs` | `50` |
| `training.early_stop_patience` | `10` |
| `training.use_amp` | `True` |
| `training.grad_clip` | `5.0` |
| `symbolic.weighting_strategy` | `static` (`learned` = gated fusion) |
| `symbolic.symbolic_weight` | `0.3` (used in `static` mode) |
| `symbolic.ltn_enabled` | `True` |

**Debug overrides** (`--debug`): batch=16, epochs=5, 500 train / 100 val samples.

Notable post-modernization fields not in the table above:

- **Visual**: `lora_target_modules` (visual projection/MLPs).
- **Question**: `max_seq_len = 64`; classifier `fusion_dim = 1280`.
- **Data**: `cache_key = (image_size, norm)`.
- **Training**: `optimizer = adamw`, cosine + linear warmup (`warmup_ratio = 0.05`), `grad_accumulation_steps`, `gradient_checkpointing`, `use_ema`, `ema_decay`.

---

## 10. Training & Evaluation Pipeline

### Training (`train.py` → `Trainer`)

1. **Build answer vocabulary** from the training split.
2. **Discover region names** via the dataset adapter (ontology-grounded with PathVQA fallback); build region→answer-index and attribute→answer-index mappings; override `num_regions`; report linkage coverage.
3. **Build dataloaders** (train shuffled with RandAugment + cache, val/test deterministic, augmentation-free; persistent workers, prefetch 2).
4. **Build the model** (`build_model`: frozen PLIP + DistilBERT with LoRA adapters, cross-attention, gate) and move to device.
5. **Loop epochs**: forward + gated (or static) symbolic fusion → `CrossEntropyLoss` (+ LTN clause loss if enabled) → AMP-scaled backward on accumulation boundaries → gradient clip → optimizer step → scheduler step. Gradient accumulation + checkpointing keep the effective batch above the 8GB limit. An **EMA** of weights is maintained for evaluation and persisted in checkpoints. Validation every epoch (or per `validate_every`), tracking neural + symbolic path accuracy and mean gate `g` per question type.
6. **Checkpointing**: periodic checkpoints `checkpoint_epoch_{N}.pt` (rolling, keeps last 3, auto-deletes older ones) + separate `best_model.pt`. Early stop after 10 val runs without improvement. Checkpoints store `ema_state_dict` alongside `model_state_dict`.
7. **TensorBoard + structured logs** to `logs/<experiment_name>/`.

### Evaluation (`evaluate.py` → `Evaluator`)

1. Loads the checkpoint (model weights + EMA, config, answer vocab, region names).
2. Rebuilds region/attribute mappings and the model from the checkpoint config (LoRA config is guaranteed identical because it is stored in the checkpoint).
3. Runs inference on `val` (default) or `test`, applies the same gated/static symbolic fusion, and writes a **structured report** (`outputs/eval_results.json`) with:
   - aggregate + **per-question-type accuracy**;
   - **calibration** (ECE) + reliability plots;
   - **uncertainty** (max-softmax, temperature scaling);
   - **BLEU/ROUGE** for free-form answers and precision/recall on anatomical entity parsing;
   - **paired significance tests** for headline comparisons.
4. Published-baseline comparison lives in `outputs/baselines.md` (accuracy + compute + citations).

### Checkpoint contents

`.pt` files store: `epoch`, `global_step`, `model_state_dict`, `ema_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_acc`, `config`, `answer_vocab`, `answer_to_idx`, `idx_to_answer`, `region_names`. LoRA adapters ride inside `model_state_dict`, so save/load needs no format change. `backfill_config()` fills fields missing from older checkpoint configs so older checkpoints stay loadable.

---

## 11. Optimizations

| Feature | Benefit |
|---|---|
| **LoRA adapters on frozen backbones** | Only ~small adapter weights train on CLIP + DistilBERT; fits RTX 4060 8GB at batch 16–32 |
| **AMP (automatic mixed precision)** | ~2× training speedup, ~40% lower VRAM via FP16 Tensor Cores (`torch.amp.GradScaler` + `autocast`, default on CUDA) |
| **cuDNN benchmark** | Auto-tunes conv algorithms at startup |
| **TF32 matmul precision** | Tensor-Core FP32 matmuls (`torch.set_float32_matmul_precision('high')`) |
| **Gradient accumulation** | Effective batch > 8GB physical limit |
| **Gradient checkpointing** | Cuts transformer activation memory when VRAM-tight |
| **EMA of weights** | Smoother eval weights (standard 2025 practice) |
| **RandAugment** | Train-time augmentation; val/test stay clean |
| **uint8 tensor cache** | Resized/cropped tensors cached once (keyed by `(image_size, norm)`) — removes repeated TIFF decode per epoch; ~4.5GB vs ~18GB fp32 for 19k train images |
| **Persistent DataLoader workers** | Workers stay alive across epochs — no image re-decode overhead |
| **Prefetch factor (2)** | Keeps the GPU fed |
| **Rolling checkpoints** | Only the last 3 periodic checkpoints kept + best model |
| **Early stopping** | Stops after 10 val runs without improvement |

---

## 12. File Structure

```
.
├── PROJECT.md               # This document (system reference — architecture, usage, conventions, config)
├── TODO.md                  # Execution roadmap (authoritative plan — phases, decisions, gotchas)
├── NOVELTIES.md              # Novelty register (claims C1–C10, status, verification, implementation specs)
├── NS-PATHVQA-AUDIT.md       # Architecture critique, ranked research proposals, publication strategy
├── train.py                 # Training entry point (Trainer)
├── evaluate.py               # Evaluation entry point (Evaluator)
├── serve.py                  # FastAPI inference server (uvicorn)
├── Dockerfile                # Container image for GPU serving
├── docker-compose.yml        # Docker Compose for GPU serving
├── requirements.txt          # Pinned deps: torch, torchvision, transformers, peft, accelerate, datasets, Pillow, tqdm, tensorboard, fastapi, uvicorn
├── pyproject.toml            # Tooling config (pytest / ruff / mypy)
├── tests/                    # Unit tests (pytest)
└── src/
    ├── api/
    │   ├── inference.py        # Model loading + predict/explain
    │   └── schemas.py          # Pydantic request/response models
    ├── data/
    │   ├── pathvqa_dataset.py  # Dataset + dataloader helpers (CLIP prep, RandAugment, cache)
    │   └── dataset_adapter.py  # Ontology-grounded region discovery (UMLS/RadLex + fallback); PathVQA/VQA-RAD/SLAKE adapters
    ├── models/
    │   ├── pathvqa_model.py    # NeuroSymbolicPathVQA, build_model
    │   ├── fusion.py            # Cross-attention transformer block
    │   ├── ema.py                # EMA of model weights
    │   ├── question/question_encoder.py    # DistilBERT + LoRA + HF tokenizer
    │   └── visual/visual_encoder.py    # PLIP + LoRA → patch tokens → 768→512 projection; MultiScaleVisualEncoder
    ├── symbolic/
    │   ├── scene_parser.py     # Neural→symbolic: region/object/attribute logits
    │   ├── query_parser.py     # Rule-based question classification
    │   ├── executor.py         # Query + scene logits → answer-vocab logits (vectorized)
    │   ├── routing.py          # Learned confidence gate (C1)
    │   ├── ltn.py               # Differentiable Logic Tensor Network auxiliary loss (C2)
    │   └── dsl.py                # DSL program compiler + differentiable soft-counting interpreter (C5, standalone)
    └── utils/
        ├── config.py            # Config dataclasses + defaults + load-time validation
        ├── metrics.py            # ECE, temperature scaling, uncertainty, paired bootstrap test
        ├── interpretability.py  # Faithfulness, attention-anatomy correspondence, failure taxonomy, counterfactuals
        ├── logging_utils.py     # Structured logging setup
        └── seed.py               # Deterministic seeding (torch/numpy/random/CUDA)
```

---

## 13. Working Conventions & Gotchas

### Working principles

- **Neuro-symbolic**: preserve the fusion and its explainability. Additive baseline must stay as ablation (`weighting_strategy=static`).
- **Medical domain**: research system, not clinical. Keep terminology accurate; no clinical claims (see `NS-PATHVQA-AUDIT.md` Discussion/Limitations framing).
- **Reproducibility**: fixed seeds, `--debug` fast path, smoke tests as gate.
- **Minimal footprint**: no new deps without justification (currently approved: `transformers`, `peft`). Symbolic pipeline passes tensors directly (no scene-graph object).
- **No git add/commit/push without explicit instruction.**
- No code outside the current `TODO.md` phase's checklist; every change: `python train.py --debug` clean, smoke tests pass, seeds fixed.

### Conventions

- **Answer vocab** built from training set at runtime. `train.py` reserves `<UNK>` for unseen answers (~14% of val/test).
- **Region names** derived from answer vocab via ontology-grounded discovery — model is dataset-dependent.
- **HF tokenizer** via `get_question_tokenizer()`.
- **Checkpoints** store model/optimizer/scheduler/scaler/EMA state, config, vocab, region names, early-stop state. `backfill_config()` fills missing fields from older configs.
- **Resume**: `--resume <path>` or `--resume auto`. Config in checkpoint wins over CLI. **RNG state is not saved** — resumed runs are seed-identical but not bit-identical (same seed/val accuracy, different data order after resume).
- **Debug runs** write to `checkpoints/debug/`, never touch production artifacts (a `--debug` run previously overwrote the real `best_model.pt` — fixed via `get_debug_config()` redirecting `paths.checkpoint_dir`/`best_model_path`).
- **AMP** enabled by default on CUDA. cuDNN benchmark + TF32 matmul set at startup — `set_seed()` disables `cudnn.benchmark` for deterministic runs (benchmark mode is incompatible with `cudnn.deterministic=True`).
- **Scaler on accumulation boundaries**: with gradient accumulation + AMP, `scaler.step()`/`scaler.update()` (and `scaler.unscale_()`) run only on accumulation boundaries — scaling per micro-step corrupts the gradient norm.
- **RandAugment** train-only (`num_ops=2`, `magnitude=9`), applied on the uint8 tensor before float/normalize so it stays stochastic per epoch; val/test stay clean.
- **uint8 tensor cache** keyed by `(image_size, norm)` at `data/cache/<key>/<split>/`. Atomic writes (`mkstemp` + `os.replace`), `weights_only=True` loads. Changing preprocessing rebuilds the cache automatically via the key — never serve stale tensors.
- **CLIP mean/std** are read from the `CLIPImageProcessor` config at runtime, never hand-typed — a typo would silently degrade CLIP features.
- **Truncated-TIFF warnings** are benign and silenced in the dataloader; the mixed-mode `.convert("RGB")` and `resize_keep_aspect` are required — don't "optimize" them away.
- **`count` questions are neural-only** (the executor skips them; DSL soft-counting is implemented but not wired in).
- **Checkpoint compatibility is explicit**: pre-modernization checkpoints (ResNet/biLSTM/word-vocab) do **not** load after the CLIP/DistilBERT upgrade — by design. LoRA adapters ride inside `model_state_dict`, but reloading requires an identical LoRA config (rank/alpha/targets) — guaranteed because config is stored in the checkpoint. Do not enable `weights_only` tricks that break `ckpt["config"]` (a `Config` object).
- **Version pinning**: `requirements.txt` pins runtime deps with `==` (torch 2.12.1, transformers 5.12.1, peft 0.20.0, …) — keep it that way when bumping `transformers`/`peft`.
- **`evaluate.py` has no prior-training-run dependency** — the HF tokenizer is self-contained; `question_vocab.json` is obsolete and removed.
- **Verification**: `pytest` (105 tests) + `ruff check .` + the 14 `__main__` smoke tests.

### Known code-level issues (tracked in `NS-PATHVQA-AUDIT.md` Appendix, not yet fixed)

A short pointer list — full detail, file/line references, and fixes are in `NS-PATHVQA-AUDIT.md`:

- Gate's documented `ent(attn)` entropy input vs. actual `LearnedGate.forward()` signature — fixed (entropy now wired in as gate input).
- `faithfulness_test()` perturbs `region_logits` but re-runs the full model — fixed (perturbation now injected directly into executor).
- `QueryAttr` in `dsl.py` uses substring containment instead of vocab scatter — fixed (now uses `mapping[valid_attr_mask]` scatter).
- `paired_bootstrap_test` has dead code (`int(acc1*n1)` computed, unused) — fixed (dead code removed).
