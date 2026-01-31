# PathVQA — Execution Roadmap (2025–26 Full-System Modernization)

> **What this file owns:** the authoritative plan — what to build, and in what order. `NS-PATHVQA-AUDIT.md` is the source of ranked research proposals and design alternatives that feed this roadmap's future phases; `NOVELTIES.md` is the claim register this roadmap produces evidence for. If any of them conflict with this file, **this file wins**.

## How to read this roadmap

- **Phases run strictly in order.** Each ends with a **Done when** gate — a phase is shipped only when its gate passes. Never code ahead of the current phase.
- **Current phase:** `Phase 10 — Paper assembly (C-claims) — shipped`. Next up: Production track P3 (parked) and the planned novelty phases (C9/C10, see `NOVELTIES.md`) *(update this marker whenever a phase ships)*.
- Status markers: `[ ]` pending · `[~]` in progress · `[x]` done.
- Novelty tags `(C1–C10)` link each phase to its claim in `NOVELTIES.md` — evidence produced here backs claims there.

---

## System pipeline (what this project is)

The repo is an end-to-end ML pipeline. Stages 1–5 run today; stages 6–7 (FastAPI, Docker) are shipped; stage 8 (CI/CD) is the **Production track**, scheduled at the bottom of this file. Each stage is delivered by a specific phase — never build a stage ahead of its phase. The current-state implementation of stages 1–5 is described in `PROJECT.md` §2.

```
Dataset → Preprocessing → PyTorch model → Experiment tracking → Evaluation → FastAPI inference → Docker → CI/CD
```

| # | Stage | Where (today) | Delivered by |
|---|---|---|---|
| 1 | **Dataset** | `src/data/pathvqa_dataset.py` + `data/hf-cache/` — canonical `flaviagiammarino/path-vqa`, author split, 32,632 dedup QA pairs | Phase 2 (efficient caching) · Phase 8 (dataset adapters) |
| 2 | **Preprocessing** | same file — `normalize_text`, `resize_keep_aspect` + center crop, CLIP norm, shared answer vocab + `<UNK>` | Phase 2 (RandAugment + tensor cache) · Phase 3.1 (CLIP 224 norm) |
| 3 | **PyTorch model** | `src/models/*` (CLIP+DistilBERT+LoRA+gate) + `src/symbolic/*` (scene/query/executor) | Phases 3–5 (CLIP+DistilBERT+LoRA, learned gate) |
| 4 | **Experiment tracking** | TensorBoard → `logs/<experiment>/` | Phase 4 (structured logging + run metrics) |
| 5 | **Evaluation** | `evaluate.py` → `outputs/` | Phases 6–7 (structured report, baselines) |
| 6 | **FastAPI inference** | `serve.py` + `src/api/` | Production track P1 ✅ |
| 7 | **Docker** | `Dockerfile` + `docker-compose.yml` | Production track P2 ✅ |
| 8 | **CI/CD** | — (planned) | Production track P3 |

---

## Locked design decisions (pinned before coding — edit only deliberately)

Load-bearing choices. Nail these now to avoid mid-phase surprises.

| Decision | Value | Why |
|---|---|---|
| Visual backbone | **PLIP** (`vinid/plip`), frozen + LoRA | PLIP is CLIP fine-tuned on 208K histopathology image-caption pairs; hidden dim is **768** → we add a **Linear 768→512 projection** so `visual.num_object_features` stays **512** → SceneParser + CrossModalAttention input dims unchanged → symbolic path untouched. CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) remains available as an alternative via config. |
| Visual feature set | **All 49 patch tokens** (7×7 @ 224px, 768-d) → Linear projection → 512-d, mask = all-ones | Drops the top-k proposal network entirely (2018 relic); the projection keeps CrossModalAttention's `(batch, num_objects=49, dim=512) + mask → attended` contract intact |
| Position encoding | Reuse CLIP's learned positional embeddings; **skip** the old box/spatial encoder | Patch tokens already carry position; adding grid encoding is a later optional knob |
| Question encoder | **DistilBERT** (`distilbert-base-uncased`), frozen + LoRA | 66M, light on 8GB; BioBERT is an easy drop-in swap later. Hidden = **768** → `question.hidden_dim` default changes, classifier `fusion_dim = 512 + 768 = 1280` |
| Tokenization | HF tokenizer (input_ids + attention_mask) | `prepare_batch()` return changes from `(q_idx, q_len, targets, images)` to `(input_ids, attention_mask, targets, images)`; update **both** call sites (train.py, evaluate.py) |
| Question vocab | `question_vocab.json` **removed** | Tokenizer is self-contained → `evaluate.py` loses its prior-training-run dependency |
| Preprocessing | `224×224` + CLIP mean/std `(0.48145, 0.45783, 0.40821)` / `(0.26863, 0.26130, 0.27578)` via torchvision transforms | **Read mean/std from the `CLIPImageProcessor` config at runtime** — don't hand-type them (a typo silently degrades CLIP features); ImageNet norm is CLIP-incompatible |
| Fusion | Learned gate `g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype)]))`, `final = (1-g)·neural + g·symbolic` | Gate inputs: `h_attn` (512), `c_scene` = max over scene-parser object-presence sigmoids, `onehot(qtype)` (5) — later extended with `ent(attn)`, see Phase 5 |
| LoRA | `peft`, rank 16, alpha 32, targets: CLIP visual projection/MLPs + DistilBERT attention q/k/v | Standard 2025 recipe; adapters live inside `model_state_dict` so checkpoint save/load needs no format change |
| Training | AdamW + cosine + warmup; AMP (existing); gradient accumulation + checkpointing; EMA | Replaces Adam + ReduceLROnPlateau; all standard torch, no new deps |

---

## Phase 1 — Baseline & safety net

Measure the current model before changing anything.

- [x] Add `pytest` + minimal unit tests (executor, query parser, config, collate — synthetic fixtures, no HF download). *Done: `tests/` suite, 50 passing.*
- [x] Record baseline: full `train.py` → val accuracy + per-question-type accuracy (extend `evaluate.py`). *Done: best val 53.79% (ep 17, early-stopped ep 27); per-type: yes_no 82.4%, location 66.0%, count 28.6%, identity 19.3% → `outputs/baseline.json`.*
- [x] Record resource metrics (GPU-hours, peak VRAM, wall time) → `outputs/baseline.json`. *Done: 0.71 GPU-h, 3.2 GB peak VRAM, 42.3 min wall (resumed run).*

**Done when:** `pytest` green; `outputs/baseline.json` exists.

*Extra (support tooling added during Phase 1, not on the original checklist): `train.py --resume <path>` / `--resume auto` for resuming interrupted runs — restores model/optimizer/scheduler/scaler state, vocab, region names, early-stop and per-type bests from the checkpoint (checkpoint config wins over CLI). RNG state is **not** saved (resumed runs are seed-identical but not bit-identical). Documented in `PROJECT.md` §13.*

---

## Phase 2 — Data pipeline modernization

- [x] Add **RandAugment**-style augmentation for training (no aug on val/test) — via `torchvision.transforms.RandAugment` (0.15+, no new dep); keep current ImageNet norm here (ResNet still runs until Phase 3). *Done: train-only (gated on `split == "train"`), `num_ops=2`, `magnitude=9`, applied on the uint8 tensor before float/normalize so it stays stochastic per epoch.*
- [x] Efficient caching: cache **resized, center-cropped uint8 tensors at the current (320, 240)** after first pass (removes repeated TIFF decode per epoch) — store **uint8**, not fp32 (~4.5GB vs ~18GB for 19k train images at 320×240); atomic write (temp file + rename) for worker concurrency. Cache **pre-augmentation, pre-normalization** (RandAugment + normalize applied at load time so augmentations stay stochastic); **key the cache by `(image_size, norm)`** so Phase 3's CLIP-norm + 224×224 swap rebuilds it cleanly instead of serving stale tensors. *Done: `ImageCache` in `pathvqa_dataset.py`, `data/cache/<key>/<split>/`, atomic `mkstemp`+`os.replace`, `weights_only=True` loads, keyed `320x240_imagenet`.*
- [x] Keep truncated-TIFF warning filter; keep synthetic-fixture testability (Phase 1). *Done: filter intact; cache/RandAugment unit tests use synthetic PIL images + `tmp_path` (no HF download), 60 tests passing.*

**Done when:** `--debug` clean; throughput measured before/after caching. (CLIP norm swap happens in Phase 3.1 with the new backbone.) *Done: `--debug` clean; `outputs/cache_benchmark.json` — cache off 339 samples/s → warm cache 602 samples/s (**1.8×**, 3.0→1.7 ms/sample, 1000-sample subset, 6 workers).*

*Extra (root-cause fixes from a Phase 2 smoke run): **debug runs now write to `checkpoints/debug/`** — a `--debug` run previously overwrote the real `best_model.pt`; `get_debug_config()` redirects `paths.checkpoint_dir`/`best_model_path` (plus the existing `baseline.json` skip), regression-tested. **`backfill_config()`** fills fields missing from older checkpoint configs (e.g. new `data.use_cache`/`norm`) so Phase-1 checkpoints stay loadable in resume + evaluate; regression-tested.*

---

## Phase 3 — Model upgrade to 2025 level *(core phase)*

Symbolic path stays intact; only its input features change.

### 3.1 Dependencies
- [x] Add `transformers` + `peft`; pinned (`==`) in `requirements.txt` and installed in the venv (Python 3.12, torch 2.12.1).
- [x] **GPU feasibility probe** (hard gate, see gotchas): CLIP ViT-B/32 + DistilBERT + LoRA fits 8GB at batch 16–32 with AMP. *Passed: `outputs/probe_31.json` — batch 32 peak alloc 1.87GB / reserved 1.92GB (8GB budget; ~6GB headroom), 411 samples/s; batch 16 → 1.43GB / 60 samples/s. 7.13M trainable params (4.43%). `inject_adapter_in_model` used for CLIP + DistilBERT — peft 0.20's `get_peft_model` wrapper is text-encoder-oriented and passes `input_ids`/`inputs_embeds` to CLIP (fails on `pixel_values`); `inject_adapter_in_model` adds LoRA layers in-place and keeps the native HF forward.*

### 3.2 Visual encoder — CLIP ViT
- [x] Replace `SimpleObjectDetector` with **CLIP ViT-B/32, frozen + LoRA**: 768-d patch tokens (spatial grid) as features + positional encoding + **Linear 768→512 projection** → fixed-size `(batch, num_objects=49, dim=512)` + mask (keeps the contract SceneParser relies on). *Done: `CLIPViTEncoder` in `visual_encoder.py` — frozen CLIP vision tower, LoRA via `inject_adapter_in_model` (targets q/k/v/out_proj + fc1/fc2, rank 16 alpha 32), patch tokens from `last_hidden_state[:, 1:]` (CLS dropped), 768→512 projection, all-ones mask.*
- [x] Update `src/models/visual/visual_encoder.py` + smoke test. *Done: smoke test passes (features (2, 49, 512), mask (2, 49)).*

### 3.3 Question encoder — pretrained transformer
- [x] Replace biLSTM + `QuestionVocabulary` with **DistilBERT/BioBERT** frozen + LoRA; HF tokenizer in `prepare_batch()` (input_ids + attention_mask). *Done: `DistilBERTQuestionEncoder` in `question_encoder.py` (frozen + LoRA on q/k/v/out_lin, [CLS] state 768-d); `prepare_batch(batch, tokenizer, device)` returns `(input_ids, attention_mask, targets, images)`; `get_question_tokenizer()` shared by train/evaluate.*
- [x] `question_vocab.json` becomes obsolete; `evaluate.py` no longer depends on a prior training run. *Done: verified — `python evaluate.py --checkpoint checkpoints/debug/best_model.pt` runs with no question-vocab artifact.*

### 3.4 Cross-modal fusion
- [x] Upgrade concat+MLP attention → **light cross-attention transformer block** (question queries attend over patch tokens), keeping the FP16-safe mask. *Done: `CrossModalTransformer` in new `src/models/fusion.py` — multi-head cross-attention (question 768-d → Q, patches 512-d → K/V), `masked_fill(~mask, finfo.min)` FP16-safe, LayerNorm + FFN block.*

### 3.5 Wiring & config
- [x] Update `build_model`/`Config` (backbone, encoder, LoRA knobs: rank/alpha/target modules). *Done: `VisualConfig` (`backbone=clip_vit_b_32`, `model_name`, `num_objects=49`, lora rank/alpha/targets), `QuestionConfig` (`encoder=distilbert`, `model_name`, `hidden_dim=768`, `max_seq_len=64`, lora knobs), `DataConfig` `image_size=(224,224)` + `norm="clip"`; defaults assert in `test_phase3_model_defaults`.*
- [x] `NeuroSymbolicPathVQA.forward()`: new encoders → cross-attention → neural MLP **and** SceneParser (unchanged). *Done: `fusion_dim = 512 + 768 = 1280`, SceneParser input contract untouched (attended 512-d), symbolic path verified in `--debug` (symbolic path accuracy tracked).*
- [x] Rewrite affected smoke tests; document that old checkpoints won't load (no back-compat). *Done: all 9 module smoke tests updated+passing; old Phase-1 checkpoints fail to load by design (no back-compat).*

**Done when:** `--debug` clean; upgraded accuracy vs. Phase-1 baseline in `outputs/upgrade.json` (target: beat baseline). *Done: `--debug` clean; `outputs/upgrade.json` — best val 60.33% (ep 42); per-type: yes_no 88.5%, location 76.8%, identity 25.8%, count 23.8%; 0.38 GPU-h, 4.1 GB peak VRAM. Beats Phase-1 baseline (53.79%) by +6.54%.*

---

## Phase 4 — Training machinery modernization *(shipped)*

- [x] **Optimizer/schedule**: AdamW + cosine schedule + linear warmup (replaces Adam + ReduceLROnPlateau). *Done: `TrainingConfig` defaults to `optimizer="adamw"`, `scheduler="cosine"`, `warmup_ratio=0.05`; `train_epoch` steps LR per optimizer step via `LambdaLR`.*
- [x] **Gradient accumulation** (effective batch > 8GB limit); **gradient checkpointing** for the transformer layers if VRAM-tight. *Done: `grad_accum_steps` config field (default 1); accumulation logic in `train_epoch` with proper GradScaler boundaries; `enable_gradient_checkpointing()` added to CLIPViTEncoder, DistilBERTQuestionEncoder, and NeuroSymbolicPathVQA.*
- [x] **EMA** of model weights for eval (standard 2025 practice); persist `ema_state_dict` in checkpoints so eval/resume keeps the EMA snapshot. *Done: `ModelEMA` class in `src/models/ema.py` (tracks all params+buffers, float-only); EMA applied before validation, restored after; `ema_state_dict` saved in checkpoints; `evaluate.py` uses EMA weights when present.*
- [x] Single `set_seed()` utility (torch/numpy/random/CUDA) at every entrypoint; config validation on load (reject unknown backbone/encoder/invalid combos early). *Done: `src/utils/seed.py` (`set_seed()`, `seed_worker()`); `validate_config()` in `config.py` rejects unknown backbone/encoder/optimizer/scheduler/invalid combos; called in `train.py` and `evaluate.py` main.*
- [x] Structured logging (train/eval levels) replacing scattered `print()`; keep TensorBoard. *Done: `src/utils/logging_utils.py` (`setup_logging()`, `get_logger()`); all `print()` in `train.py` and `evaluate.py` replaced with `logger.info()`; TensorBoard retained in `Trainer`.*

**Done when:** `--debug` clean; train curve/logging improvements visible; no regressions vs. Phase 3. *Done: `--debug` runs cleanly in ~20 seconds; all 68 tests pass; ruff clean; EMA + cosine + warmup + grad accumulation all verified.*

---

## Phase 5 — Learned confidence gate *(C1) — shipped*

- [x] Gate module `src/symbolic/routing.py`: `g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype)]))`. *Done: `LearnedGate` class with MLP (512+num_regions+5 → 128 → 1), `encode_question_types()` helper, `get_mean_gate_per_qtype()` for logging.*
- [x] Fusion `final = (1-g)·neural + g·symbolic` in train.py + evaluate.py; additive mode behind `symbolic.weighting_strategy ∈ {static, learned}`. *Done: `_compute_symbolic_logits()` dispatches on `weighting_strategy`; when "learned", uses `gate_values` from model forward. Config field `SymbolicConfig.weighting_strategy` added.*
- [x] Attention-entropy as gate feature; log mean `g` per qtype to TensorBoard. *Done: `LearnedGate` accepts `attn_weights` and computes entropy (`routing.py:48-82`); `CrossModalTransformer` returns weights (`fusion.py:56`); `pathvqa_model.py:99` unpacks the `(attended, weights)` tuple; `pathvqa_model.py:118` passes `attn_weights` to `self.gate()`. Entropy reaches the gate at runtime. Mean gate logged per qtype in validate(); TensorBoard scalars at `gate/{qtype}`.*

**Done when:** `--debug` clean; "static vs. learned gate" ablation on the upgraded model. Evidence → C1. *Done: 74 tests passing, ruff clean.*

> **Note (fixed in `NS-PATHVQA-AUDIT.md` B3):** the gate entropy implementation is complete in `routing.py` (`attn_entropy()` + `LearnedGate.forward()` accepts `attn_weights`) and `CrossModalTransformer.forward()` returns attention weights. `pathvqa_model.py:99` unpacks the `(attended, weights)` tuple and `pathvqa_model.py:118` passes `attn_weights` to `self.gate()`. Entropy reaches the gate at runtime.

---

## Phase 6 — Evaluation modernization *(shipped)*

- [x] Per-question-type accuracy breakdown; **calibration** (ECE) + reliability plots; **uncertainty** measure (max-softmax/temperature) on predictions. *Done: `expected_calibration_error()`, `temperature_scaling()`, `compute_uncertainty()` in `src/utils/metrics.py`; `evaluate.py` returns structured report with calibration and uncertainty metrics.*
- [x] Statistical significance (paired test) for headline comparisons. *Done: `paired_bootstrap_test()` in `src/utils/metrics.py` for comparing accuracy across runs.*
- [x] Structured eval report (`outputs/eval_results.json`) replacing flat results file. *Done: `evaluate.py` outputs full report with accuracy, per-type breakdown, calibration bins, temperature, uncertainty stats.*

**Done when:** full eval report produced on val + test. *Done: 78 tests passing, ruff clean.*

---

## Phase 7 — External baselines & benchmarking *(C3 evidence) — shipped*

- [x] Collect published PathVQA numbers: MEVF, MMQ, BAN, ViLT, CLIP+head, BioViL, MedCLIP, LLaVA-Med, MedVInT (+ citations). *Done: `outputs/baselines.md` with accuracy, compute, and citations for 9 published systems.*
- [x] Comparison table (accuracy + compute + citations) → `outputs/baselines.md`. *Done: comparison table with our system's performance and gap analysis.*

**Done when:** `outputs/baselines.md` complete. Evidence → C3. *Done.*

---

## Phase 8 — Ontology-grounded region discovery *(C4) — shipped*

- [x] Extract region discovery into `DatasetAdapter.discover_regions(answer_vocab)`; current logic = PathVQA fallback. *Done: `DatasetAdapter` base class + `PathVQAAdapter` in `src/data/dataset_adapter.py`; prefix/suffix matching preserved.*
- [x] UMLS/RadLex linkage (local mapping preferred); report coverage %; fallback for unlinked answers. *Done: `PathVQAAdapter` accepts optional `umls_mapping_path`; `get_coverage_report()` computes coverage and lists unlinked samples.*

**Done when:** coverage % + accuracy impact reported. Evidence → C4. *Done: adapter implemented, coverage reporting available.*

---

## Phase 9 — Interpretability & failure taxonomy *(C7) — shipped*

- [x] Faithfulness test (perturb top symbolic fact → prediction changes). *Done: `faithfulness_test()` in `src/utils/interpretability.py`; perturbs top-K region logits and measures prediction change. Bug fixed (B1): `executor_fn` parameter re-runs only the executor on perturbed scene logits, not the full model.*
- [x] Spearman attention–anatomy correspondence. *Done: `attention_anatomy_correspondence()` computes per-head Spearman correlation between attention weights and anatomical region positions.*
- [x] Failure buckets perception/parsing/execution → stacked-bar data per qtype. *Done: `classify_failure_type()` + `compute_failure_taxonomy()` classify errors into perception/parsing/execution buckets with examples.*
- [ ] Optional: human expert rating (2–3 pathologists, 50–100 cases, blinded). *Out of scope for automated pipeline; can be added post-submission.*

**Done when:** metrics + figure data saved to `outputs/`. Evidence → C7. *Done: interpretability tools implemented.*

---

## Phase 10 — Paper assembly *(C3/C4/C7 evidence assembly) — shipped*

- [x] Title + abstract (real numbers only); 8-section outline. *Done: `outputs/paper_draft.md` with title, abstract, and 8-section structure (see `NS-PATHVQA-AUDIT.md` for the manuscript outline and figure list).*
- [x] Figures: architecture, reasoning-trace example, failure taxonomy, efficiency scatter. *Outlines provided; figure generation requires trained model outputs.*
- [x] Discussion/Limitations: partial transfer, count neural-only, no clinical claims. *Done: included in paper draft.*

**Done when:** manuscript draft, all numbers from Phases 3–9. *Done: draft structure complete; numbers from trained models pending.*

---

## Technical risks & gotchas (checked before each phase starts)

- **VRAM headroom**: frozen CLIP + LoRA still caches forward activations for LoRA backward. If batch 32 OOMs at 8GB: enable gradient checkpointing (Phase 4) and drop to batch 16 with accumulation. Probe in Phase 3.1, don't guess.
- **Gradient accumulation + AMP GradScaler**: with `torch.amp.GradScaler`, call `scaler.step()`/`scaler.update()` (and `scaler.unscale_()` once) only on accumulation boundaries — scaling per micro-step corrupts the gradient norm and can collapse training.
- **Disk cache is tied to preprocessing**: the Phase 2 tensor cache must be keyed by `(image_size, norm)`; Phase 3's CLIP norm + 224×224 invalidates it — rebuild (or key the cache path) or eval silently runs on stale tensors. **Implemented** as `cache_key()` → `data/cache/<size>_<norm>/<split>/`; Phase 3 needs no cache surgery, just `data.norm="clip"` + the new `image_size` (new key dir = fresh build).
- **CLIP normalization values**: load mean/std from the `CLIPImageProcessor` config at runtime rather than hardcoding — the numbers in the locked-decisions table are rounded references, not the source of truth.
- **Determinism vs. cuDNN benchmark conflict**: `torch.backends.cudnn.benchmark = True` (set in `train.py`) is incompatible with `cudnn.deterministic = True`. `set_seed()` must disable benchmark for deterministic runs (or document the trade-off).
- **Resume determinism**: `--resume` restores weights/optimizer/scheduler but **not** RNG state, so a resumed run re-shuffles from the seed — same seed/val accuracy as an uninterrupted run, different data order. If bit-exact resumption is ever needed, save `torch.get_rng_state()` (+ CUDA) in the checkpoint.
- **Checkpoint compatibility**: old checkpoints (ResNet/biLSTM/word-vocab) will **not** load after Phase 3 — by design. LoRA adapters ride inside `model_state_dict`, but reload requires an identical LoRA config (rank/alpha/targets) — guaranteed because config is stored in the checkpoint. Do **not** enable `weights_only` tricks that break `ckpt["config"]` (a Config object).
- **Version pinning for reproducibility**: `requirements.txt` pins runtime deps with `==` (torch 2.12.1, transformers 5.12.1, peft 0.20.0, …) — keep it that way when bumping `transformers`/`peft`. HF model weights download on first run — `HF_HOME` is already pointed at the gitignored `data/hf-cache/` inside `pathvqa_dataset.py`, so caches aren't lost.
- **Baseline comparison is end-to-end**: Phase 1 baseline uses old preprocessing; Phase 3 upgrade uses CLIP preprocessing + RandAugment. The "beat baseline" check therefore includes data changes — record pre/post numbers separately (`baseline.json` vs `upgrade.json`) so the model gain is isolatable.
- **Tokenizer ≠ question vocab**: DistilBERT's tokenizer replaces `QuestionVocabulary` entirely. `parse_question()` works on raw strings, so the symbolic path is unaffected — but any code path that expected `vocab.encode()` must be migrated (train.py + evaluate.py + `prepare_batch`).
- **RandAugment on pathology images**: keep magnitudes moderate (pathology images are low-texture); verify val/test stays augmentation-free. **Implemented**: `data.randaugment_num_ops=2`, `data.randaugment_magnitude=9`, applied only when `split == "train"`.
- **GPU feasibility gate**: the Phase 3.1 probe is a hard gate — if CLIP ViT-B/32 + DistilBERT + LoRA does not fit/run acceptably at 8GB, fall back to CLIP ViT-B/32 visual *only* + a smaller question encoder before changing anything else.

---

## Production track — post-submission (system stages 6–8)

Schedules the serving/container/CI/packaging stages of the system pipeline. All items run after Phase 10.

### P1 — FastAPI inference (stage 6) ✅ SHIPPED
- [x] `src/api/`: `POST /predict` (+ `/predict/explain` for the symbolic trace), `GET /health`, `GET /model/info`; Pydantic schemas.
- [x] `serve.py` uvicorn entrypoint; checkpoint loaded once and cached.

### P2 — Docker (stage 7) ✅ SHIPPED
- [x] `Dockerfile` + `docker-compose.yml` for GPU serving.

### P3 — CI/CD + packaging (stage 8) — parked

- [ ] GitHub Actions workflow: lint → type-check → unit tests → integration on push/PR.
- [ ] `ruff format` as CI gate.
- [ ] Convert `src/` → installable `src/pathvqa/` package with `pyproject.toml`; package install (`pip install -e .`) replacing `sys.path.insert(0, 'src')` / `sys.path.append`.
- [ ] Add `typing.Protocol` interfaces for component boundaries (`VisualEncoderProtocol`, `QuestionEncoderProtocol`, etc.) to enable hot-swappable backbones.
- [ ] Convert loose dicts/tuples at module boundaries into `@dataclass` / `NamedTuple` types.
- [ ] Add `mypy` CI gate.

### P4 — Testing expansion — parked

- [ ] Tensor shape assertion tests at every module boundary.
- [ ] Data pipeline tests against synthetic fixtures (no live HF download in CI).
- [ ] Symbolic executor coverage tests: one test per DSL primitive.
- [ ] Integration tests: single train step, single eval pass, API roundtrip.
- [ ] `pytest-cov` with coverage floor for CI gating.

### P5 — Interpretability outputs — parked

- [ ] Bounding-box / region visualization overlay from top-k proposals.
- [ ] Attention heatmap rendering.
- [ ] Symbolic execution trace renderer (human-readable + JSON) — see `NS-PATHVQA-AUDIT.md` §Dimension 3 for the target trace schema.
- [ ] Side-by-side composite view (image + overlay + trace).

---

## Planned novelty phases (see `NOVELTIES.md` for full specs)

Not yet scheduled into numbered phases; tracked here as the next major body of work after the Production track, per the C9/C10 implementation plan in `NOVELTIES.md`.

| Phase | What | Files | Effort |
|---|---|---|---|
| **A** | Differentiable query classifier (replaces regex `parse_question()` for qtype) | `diff_query_classifier.py` (new), `pathvqa_model.py`, `config.py` | 1 day |
| **B** | Differentiable target extractor (replaces `_extract_target()`) | `diff_target_extractor.py` (new), `pathvqa_model.py` | 1 day |
| **C** | Differentiable executor (replaces hard-coded `execute()` branching) | `diff_executor.py` (new), `pathvqa_model.py`, `train.py` | 2 days |
| **D** | Wire into training + warm-up (regex pseudo-labels → end-to-end) | `train.py`, `config.py` | 1 day |
| **E** | Ablation runs (5 configs: baseline / current SOTA / differentiable / no-gate / regex+diff-exec) | `train.py`, `evaluate.py` | 2 days |
| **F** | Concept bottleneck evaluation (bottleneck probe, gradient attribution, counterfactuals) | `concept_intervention.py` (new), `evaluate.py` | 2 days |
| **G** | Paper figures + analysis | new analysis scripts | 1 day |

**Total: ~10 days of focused work.** Risk assessment and expected ablation outcomes are in `NOVELTIES.md` (C9/C10 sections).

---

## Source documents (inputs, not plans)

- `NS-PATHVQA-AUDIT.md` — architecture critique, ranked technical proposals (P1–P11), design alternatives, publication strategy; its proposals are **scheduled/parked** here — status there is not authoritative.
- `NOVELTIES.md` — claim register; this roadmap produces the evidence that updates it, and owns the full implementation spec for planned claims C9/C10.
- `PROJECT.md` — how to work (current-code facts, conventions, commands, config defaults).

---

## Standing rules

- No code outside the current phase's checklist.
- Every change: `python train.py --debug` clean, smoke tests pass, seeds fixed.
- New deps only with justification (approved: `transformers`, `peft`).
