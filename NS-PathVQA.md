<!-- markdownlint-disable MD013 -->
# Project Detail — NS-PathVQA (Neuro-Symbolic Pathology VQA)

> Portfolio write-up for this project (filled from a résumé template). Every number below is measured and traceable to a repo artifact (`outputs/baseline.json`, `outputs/upgrade.json`, `outputs/cache_benchmark.json`, `outputs/eval_results.json`, `outputs/probe_31.json`, `outputs/baselines.md`) — nothing invented. Technical reference now lives in `DOCS.md`.

## The rule check

Every bullet below carries a number, a tool, or a decision. Sections marked **[SHIPPED]** exist in the repo today.

---

## 1. Identity

- **Item type:** Personal research + engineering project (ML/AI).
- **Title:** Neuro-Symbolic Pathology Visual Question Answering (NS-PathVQA)
- **Dates:** Oct 2025 – Aug 2026
- **Status:** Completed — Phases 1–10 + Production track P1–P5 shipped; system fully trained, evaluated, served via API/Docker, CI-gated.
- **GitHub / demo / paper links:** Private repository; paper draft assembled (`outputs/paper_draft.md`). No public demo.
- **Solo or team?** Solo — owned end-to-end: problem framing, architecture, data pipeline, model, training, evaluation, serving, containerization, documentation.

## 2. One-line story (your lead bullet)

Built a neuro-symbolic medical VQA system (frozen PLIP + DistilBERT with LoRA, learned confidence-gated symbolic reasoning) trained on **32,632 pathology QA pairs**, reaching **59.9% test accuracy** in **0.38 GPU-hours on a single RTX 4060 (8 GB)** — a **+12% relative** gain over its own deep baseline at ~100× fewer trainable parameters than LLM-based systems.

## 3. Problem & Motivation

- **Problem:** Visual Question Answering over histopathology images (PathVQA benchmark) — answering natural-language clinical-education questions ("is the tumor malignant?", "where is the lesion?") about microscopy images. Audience: medical AI research; explainable clinical-decision-support research (explicitly non-clinical deployment).
- **Why hard/interesting:** Published SOTA (LLaVA-Med, MedVInT, PathChat) uses 7B+ parameter multimodal LLMs needing multi-A100 clusters and produces unauditable black-box answers. Medical answers need *reasons*. Getting competitive accuracy with an interpretable reasoning chain on an 8 GB consumer GPU was the engineering + research challenge.
- **Why this approach:** Neuro-symbolic fusion gives both worlds — a neural path for representation learning and a rule-based symbolic path (scene parsing → query parsing → execution) whose intermediate facts are human-readable. A learned per-sample gate decides how much to trust each path instead of a hand-tuned weight. Simpler alternatives rejected: pure fine-tuned CNN+LSTM (no interpretability, weaker accuracy) and prompt-a 7B VLM (infeasible hardware, no auditability, no reproducibility on consumer hardware).

## 4. Data / Inputs

- **Source:** PathVQA (`flaviagiammarino/path-vqa` on HuggingFace; canonical mirror of the UCSD-AI4H dataset, official author split, deduplicated). Public research license; no privacy issues (public pathology images).
- **Size:** 32,632 QA pairs over 5,044 images — Train 19,654 / Val 6,259 / Test 6,719; 5 question types (yes/no, identity, location, attribute, count).
- **What I did to it (all coded, none shipped with the dataset):**
  - Text normalization for questions/answers; runtime-built answer vocabulary from the training split with `<UNK>` reserved for unseen answers (~14% of val/test answers).
  - Aspect-preserving resize + center-crop to 224×224 (no distortion; handles ~5% portrait images and mixed CMYK/RGB/L modes and truncated TIFFs via `.convert("RGB")` + warning filtering).
  - CLIP mean/std normalization read at runtime from the `CLIPImageProcessor` config (never hardcoded — typo would silently degrade features).
  - Train-only RandAugment (num_ops=2, magnitude=9) applied on uint8 tensors pre-normalization so augmentation stays stochastic per epoch.
  - Atomic-write uint8 tensor cache keyed by `(image_size, norm)` — 19k images cached at ~4.5 GB vs ~18 GB fp32, rebuild-safe when preprocessing changes.
  - Ontology-grounded anatomical region discovery: region names extracted from the answer vocabulary and linked to UMLS/RadLex organ-system hierarchy (10 systems, 140+ organ terms), with coverage reporting.
- **Split strategy:** Official author split (train/val/test) — chosen because all published PathVQA baselines (MEVF, MMQ, BAN…) report on it, making numbers directly comparable.
- **Quality issues handled:** truncated TIFF files (benign warnings silenced), mixed image color modes, class/type imbalance in question types (count questions are rare: 21 in val), unseen-answer rate ~14%.

## 5. System / Model / Approach

**ML path:**

- **Backbones and why:** **PLIP** (`vinid/plip`) visual encoder — CLIP fine-tuned on 208K histopathology image–caption pairs, i.e., domain-pretrained features for free without training cost; kept **frozen + LoRA** (rank 16, alpha 32). **DistilBERT** question encoder (66M params) — light enough for 8 GB VRAM, frozen + LoRA on attention q/k/v. Both projected into a shared space (patches 768→512 via linear projection; question [CLS] 768-d).
- **Custom modules designed:**
  - **CrossModalTransformer** (`src/pathvqa/models/fusion.py`) — multi-head cross-attention where question tokens query 49 ViT patch tokens (FP16-safe masking).
  - **SceneParser** (`src/pathvqa/symbolic/scene_parser.py`) — neural→symbolic bridge predicting region logits, object presence, and attributes (17 colors / 9 shapes / 11 sizes) from attended features.
  - **QueryParser** (`src/pathvqa/symbolic/query_parser.py`) — regex question-type classifier (identity/location/yes_no/attribute/count) with target extraction against the answer vocab; superseded as default by a **learned differentiable classifier + target extractor** (`diff_query_classifier.py`, `diff_target_extractor.py`), with the regex path retained for ablations.
  - **VectorizedSymbolicExecutor** (`src/pathvqa/symbolic/executor.py`) — maps scene facts + parsed query to answer-vocabulary logits, vectorized with `torch.scatter_add_`; paired with a fully differentiable executor (`diff_executor.py`) that mixes branch logits via soft question-type weights.
  - **SymbolicFusionEngine** (`src/pathvqa/symbolic/poe.py`) — the single shared fusion implementation used by training, evaluation, and the API layer; routes count questions through the DSL interpreter when patch features are available.
  - **LearnedGate** (`src/pathvqa/symbolic/routing.py`, novelty C1) — per-sample confidence gate `g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)]))`; fusion is a weighted geometric product of experts in log space, `log p = (1−g)·log p_neural + g·log p_symbolic` (NLLLoss on fused log-probs). Static convex mode retained behind config as the ablation baseline.
  - **MedicalLogicTensorNetwork** (`src/pathvqa/symbolic/ltn.py`, C2) — differentiable fuzzy-logic auxiliary loss (5 clauses incl. region–object coherence) added to cross-entropy.
  - **DSLProgramCompiler + DifferentiableDSLInterpreter** (`src/pathvqa/symbolic/dsl.py`, C5) — typed AST programs (`Filter`, `Count`, `Verify`, …) executed differentiably with soft counting `Σ σ(Wv_i)` + Gaussian-kernel count discretization.
  - **MultiScaleVisualEncoder** (C6) — global view + 4 high-res quadrant crops fused by a learned per-token blend MLP, not a sigmoid gate (config-gated).
  - **Interpretability suite** (`src/pathvqa/utils/interpretability.py`, C7) — faithfulness testing, attention–anatomy Spearman correspondence, perception/parsing/execution failure taxonomy, counterfactual explanations.
- **Losses / optimizer / hyperparameters:** NLL on the fused PoE log-probabilities (after a 3-epoch question-type warm-up on plain cross-entropy), plus auxiliary losses: LTN clause satisfaction `1 − satisfaction`, supervised scene-head losses (region BCE / attribute CE / yes-no consistency) and an ontology sibling regularizer; AdamW, lr 1e-4, cosine schedule with 5% linear warmup, batch 32, AMP (FP16 GradScaler), grad clip 5.0, gradient accumulation + activation checkpointing, EMA of weights (β=0.999), early stopping patience 10, max 50 epochs, seed 42.
- **Unusual elements:** learned neuro-symbolic fusion gate conditioned on 4 heterogeneous signals; LTN fuzzy-logic auxiliary loss (first application to medical VQA); differentiable DSL soft counting; ontology-grounded region discovery; architecture-grounded failure taxonomy; fully calibrated evaluation protocol.

## 6. Build & Training

- **Framework/versions:** PyTorch 2.12.1 (CUDA 13.0), torchvision, transformers 5.12.1, peft 0.20.0, datasets, FastAPI 0.139.0, uvicorn 0.49.0 — all pinned `==` in `requirements.txt`.
- **Hardware:** 1× NVIDIA RTX 4060 (8 GB VRAM). Final model trains in **0.376 GPU-h (~23 min wall, 50 epochs)**; pre-upgrade baseline took 0.706 GPU-h. Peak VRAM 4.1 GB (final) vs 3.2 GB (baseline).
- **Distributed/parallel:** None needed — single-GPU design point; dataloader parallelism (6 persistent workers, prefetch 2) keeps GPU fed at ~600 samples/s.
- **Config system:** Typed dataclass hierarchy (`Config → {Data, Visual, Question, Training, Symbolic, Path}Config`) with JSON save/load and load-time validation (rejects invalid backbone/encoder combos before any GPU time is spent). `--debug` profile (500/100 samples, batch 16, 5 epochs, isolated checkpoint dir) enables ~20-second iteration loops.
- **Experiment tracking:** TensorBoard scalars (train/val loss, per-question-type accuracies, mean gate value `g` per question type at `gate/{qtype}`) + structured logging to `logs/<experiment>/` + machine-readable result JSONs in `outputs/`.
- **Iterations/sweeps:** Hardware-feasibility probes (batch 16 vs 32 VRAM/throughput), baseline → modernized-model comparison, static-vs-learned-gate ablation, neural-only (`--no-symbolic`) ablation, cache on/off throughput benchmark — every run recorded as JSON evidence.

## 7. Evaluation / Testing

- **Metrics used and why:** Aggregate + per-question-type accuracy (medical errors are not uniform across question types); Expected Calibration Error with temperature scaling + reliability bins, max-softmax uncertainty, and split-conformal prediction sets (clinicians need to know when the model doesn't know); paired bootstrap significance tests for headline comparisons (single-split accuracy differences are noisy); optional concept-bottleneck interventions probing whether scene concepts are causally used.
- **Baseline beaten:** Own Phase-1 baseline (ResNet + biLSTM + word-vocab, same data): 53.79% val → **60.33% val (+6.54 pts absolute, +12.2% relative)**; final model **59.87% test**. Also exceeds published compute-comparable baselines (MEVF 53.8%, BAN 54.2%, MMQ 55.1%) at 0.38 GPU-h, documented with citations in `outputs/baselines.md`; positioned honestly as a Pareto trade-off vs MedCLIP (63.1%) / LLaVA-Med (67.2%), which need 10–50+ GPU-h and 7B+ parameters.
- **Statistical validation:** Paired bootstrap tests in the eval report; ECE 0.231 reported with temperature-scaled calibration; fixed seed 42; deterministic seeding utility across torch/numpy/random/CUDA.
- **Ablations / failure analysis:** (a) neural-only vs neuro-symbolic; (b) static fixed-weight fusion vs learned gate (the C1 ablation matrix); (c) faithfulness test — perturbing the top symbolic fact and measuring prediction flips via executor injection; (d) failure taxonomy bucketing every error into perception/parsing/execution stages; (e) attention–anatomy correspondence via per-head Spearman correlation.
- **Test suite:** 208 pytest tests passing (~26 s, ≥80% coverage enforced) + `ruff format`/`ruff check` + strict mypy over 39 source files + 16 module-level smoke tests (`__main__` guards), all gated in GitHub Actions CI.

## 8. Engineering / Infrastructure

- **Package structure [SHIPPED]:** clean `src/pathvqa/` layout — `models/` (visual, question, fusion, ema, protocols), `symbolic/` (scene_parser, query_parser + learned variants, executor + differentiable executor, poe fusion engine, routing, ltn, dsl, aux losses), `data/` (dataset with uint8 cache, ontology adapters), `utils/` (config, metrics, conformal, concept_intervention, interpretability, seed, logging), `api/` (inference, schemas) — ~4,800 LOC Python plus `tests/`.
- **Docker [SHIPPED]:** `Dockerfile` + `docker-compose.yml` with GPU passthrough (NVIDIA runtime), read-only checkpoint mount, `CHECKPOINT_PATH` env selection.
- **CI/CD [SHIPPED]:** GitHub Actions pipeline (`.github/workflows/ci.yml`) running the four gates on every push/PR: `ruff format --check`, `ruff check`, strict `mypy src/`, and `pytest` with an ≥80% coverage floor; package installable via `uv pip install -e .`.
- **Serving/API [SHIPPED]:** FastAPI service (`serve.py`) — `POST /predict` (base64 image + question → answer + confidence), `POST /predict/explain` (adds the full symbolic execution trace), `GET /health`, `GET /model/info`; checkpoint loaded once and cached; Pydantic schemas.
- **Reproducibility [SHIPPED]:** pinned dependencies, fixed seeds, config-in-checkpoint (backfilled for forward compatibility of older checkpoints), structured eval reports, auto-download + pinned HF cache, atomic tensor caches, documented conventions/gotchas in a consolidated reference (`DOCS.md`) plus agent rules (`AGENTS.md`).
- **Tests/linting [SHIPPED]:** 208 pytest tests (executor, query parser, routing, PoE fusion parity, config, dataset/cache, ontology, differentiable pipeline, AMP regression, conformal, API roundtrip…), ruff clean, strict mypy clean, 16 smoke tests.
- **Orchestration:** N/A by design — single-GPU research system; complexity budget spent on the model, not schedulers.

## 9. Deployment

- Served locally via FastAPI/uvicorn on :8000 inside Docker (GPU inference). Single-user research deployment — latency dominated by two frozen-transformer forwards (~sub-second per query on RTX 4060). No public hosting, uptime/traffic metrics not applicable.

## 10. Numbers at a glance

| Quantity | Value |
|---|---|
| Dataset size | 32,632 QA pairs (19,654 / 6,259 / 6,719), 5 question types |
| Baseline result | 53.79% val acc (ResNet+biLSTM), 0.71 GPU-h, 3.2 GB VRAM |
| Final result | **60.33% val / 59.87% test**, per-type test: yes/no 87.7%, location 72.4%, identity 26.5%, count 33.3% |
| Improvement | +6.54 pts absolute = **+12.2% relative** over own baseline; beats published MEVF/BAN/MMQ (54–55%) |
| Calibration / significance | ECE 0.231 with temperature scaling; paired bootstrap tests in eval report |
| Hardware / cost | 1× RTX 4060 8 GB; **0.38 GPU-h** full training (~23 min); peak 4.1 GB VRAM |
| Parameters | ~270M frozen backbones + **11.9M trainable (4.4%)** via LoRA/gates/heads |
| Experiments / runs | Feasibility probes (2), baseline, upgraded model, ≥2 fusion ablations, cache benchmark — all logged as JSON artifacts |
| Dataloader throughput | 339 → **602 samples/s (1.78×)** via uint8 tensor cache (1000-sample benchmark, 6 workers) |
| Cache footprint | ~18 GB fp32-equivalent → **~4.5 GB** uint8 |
| Codebase | ~4,800 LOC `src/` + `tests/`, **208 passing pytest tests**, ruff/mypy-clean, 16 smoke tests |
| Deployment | FastAPI (4 endpoints) + Docker GPU serving |

## 11. Challenges & Lessons

- **Hardest problem — 8 GB VRAM vs 2025-grade transformers:** CLIP+DistilBERT+LoRA looked infeasible until measured. Ran a dedicated feasibility probe *before* building (`outputs/probe_31.json`: batch 32 → 1.87 GB peak, 411 samples/s), then combined LoRA-on-frozen-backbones + AMP + gradient accumulation + activation checkpointing. Lesson: probe first, never guess memory.
- **Silent-failure debugging:** a debug-mode run once overwrote the production `best_model.pt` (root cause: shared checkpoint paths). Fixed by redirecting debug artifacts to `checkpoints/debug/` with a regression test. Similarly, discovered that mixing AMP `GradScaler` with gradient accumulation corrupts gradients unless `scaler.step()/update()` run only on accumulation boundaries — fixed and documented. Also hit peft's text-oriented `get_peft_model` failing on CLIP pixel inputs; solved with `inject_adapter_in_model` to keep native HF forwards. Lesson: instrument the exact interaction points of libraries, and make destructive paths structurally impossible.
- **Honest negative results:** count questions remain the weakest type; the differentiable DSL soft-counting fix is implemented *and wired into fusion* (count queries route through the DSL interpreter via the shared PoE engine), and the remaining accuracy gap is documented rather than hidden. Cross-dataset adapters (VQA-RAD, SLAKE) demonstrate transfer *potential* but are claimed only as such.
- **What I'd do differently / scale next:** add CI/CD from week one (it was parked deliberately, and is now in place gating every push); the regex query parser — originally flagged as future work to close the gradient dead-end at the executor boundary — has since been replaced by a learned differentiable classifier with a differentiable executor as the default configuration; evaluate on VQA-RAD/SLAKE to prove transfer. Scaling path: export fused ONNX/TorchScript artifact, batch pre-warmed inference behind the existing FastAPI service, horizontal scale with a model registry.

## 12. Draft bullets

- Built a neuro-symbolic pathology VQA system (frozen PLIP + DistilBERT with LoRA, cross-attention fusion, rule-based symbolic executor) achieving **59.9% test accuracy on 32,632 PathVQA QA pairs** — beating published BAN/MMQ baselines while training in **0.38 GPU-hours on a single RTX 4060 (8 GB)**.
- Designed a learned per-sample confidence gate blending neural and symbolic log-probabilities conditioned on attention state, scene confidence, question type, and attention entropy; improved validation accuracy **+6.54 pts (+12.2% rel.)** over the ResNet+biLSTM baseline with an interpretable execution trace per prediction.
- Engineered the full MLOps loop — typed validated configs, TensorBoard tracking, calibrated evaluation (ECE, temperature scaling, conformal sets, paired bootstrap), **208 pytest tests** behind CI, FastAPI inference with symbolic explanation endpoint, and Docker GPU deployment — with a uint8 tensor cache lifting dataloader throughput **1.78× to 602 samples/s**.

---

## 13. Internship / research position section

N/A — solo personal project, no supervisor or reporting line.

---

## Extra: interview-proof answers

- **Why this architecture?** Three constraints drove it: (1) 8 GB VRAM rules out full fine-tuning of any pretrained backbone → frozen encoders + LoRA (rank 16) train only 4.4% of parameters; (2) pathology benefits from domain-pretrained features → PLIP (208K histopathology caption pairs) over generic CLIP at identical cost; (3) medical answers must be auditable → the symbolic path emits explicit facts (regions, colors, shapes) and a per-question reasoning trace, and a learned gate makes "when to trust the rules" itself learnable rather than a magic constant like the standard `neural + w·symbolic`.
- **Why these tools?** PyTorch + HuggingFace `transformers`/`peft` because LoRA injection had to coexist with native HF forward paths (peft's wrapper broke on CLIP pixel inputs; `inject_adapter_in_model` didn't). FastAPI over Flask for async + automatic OpenAI-compatible schema via Pydantic. No experiment-tracking SaaS (MLflow/W&B) — TensorBoard + JSON artifacts keep the footprint minimal and the repo self-contained/reproducible offline.
- **Why this loss/training recipe?** Cross-entropy for the closed answer vocabulary, plus the LTN clause-satisfaction auxiliary loss to regularize scene-parser internal consistency (region↔object coherence) — computationally free since it reuses existing logits. AdamW + cosine + warmup replaced Adam + ReduceLROnPlateau after Phase-1 instability; EMA weights gave smoother validation behavior at zero serving cost.
- **How did you process the data?** Decode (tolerating truncated TIFFs/mixed color modes) → aspect-preserving resize + center crop 224×224 → train-only RandAugment on uint8 → float conversion + runtime-read CLIP normalization. Two decisions mattered most: caching post-resize *uint8 tensors* keyed by `(image_size, norm)` (1.78× throughput, automatic invalidation on preprocessing change) and reading CLIP stats from the processor config instead of hardcoding (a typo silently degrades pretrained features).
- **How did you evaluate?** Author split for comparability with every published number; per-question-type breakdown because yes/no (87.7%) masks identity failures (26.5%); calibration + uncertainty because a medical system must express doubt; paired bootstrap for every headline claim; ablation matrix (static vs learned gate, with/without symbolic path) to attribute gains to components rather than preprocessing.
- **What failed, and how did you optimize it?** Concrete story: a `--debug` run silently replaced the production checkpoint → introduced path isolation + regression test. AMP + gradient accumulation produced collapsing gradients → traced to per-micro-step scaler updates, moved step/update to accumulation boundaries. Each fix is documented in the roadmap gotchas so it can't recur.
- **How would you deploy/scale it?** Already: FastAPI + Docker GPU serving with health/model-info endpoints and cached checkpoint loading. Next: TorchScript/ONNX export of the fused graph, dynamic batching with pre-warmed sessions, quantized INT8 variant for CPU edge serving (the 11.9M-trainable-parameter design makes distillation cheap), and a model registry keyed by checkpoint config for safe rollbacks.
