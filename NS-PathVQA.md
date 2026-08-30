<!-- markdownlint-disable MD013 -->
# Project Detail — NS-PathVQA (Neuro-Symbolic Pathology VQA)

> Portfolio write-up for this project (filled from `PROJECT_TEMPLATE.md`). Numbers are traceable to artifacts currently in the repo: `outputs/canonical_seed42{,_val,_test}.json` and `outputs/canonical_seed42_release_manifest.json` (canonical run), `outputs/final_repro_{val,test}.json` + `outputs/final_image_dependence/` (retained-checkpoint reproduction), `outputs/baseline.json`, `outputs/cache_benchmark.json`, plus `CHANGELOG.md`, `AUDIT.md`, `RUN.md`. Historical numbers from `outputs/upgrade.json` are labelled as such and are **not** headline results. Technical reference lives in `CHANGELOG.md`; evidence status in `AUDIT.md`.

## The rule check

Every bullet below carries a number, a tool, or a decision. **[SHIPPED]** means the item exists in the repo today; **[NOT DONE]** means the tooling exists but the evidence does not. No clinical or SOTA claim is made — the canonical run is a single seed.

---

## 1. Identity

- **Item type:** Personal research + engineering project (ML/AI).
- **Title:** Neuro-Symbolic Pathology Visual Question Answering (NS-PathVQA)
- **Dates:** Oct 2025 – Aug 30, 2026 (canonical run completed August 30, 2026).
- **Status:** Core system + canonical seed-42 run complete; A–K matrix (single seed), blinded expert adjudication, authorized PathMMU evaluation, and manuscript/release are **incomplete** — repository is explicitly **not conference-ready** (`AUDIT.md` §1).
- **GitHub / demo / paper links:** Private repository; no public demo. A paper draft existed in an earlier ignored-output workspace but **no manuscript file is present in this checkout** (`AUDIT.md` §5.6).
- **Solo or team?** Solo — owned problem framing, data pipeline, model, training, evaluation, serving, containerization, and documentation.

## 2. One-line story (your lead bullet)

Built a neuro-symbolic pathology VQA system (frozen PLIP + DistilBERT with LoRA, learned question parser + differentiable symbolic executor + DSL, log-space product-of-experts at fixed weight 0.3) on **32,632 PathVQA QA pairs**, reaching **60.28% validation / 60.01% test accuracy** in **1.356 GPU-hours on one RTX 4060 Laptop GPU (8 GB)** — a single-seed result, not a component-effect or clinical claim.

## 3. Problem & Motivation

- **Problem:** Visual Question Answering over histopathology images (PathVQA benchmark) — answering clinical-education questions ("is the tumor malignant?", "where is the lesion?") from microscopy images. Audience: medical-AI research and explainable-decision-support research; explicitly non-clinical deployment.
- **Why hard/interesting:** 7B+ parameter multimodal LLMs (LLaVA-Med, MedVInT, PathChat) dominate published scores but need multi-A100 compute and emit unauditable answers. Getting a competitive, inspectable reasoning chain to run on an 8 GB consumer GPU was the engineering + research constraint.
- **Why this approach:** Neural path learns representations; symbolic path (scene parsing → query parsing → execution) emits human-readable region/attribute facts and a per-prediction trace; a single shared `SymbolicFusionEngine` combines them in log space. Simpler alternatives rejected: pure fine-tuned CNN+LSTM (no audit trail, weaker accuracy — measured 53.79% val in `outputs/baseline.json`) and prompt-tuning a 7B VLM (infeasible on 8 GB, no reproducible audit). **Honesty boundary:** a learned per-sample gate is implemented as a *comparison condition* only; both the retained legacy checkpoint and the canonical checkpoint use **static** PoE weight `g=0.3`, so no learned-gate gain is claimed (`AUDIT.md` §2).

## 4. Data / Inputs

- **Source:** PathVQA (`flaviagiammarino/path-vqa`, pinned HF revision `1685832883334b5bb5beaf4e4b333fdeecaa4ad9`), public research data, no privacy concerns; split provenance/fingerprints recorded in every eval artifact.
- **Size:** **32,632 QA pairs** — Train 19,654 / Val 6,259 / Test 6,719; 5 question types (yes/no, identity, location, attribute, count); answer vocabulary of **3,226 classes** built from train.
- **What I did to it (coded, not shipped with the dataset):**
  - Question/answer text normalization; runtime answer vocab with `<UNK>` for unseen answers. Coverage 85.62% val / 85.47% test; accuracy on covered answers 70.01% / 69.98%, on unseen 0% (cannot match exactly) — `AUDIT.md` §3.
  - 224×224 aspect-preserving resize + center crop; `.convert("RGB")` for mixed CMYK/RGB/L modes and truncated TIFFs; CLIP mean/std read at runtime from `CLIPImageProcessor` (never hardcoded).
  - Train-only RandAugment (2 ops, magnitude 9) applied on uint8 tensors before float/normalize.
  - Atomic uint8 tensor cache keyed by `(image_size, norm, TRANSFORM_VERSION)`; full train cache ≈**4.8 GB** on disk (`AUDIT.md` §6); a historical 1,000-sample, 6-worker benchmark measured **338.9 → 602.4 samples/s (1.78×)** for cache-off → warm cache (`outputs/cache_benchmark.json`, 320×240/ImageNet-era configuration — not a canonical-run number).
  - Ontology-grounded region discovery: 40 region names derived from the answer vocab and linked to UMLS/RadLex organ-system hierarchy (10 systems, 140+ terms), coverage reported by the adapter.
- **Split strategy:** Official author train/val/test split — chosen for comparability with published PathVQA work; validation for model selection, test reserved for the frozen recipe.
- **Quality issues handled:** truncated TIFFs, mixed color modes, question-type imbalance (count: 21 val / 18 test), ~14% unseen answers, repeated questions per image (handled as image-content clusters in uncertainty estimation).

## 5. System / Model / Approach

**ML path — canonical configuration (`outputs/canonical_seed42_val.json` → `config`):**

- **Backbones and why:** **PLIP** (`vinid/plip`) visual encoder — CLIP fine-tuned on ~208K histopathology image–caption pairs gives domain features at zero training cost — frozen + LoRA (r16, α32), all 49 patch tokens projected 768→512. **DistilBERT** (`distilbert-base-uncased`, 768-d) question encoder — fits 8 GB, frozen + LoRA on q/k/v/out. (The retained pre-canonical checkpoint instead used `openai/clip-vit-base-patch32`, regex/hard path, no DSL, `g=0.3`.)
- **Custom modules designed:**
  - `CrossModalTransformer` (`models/fusion.py`) — multi-head cross-attention, question tokens query the 49 patch tokens with FP16-safe masking.
  - `SceneParser` (`symbolic/scene_parser.py`) — region logits, object presence, attribute heads (17 colors / 9 shapes / 11 sizes); dormant density head removed.
  - Dual query parser (`symbolic/query_parser.py` + `diff_query_classifier.py`, `diff_target_extractor.py`) — regex rules retained; **learned soft qtype + soft region target** is the canonical default, with a no-region class and weak parser-derived supervision.
  - Executors (`symbolic/executor.py`, `diff_executor.py`) — vectorized hard `scatter_add_` executor and differentiable branch mixing driven by the learned parser; yes/no evidence only for region targets.
  - `SymbolicFusionEngine` (`symbolic/poe.py`) — one shared log-space PoE implementation for train/eval/API; per-row `symbolic_used` guard keeps rows with no written evidence neural-only; count queries with patch features route through the DSL interpreter.
  - `LearnedGate` (`symbolic/routing.py`) — 4-signal confidence gate `g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)]))`; **implemented as an ablation only**, absent from both reported checkpoints.
  - `MedicalLogicTensorNetwork` (`symbolic/ltn.py`) — fuzzy-logic auxiliary clauses (organ-system co-occurrence / morphology), weight 0.1.
  - `DSLProgramCompiler` + `DifferentiableDSLInterpreter` (`symbolic/dsl.py`) — typed AST (`Filter`, `Count`, `Verify`, …), soft counting over vocab-derived count answers, log-space evidence with a finite `-30` mask floor stored in float32.
  - `MultiScaleVisualEncoder` (global + 4 quadrant crops, learned blend MLP) — config-gated, **off** in the canonical checkpoint.
  - Interpretability suite (`utils/interpretability.py`, `utils/concept_intervention.py`) — failure taxonomy, counterfactual concept suppression/re-decode, trace schema, `--intervention` CLI.
- **Losses / optimizer / hyperparameters:** NLL on fused log-probs after a 3-epoch qtype warm-up (weight 0.1); auxiliaries region BCE 0.5 / attribute CE 0.3 / yes-no consistency 0.2 / ontology-sibling 0.1 / target 0.1, LTN 0.1. AdamW, lr 1e-4, cosine + 5% warmup, batch 32, AMP fp16, grad clip 5.0, grad accumulation + activation checkpointing, EMA (β=0.999), early-stop patience 10, 50 epochs max, seed 42, deterministic mode (best-effort CUDA kernels). Fusion: `log p = (1−g)·log p_neural + g·log p_symbolic`, fixed `g=0.3` in reported checkpoints.

## 6. Build & Training

- **Framework/versions (pinned `requirements.txt`, Python 3.12.13):** torch 2.12.1+cu130, torchvision 0.27.1, transformers 5.12.1, peft 0.20.0, accelerate 1.14.0, datasets 5.0.0, Pillow 12.3.0, tensorboard 2.21.0, fastapi 0.139.0, uvicorn 0.49.0.
- **Hardware (measured, canonical):** 1× NVIDIA RTX 4060 Laptop GPU (8 GB); **4,882.3 s = 1.356 GPU-hours**, mean 123.1 s/epoch over **33 epochs** (best epoch index 22, 60.2812% val), peak allocated VRAM **2,857.7 MB** (`outputs/canonical_seed42.json`; device recorded in eval runtime metadata). Historical, non-headline: baseline 0.706 GPU-h / 3,245.2 MB; 2026-08 upgrade 0.376 GPU-h / 4,105.6 MB (`outputs/baseline.json`, `outputs/upgrade.json`) — `AUDIT.md` warns legacy GPU-hour/VRAM claims are inconsistent and must not be promoted.
- **Parallel / config:** single-GPU design point; DataLoader workers (6 default, 0 in canonical eval because the execution environment restricted workers). Typed dataclass config hierarchy with JSON save/load + load-time validation; `--debug` profile (500/100 samples, batch 16, 5 epochs, isolated `checkpoints/debug/`).
- **Tracking:** TensorBoard scalars (loss, per-qtype accuracy, gate stats), structured logs in `logs/<experiment>/`, machine-readable JSON artifacts with config, dataset revision/fingerprint, checkpoint SHA-256, seed, per-example predictions, image clusters and runtime.
- **Iterations:** canonical sequence = bounded CPU smoke → repaired 10-epoch diagnostic → full early-stopped training → val/test eval → 4 image controls → paired cluster analysis → release manifest (88 hashed source files + checkpoint hash `d84e0d…`, `outputs/canonical_seed42_release_manifest.json`). A–K runner/analysis/recipe tools were only smoke-tested in debug (A/I, 100 examples) — plumbing, not evidence.

## 7. Evaluation / Testing

- **Metrics and why:** accuracy + per-question-type breakdown (yes/no 88.05% masks identity 26.30% on canonical val); ECE with held-out-image-group temperature fitting (canonical val **0.1748**, test **0.1801**, T≈2.02); split-conformal sets — 100% coverage but average set size **3,226 = the entire answer vocabulary**, so explicitly *not* useful narrowing (`AUDIT.md` §6); 95% image-cluster bootstrap intervals (retained checkpoint: val [58.38%, 61.50%], test [58.21%, 61.49%], 832/858 clusters); image-dependence paired controls with cluster sign tests + McNemar.
- **Comparison point:** own Phase-1 ResNet+biLSTM baseline **53.79% val** (`outputs/baseline.json`, 0.706 GPU-h) → canonical **60.28% val = +6.49 pts absolute / +12.1% relative**. Caveat: baseline used 320×240/ImageNet preprocessing, canonical uses 224²/CLIP norm — this is an end-to-end pipeline comparison, not a controlled single-component gain. Retained legacy checkpoint reproduction under current code: **59.9457% val / 59.8154% test** (`outputs/final_repro_{val,test}.json`). Published literature numbers are protocol-incompatible; no `outputs/baselines.md` or other literature table exists in this checkout, so no matched published-baseline claim is made.
- **Statistical validation:** 10,000-resample cluster bootstraps over 832 image-content groups for canonical controls (blank −17.48 pp [−18.81, −16.15]; shuffled −17.40 pp [−18.73, −16.07]; image-only −42.18 pp [−43.95, −40.43]); fixed seed 42; group-disjoint temperature/ECE protocol (numbers above are from the canonical eval artifacts). **Note:** earlier calibration/conformal artifacts were superseded after finding same-image rows treated as independent; no regenerated calibration numbers are claimed beyond the canonical artifacts.
- **Ablations / failure analysis [NOT DONE]:** A–K matrix is **defined** in `utils/ablation.py` (A–H = regex/learned × hard/differentiable × static/learned factorial; I = neural-only; J/K = LTN / all-aux loss controls) with runner `scripts/run_ablations.py`, but the matrix has **not been run** — only a debug plumbing smoke. Failure taxonomy, attention–anatomy correlations, and concept interventions are implemented tooling, not quantified headline results.
- **Test suite [SHIPPED]:** **299 pytest tests passing**, ≥80% coverage gate (86.35% at the canonical run), `ruff format --check`, `ruff check`, strict `mypy src/` (40 modules), and 16 `__main__` smoke guards, all gated by GitHub Actions.

## 8. Engineering / Infrastructure

- **Package structure [SHIPPED]:** `src/pathvqa/` — 40 Python modules, ≈**6,000 LOC** (`models/`, `symbolic/`, `data/`, `utils/`, `api/`); tests ≈5,400 LOC across 30 files.
- **Docker [SHIPPED]:** `Dockerfile` + `docker-compose.yml` with NVIDIA runtime GPU passthrough, read-only checkpoint mount, `CHECKPOINT_PATH` selection; serves :8000.
- **CI/CD [SHIPPED]:** `.github/workflows/ci.yml` runs the four gates (ruff format, ruff check, strict mypy, pytest with ≥80% coverage) on push/PR to `main`; `uv pip install --system -e ".[dev]"` in CI.
- **Serving/API [SHIPPED]:** FastAPI (`serve.py`) — `POST /predict`, `POST /predict/explain` (model-derived symbolic trace), `GET /health`, `GET /model/info`; Pydantic schemas; checkpoint loaded once and cached; async routes with blocking inference in `asyncio.to_thread`; 19 offline API tests.
- **Reproducibility [SHIPPED]:** pinned requirements, fixed seeds, config embedded in checkpoint + `backfill_config()` legacy migration, pinned dataset revision/HF cache, atomic tensor caches, provenance in every eval artifact, consolidated docs (`AGENTS.md`, `CHANGELOG.md`, `AUDIT.md`, `RUN.md`), release manifest (88 source hashes + checkpoint SHA-256 — must be regenerated after documentation edits).
- **Tests/linting [SHIPPED]:** 299 pytest tests, ruff + strict mypy clean, 16 smoke guards, regression tests for PoE parity, DSL fp16 floor, AMP scatter dtype, yes/no BCE precision, config/API/cache behavior.
- **Orchestration:** N/A by design — single-GPU research system.
- **Outstanding [NOT DONE]:** single-seed A–K coverage matrix, blinded expert adjudication (89-case package exported to `outputs/final_expert_review/`, no responses), PathMMU evaluation (requires dataset access + image licenses), manuscript, license review and release archive.

## 9. Deployment

- Served locally through FastAPI/uvicorn on :8000 (Docker + GPU). Single-user research deployment; **no public hosting**. Latency, throughput, uptime, and request volume were **not measured** — not claimed.

## 10. Numbers at a glance

| Quantity | Value |
|---|---|
| Dataset size | 32,632 QA pairs (19,654 / 6,259 / 6,719), 3,226 answer classes, 5 qtypes |
| Baseline result | 53.79% val (ResNet+biLSTM, 0.706 GPU-h, 3,245 MB peak; `outputs/baseline.json`) |
| Canonical result | **60.28% val (3,773/6,259; ECE 0.1748) / 60.01% test (4,032/6,719; ECE 0.1801)** |
| Per-type (canonical test) | yes/no 87.34%, location 75.87%, identity 26.63%, count 38.89% (18 examples) |
| Improvement over own baseline | +6.49 pts absolute = **+12.1% relative** (end-to-end pipeline change, not a single component) |
| Retained-checkpoint reproduction | 59.9457% val / 59.8154% test under current code; static `g=0.3`, regex/hard, no DSL |
| Statistical evidence | 10,000-resample image-cluster bootstrap (832 groups); 95% CIs; paired sign/McNemar on image controls; conformal sets non-informative (avg size 3,226) |
| Hardware / cost | 1× RTX 4060 Laptop GPU 8 GB; **1.356 GPU-h (4,882.3 s)**, 33 epochs, peak **2,857.7 MB** |
| Training config | PLIP + DistilBERT frozen + LoRA r16/α32, learned parser, differentiable executor, DSL on, static PoE g=0.3, seed 42 |
| Experiments recorded | baseline, historical upgrade, cache benchmark, canonical sequence (smoke + diagnostic + train + eval + 4 controls + manifest); A–K matrix **not run** |
| Data throughput | 338.9 → 602.4 samples/s (1.78×) uint8 cache, 6 workers — historical 1,000-sample benchmark; full train cache ≈4.8 GB |
| Codebase | 40 `src/` modules ≈6,000 LOC + 30 test files ≈5,400 LOC; **299 passing tests**, 86.35% coverage; 16 smoke tests; ruff/mypy clean |
| Deployment | FastAPI (4 endpoints) + Docker GPU serving; latency/uptime not measured |
| Research readiness | A–K matrix, expert adjudication, PathMMU, manuscript/release: **not done** |

## 11. Challenges & Lessons

- **Hardest problem — fp16 DSL evidence caused divergent training:** the first canonical attempt hit infinite answer NLL at epoch 3; diagnosis isolated a `-1e9` DSL mask floor cast into fp16 evidence (`-inf`) for some gold answers, plus a yes/no BCE clamp that rounded `1−eps` to 1.0 in fp16. Fix: finite `-30` log-evidence floor, float32 fusion evidence buffer, float32 promotion before BCE, and a fail-fast nonfinite-loss check with component/row diagnostics. A repaired 10-epoch diagnostic passed; regression tests cover both paths (`RUN.md` §4, `CHANGELOG.md` §9). Lesson: inspect dtype interactions at custom-op boundaries, never guess.
- **Silent calibration invalidation:** earlier artifacts treated repeated questions per image as independent and reused the same split for temperature fitting and ECE. Fixed to image-content-group partitioning with disjoint fit/eval groups and group-maximum conformal scores; because old JSON artifacts were not regenerated under the corrected protocol, superseded numbers were withdrawn rather than republished.
- **Retracted contribution:** an earlier paper draft credited a learned gate with a ~6.5-point gain. Checkpoint inspection showed static PoE (`g=0.3`), no gate parameters, regex/hard execution in the retained model. The claim was removed; learned gating is reported only as an implemented comparison condition. Lesson: verify claims against the serialized artifact, not the intended design.
- **Earlier engineering fixes:** a debug run once overwrote the production checkpoint → debug artifacts isolated to `checkpoints/debug/`; AMP `GradScaler` with gradient accumulation corrupted gradients unless `step()/update()` run only on accumulation boundaries; peft's `get_peft_model` broke on CLIP pixel inputs → `inject_adapter_in_model`.
- **What's next:** run the single-seed, descriptive A–K matrix (no effect claims from one seed), collect blinded expert adjudication, evaluate PathMMU after access approval, write the manuscript from canonical artifacts, regenerate the release manifest, and only then archive/release. Scale path: exported fused graph behind batched FastAPI inference, with a config-keyed model registry for rollback.

## 12. Draft bullets

- Built a neuro-symbolic pathology VQA system (frozen PLIP + DistilBERT with LoRA, learned parser, differentiable symbolic executor + DSL, log-space PoE at g=0.3) reaching **60.28% validation / 60.01% test accuracy on 32,632 PathVQA QA pairs**, trained in **1.356 GPU-hours on a single RTX 4060 Laptop GPU** — reported as a single-seed result, not a clinical or SOTA claim.
- Implemented a shared per-row product-of-experts fusion engine with learned question routing, differentiable DSL counting, a logic-tensor-network auxiliary loss, and an image-group-disjoint calibration/uncertainty protocol; **retracted an unsupported learned-gate gain** after checkpoint inspection showed static `g=0.3`, and kept learned gating as an A–K comparison condition.
- Shipped the full engineering loop — typed configs, provenance-rich JSON artifacts, FastAPI with a symbolic-trace endpoint, Docker GPU serving, CI with **299 pytest tests (86.35% coverage)**, strict mypy, and a release manifest hashing **88 source files** — while a uint8 tensor cache gave a measured **1.78× dataloader speedup** in a historical benchmark.

---

## 13. Internship / research position section

N/A — solo personal project, no supervisor or reporting line.

---

## Extra: interview-proof answers

- **Why this architecture?** Three constraints: (1) 8 GB VRAM → frozen backbones + LoRA r16/α32 instead of full fine-tuning; (2) pathology domain gap → PLIP (≈208K histopathology caption pairs) at identical training cost to generic CLIP; (3) medical answers must be auditable → a symbolic path emitting region/attribute facts and a traced execution. Fusion is a *fixed* log-space PoE (`g=0.3`) in both reported checkpoints; a learned 4-signal gate exists only as an ablation because no matched experiment established its benefit.
- **Why these tools?** PyTorch + HuggingFace `transformers`/`peft` because LoRA had to coexist with native HF forwards (`inject_adapter_in_model`, not `get_peft_model`, for CLIP pixel inputs). FastAPI + Pydantic for async serving and typed schemas. TensorBoard + JSON artifacts instead of a tracking SaaS to stay offline/reproducible. CI installs via `uv pip install --system -e ".[dev]"`; local runtime uses pinned `requirements.txt`.
- **Why this loss/training recipe?** Cross-entropy/NLL over a closed 3,226-class answer vocabulary, with LTN clause satisfaction (0.1) reusing scene logits to enforce anatomy↔morphology consistency; auxiliary region/attribute/yes-no + ontology losses (0.5/0.3/0.2/0.1) supervise the neural→symbolic bridge. AdamW + cosine + warmup and EMA replaced an unstable early Adam + ReduceLROnPlateau recipe; early stopping at patience 10 ended canonical training at epoch 33 (best index 22).
- **How did you process the data?** Decode (truncated TIFFs / mixed modes → RGB) → 224² aspect-preserving resize + center crop → train-only RandAugment on uint8 → float + CLIP stats read from the processor config. Key decisions: cache post-resize **uint8** tensors keyed by `(image_size, norm, TRANSFORM_VERSION)` (measured 1.78× throughput; ≈4.8 GB full cache), and never hardcode normalization constants.
- **How did you evaluate?** Official split for comparability; per-type accuracy because aggregate accuracy hides 26% identity performance; image-content-group bootstrap intervals because repeated questions on one image are dependent; four image conditions to test image dependence; held-out-group temperature/ECE and conformal tooling because a medical system must express calibrated doubt. Conformal sets currently cover the whole vocabulary (avg 3,226) and are reported as non-informative; the A–K matrix has not been run, so no component-effect claim is made.
- **What failed, and how did you optimize it?** Concrete canonical story: infinite NLL at epoch 3 → traced to fp16 `-1e9` DSL mask becoming `-inf`, fixed with a finite `-30` floor + float32 evidence and a fail-fast nonfinite check; yes/no BCE fp16 clamp → float32 promotion; earlier calibration protocol treated same-image rows as independent → group-disjoint protocol, with superseded numbers withdrawn. Also retracted the learned-gate claim after inspecting the checkpoint's serialized config.
- **How would you deploy/scale it?** Already: FastAPI + Docker GPU serving with health/model-info endpoints and cached checkpoint loading. Next: export the fused graph (TorchScript/ONNX), dynamic batching with pre-warmed sessions, a quantized CPU variant for edge serving, and a config-keyed model registry for safe rollbacks — plus the outstanding experiments (A–K, expert review, PathMMU) before any research submission.
