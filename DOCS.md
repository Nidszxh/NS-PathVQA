# NS-PathVQA — Consolidated Documentation

> Merged on 2026-08-24 from `PROJECT.md`, `NOVELTIES.md`, `TODO.md` and
> `NS-PATHVQA-AUDIT.md` (all deleted; this file is now the single reference).
> Division of labor: **README.md** = quick start · **AGENTS.md** = working rules +
> commands + config defaults · **DOCS.md** (this file) = system reference,
> novelty register, history, audit outcomes · **NS-PathVQA.md** = personal portfolio
> write-up (not project docs).

---

## 1. System overview

Neuro-symbolic Visual Question Answering over histopathology images (PathVQA:
32,632 QA pairs). A neural path (frozen PLIP + DistilBERT with LoRA adapters,
cross-attention fusion) is fused with an interpretable symbolic path
(scene parser → query parser → executor) by a learned per-sample confidence
gate. Fusion is a **weighted geometric Product-of-Experts in log space**:

```
g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)]))     # or static w = symbolic_weight
log p = (1 − g) · log p_neural + g · log p_symbolic            # NLLLoss on fused log-probs
```

Single implementation: `SymbolicFusionEngine` (`src/pathvqa/symbolic/poe.py`);
train.py / evaluate.py / api/inference.py all delegate to it.

**Headline results:** 53.79% baseline val acc → **60.33% val / 59.87% test**
(+6.54 pts), trained in **0.38 GPU-hours** at **4.1 GB VRAM** on one RTX 4060
(8 GB) — ~100× fewer trainable params than LLM-based medical VQA systems.
Measured artifacts: `outputs/baseline.json`, `outputs/upgrade.json`,
`outputs/cache_benchmark.json`, `outputs/eval_results.json`, `outputs/baselines.md`.

## 2. Architecture reference

### 2.1 Component map

| # | Component | File(s) | Essence |
|---|---|---|---|
| 1 | Visual encoder | `models/visual/visual_encoder.py` | PLIP frozen + LoRA (r16 α32); all 49 patch tokens (7×7 @224, 768-d) → Linear 768→512 → `(B,49,512)` + all-ones mask; CLIP positional embeddings reused (no box encoder). Optional `MultiScaleVisualEncoder` behind `use_multiscale` (off by default): global + 4 quadrant crops fused by a learned blend MLP (`fused = MLP([global ‖ local])`) — *not* a sigmoid gate |
| 2 | Question encoder | `models/question/question_encoder.py` | DistilBERT frozen + LoRA (r16 α32), hidden 768; HF tokenizer |
| 3 | Cross-modal fusion | `models/fusion.py` | Cross-attention block: question queries attend over 49 patch tokens (FP16-safe mask) |
| 4 | Neural classifier | `models/pathvqa_model.py` | MLP over `[attended ‖ q_state]` → answer logits |
| 5 | Scene parser | `symbolic/scene_parser.py` | Region logits (num_regions), object presence (sigmoid), attribute heads: 17 colors / 9 shapes / 11 sizes (the dormant density head was removed — it was produced but consumed by nothing) |
| 6 | Query parser | `symbolic/query_parser.py`, `diff_query_classifier.py`, `diff_target_extractor.py` | Two interchangeable parsers: **regex** (rule-based identity/location/yes_no/attribute/count) and **learned** (default: soft qtype `(B,5)` + soft target attention over regions, both differentiable) |
| 7 | Executor | `symbolic/executor.py`, `diff_executor.py` | **hard**: vectorized scatter_add mapping scene+query→answer vocab (count branch = zeros); **differentiable** (default): branch logits mixed via soft_qtype weights |
| 8 | Gate & fusion | `symbolic/routing.py`, `symbolic/poe.py` | Learned gate (4 signals incl. attention entropy); PoE engine shared by train/eval/API; count queries with patch features route through the DSL interpreter |
| 9 | Main model | `models/pathvqa_model.py` | Glue; exposes `soft_qtype`/`soft_target`/`qtype_logits`/`target_logits` when learned pipeline active. Factory: `build_model(config, answer_vocab_size, attribute_mappings=None)` |
| 10 | Aux losses | `symbolic/aux_losses.py`, `ontology_loss.py` | Region BCE 0.5 / attribute CE 0.3 / yes-no consistency 0.2 + ontology sibling regularizer 0.1 |
| 11 | Dataset adapter | `data/dataset_adapter.py` | Ontology-grounded region discovery (UMLS/RadLex linkage + PathVQA fallback, coverage % reported); PathVQA shipped, VQA-RAD/SLAKE implemented-not-trained |

### 2.2 Question flow

1. Image → PLIP patches → 512-d tokens; question → DistilBERT state.
2. Cross-attention: question attends to patch tokens.
3. Scene parser predicts symbolic facts from attended features.
4. Query parser classifies question (learned soft or regex hard).
5. Executor maps facts+query → answer-vocab logits (DSL interpreter for counts).
6. Gate blends both paths in log space (PoE) → NLL loss / prediction.

### 2.3 Checkpoints & invariants

- `.pt` stores: epoch, global_step, model/EMA/optimizer/scheduler/scaler state,
  best_val_acc, config, answer_vocab(+to_idx/idx_to_answer), region_names,
  early-stop state. LoRA rides inside model_state_dict.
  `backfill_config()` keeps older configs loadable; checkpoint config wins on resume.
- Invariants: fusion must stay log-space PoE · `soft_target` sized from
  `len(region_names)` never stale `config.symbolic.num_regions` · scene-head ops
  needing matching dtypes under autocast cast sources to buffer dtype ·
  region names derive from answer vocab (`build_region_names()`) → dataset-dependent model.

## 3. Locked design decisions

| Decision | Value | Why |
|---|---|---|
| Backbone | PLIP (`vinid/plip`) frozen+LoRA; 768→512 projection keeps symbolic path dims intact | CLIP fine-tuned on 208K pathology pairs; CLIP ViT-B/32 stays available via config |
| Features | All 49 patch tokens, no proposal network, mask all-ones | Keeps `(B,49,512)` contract; drops 2018-relic top-k |
| Positions | Reuse CLIP positional embeddings | Patches already carry position |
| Text | DistilBERT frozen+LoRA; HF tokenizer; question_vocab.json removed | Light on 8 GB; eval loses prior-run dependency |
| Preprocess | 224², CLIP mean/std read from `CLIPImageProcessor` at runtime | Hand-typed constants silently degrade features; ImageNet norm incompatible |
| Training | AdamW + cosine + warmup, AMP, grad accumulation + checkpointing, EMA | Replaces Adam+ReduceLROnPlateau; no new deps |

## 4. Novelty register (C1–C10)

| ID | Claim | Where | Differentiator vs prior work |
|---|---|---|---|
| C1 | Learned per-sample confidence gate fusing neuro-symbolic experts as weighted geometric PoE in log space | `routing.py` + `poe.py` | Bao et al. 2021 gate is fixed & unconditioned; typical fusion is fixed logit sums. Ours: end-to-end learned, conditions on 4 signals (attention state, scene confidence, qtype, attention entropy) |
| C2 | Logic Tensor Network as differentiable auxiliary loss enforcing domain consistency (organ-system co-occurrence, morphology clauses) | `ltn.py` | LTNs not previously used as VQA aux loss for pathology; clause "irregular" resolved by name (`SHAPE_VALUES.index`) |
| C3 | Resource-efficiency: full neuro-symbolic training on RTX 4060 8 GB (0.38 GPU-h to SOTA-competitive) | whole pipeline | LLaVA-Med/MedVInT/PathChat need 7B+ params, multi-A100; we trade ~35 pts vs GPT-4-class for ~100× fewer trainable params + auditability |
| C4 | Ontology-grounded region discovery linking answer vocab to UMLS/RadLex organ-system hierarchy (10 systems, 140+ terms) | `dataset_adapter.py` | Region set derived from data+ontology rather than hand-picked; transfer story for VQA-RAD/SLAKE |
| C5 | DSL program synthesis + differentiable soft counting routed into fusion for count queries | `dsl.py` + `poe.py` | Wired via SymbolicFusionEngine count routing; frame as supporting component of the differentiable story, not standalone program-synthesis loop |
| C6 | Multi-scale morphological encoding (global + 4 high-res crops, learned blend) | `visual_encoder.py` | MR-PLIP = multi-resolution *pre-training*; ViLa-MIL = MIL classification; ours = architectural multi-scale at ViT-B/32 inference, off by default |
| C7 | Architecture-grounded failure taxonomy: perception / parsing / execution buckets mapped to components; name-based attribution when region names available | `interpretability.py` | Prescriptive (targets components) vs descriptive error labeling (GPT-4 USMLE, MedThink, GEMeX); accepts production CollateOutput batches |
| C8 | Calibrated evaluation: ECE + temperature scaling, max-softmax uncertainty, split-conformal sets, paired bootstrap significance | `metrics.py`, `conformal.py`, `evaluate.py` | Most medical VQA reports accuracy only; addresses SURE-VQA 2024 critique. BLEU/ROUGE/entity metrics are NOT implemented (answers are short classification strings) |
| C9 | Fully differentiable symbolic pipeline: learned qtype classifier + target extractor + differentiable executor, 3-epoch warm-up | `diff_*.py` | Gradients reach the text encoder through the symbolic path; default since Phase 10 |
| C10 | Concept-bottleneck interventions testing causal use of intermediate concepts (probe/attribution/counterfactual) | `concept_intervention.py`, `evaluate.py --intervention` | Intervention tests on scene concepts, not post-hoc saliency |

Honest accounting (not claimed novel): LoRA adapters, EMA, AMP, conformal
calibration itself, cross-attention fusion, RandAugment — established techniques.

## 5. History — six delivery tracks

The original roadmap used ten sequential build phases plus five production
stages, and the architecture audit added eleven ranked proposals. Grouped here
by theme so related work sits together; original labels kept in parentheses for
traceability.

| Track | Absorbs | Delivered |
|---|---|---|
| **T1 — Data & baseline** | Phases 1–2 | ResNet+biLSTM baseline recorded: 53.79% val (`outputs/baseline.json`); RandAugment + atomic uint8 tensor cache → 1.78× dataloader throughput (`outputs/cache_benchmark.json`) |
| **T2 — Modern model core** | Phase 3; audit proposal P6 | PLIP + DistilBERT, frozen + LoRA, cross-attention fusion → **60.33% val / 59.87% test, 0.38 GPU-h, 4.1 GB VRAM** (`outputs/upgrade.json`) |
| **T3 — Training machinery** | Phase 4 | AdamW + cosine + warmup, AMP on accumulation boundaries, grad accumulation + activation checkpointing, EMA, rolling checkpoints |
| **T4 — Learned neuro-symbolic fusion** | Phase 5; audit proposals P1, P2, P4, P5; novelty C9 (C5 wiring came with the 2026-08 audit) | Learned confidence gate C1; PoE fusion in log space; attention entropy as gate input (P4); faithfulness-test fix (P2); end-to-end differentiable symbolic supervision (P1); fully differentiable pipeline as default (C9); DSL interpreter routed for count queries via `SymbolicFusionEngine` |
| **T5 — Evaluation, ontology & interpretability** | Phases 6, 7, 8, 9; audit proposals P7, P8; novelties C4, C7, C8, C10 | Calibrated evaluation suite (ECE/temperature/conformal — P8), paired bootstrap, external baselines table, ontology-grounded region discovery (C4) with executor fallback (P7), failure taxonomy (C7), concept-bottleneck interventions (C10) |
| **T6 — Paper & production** | Phase 10; production stages P1–P5; novelty C3 evidence | Paper draft (`outputs/paper_draft.md`), FastAPI serving, Docker GPU deploy, CI + packaging, test expansion, interpretability API outputs |

Novelty claims not listed above: **C2** (LTN auxiliary loss) shipped with T4's
symbolic-training work; **C3** (resource efficiency) is the cross-cutting claim
evidenced by T2/T3 measurements plus T5's baselines table.

Ablation matrix A–E (Baseline / SOTA / Diff / NoGate / Regex+Diff):
definitions in §4 above; runner `scripts/run_ablations.py`,
`--ablation A…E` on train.py/evaluate.py.

Audit proposals that remain **future work**: typed-relational scene graph,
DeepProbLog-style probabilistic logic programs, cross-dataset transfer
(VQA-RAD, SLAKE adapters exist but untrained).

Key design alternatives considered and rejected: arithmetic-logit blending
(kept only as ablation), hand-crafted region atlases, LLM-mediated routing,
full scene-graph supervision (no labels).

2026-08-24 full-codebase audit: every file reviewed; defects fixed and recorded
in the completion ledger at the bottom of this file. A second full-codebase
audit (2026-10) re-reviewed the tree; its findings are in `Audit.md` and the
resulting changes are summarized below. Suite at close: 260 tests, ~90 s,
86.5% coverage; ruff + mypy strict clean; legacy-checkpoint loading, real API
inference and every CLI entry point covered by tests.

## 6. Publication strategy (essentials)

- **Narrative:** interpretable neuro-symbolic VQA at consumer-GPU cost; auditability
  (reasoning traces, concept interventions, conformal sets) vs black-box LLM systems.
  Do NOT claim accuracy parity with LLM methods; claim Pareto trade-off.
- **Manuscript skeleton:** Intro (auditability gap) → Related work (medical VQA,
  neuro-symbolic, parameter-efficient tuning) → Method (§2 here) → Experiments
  (ablations A–E, calibration, taxonomy, interventions) → Limitations (count
  accuracy, dataset-dependent regions, single-dataset training) → Conclusion.
- **Reviewer risks:** C1 vs Bao et al. 2021 (answer: learned + 4-conditioned +
  sample-adaptive); C5 dilution (frame as wired component); accuracy gap (Pareto).
- Interpretability metric definitions and reasoning-trace schema: see
  `utils/interpretability.py` docstrings and `/predict/explain` response schema.

## 7. Gotchas beyond AGENTS.md

- Answer `<UNK>` covers ~14% of val/test (vocab built from train).
- `HF_HOME` pinned to `data/hf-cache/`; first run downloads PathVQA.
- Mixed image modes (CMYK/RGB/L) — always `.convert("RGB")`.
- Cache key includes `(image_size, norm, TRANSFORM_VERSION)` — bump
  `TRANSFORM_VERSION` in `pathvqa_dataset.py` whenever preprocessing changes.
- Debug runs write only `checkpoints/debug/`; RNG state is not saved in checkpoints.
- Pre-DSL checkpoints (e.g. the Aug-18 `best_model.pt`) have no `dsl_interpreter.*`
  keys; `is_pre_dsl_checkpoint()` (`_compat.py`) flips `symbolic.dsl_enabled=False`
  at all three load sites so the model rebuilds as the architecture that saved
  them (random-init of the interpreter would silently mis-evaluate). Reproduced
  under 2026-10 code: 59.55% val / 59.73% test vs the Aug-era 60.33/59.87
  headline — drift affects both splits in the same direction and
  `train.py --debug` + `evaluate.py` agree exactly on identical weights, so the
  two rulers are consistent.

---

## Audit completion ledger (2026-08-24)

Full-codebase bug audit: every file below was reviewed and is **complete** — no known outstanding defects. One-liner = what was wrong / what changed.

| File | Status | Note |
|---|---|---|
| `src/pathvqa/symbolic/poe.py` | ✅ NEW | `SymbolicFusionEngine` — single shared PoE fusion; replaces 3 drifted copies; soft_target sized from `len(region_names)` |
| `train.py` | ✅ | Delegates fusion to engine; UNK vocab used real index; merged duplicate parse loop in validate(); dead imports removed |
| `evaluate.py` | ✅ | Delegates fusion to engine; loud intervention-import failure (was silent `pass`); Optional-typed signatures |
| `serve.py` | ✅ | Client errors → 400, internal bugs → logged 500 (was blanket 400 leaking internals) |
| `src/pathvqa/symbolic/executor.py` | ✅ | Early-exit zeros on correct device; fixed batch-mismatch in `__main__` demo |
| `src/pathvqa/utils/metrics.py` | ✅ | Temperature-scaling tensor created on logits' device |
| `src/pathvqa/utils/conformal.py` | ✅ | Uncertainty flag + set-size tensors on logits' device |
| `src/pathvqa/utils/interpretability.py` | ✅ | No fabricated patch indices in reasoning trees; taxonomy accepts CollateOutput/BatchData + legacy dicts; name-based parsing-vs-execution attribution |
| `src/pathvqa/data/pathvqa_dataset.py` | ✅ | `max_samples=0` honored (`is not None`); unknown split → ValueError with valid options |
| `src/pathvqa/data/dataset_adapter.py` | ✅ | Warns instead of failing silently on missing custom ontology/UMLS paths |
| `src/pathvqa/symbolic/routing.py` | ✅ | Docstring matches PoE log-space reality; accepts pre-parsed Query objects (crashed before); missing import |
| `src/pathvqa/symbolic/aux_losses.py` | ✅ | Misleading "no_grad-like context" comment replaced with accurate autocast explanation |
| `src/pathvqa/symbolic/ltn.py` | ✅ | "irregular" resolved via `SHAPE_VALUES.index()` (was hardcoded index 0) |
| `src/pathvqa/models/pathvqa_model.py` | ✅ | MultiScale built once (was constructed twice → double HF load); stale conversational comment removed |
| `src/pathvqa/models/protocols.py` | ✅ | Removed dormant `scene_density_logits` from `scene_head_dict()` (density head deleted) |
| `src/pathvqa/models/visual/visual_encoder.py` | ✅ | Comment corrected: blend MLP, not sigmoid gate |
| `tests/test_integration.py` | ✅ | Val-loader patch added → suite 445 s → ~26 s total |
| `tests/test_audit_regressions.py` | ✅ | +10 regression tests: PoE parity/device, soft_target sizing, no-fabrication, taxonomy interfaces, name-based attribution |
| `pyproject.toml` | ✅ | filterwarnings matched actual warning text ("Truncated File Read") |

**Verification at close (2026-10 re-audit):** ruff format/check clean · mypy strict clean (40 files) · pytest 260 passed, 86.5% coverage · 16/16 `__main__` smoke tests exit 0 · legacy checkpoint load and real `/predict` + `/predict/explain` covered by tests.
