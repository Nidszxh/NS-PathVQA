# NS-PathVQA — Change Log and System Reference

> Consolidates the project history and system reference formerly spread across
> `PROJECT.md`, `NOVELTIES.md`, `TODO.md`, and `NS-PATHVQA-AUDIT.md`.
> Division of labor: **README.md** = quick start · **AGENTS.md** = working rules +
> commands + config defaults · **CHANGELOG.md** (this file) = system reference,
> novelty register, history, audit outcomes · **NS-PathVQA.md** = personal portfolio
> write-up (not project docs).

---

## 1. System overview

Neuro-symbolic Visual Question Answering over the pinned Hugging Face PathVQA
revision (19,654 train / 6,259 validation / 6,719 test QA pairs). A neural path
(frozen CLIP or PLIP ViT-B/32 + LoRA, plus frozen DistilBERT + LoRA, with
cross-attention) is combined with a symbolic path (scene parser → query parser →
executor) by a weighted geometric Product-of-Experts in log space. The retained
checkpoint used `openai/clip-vit-base-patch32`; PLIP (`vinid/plip`) is the
current default for newly trained models. The retained `best_model.pt` uses **static**
weighting; learned gating is implemented as a comparison condition but is not
present in that checkpoint:

```
g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)]))     # or static w = symbolic_weight
log p = (1 − g) · log p_neural + g · log p_symbolic            # NLLLoss on fused log-probs
```

Single implementation: `SymbolicFusionEngine` (`src/pathvqa/symbolic/poe.py`);
train.py / evaluate.py / api/inference.py all delegate to it.

Recorded current-code reproduction of the retained checkpoint: 59.9457% validation
(3,752/6,259) and 59.8154% test (4,019/6,719), single checkpoint. The saved
artifacts record CUDA runtime metadata, and the current environment reports one CUDA
device; the exact hardware/runtime manifest should accompany any final release.
Historical 60.33% / 59.87%, GPU-hour, and VRAM claims are not current headline
results. See `AUDIT.md` for evidence status and the run gate.

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
| 7 | Executor | `symbolic/executor.py`, `diff_executor.py` | **hard**: vectorized scatter_add mapping scene+query→answer vocab (count branch = zeros; yes/no writes binary log-odds only for region targets); **differentiable** (default): branch logits mixed via soft_qtype weights |
| 8 | Gate & fusion | `symbolic/routing.py`, `symbolic/poe.py` | Learned gate (4 signals incl. attention entropy); PoE engine shared by train/eval/API; per-row evidence check (`trace["symbolic_used"]`) keeps the neural distribution when the symbolic path wrote nothing; learned qtype/target drive the hard executor; count queries with patch features route through the DSL interpreter |
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
| Features | All 49 patch tokens, no proposal network, mask all-ones | Keeps `(B,49,512)` contract; removes obsolete top-k proposals |
| Positions | Reuse CLIP positional embeddings | Patches already carry position |
| Text | DistilBERT frozen+LoRA; HF tokenizer; question_vocab.json removed | Light on 8 GB; eval loses prior-run dependency |
| Preprocess | 224², CLIP mean/std read from `CLIPImageProcessor` at runtime | Hand-typed constants silently degrade features; ImageNet norm incompatible |
| Training | AdamW + cosine + warmup, AMP, grad accumulation + checkpointing, EMA | Replaces Adam+ReduceLROnPlateau; no new deps |

## 4. Novelty register (C1–C10)

| ID | Claim | Where | Differentiator vs prior work |
|---|---|---|---|
| C1 | Learned per-sample confidence gate is implemented as an ablation | `routing.py` + `poe.py` | The retained checkpoint uses static weighting. No learned-gate benefit is established. |
| C2 | Logic Tensor Network as differentiable auxiliary loss enforcing domain consistency (organ-system co-occurrence, morphology clauses) | `ltn.py` | LTNs not previously used as VQA aux loss for pathology; clause "irregular" resolved by name (`SHAPE_VALUES.index`) |
| C3 | Resource-efficiency is a measurement goal, not a comparative claim | whole pipeline | The canonical seed-42 run records 1.356 GPU-hours and peak VRAM; one run on one GPU does not establish comparative efficiency. |
| C4 | Ontology-grounded region discovery linking answer vocab to UMLS/RadLex organ-system hierarchy (10 systems, 140+ terms) | `dataset_adapter.py` | Region set derived from data+ontology rather than hand-picked; transfer story for VQA-RAD/SLAKE |
| C5 | DSL program synthesis + differentiable soft counting routed into fusion for count queries | `dsl.py` + `poe.py` | Wired via SymbolicFusionEngine count routing; emits log-space evidence over vocabulary-derived count answers; frame as supporting component of the differentiable story, not standalone program-synthesis loop |
| C6 | Multi-scale morphological encoding (global + 4 high-res crops, learned blend) | `visual_encoder.py` | MR-PLIP = multi-resolution *pre-training*; ViLa-MIL = MIL classification; ours = architectural multi-scale at ViT-B/32 inference, off by default |
| C7 | Architecture-grounded failure taxonomy: perception / parsing / execution buckets mapped to components; name-based attribution when region names available | `interpretability.py` | Prescriptive (targets components) vs descriptive error labeling (GPT-4 USMLE, MedThink, GEMeX); accepts production CollateOutput batches |
| C8 | Evaluation tooling for ECE, temperature scaling, max-softmax uncertainty, split-conformal sets, and paired comparisons | `metrics.py`, `conformal.py`, `evaluate.py` | Tool availability does not establish empirical calibration or significance; final measurements require refreshed artifacts. BLEU/ROUGE/entity metrics are not implemented. |
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
| **T2 — Modern model core** | Phase 3; audit proposal P6 | PLIP + DistilBERT, frozen + LoRA, cross-attention fusion. Historical `outputs/upgrade.json` reports 60.33% val / 59.87% test; the current `outputs/final_repro_{val,test}.json` artifacts report 59.9457% val / 59.8154% test. |
| **T3 — Training machinery** | Phase 4 | AdamW + cosine + warmup, AMP on accumulation boundaries, grad accumulation + activation checkpointing, EMA, rolling checkpoints |
| **T4 — Neuro-symbolic fusion** | Phase 5; audit proposals P1, P2, P4, P5; novelty C9 | PoE fusion in log space; learned confidence gate is implemented but absent from the saved `best_model.pt` (`weighting_strategy=static`). That legacy checkpoint loads as regex parser + hard executor with DSL disabled; learned/differentiable modules belong to current configurations and require their own matched checkpoints. |
| **T5 — Evaluation, ontology & interpretability** | Phases 6, 7, 8, 9; audit proposals P7, P8; novelties C4, C7, C8, C10 | Calibrated evaluation suite (ECE/temperature/conformal — P8), paired bootstrap, external baselines table, ontology-grounded region discovery (C4) with executor fallback (P7), failure taxonomy (C7), concept-bottleneck interventions (C10) |
| **T6 — Paper & production** | Phase 10; production stages P1–P5; novelty C3 evidence | A paper draft was generated in an earlier ignored-output workspace but is not present in the current checkout; FastAPI serving, Docker GPU deploy, CI + packaging, test expansion, interpretability API outputs |

These entries describe implemented components, not demonstrated novelty. The
paper frames the work as a controlled system/evaluation study; claims about
component benefit require the matched experiments and external validation.

Matched ablations A–K: A–H are the full regex/learned parser × hard/differentiable
executor × static/learned weighting factorial; I is neural-only; J disables LTN;
K disables LTN and auxiliary symbolic losses. Definitions are in
`src/pathvqa/utils/ablation.py`; runner `scripts/run_ablations.py --seeds 42 43 44`.
The retained `best_model.pt` predates explicit parser/executor config fields;
`backfill_config()` maps it to regex + hard execution and static weighting
(variant A). It is not a learned-gate checkpoint. Gate or parser/executor
effects must be established with matched, repeated-seed runs before making a
contribution claim.

Audit proposals that remain **future work**: typed-relational scene graph,
DeepProbLog-style probabilistic logic programs, cross-dataset transfer
(VQA-RAD, SLAKE adapters exist but untrained).

Key design alternatives considered and rejected: arithmetic-logit blending
(kept only as ablation), hand-crafted region atlases, LLM-mediated routing,
full scene-graph supervision (no labels).

Earlier full-codebase audit history is retained below. Current findings and
verification status are maintained in `AUDIT.md`; do not treat prior suite
results as verification of the latest changes.

## 6. Publication strategy (essentials)

- **Narrative:** a reproducible controlled study of neuro-symbolic fusion on
  closed-vocabulary PathVQA. Do not claim learned-gate gains, Pareto superiority,
  clinical reliability, or validated explanations without supporting results.
- **Manuscript skeleton:** Intro (auditability gap) → Related work (medical VQA,
  neuro-symbolic, parameter-efficient tuning) → Method (§2 here) → Experiments
  (ablations A–K, calibration, taxonomy, interventions) → Limitations (count
  accuracy, dataset-dependent regions, single-dataset training) → Conclusion.
- **Reviewer risks:** C1 vs Bao et al. (answer: learned + 4-conditioned +
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
- Pre-DSL checkpoints (e.g. the retained legacy `best_model.pt`) have no `dsl_interpreter.*`
  keys; `is_pre_dsl_checkpoint()` (`_compat.py`) flips `symbolic.dsl_enabled=False`
  at all three load sites so the model rebuilds as the architecture that saved
  them (random-init of the interpreter would silently mis-evaluate). Reproduced
  with the current evaluation code: 59.9457% validation / 59.8154% test. Earlier
  headline values are superseded; see `AUDIT.md` for complete artifacts.

---

## 8. Latest audit round (August 28, 2026, pre-master-run)

A second code review found symbolic-path defects that earlier audits had missed.
All are fixed with regression tests; the retained checkpoint was re-evaluated
afterwards (numbers in `AUDIT.md`).

| Finding | Effect | Fix / boundary |
|---|---|---|
| Yes/no rows with non-region targets scattered region logits into the answer vocabulary | A region class could outrank `yes`/`no`; affected 2,159/3,114 validation yes/no rows for the retained checkpoint | `executor.py` writes binary evidence only for matching region targets; other yes/no rows emit nothing and stay neural-only |
| Unused symbolic rows were blended with `log_softmax(zeros)` | Count rows without a DSL interpreter (and rows with no evidence) were pulled toward the uniform distribution at weight `g` | `poe.py` computes per-row `symbolic_used` from written evidence and returns `log p_neural` unchanged for unused rows |
| DSL emitted raw similarities in `[0,1]` | Non-target zeros outranked the predicted count after the fusion `log_softmax`, inverting the signal | `dsl.py` emits log evidence on candidates and masks non-candidates; count candidates are derived from the answer vocabulary (`count_answer_values`) instead of a fixed 1–5 map |
| Learned parser did not reach the hard executor | The learned+hard ablation executed regex queries, so the parser factor was not actually varied | `poe.py::_effective_queries` uses the argmax qtype and the most probable region target (above chance) for the hard executor, and recompiles the DSL program |
| API built the model without attribute mappings | The `QueryAttr` DSL branch was silently disabled when serving, unlike train/eval | `api/inference.py` passes `self.attribute_mappings` to `build_model` |
| `evaluate.py` pinned too few architecture fields | `dsl_enabled`, region count, and LoRA ranks could drift through `--config` while loading the same checkpoint | `_ARCHITECTURE_FIELDS` extended; `validate_config` checks gate range, conformal α, epochs, batch size, image size |
| Accuracy had no uncertainty interval | A single split number without sampling uncertainty invites over-reading | `evaluate.py` reports a 95% percentile bootstrap over image-content clusters; `scripts/analyze_ablations.py` summarises repeated seeds and paired differences; `scripts/select_recipe.py` freezes the recipe on validation only |

**Final verification for this round (August 29, 2026):** `ruff format --check .`,
`ruff check .`, and strict `mypy src/` pass; full pytest is **299 passed at
87.20% coverage**; a debug E2E run (learned parser + differentiable executor +
DSL) evaluated 200 validation and 100 test examples; the retained checkpoint was
re-evaluated on both full splits (59.9457% / 59.8154%) and on all four full-split
image conditions (actual 59.95%, blank 45.77%, shuffled 43.27%, empty question
7.89%); the API answered a real PathVQA image through `/predict` and
`/predict/explain`; and the A–K runner/analysis/recipe tools completed a
two-variant debug smoke. None of these is the master run.

## 9. Canonical run and mixed-precision loss fixes (August 30, 2026)

The authorized canonical sequence completed: bounded smoke; a repaired 10-epoch
diagnostic; canonical training; full validation/test checkpoint evaluation;
actual, blank, shuffled, and image-only validation controls; paired
image-cluster analysis; and release-manifest generation. The A–K matrix,
expert adjudication, and PathMMU evaluation remain future work.

The initial canonical training attempt produced an infinite answer NLL at
epoch 3. Diagnostics isolated the immediate fault to DSL non-candidate logits:
the `-1e9` floor was cast into an fp16 evidence tensor and became `-inf` for
some gold answers. The DSL now uses a finite `-30` floor and the fusion stores
DSL evidence in float32. A separate yes/no auxiliary BCE precision issue was
fixed by promoting probabilities to float32 before clamping and BCE. A
fail-fast check now emits component/row diagnostics and stops before
backpropagation if total loss is nonfinite. Regression coverage was added for
both precision paths. The repaired diagnostic passed 10 epochs; full local
gates passed after these changes.

Canonical training stopped normally after 33 epochs at the configured
early-stopping patience, with best validation accuracy 60.2812% at recorded
epoch index 22. It used 1.356 GPU-hours on one RTX 4060 Laptop GPU, with
2,857.7 MB peak VRAM. The resulting checkpoint measured 60.28% validation
(3,773/6,259; ECE 0.1748) and 60.01% test (4,032/6,719; ECE 0.1801).
Canonical image controls measured 60.28% actual, 42.80% blank, 42.88%
shuffled, and 18.10% image-only. Cluster-aware paired intervals and all paths
are recorded in `RUN.md` and `AUDIT.md`.

**Interpretation limits:** conformal coverage was 100% with an average set
size of all 3,226 answer classes, so it is not useful set narrowing. Group
bootstrap inference assumes exchangeability/independence between distinct
image groups. Regex-derived parser labels remain weak supervision; the
available evidence cannot establish overlap between PLIP pretraining and
PathVQA. The benchmark results do not establish clinical reliability. CUDA
deterministic mode is best-effort for attention backward kernels.

The current workspace's `checkpoints/best_model.pt` is the canonical
checkpoint; the pre-run legacy checkpoint is preserved at
`checkpoints/best_model.pre_canonical.pt`. Both are gitignored. The release
manifest records the canonical checkpoint hash and 88 source-file hashes; it
must be regenerated after subsequent documentation edits before archiving a
final release. No files were staged or committed.

## Earlier audit completion ledger

Full-codebase bug audit at the time: the files below were reviewed and the
listed defects fixed. That ledger is historical — it is **not** a claim that no
defects remained, as the latest round above demonstrates. One-liner = what was
wrong / what changed.

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

**Historical verification:** an earlier suite run had 260 passing tests and 86.5% coverage. Current implementation verification is recorded in `AUDIT.md`; do not treat that historical count as validation of current changes.
