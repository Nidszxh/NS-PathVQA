# AUDIT.md: Repository Audit and Submission Readiness

## 1. Summary

| Area | Status |
|---|---|
| Implementation findings | Resolved |
| Engineering gates (ruff, mypy, pytest) | Passing |
| Retained-checkpoint reproduction and image controls | Complete; separate from canonical run |
| Canonical seed-42 run, evaluation, controls, and manifest | Complete (August 30, 2026) |
| A–K matrix (single seed, descriptive) | Not started; no training-seed variance is estimated |
| Expert adjudication | Not done; package exported, no reviewer responses |
| PathMMU external evaluation | Not done; requires approved dataset access |
| Manuscript | Working draft, not submission-ready |

**Decision:** The canonical seed-42 run and its prescribed evaluation/control/manifest stages are complete. The A–K matrix, expert adjudication, PathMMU evaluation, and manuscript/release work remain incomplete; the repository is not ready for conference submission. The canonical run is a single-seed result and does not establish clinical reliability.

## 2. Retained checkpoint and contribution

- The pre-canonical retained checkpoint is preserved as `checkpoints/best_model.pre_canonical.pt`; it is a **static product-of-experts (PoE) checkpoint** using the regex query parser and hard executor. In this run workspace, `checkpoints/best_model.pt` is now the canonical seed-42 learned/differentiable checkpoint. Both files are gitignored and are not present in a fresh clone.
- Its serialized config predates explicit parser/executor fields. `backfill_config()` maps it to regex/hard, and the missing DSL parameters disable the DSL interpreter at load time.
- The state dictionary has no learned gate parameters.
- Fusion is `log p = (1 - g) log p_neural + g log p_symbolic`, with fixed `g = 0.3`.
- The earlier paper's learned-gate contribution and its 6.5-point gain were unsupported and have been removed. Learned weighting remains a comparison condition only.
- The draft is narrowed to the model the checkpoint actually represents and makes no clinical deployment claim.

## 3. Headline metrics (retained checkpoint)

Fresh full-split CUDA evaluations under current code (one NVIDIA RTX 4060 Laptop GPU, 8 GB). Artifacts: `outputs/final_repro_val.json`, `outputs/final_repro_test.json` (per-example predictions, provenance, answer coverage, image-cluster confidence intervals, calibration, conformal statistics).

| Split | Correct / total | Accuracy | 95% image-cluster bootstrap CI | Image clusters |
|---|---:|---:|---|---:|
| Validation | 3,752 / 6,259 | 59.9457% | [58.38%, 61.50%] | 832 |
| Test | 4,019 / 6,719 | 59.8154% | [58.21%, 61.49%] | 858 |

| Split | Answer-vocab coverage | Accuracy on covered answers | Accuracy on unseen answers |
|---|---:|---:|---:|
| Validation | 85.62% (5,359/6,259) | 70.01% | 0% |
| Test | 85.47% (5,743/6,719) | 69.98% | 0% |

Unseen answers map to `<UNK>` and cannot be matched exactly, hence 0%.

| Split | yes/no | identity | location | count | ECE | Fitted T |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 88.47% (2,755/3,114) | 25.01% (679/2,715) | 76.53% (313/409) | 23.81% (5/21) | 0.1745 | 1.544 |
| Test | 87.82% (2,934/3,341) | 26.25% (769/2,929) | 71.93% (310/431) | 33.33% (6/18) | 0.1806 | 1.542 |

**Calibration/conformal figures in the retained artifacts above are superseded.** A further audit found that rows from the same image had been treated as independent calibration units, and temperature fitting and ECE reporting used the same split. The current implementation instead partitions by image-content group, fits temperature on calibration groups, and computes held-out ECE on disjoint image groups. Conformal calibration uses the maximum nonconformity score within each calibration image group and evaluates question-level and simultaneous group coverage. Existing JSON artifacts have not been regenerated under this corrected protocol, so no updated calibration or conformal numbers are claimed here. These procedures still rely on exchangeability of distinct image groups; PathVQA's fixed benchmark split and model-selection history limit the scope of formal coverage claims.

**Retained-checkpoint configuration:** `openai/clip-vit-base-patch32` (frozen, LoRA r=16/α=32, 49 patch tokens), frozen DistilBERT + LoRA, 40 region names, static `g=0.3`, regex/hard path, no DSL interpreter, 42 training epochs, best stored validation accuracy 60.33%. The stored training-time figure is not the current-code reproduction.

**Provenance recorded by the evaluator:** pinned dataset revision `1685832883334b5bb5beaf4e4b333fdeecaa4ad9`, split fingerprint and counts, checkpoint SHA-256, full config, seed, per-example predictions, exact raw-answer accuracy, answer-vocabulary coverage, runtime device, and dataset-derived image-cluster IDs.

**Split sizes:** 19,654 train / 6,259 validation / 6,719 test. Validation is for model selection; test is reserved for the frozen final recipe.

The older 60.33% / 59.87% figures and the superseded 59.55% / 59.73% image-control figures are not headline reproduction results. The legacy GPU-hour figures are inconsistent and must not be used.

## 4. Resolved implementation findings

| Finding | Resolution | Evidence / boundary |
|---|---|---|
| Multi-scale setting silently ignored by callers | `NeuroSymbolicPathVQA.forward(use_multiscale=None)` inherits the construction setting; the crop branch runs at train/eval/serve. | `src/pathvqa/models/pathvqa_model.py`. Multi-scale is off in the released checkpoint. |
| Ablation definitions could drift | One A-K matrix in `src/pathvqa/utils/ablation.py` used by train, evaluate and runner. A-H = 2x2x2 parser/executor/gate factorial; I = neural-only; J/K remove LTN, then all auxiliary symbolic losses. | `tests/test_ablation.py` |
| Interrupted ablation matrix lost state or overwrote a different seed recipe | Runner writes an atomic progress artifact before execution and after each seed, tracks pending/running/success/failure, and includes seeds in the default output name. | `scripts/run_ablations.py`; fake-runner tests |
| Paper claimed a learned gate and misidentified parser/executor | Checkpoint is static log-space PoE (g=0.3), regex/hard, no gate parameters. | Direct checkpoint inspection and fresh evaluation artifacts; manuscript now matches |
| Synthetic paper figures | Figure generator consumes evaluation artifacts; synthetic examples only under explicit `--mock`, watermarked. | `scripts/generate_paper_figures.py`; mock output is never evidence |
| Missing provenance in training/evaluation | Artifacts capture config, dataset revision/fingerprint/split sizes, checkpoint path/hash, seed, image condition, per-example predictions, vocabulary coverage, runtime details. | `train.py`, `evaluate.py`; old artifacts remain incomplete and are not promoted |
| Legacy differentiable-pipeline migration | `backfill_config()` checks missing fields and maps legacy `differentiable_pipeline=True` to learned parser + differentiable executor. | `src/pathvqa/utils/config.py`; migration tests |
| Differentiable executor over-reported symbolic use | Counts that bypass symbolic execution are marked not symbolically used; metrics emitted by the shared fusion path. | `src/pathvqa/symbolic/poe.py`; executor tests |
| Yes/no evidence treated raw region logits as probabilities | Dedicated yes/no presence logits; described as uncalibrated evidence. | `symbolic/executor.py`, `diff_executor.py`; no post-hoc calibration claimed |
| Calibration treated repeated questions per image as independent; temperature and ECE reused the same examples | Temperature fitting/evaluation and conformal calibration/evaluation now split by image-content group. Group conformal scores are per-image maxima; the metric reports question coverage and simultaneous group coverage. | `evaluate.py`, `src/pathvqa/utils/conformal.py`, `src/pathvqa/utils/metrics.py`; prior calibration/conformal artifact values are superseded and must be regenerated |
| Frozen encoders could remain in train mode | Frozen visual/question backbones override `train()` to stay in eval mode. | `visual_encoder.py`, `question_encoder.py` |
| Determinism unsupported | Deterministic mode and per-worker seeding; cuBLAS workspace set before CUDA seeding/init. | `src/pathvqa/utils/seed.py`; uses `warn_only`, so exact determinism is not promised for unsupported ops/hardware |
| API counterfactual was a fabricated no-change result | API suppresses the selected predicted concept and re-decodes through the model. | `src/pathvqa/api/inference.py`; a model intervention, not clinical causality |
| Attribute mappings differed across train/eval/serve | All build paths pass the same mappings. | `train.py`, `evaluate.py`, `api/inference.py` |
| Dead density head / unused count head / relation placeholder | Density head removed with legacy-checkpoint migration; DSL count head called; relation argument consumed by relation-offset scoring. | `src/pathvqa/_compat.py`, `symbolic/dsl.py`; spatial relation coverage remains limited |
| Learned target extractor had no representation for questions without a region referent | Fresh configs add a no-region class, weakly supervised from the parsed question target; runtime abstains when its probability wins. Legacy checkpoint configs retain the old head shape. | `src/pathvqa/models/pathvqa_model.py`, `src/pathvqa/symbolic/aux_losses.py`, `src/pathvqa/symbolic/poe.py`; labels are parser-derived weak supervision, not visual ground truth |
| Spatial DSL compilation defaulted all location relations to `located_in` | Compiler maps explicit directional/containment/near language to supported relation labels and uses `located_in` only as a fallback. | `src/pathvqa/symbolic/dsl.py`; phrase extraction is heuristic and does not establish spatial grounding |
| Region order, conformal max set size, ontology checks, failure classification | Stable region order; configured set-size limit used; ontology lookup and failure inputs corrected. | Adapter, conformal, ontology, evaluation modules; failure taxonomy remains heuristic |
| Autocast / EMA / GPU-hour / resume inconsistencies | Shared PoE in train/eval/serve; mixed-precision casts match buffers; EMA tracks floating values and restores live state; GPU-hours counts allocated GPU processes only; resume reapplies explicit CLI overrides. | Relevant modules and regression tests; resource totals need a real run |
| Evaluation could replace checkpoint architecture config | Checkpoint architecture is authoritative; compatible runtime/data overrides only. | `evaluate.py`; architecture-changing ablations must be applied before model build (strict loading is intentional) |
| Image-dependence and external evaluation tooling absent | Added actual/blank/shuffled/image-only conditions, paired bootstrap + exact McNemar report, blind expert-review export, PathMMU evaluator (reports coverage and skipped cases). | `scripts/analyze_image_dependence.py`, `export_expert_review.py`, `evaluate_pathmmu.py`; tooling is not completed validation |
| Image-dependence intervals treated repeated questions as independent | Content-derived image cluster IDs; cluster-bootstrap intervals and image-cluster sign tests. | `evaluate.py`, `scripts/analyze_image_dependence.py`, `tests/test_image_dependence.py` |
| API route tests stalled in AnyIO/TestClient | Async FastAPI routes; blocking prediction via `asyncio.to_thread`; tests use `httpx.AsyncClient` with ASGI transport. | `serve.py`, `tests/test_api.py`; 19 API tests pass offline with cached models |

## 5. Evidence status by research area

### 5.0 Scientific assumptions checked in this pre-run pass
- PathVQA is a closed-vocabulary benchmark assembled from pathology images and question-answer pairs; the source paper describes question construction from image captions and pathology textbooks with manual checking. Its benchmark accuracy does not imply clinical validity. [He et al., PathVQA](https://arxiv.org/abs/2003.10286).
- Repeated questions for one image are dependent observations for uncertainty estimation. This pass changed calibration units to image-content groups; group maximum scores give a conservative simultaneous-within-group construction, conditional on exchangeability between groups. This does not eliminate fixed-split/model-selection concerns. [Dunn, Wasserman & Ramdas, hierarchical conformal prediction](https://arxiv.org/abs/1809.07441).
- Temperature scaling is a held-out calibration procedure; the current evaluator uses disjoint image groups for fit and ECE assessment. No corrected numeric calibration artifact has been generated yet. [Guo et al., On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
- PLIP's published pretraining corpus (OpenPath) is large and pathology-specific. The available evidence does not establish whether any PathVQA source images/text overlap its pretraining data; therefore this audit cannot certify absence of pretraining contamination. [Huang et al., PLIP](https://www.nature.com/articles/s41591-023-02504-3).
- Regex-derived qtype/target labels are proxies for intended semantics, not annotated parser ground truth. The new target head's loss is explicitly weak supervision. Parser accuracy and anatomical grounding require independent annotation or expert adjudication.

### 5.1 Experimental validation: canonical run complete, A–K results outstanding
- The A-K full factorial and controls (A-H parser x executor x weighting; I-K neural-only / loss controls) are implemented.
- `scripts/run_ablations.py` defaults to seed 42 and `--keep-checkpoints best+last` (best weights plus the final-epoch snapshot, ~2.6 GB per run), preserves per-seed validation artifacts, and aggregates including failures. It does not select on test. Multiple seeds remain available via `--seeds` but are not part of the current protocol.
- The canonical seed-42 run, full checkpoint validation/test evaluation, image controls, and release manifest are complete; the A–K matrix is **not started**. The canonical run used one RTX 4060 Laptop GPU and its measured runtime/provenance are recorded in `outputs/canonical_seed42.json` and `outputs/canonical_seed42_release_manifest.json`.
- Protocol: seed 42 across A–K, validation only. The result is descriptive coverage and must not be reported as a component-effect estimate; a single seed cannot separate a variant effect from seed noise.
- If effect claims are added later, the protocol becomes repeated-seed: several seeds per variant, mean ± sample standard deviation, paired per-seed differences, and test evaluation only after selection. `scripts/select_recipe.py` refuses aggregates with fewer than two completed seeds so a single-seed ranking cannot be reported as a frozen recipe.

### 5.2 Image dependence: retained-checkpoint controls

The table below is the earlier retained-checkpoint control set. It is distinct
from the canonical-checkpoint results that follow.

All four conditions were evaluated on the full 6,259-question validation split with the same checkpoint. Image IDs come from exact decoded RGB content of the pinned dataset (no model inference), giving 832 image clusters. Intervals are 10,000-resample cluster bootstraps.

| Condition | Accuracy | Paired diff. vs actual | 95% cluster interval | Sign-test p (non-tied clusters) | McNemar p |
|---|---:|---:|---|---:|---:|
| Actual image | 59.95% | n/a | n/a | n/a | n/a |
| Blank image | 45.77% | -14.17 pts | [-15.49, -12.82] | 2.01x10^-80 (557) | 2.97x10^-164 |
| Shuffled image | 43.27% | -16.68 pts | [-17.97, -15.42] | 1.93x10^-102 (569) | 7.01x10^-213 |
| Empty question | 7.89% | -52.05 pts | [-53.52, -50.61] | 3.19x10^-231 (793) | underflow (<10^-323) |

Report: `outputs/final_image_dependence/val_paired_comparisons_clustered.json` (10,000-resample bootstraps). Intervals preserve repeated QA pairs within a cluster, but inference assumes distinct content clusters are independent. This supports image dependence on PathVQA; it does not validate anatomical grounding or clinical reliability. Repeat after the final matched-seed recipe is chosen.

### 5.2.1 Canonical seed-42 checkpoint controls (August 30, 2026)

The canonical checkpoint was evaluated on all 6,259 validation examples under
actual, blank, shuffled, and image-only conditions. The four artifacts share
the same question rows and targets; shuffled image-source rows differ by
design. Cluster-aware paired intervals use 10,000 bootstrap resamples over
832 image-content groups.

| Condition | Accuracy | Difference vs actual | 95% image-cluster interval |
|---|---:|---:|---|
| Actual image | 60.28% (3,773/6,259) | — | — |
| Blank image | 42.80% (2,679/6,259) | −17.48 pp | [−18.81, −16.15] pp |
| Shuffled image | 42.88% (2,684/6,259) | −17.40 pp | [−18.73, −16.07] pp |
| Image only | 18.10% (1,133/6,259) | −42.18 pp | [−43.95, −40.43] pp |

Artifacts: `outputs/canonical_seed42_image_dependence/val_{actual,blank,shuffled,image_only}.json`, `val_paired_comparisons.json`, and `val_cluster_analysis.json`. These paired controls support image dependence on this benchmark split, not anatomical grounding or clinical reliability. Group-level uncertainty still assumes exchangeability/independence between distinct image groups and does not eliminate model-selection or fixed-split limitations.

### 5.3 Expert and external validation: pending
- `scripts/export_expert_review.py` produced an 89-case blinded review package, a separate adjudication key, and analyst-only sampling weights in `outputs/final_expert_review/`. No experts have adjudicated it.
- The PathMMU evaluator reports score coverage and skipped cases. The official [PathMMU dataset](https://huggingface.co/datasets/jamessyx/PathMMU) requires access approval, and its [acquisition instructions](https://github.com/PathMMU-Benchmark/PathMMU/blob/main/data/instructions.md) require obtaining some source images from original providers. No external score is claimed until access and all local image assets are in place.

### 5.4 Interpretability and reliability: claims narrowed, evidence pending
- Counterfactual output performs concept suppression and model re-decoding. The canonical evaluation artifacts report conformal coverage, but prediction sets average all 3,226 answer classes and therefore do not narrow the answer vocabulary. Neither the intervention nor conformal output is evidence of clinical faithfulness.
- The paper describes traces as model-generated, does not equate attention with explanation, and makes no causal clinical reasoning claim.
- Concept interventions, expert trace review, and reliability analysis need final run artifacts and/or human ratings.

### 5.5 Novelty and literature comparison
- PathVQA is the in-domain benchmark. K-PathVQA (knowledge-graph-based) and Path-RAG (knowledge-guided region retrieval) are directly related. PathMMU is expert-reviewed but multiple-choice, so results are not comparable without a matched protocol.
- The contribution is framed as a controlled system/evaluation study; novelty remains modest until matched ablations and external/expert evidence exist.

### 5.6 Manuscript, baselines, release
- No manuscript file is present in the current workspace. A manuscript update remains necessary and should use the canonical artifacts while preserving the distinction from retained-checkpoint reproduction results. Still needed: the seed-42 A–K coverage matrix, expert and external validation, and venue formatting.
- `outputs/baselines.md` separates protocol-incompatible literature results and does not present them as matched baselines. Old unsupported numbers are excluded.
- The canonical manifest `outputs/canonical_seed42_release_manifest.json` records SHA-256 hashes for 88 source files (including the current run documentation) and the canonical checkpoint, plus dataset revision and runtime metadata. Regenerate it after any future source or documentation edits before archiving a final release. Checkpoints/data are not committed to Git; a release archive or stable external location and license review are still needed before public release.

## 6. Verification record

This section records code verification and the completed canonical run. Historical entries are retained with their dates and scope; they should not be mistaken for results from the canonical checkpoint.

- `ruff format --check .` passed; `ruff check .` passed.
- `mypy src/` passed (40 source files, strict).
- Current pre-run pass: full `pytest -q --basetemp=scratch/pytest` completed with coverage **86.35%**, above the configured 80% gate. The focused changed-module suite also passed. `ruff format --check .`, `ruff check .`, strict `mypy src/` (40 files), and `git diff --check` passed after the code changes.
- The focused API suite passes all 19 tests in offline mode with cached model assets, covering routing, inference, lifespan, and error handling.
- A stale conformal test expectation was updated for the exact `n+1` quantile (when `k > n`, `tau = 1` includes every class).
- The trainer artifact writer handles generic iterable loaders with optional dataset provenance; real DataLoaders still record dataset metadata.
- **August 30, 2026 canonical run:** bounded CPU smoke completed (five debug epochs; workers set to 0) and an 8-example validation check passed. The first full training attempt exposed an infinite answer NLL at epoch 3. The DSL mask was changed from an fp16-overflowing `-1e9` to a finite `-30` floor with float32 evidence storage; yes/no auxiliary BCE now promotes probabilities to float32 before clamp/loss; nonfinite objective detection fails fast with diagnostics. The repaired 10-epoch diagnostic completed with finite train/validation losses. Targeted regression tests and the full gates passed after the code fixes: formatting, lint, strict mypy, full pytest (86.35% coverage), and `git diff --check`.
- Canonical training then completed 33 epochs with configured early stopping (10 consecutive non-improving validation runs); best validation epoch index 22, best validation accuracy 60.2812%. It used 1.356 GPU-hours, one RTX 4060 Laptop GPU, and peaked at 2,857.7 MB VRAM. Full evaluation was 60.28% validation (3,773/6,259; ECE 0.1748) and 60.01% test (4,032/6,719; ECE 0.1801). Canonical image controls and cluster-aware intervals are recorded in §5.2.1. The conformal reports have 100% coverage but average set size 3,226 (the entire answer vocabulary), so they do not provide useful set narrowing.
- Canonical artifact paths and hashes are listed in `RUN.md` and `outputs/canonical_seed42_release_manifest.json`. No files were staged or committed. CUDA deterministic mode is best-effort: attention backward kernels may be nondeterministic.
- Older bounded debug runs, API smoke, cache checks, and A–K tool smoke below are historical checks and do not substitute for the canonical full-split run.
- API smoke with the retained checkpoint on CUDA: `/health` and `/model/info` return the checkpoint/device/40-region metadata; `/predict` and `/predict/explain` answer a real PathVQA validation image and return a model-derived trace without error.
- Cache verified: uint8 cache `data/cache/224x224_clip_v1` matches current preprocessing byte-for-byte on probes across train (500 indices 0–499, plus 19,654 rows after prefill), val, and test splits (max absolute difference `0.0`). Train cache was incomplete (500 of 19,654 rows, left over from the debug run) and has been prefilled to full coverage in 73 s; disk usage grew 4.8 GB. `TRANSFORM_VERSION` remains `v1` and is unchanged.
- The earlier run workspace had its train-image cache prefilled and `scratch/pytest` plus intermediate checkpoints cleaned. A later repository cleanup removed `scratch/`, Python/tool caches, the HF dataset/model cache, the image tensor cache, and superseded training checkpoints. Canonical and retained-checkpoint artifacts remain; future runs must redownload/rebuild the data caches.
- Protocol revised: A–K matrix is now single-seed (seed 42) and descriptive only. `scripts/run_ablations.py` defaults to `--seeds 42` and `--keep-checkpoints best+last`; `scripts/select_recipe.py` refuses aggregates with fewer than two completed seeds. Docs (`RUN.md`, `README.md`, `AGENTS.md`) and citation of the `outputs/final_expert_review/` location were updated. The retained-checkpoint reproduction, image controls, and downstream artifacts are untouched.
- Ablation tooling smoke (`--debug --ablations A I --seeds 7`, 100-example validation): the runner trained both variants, wrote an atomic aggregate (`outputs/ablation_smoke_debug.json`), `analyze_ablations.py` produced mean/std and paired differences, and `select_recipe.py` froze a validation-only choice. These debug numbers are a plumbing check and are **not** evidence about the variants.

## 7. Outstanding work (evidence or access, not unimplemented code)

1. **A–K matrix (seed 42):** run the eleven variants once, validation only, and report the result as descriptive coverage with the single-seed limitation stated. Do not present the deltas as effects. If effect claims are added, switch to a repeated-seed protocol and re-derive the table.
2. **Image controls:** canonical seed-42 controls are complete. If a final recipe is selected after a future A–K/repeated-seed study, rerun controls for that selected checkpoint.
3. **Expert adjudication:** conduct blinded review from `outputs/final_expert_review/`; no responses exist yet.
4. **PathMMU:** run after an authorized download/access grant and required image assets; report scored coverage.
5. **Manuscript and release:** update any manuscript tables/figures from the canonical artifacts; regenerate the manifest after documentation/manuscript edits; archive source/data/checkpoint artifacts; complete citations, venue formatting, license review, and release location.
