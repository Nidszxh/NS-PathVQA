Codebase Audit Report: PathVQA
1. High-Priority Findings
1.1 Multi-Scale Visual Fusion Is Disabled in Production

Files: pathvqa_model.py:110,117; visual_encoder.py:111

NeuroSymbolicPathVQA.forward() defaults to use_multiscale=False, and no production caller overrides it. Although build_model() constructs MultiScaleVisualEncoder when configured, the scale-fusion MLP and crop branch are never exercised. Consequently, multi-scale configurations silently use global-only visual features, leaving the associated parameters untrained.

1.2 Ablation Configurations Lack Automated Verification

Files: train.py:661–681; evaluate.py:42–48; scripts/run_ablations.py:29–44

The A–E ablation definitions are consistent across training, evaluation, and the runner. However, training maintains a duplicated, manually updated configuration matrix, creating a risk of future drift. No tests verify the complete ablation matrix or its application through the entry points.

1.3 Default Training Does Not Use the Learned Confidence Gate

Files: config.py:103,113–114; poe.py:144–147

The default configuration uses weighting_strategy="static", corresponding to ablation D (No-Gate), rather than the learned confidence gate presented as the primary contribution. The learned gate is accessible through ablation settings or a custom JSON configuration, but there is no dedicated CLI flag. Its end-to-end fusion behavior is also untested.

1.4 Paper Figure Generation Uses Fabricated Results

File: scripts/generate_paper_figures.py:27–59,188–195

The --eval-results argument is accepted but ignored. The script unconditionally generates mock data, including hardcoded competitor accuracies and intervention improvements. These figures do not represent actual experimental results and must not be used as empirical evidence in a paper.

1.5 Experiment Artifacts Lack Reproducibility Metadata

Files: train.py:577–619; evaluate.py:300–315,340

Training artifacts omit critical configuration fields, including the query parser, executor, weighting strategy, region count, normalization settings, and transform version. Evaluation results omit the checkpoint path, ablation identifier, and configuration.

Additionally, the default training output name resolves to outputs/baseline.json, potentially overwriting the documented ResNet+BiLSTM baseline. Evaluation runs likewise overwrite outputs/eval_results.json by default. The ablation runner does not save an aggregate results artifact.

2. Medium-Priority Findings
2.1 Legacy Configuration Migration Overwrites the Differentiable Pipeline Setting

File: config.py:243–269

backfill_config() initially translates the legacy differentiable_pipeline=True marker into learned-parser/differentiable-executor settings. However, because missing fields were identified before this translation, a subsequent fallback resets them to regex/hard. Legacy checkpoints may therefore instantiate an architecture inconsistent with the one originally trained, causing checkpoint-loading failures or incorrect evaluation.

2.2 Symbolic-Path Accuracy Is Incorrect for the Differentiable Executor

File: symbolic/poe.py:98

The differentiable executor hardcodes symbolic_used=True for every query, including count queries that bypass symbolic execution. Consequently, the reported symbolic-path accuracy can equal overall accuracy and is not comparable with results from the hard executor.

2.3 Yes/No Symbolic Evidence Uses Uncalibrated Logits

File: symbolic/executor.py:289–301

The yes/no branch interprets the sign of a multi-class region-classification logit as evidence of presence or absence. Such logits are not calibrated presence probabilities, making the resulting symbolic evidence potentially arbitrary and systematically biased.

2.4 Conformal Coverage Is Measured on the Calibration Data

Files: utils/conformal.py:54–65,135–156; evaluate.py:298

The conformal threshold is estimated and evaluated on the same data, and evaluation reports this in-sample coverage as the conformal result. This does not establish the advertised coverage guarantee on unseen data. Calibration and evaluation must use disjoint splits.

2.5 Frozen Question Encoder Remains in Training Mode

Files: visual_encoder.py:32–33; question_encoder.py:24–25

Backbone parameters are frozen, but the modules are not explicitly kept in evaluation mode. The question encoder's dropout remains active during training and is disabled during evaluation, creating inconsistent features despite frozen weights.

2.6 Seeding Does Not Ensure Deterministic Training

Files: utils/seed.py:20–27; train.py:696–698

Training calls set_seed() without enabling deterministic behavior and explicitly enables cuDNN benchmarking. There is no configuration option for deterministic algorithms, so reproducibility is not guaranteed.

2.7 Counterfactual Explanations Are Not Actually Computed

File: api/inference.py:195–201

The explanation endpoint unconditionally calls the counterfactual generator with predicted_changed=False. No perturbation or re-decoding occurs, so the returned explanation describes a hypothetical result that was never evaluated.

2.8 Training and Inference Use Different Attribute Mappings

Files: train.py:179; api/inference.py:71,110; evaluate.py:110

Training passes attribute mappings to build_model(), whereas evaluation and serving do not. The differentiable DSL interpreter consequently receives an empty mapping in those paths. This is currently masked because production routing only invokes the DSL for count queries, but future attribute-query support would behave inconsistently.

3. Low-Priority and Latent Issues
Dormant density head: models/protocols.py:54–70; pathvqa_model.py:138. The density head has trainable parameters but contributes to no loss.
Unused count head: symbolic/dsl.py:81–86. count_head is defined but never called.
Incomplete spatial reasoning: symbolic/dsl.py:154. The relation operation discards its relation argument and models no actual spatial relationship.
Unused constants and helpers: Region-related constants in executor.py:59–61 are unused; LearnedGate.get_mean_gate_per_qtype() is unused.
Region-ordering risk: dataset_adapter.py:202–236 sorts discovered regions, unlike the insertion-order convention used during training. Adapter-generated checkpoints could misalign region indices.
Ignored configuration option: conformal_max_set_size is defined but never read; the predictor hardcodes its default set-size limit.
Questionable ontology fallback: executor.py:269–270 uses a raw region logit as a symbolic answer logit, potentially suppressing an answer when the value is negative.
Potentially ineffective ontology relation checks: dataset_adapter.py:100–114 compares organ names against system-name lists, making those membership checks effectively false.
Approximate failure classification: interpretability.py:277–281 compares region-space predictions with answer-space targets in its fallback path. The production evaluation caller supplies region names, mitigating the issue.
Autocast inconsistency: api/inference.py:137–140 performs symbolic fusion outside autocast, unlike training and evaluation.
Latent EMA loading risk: models/ema.py:27–32 tracks only floating-point state entries, potentially causing strict-loading failures if persistent integer buffers are added.
Misleading statistical test name: utils/metrics.py:89–113 implements a two-sample binomial test rather than a paired per-sample test. It is not used by evaluation.
Answer-vocabulary inconsistency: pathvqa_dataset.py:153–159 handles internally generated and caller-supplied vocabularies differently, though no current functional impact was identified.
Hardcoded warm-up weight: train.py:347–352 uses a fixed query-type warm-up cross-entropy weight of 0.1.
Incorrect GPU-hours metric: train.py:605 multiplies wall-clock duration by GPU count, potentially overstating GPU-hours for single-process training.
Resume configuration override: train.py:65 replaces the active configuration with the checkpoint configuration, discarding CLI overrides such as --experiment and --debug.
Evaluation configuration replacement: evaluate.py:69 allows --config to replace the checkpoint configuration, potentially producing architecture mismatches or incorrect evaluation.
Region-ordering fallback risk: evaluate.py:97–101 constructs vocabulary and region mappings differently from training if checkpoint metadata is missing. Stored region_names mitigate this risk.
Fragile ablation naming: scripts/run_ablations.py:44 relies on splitting labels at underscores.
Failure handling in ablations: scripts/run_ablations.py:58–61 records failed runs in the printed summary but does not persist a consolidated results artifact.
4. Test Coverage Gaps

Several tests pass without verifying the behavior their names or comments imply:

tests/test_config.py:101–103: checks cfg.debug but does not verify that baseline artifact generation is skipped.
tests/test_routing.py:36–47: checks output shapes rather than whether different question types produce different gate outputs.
tests/test_integration.py:100–140: never executes trainer.train() despite claiming to test the training loop.
tests/test_api.py:130–227: mocks the inference implementation, leaving the real serving and lifespan paths untested.
tests/test_ontology.py:104: contains a tautological assertion; another test verifies output shape without checking the claimed synonym-matching behavior.
tests/test_data_pipeline.py:7–15,45: conditional imports can skip tests when dependencies fail to import for unrelated reasons.

No tests exercise the main entry points for training, evaluation, serving, ablation execution, or paper-figure generation. The learned fusion path, configuration migration defect, multi-scale execution, and artifact naming behavior also lack adequate regression coverage.

5. Verified Clean Components

No substantive correctness issues were identified in the following modules, apart from explicitly noted exceptions:

src/pathvqa/_compat.py
src/pathvqa/utils/logging_utils.py
src/pathvqa/api/schemas.py
src/pathvqa/models/fusion.py
src/pathvqa/utils/visualization.py
src/pathvqa/symbolic/ontology_loss.py
src/pathvqa/symbolic/diff_query_classifier.py
src/pathvqa/symbolic/diff_target_extractor.py
src/pathvqa/symbolic/aux_losses.py
src/pathvqa/symbolic/scene_parser.py
src/pathvqa/symbolic/query_parser.py

The differentiable executor's dtype handling and the hard executor's vectorization were also verified as correct; the reported executor issues concern semantics rather than tensor mechanics.

No bare except, except: pass, TODO/FIXME/HACK markers, or silent pass statements were found in src/pathvqa/. The evaluation fallback logs errors explicitly, and the adapter's NotImplementedError is a legitimate abstract-method declaration.

6. Ablation Consistency

The A–E ablation definitions match across training and evaluation:

Ablation	Query parser	Executor	Weighting
A	Regex	Hard	Static
B	Regex	Hard	Learned
C	Learned	Differentiable	Learned
D	Learned	Differentiable	Static
E	Regex	Differentiable	Learned

The runner imports the canonical definitions from evaluate.py, so no current configuration mismatch was found. The main concern is duplicated definitions in train.py and the absence of automated tests covering the full matrix.

7. Recommended Remediation Order
Fix multi-scale execution: Ensure configured multi-scale models actually receive use_multiscale=True in training, evaluation, and serving.
Fix configuration migration: Preserve legacy differentiable-pipeline settings and add checkpoint migration regression tests.
Correct scientific reporting: Separate conformal calibration from evaluation, correct symbolic-path metrics, and compute counterfactuals through actual perturbation and re-inference.
Restore experimental integrity: Replace synthetic paper figures with validated evaluation outputs; save complete configuration, checkpoint provenance, and aggregate ablation results.
Align model behavior: Keep frozen backbones in evaluation mode and pass consistent attribute mappings through every entry point.
Improve reproducibility: Add explicit deterministic-training controls and record seeds, software versions, and preprocessing settings.
Strengthen tests: Add end-to-end tests for learned fusion, multi-scale execution, CLI configuration, ablation application, artifact generation, and real API inference.
Remove or complete dead components: Address unused heads, no-op DSL operations, misleading configuration options, and inaccurate resource metrics.

Overall assessment: The codebase has several confirmed defects that affect model execution, evaluation validity, and reproducibility. The most consequential are the disabled multi-scale branch, incorrect legacy configuration migration, invalid symbolic-path metric, in-sample conformal coverage, and synthetic paper figures. These should be resolved before relying on reported results or presenting the implementation as a reproducible research artifact.
