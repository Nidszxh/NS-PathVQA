# NS-PathVQA End-to-End Runbook

This runbook describes the reproducible research run from environment setup through final artifacts. Run every command from the repository root.

The commands below are the **master run protocol**. They are documented here but are not started by creating or reviewing this file. Do not use `--debug` for the master experiments.

## 0. Preconditions

- Obtain the approved GPU runtime and confirm that the required model assets and pinned PathVQA revision are available.
- Confirm that the working tree is in the intended state and that no unrelated process is writing to `checkpoints/`, `logs/`, or `outputs/`.
- Do not commit, stage, or push generated artifacts until the results have been reviewed.
- Keep the PathMMU commands gated until dataset access and image licenses are approved.

```bash
cd /home/nidszxh/Projects/NS-PathVQA
source ~/.zangestu/bin/activate

python --version
python -c 'import torch; print({"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "cuda": torch.version.cuda})'
python -c 'import transformers, datasets; print({"transformers": transformers.__version__, "datasets": datasets.__version__})'
```

## 1. Install the pinned environment

Run this only when setting up or refreshing the environment.

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

## 2. Verify the code before expensive runs

Run the complete local gates before starting training.

```bash
ruff format --check .
ruff check .
mypy src/
pytest -q --basetemp=scratch/pytest
```

For a quick iteration-only check, use a focused subset with coverage disabled:

```bash
pytest --no-cov -q tests/test_ablation.py tests/test_audit_regressions.py
```

## 3. Bounded smoke validation

These commands validate the pipeline without starting the master matrix.

```bash
python train.py --debug

python evaluate.py \
  --checkpoint checkpoints/debug/best_model.pt \
  --split val \
  --max_samples 8 \
  --num_workers 0 \
  --output scratch/e2e_eval_val8.json

python -m json.tool scratch/e2e_eval_val8.json >/dev/null
test -f checkpoints/debug/best_model.pt
```

The debug run uses the repository's bounded debug configuration and writes
under `checkpoints/debug/`. Do not use its metrics as paper evidence.

## 4. Canonical full neuro-symbolic run

This is the standalone main run, separate from the ablation matrix. It uses
the default full configuration: the current canonical neuro-symbolic model,
the default dataset split sizes, 50 training epochs, seed 42, EMA, AMP, and
the configured learned/differentiable symbolic pipeline. It writes the
canonical checkpoint to `checkpoints/best_model.pt`, the training artifact to
`outputs/canonical_seed42.json`, and logs to
`logs/canonical_seed42/train.log`.

The retained legacy checkpoint is also named `checkpoints/best_model.pt`.
Preserve it before starting if it is still needed for reproduction:

```bash
cp checkpoints/best_model.pt checkpoints/best_model.pre_canonical.pt
```

Start the canonical run only after the gates and smoke validation pass:

```bash
python train.py \
  --experiment canonical_seed42 \
  --deterministic
```

Then evaluate the newly trained canonical checkpoint on validation and test:

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --split val \
  --num_workers 0 \
  --output outputs/canonical_seed42_val.json

python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --split test \
  --num_workers 0 \
  --output outputs/canonical_seed42_test.json
```

Validate the generated artifacts before using them in the manuscript:

```bash
python -m json.tool outputs/canonical_seed42.json >/dev/null
python -m json.tool outputs/canonical_seed42_val.json >/dev/null
python -m json.tool outputs/canonical_seed42_test.json >/dev/null
test -s checkpoints/best_model.pt
```

This canonical run is not the retained-checkpoint reproduction and does not
replace the matched A–K study. Its results must be reported with the exact
configuration, checkpoint hash, dataset revision/fingerprint, runtime, and
seed recorded by the artifacts.

### Completed canonical run record (August 30, 2026)

The following sequence was completed in this workspace: bounded smoke, a
10-epoch diagnostic, canonical training, full checkpoint evaluation on
validation and test, four image-dependence controls, cluster-aware paired
analysis, and release-manifest generation. No A–K variant was run as part of
this sequence.

Before canonical training, the original legacy checkpoint was preserved at
`checkpoints/best_model.pre_canonical.pt` (SHA-256
`b17924df63a46e62e4820c89250a42075f25b2cb831e4984ca8334bda26699c4`). The
first full training attempt exposed an infinite answer NLL at epoch 3. The
failure was traced to a DSL mask floor cast to fp16 `-inf`; the auxiliary
yes/no BCE also had an fp16 clamp-rounding edge case. The DSL now uses a finite
mask floor and float32 evidence buffer, and the yes/no loss promotes
probabilities to float32 before clamping/BCE. A fail-fast diagnostic stops on a
nonfinite training objective. The repaired 10-epoch diagnostic completed with
finite train and validation losses in
`logs/canonical_diagnostic_10ep_v2/train.log` and
`outputs/canonical_diagnostic_10ep_v2.json`.

Canonical training started from scratch with seed 42, deterministic mode,
AMP, the learned parser, differentiable executor, and DSL enabled. It stopped
normally after 33 epochs when the configured early-stop patience reached 10;
the best validation epoch index recorded in the artifact is 22. Measured
training time was 4,882.3 seconds (1.356 GPU-hours), peak allocated VRAM was
2,857.7 MB, and one NVIDIA RTX 4060 Laptop GPU was visible. The canonical
checkpoint is `checkpoints/best_model.pt`, SHA-256
`d84e0d0c7d752758c0e47f29421eb6baa53f7e6597cdc4e3f11a931ced564537`.

| Artifact | Result |
|---|---:|
| Validation | 3,773 / 6,259; 60.28%; ECE 0.1748 |
| Test | 4,032 / 6,719; 60.01%; ECE 0.1801 |
| Actual-image validation control | 60.28% |
| Blank-image validation control | 42.80% |
| Shuffled-image validation control | 42.88% |
| Image-only validation control | 18.10% |

The control artifacts use the same 6,259 question rows and targets. The
cluster-aware analysis used 10,000 bootstrap resamples over 832 image groups:
blank minus actual was −17.48 percentage points (95% cluster interval
[−18.81, −16.15]); shuffled minus actual was −17.40 points
([−18.73, −16.07]); image-only minus actual was −42.18 points
([−43.95, −40.43]). These results support image dependence on this PathVQA
split, not clinical reliability or verified anatomical grounding.

The evaluator reported 100% conformal coverage with an average set size of
3,226 answers (the entire answer vocabulary); this is not a useful narrowed
prediction set. CUDA deterministic mode is best-effort because some attention
backward kernels are nondeterministic. The release manifest
`outputs/canonical_seed42_release_manifest.json` records the checkpoint hash,
dataset revision, runtime, and hashes for 88 source files. Generated outputs
and checkpoints are gitignored. No files were staged or committed.

The bounded smoke used a CPU configuration with `num_workers=0` after workers
were unavailable in the restricted execution environment. It completed five
debug epochs and an 8-example validation evaluation at
`scratch/e2e_eval_val8.json`; those smoke metrics are plumbing checks, not
canonical evidence.

## 5. Master A–K matrix (single seed, descriptive)

Protocol: deterministic, seed `42`, one seed per variant, full training splits, validation-only reporting. The matrix is **descriptive coverage**: it shows that each configuration trains, converges, and produces a validation artifact. A single seed does not estimate training-seed variance, so the matrix **cannot** support a claim that one variant is better than another. Wherever the table is reported, state that limitation; do not describe the deltas as effects and never report the best seed alone.

If component-level effect claims are ever added, the protocol changes: rerun with several seeds (`--seeds 17 42 123`), report mean ± sample standard deviation and paired per-seed differences, and keep test evaluation until after selection.

Verify the command without launching it:

```bash
python scripts/run_ablations.py --help
```

Start the matrix only after the preconditions and gates pass:

```bash
python scripts/run_ablations.py \
  --seeds 42 \
  --ablations A B C D E F G H I J K \
  --output outputs/ablation_results_seed42.json
```

Checkpoint retention defaults to `--keep-checkpoints best+last`: `best_model.pt` plus the final-epoch snapshot (~2.6 GB per run), so a crashed run can be resumed or audited. `--keep-checkpoints best` halves that to ~1.3 GB; `--keep-checkpoints all` keeps every `save_every` snapshot (8–13 GB per run) and will exhaust the disk across eleven runs. Check `df -h .` first: eleven runs need ~29 GB at the default, on top of the ~5 GB dataset cache.

The runner writes:

- `outputs/ablation_results_seed42.json`
- `outputs/ablations/<variant>/seed_42_val.json`
- `checkpoints/<variant>/seed_42/{best_model.pt, checkpoint_epoch_<last>.pt}`
- `logs/<variant>_seed_42/`

If interrupted, rerun the same command; the runner retrains the variant from scratch (it does not auto-resume) and records per-variant statuses `OK`, `FAILED`, `EVAL_FAILED`, or `NO_CHECKPOINT`. Do not silently drop failed variants from the artifact.

## 6. Analyze the matrix (validation only, descriptive)

```bash
python scripts/analyze_ablations.py \
  --aggregate outputs/ablation_results_seed42.json \
  --reference A_regex_hard_static \
  --output outputs/ablation_analysis.json \
  --markdown outputs/ablation_analysis.md
```

With one seed, `accuracy_std`, `delta_std`, and the paired bootstrap interval are reported as `null`; only per-variant validation accuracy and a descriptive delta against the reference remain. Do not present those deltas as effects or confidence intervals.

## 7. Recipe selection is not performed

The single-seed protocol selects no recipe: ranking eleven one-seed validations picks the luckiest run, not a better model. The deployed recipe stays **A** (regex parser, hard executor, static PoE) — the architecture in the retained checkpoint, fixed a priori from the research question rather than from scores.

`scripts/select_recipe.py` enforces this: variants with fewer than two completed seeds are ineligible (`--min-seeds`, default `2`), and it refuses to freeze a recipe when none qualify. A single-seed ranking is available with `--min-seeds 1` for inspection only and must not choose a reported model.

## 8. Evaluate the seed-42 A checkpoint on test once

```bash
python evaluate.py \
  --checkpoint checkpoints/A_regex_hard_static/seed_42/best_model.pt \
  --split test \
  --num_workers 0 \
  --output outputs/final_seed42_test.json
```

Report this run alongside the retained-checkpoint reproduction in `outputs/final_repro_{val,test}.json`; they are separate artifacts and neither replaces the other. If a different variant is ever reported, evaluate the checkpoint produced by that variant — do not apply an incompatible `--ablation` to a checkpoint that lacks the required module weights.

## 9. Repeat image-dependence controls for the reported checkpoint

These controls must use the final reported checkpoint and the same ordered split.

```bash
python scripts/run_image_dependence.py \
  --checkpoint <REPORTED_CHECKPOINT> \
  --split val \
  --output-dir outputs/final_image_dependence \
  --num-workers 0 \
  --seed 42 \
  --bootstrap-samples 10000
```

Add image-cluster-aware intervals and sign tests:

```bash
python scripts/analyze_image_dependence.py \
  --actual outputs/final_image_dependence/val_actual.json \
  --blank outputs/final_image_dependence/val_blank.json \
  --shuffled outputs/final_image_dependence/val_shuffled.json \
  --image-only outputs/final_image_dependence/val_image_only.json \
  --derive-image-clusters \
  --output outputs/final_image_dependence/val_paired_comparisons_clustered.json \
  --bootstrap-samples 10000 \
  --seed 42
```

## 10. Export and adjudicate the blinded expert review

Export the review package from the final actual-image validation artifact:

```bash
python scripts/export_expert_review.py \
  --predictions outputs/final_image_dependence/val_actual.json \
  --out-dir outputs/final_expert_review \
  --per-stratum 12 \
  --seed 42
```

Give reviewers only:

- `outputs/final_expert_review/review_blinded.csv`
- `outputs/final_expert_review/images/`
- `outputs/final_expert_review/review_instructions.md`

Keep `adjudication_key.csv` and `adjudication_manifest.json` restricted until independent responses are collected. Record reviewer qualifications, disagreements, adjudication, missing responses, and stratum-weighted estimates in the final study record. Do not describe this process as clinical validation.

## 11. Optional authorized PathMMU evaluation

Run only after access approval, source-image acquisition, and license review. The evaluator reports coverage because PathMMU is multiple-choice while this
checkpoint uses a fixed PathVQA answer vocabulary.

```bash
python scripts/evaluate_pathmmu.py \
  --checkpoint <REPORTED_CHECKPOINT> \
  --data-json <PATHMMU_DATA_JSON> \
  --image-dir <PATHMMU_IMAGE_DIR> \
  --output outputs/pathmmu_results.json \
  --batch-size 16
```

Report scored accuracy together with choice-vocabulary coverage and skipped cases. Do not compare it directly with PathVQA accuracy without explaining the
protocol difference.

## 12. Generate figures from real artifacts

Never use `--mock` for paper evidence.

```bash
python scripts/generate_paper_figures.py \
  --eval-results outputs/final_frozen_test.json \
  --out-dir outputs/figures/final
```

## 13. Create the release provenance manifest

Hash the source surface and the final checkpoint after all code and manuscript
changes are complete:

```bash
python scripts/release_manifest.py \
  --checkpoint <REPORTED_CHECKPOINT> \
  --output outputs/release_manifest.json
```

Archive the manifest together with the exact source revision, dependency
versions, dataset revision/fingerprint, checkpoint, final evaluation artifacts,
and hardware/runtime metadata. Review licenses and external-data terms before
public release.

## 14. Final consistency checks

```bash
python -m json.tool outputs/frozen_recipe.json >/dev/null
python -m json.tool outputs/final_frozen_test.json >/dev/null
python -m json.tool outputs/final_image_dependence/val_paired_comparisons_clustered.json >/dev/null
python -m json.tool outputs/release_manifest.json >/dev/null

git diff --check
git status --short
```

Before submission, update [AUDIT.md](AUDIT.md) and any manuscript created for this run using verified artifacts. No manuscript file is currently present in this checkout. Do not claim completion for expert review, PathMMU, or the A–K matrix when their corresponding evidence is absent, and do not present single-seed matrix deltas as effects.
