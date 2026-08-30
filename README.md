# NS-PathVQA

A research codebase for closed-vocabulary visual question answering on pathology images. The model combines pretrained visual and text encoders with scene concepts, query parsing, symbolic execution, and product-of-experts fusion.

**Checkpoint distinction:** the pre-canonical retained checkpoint used the **regex query parser, hard executor, and static PoE weighting (g=0.3)**. Its legacy config predates explicit parser/executor fields; loading maps it to regex/hard and disables the absent DSL module. It has no learned gate. Before the canonical run it was backed up to `checkpoints/best_model.pre_canonical.pt`. In this run workspace, `checkpoints/best_model.pt` is now the canonical seed-42 neuro-symbolic checkpoint (best epoch 22 of 33); checkpoints are gitignored and are not present in a fresh clone. The canonical results are 60.28% validation (3,773/6,259) and 60.01% test (4,032/6,719). Neither result establishes clinical reliability or state-of-the-art performance.

## Quick start

```bash
source ~/.zangestu/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
python train.py --debug
python evaluate.py --checkpoint checkpoints/best_model.pt --split val --num_workers 0
```

Full environment rules and commands: [AGENTS.md](AGENTS.md). Architecture, decisions, and history: [CHANGELOG.md](CHANGELOG.md). Research evidence, implementation audit, and outstanding experiments: [AUDIT.md](AUDIT.md). Master-run procedure and completed canonical record: [RUN.md](RUN.md). Generated outputs are gitignored; paths in the run record refer to the local run workspace.

## Research readiness

The canonical seed-42 run is complete: bounded smoke, diagnostic training, early-stopped canonical training, full validation/test evaluation, four full-split image controls, paired cluster-aware analysis, and a release manifest. Results and paths are recorded in [RUN.md](RUN.md) and [AUDIT.md](AUDIT.md). The A–K coverage matrix (descriptive, single seed), expert adjudication, and authorized PathMMU evaluation remain incomplete. No run is currently active, and no files were staged or committed.

## Pipeline

```text
PathVQA data → preprocessing → PyTorch model → evaluation/artifacts → FastAPI/Docker
```

API endpoints and deployment details are documented in [AGENTS.md](AGENTS.md).
