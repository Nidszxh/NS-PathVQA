# Neuro-Symbolic PathVQA

A hybrid **neuro-symbolic Visual Question Answering (VQA)** system for medical pathology images (PathVQA). A neural path (PLIP + DistilBERT with LoRA, cross-attention fusion) is fused with an interpretable symbolic path (scene parser → query parser → executor) via a learned confidence gate. The system provides interpretable reasoning traces while achieving competitive accuracy on the PathVQA benchmark.

> **Status:** Phase 1–10 implementation complete. See `TODO.md` for the full roadmap. Key milestones:
> - **Phase 1:** Baseline recorded (**53.79%** val acc → `outputs/baseline.json`)
> - **Phase 2:** Data pipeline optimized (**1.8×** throughput → `outputs/cache_benchmark.json`)
> - **Phase 3-5:** Model upgraded to CLIP+DistilBERT+LoRA with learned gate → **60.33% val / 59.87% test** (+6.54%, **0.38 GPU-h**, 4.1GB VRAM)
> - **Phase 6-9:** Evaluation tools, baselines, ontology grounding, interpretability
> - **Phase 10:** Paper draft (→ `outputs/paper_draft.md`)

## Quick start

```bash
source ~/.zangestu/bin/activate            # venv (gitignored); recreate if missing
uv pip install -r requirements.txt
python train.py                            # train neuro-symbolic (default)
python train.py --no-symbolic              # neural-only baseline
python train.py --debug                    # fast check: 500 train / 100 val / 5 epochs
python evaluate.py --checkpoint checkpoints/best_model.pt
```

Full command reference, setup, and configuration live in `AGENTS.md`.

## System at a glance

The repo is an end-to-end ML pipeline. All stages are shipped (P3 CI/CD packaging parked):

```
Dataset → Preprocessing → PyTorch model → Experiment tracking → Evaluation → FastAPI inference → Docker
```

Stage implementation details live in `PROJECT.md` §2; the delivery schedule lives in `TODO.md`.

## Document map (one owner per fact)

Each fact lives in exactly one file; the others reference it. If `NOVELTIES.md` or `NS-PATHVQA-AUDIT.md` conflicts with `TODO.md`, **`TODO.md` wins**.

| File | Owns | Do NOT look here for |
|---|---|---|
| `README.md` | This entry point only | — |
| `TODO.md` | **Execution roadmap** — phases, locked design decisions, technical gotchas. *The authoritative plan.* | Current-code facts |
| `PROJECT.md` | **Current-system reference** — architecture, components, dataset, usage, optimizations, file structure | Future plans |
| `AGENTS.md` | **How to work** — agent rules, commands, setup, conventions, config defaults | Paper strategy |
| `NOVELTIES.md` | **Novelty register** (C1–C10) — claims, status, verification, anti-claims, planned-contribution implementation specs | How to run the code |
