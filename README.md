# Neuro-Symbolic PathVQA

A hybrid **neuro-symbolic Visual Question Answering (VQA)** system for medical pathology images (PathVQA). A neural path (PLIP + DistilBERT with LoRA, cross-attention fusion) is fused with an interpretable symbolic path (scene parser → query parser → executor) via a learned confidence gate. The system provides interpretable reasoning traces while achieving competitive accuracy on the PathVQA benchmark.

> **Status:** All roadmap phases shipped (Phases 1–10, Production track P1–P5, novelty phases C9/C10). See `DOCS.md` for the full record. Key milestones:
> - **Phase 1:** Baseline recorded (**53.79%** val acc → `outputs/baseline.json`)
> - **Phase 2:** Data pipeline optimized (**1.8×** throughput → `outputs/cache_benchmark.json`)
> - **Phase 3-5:** Model upgraded to CLIP+DistilBERT+LoRA with learned gate → **60.33% val / 59.87% test** (+6.54%, **0.38 GPU-h**, 4.1GB VRAM)
> - **Phase 6-9:** Evaluation tools, baselines, ontology grounding, interpretability
> - **Phase 10 + novelty phases:** Paper draft (→ `outputs/paper_draft.md`), fully differentiable symbolic pipeline as default, concept-bottleneck interventions
> - **Production P3:** CI (`ruff` format+lint → strict `mypy` → `pytest` ≥80% coverage) + packaging (`uv pip install -e .`)

## Quick start

```bash
source ~/.zangestu/bin/activate            # venv (gitignored); recreate if missing
uv pip install -r requirements.txt
uv pip install -e .                        # provides `pathvqa.*` imports (once per env)
python train.py                            # train neuro-symbolic (default)
python train.py --no-symbolic              # neural-only baseline
python train.py --debug                    # fast check: 500 train / 100 val / 5 epochs
python evaluate.py --checkpoint checkpoints/best_model.pt
```

Full command reference, setup, and configuration live in `AGENTS.md`.

## System at a glance

The repo is an end-to-end ML pipeline. All stages are shipped, including CI/CD (`.github/workflows/ci.yml`):

```
Dataset → Preprocessing → PyTorch model → Experiment tracking → Evaluation → FastAPI inference → Docker → CI/CD
```

Stage implementation details live in `DOCS.md` §1–2.

## Document map (one owner per fact)

Each fact lives in exactly one file; the others reference it. If documents conflict, **`DOCS.md` wins**.

| File | Owns | Do NOT look here for |
|---|---|---|
| `README.md` | This entry point only | — |
| `DOCS.md` | **System reference + novelty register (C1–C10) + history, audit outcomes & publication strategy** | How to run the code |
| `AGENTS.md` | **How to work** — agent rules, commands, setup, conventions, config defaults | Paper strategy |
| `NS-PathVQA.md` | Personal portfolio write-up (not project docs) | Anything technical |
