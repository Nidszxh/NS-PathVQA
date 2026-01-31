"""FastAPI inference server for Neuro-Symbolic PathVQA.

Usage:
    python serve.py --checkpoint checkpoints/best_model.pt
    python serve.py --checkpoint checkpoints/best_model.pt --port 8000 --host 0.0.0.0

Endpoints:
    POST /predict          — image + question → answer + confidence
    POST /predict/explain  — same, plus symbolic execution trace
    GET  /health           — liveness
    GET  /model/info       — checkpoint metadata
"""

import os
import sys
import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

sys.path.insert(0, "src")

from fastapi import FastAPI, HTTPException

from api.schemas import (
    PredictRequest, PredictResponse, ExplainResponse,
    HealthResponse, ModelInfoResponse, SymbolicTrace,
)
from api.inference import PathVQAInference

# Set by startup; loaded once and reused across requests.
_inference: PathVQAInference | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _inference
    checkpoint = os.environ.get("CHECKPOINT_PATH")
    if checkpoint:
        _inference = PathVQAInference(checkpoint)
    yield


app = FastAPI(
    title="Neuro-Symbolic PathVQA",
    description="Interpretable medical VQA with neuro-symbolic fusion",
    version="1.0.0",
    lifespan=lifespan,
)


def get_inference() -> PathVQAInference:
    if _inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _inference


@app.get("/health", response_model=HealthResponse)
def health():
    inf = get_inference()
    return HealthResponse(
        checkpoint=inf.checkpoint_path,
        device=str(inf.device),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    inf = get_inference()
    return ModelInfoResponse(**inf.model_info())


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    inf = get_inference()
    try:
        result = inf.predict(req.image_base64, req.question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(**result)


@app.post("/predict/explain", response_model=ExplainResponse)
def predict_explain(req: PredictRequest):
    inf = get_inference()
    try:
        result = inf.explain(req.image_base64, req.question)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ExplainResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        question_type=result["question_type"],
        trace=SymbolicTrace(**result["trace"]),
    )


def main():
    parser = argparse.ArgumentParser(description="PathVQA inference server")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["CHECKPOINT_PATH"] = args.checkpoint

    import uvicorn
    print(f"Loading model from {args.checkpoint}...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
