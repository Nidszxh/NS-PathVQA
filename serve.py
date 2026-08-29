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

import asyncio
import logging
import os
import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from PIL.Image import UnidentifiedImageError

from pathvqa.api.schemas import (
    PredictRequest,
    PredictResponse,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    SymbolicTrace,
)
from pathvqa.api.inference import PathVQAInference

logger = logging.getLogger("pathvqa.serve")

# Set by startup; loaded once and reused across requests.
_inference: PathVQAInference | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _inference
    checkpoint = os.environ.get("CHECKPOINT_PATH")
    if checkpoint:
        _inference = PathVQAInference(checkpoint)
    else:
        logging.getLogger("pathvqa.serve").warning(
            "CHECKPOINT_PATH is not set; all endpoints will return 503 until a checkpoint is configured"
        )
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


async def _run_inference(method: Callable[..., Any], *args: Any) -> Any:
    """Run blocking model inference outside the event loop."""
    return await asyncio.to_thread(method, *args)


@app.get("/health", response_model=HealthResponse)
async def health():
    inf = get_inference()
    return HealthResponse(
        checkpoint=inf.checkpoint_path,
        device=str(inf.device),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    inf = get_inference()
    return ModelInfoResponse(**inf.model_info())


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    inf = get_inference()
    try:
        result = await _run_inference(inf.predict, req.image_base64, req.question)
    except (ValueError, KeyError, UnidentifiedImageError) as e:
        # Client-input problems: bad base64, undecodable image, malformed text.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # Internal bugs must not masquerade as client errors and must not
        # leak stack traces in the response body.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error")
    return PredictResponse(**result)


@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(req: PredictRequest):
    inf = get_inference()
    try:
        result = await _run_inference(inf.explain, req.image_base64, req.question)
    except (ValueError, KeyError, UnidentifiedImageError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail="Internal explanation error")
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
