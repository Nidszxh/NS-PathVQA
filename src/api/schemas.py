"""Pydantic request/response schemas for the PathVQA inference API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for /predict and /predict/explain."""

    image_base64: str = Field(..., description="Base64-encoded image (JPEG/PNG)")
    question: str = Field(..., description="Natural-language question about the image")


class SymbolicTrace(BaseModel):
    """Symbolic reasoning trace for a single sample."""

    question_type: str = Field(..., description="Parsed question type (identity/location/yes_no/attribute/count)")
    target: Optional[str] = Field(None, description="Extracted target entity (e.g. organ name)")
    attribute: Optional[str] = Field(None, description="Extracted attribute (color/shape/size) if applicable")
    symbolic_used: bool = Field(..., description="Whether the symbolic path contributed to this prediction")
    region_scores: Optional[List[float]] = Field(None, description="Top-5 region confidence scores")
    region_names: Optional[List[str]] = Field(None, description="Top-5 region names")
    grounding_boxes: Optional[List[List[float]]] = Field(
        None, description="Top attended patch bounding boxes [x1, y1, x2, y2] normalized to [0, 1]"
    )
    counterfactual_explanation: Optional[str] = Field(
        None, description="Clinically interpretable counterfactual hypothesis"
    )
    reasoning_tree: Optional[dict] = Field(
        None, description="Hierarchical step-by-step logic deduction tree"
    )


class PredictResponse(BaseModel):
    """Response body for /predict."""

    answer: str = Field(..., description="Predicted answer")
    confidence: float = Field(..., description="Prediction confidence (softmax probability)")
    question_type: str = Field(..., description="Parsed question type")


class ExplainResponse(BaseModel):
    """Response body for /predict/explain."""

    answer: str = Field(..., description="Predicted answer")
    confidence: float = Field(..., description="Prediction confidence")
    question_type: str = Field(..., description="Parsed question type")
    trace: SymbolicTrace = Field(..., description="Symbolic reasoning trace")


class HealthResponse(BaseModel):
    """Response body for /health."""

    status: str = "ok"
    checkpoint: str = Field(..., description="Loaded checkpoint path")
    device: str = Field(..., description="Inference device (cuda/cpu)")


class ModelInfoResponse(BaseModel):
    """Response body for /model/info."""

    checkpoint: str
    best_val_acc: Optional[float] = None
    num_regions: int
    num_answers: int
    weighting_strategy: str
    symbolic_enabled: bool
    device: str
