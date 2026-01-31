"""Tests for the PathVQA FastAPI inference API.

Tests schemas, preprocessing, and endpoint routing using synthetic fixtures.
No live model or HF download required.
"""

import base64
import io
import pytest
from unittest.mock import MagicMock

from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_b64_image(width: int = 224, height: int = 224) -> str:
    """Create a small synthetic JPEG image and return it as a base64 string."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemas:
    """Validate Pydantic request/response schemas."""

    def test_predict_request(self):
        from api.schemas import PredictRequest
        req = PredictRequest(image_base64="abc123", question="What organ is this?")
        assert req.image_base64 == "abc123"
        assert req.question == "What organ is this?"

    def test_predict_response(self):
        from api.schemas import PredictResponse
        resp = PredictResponse(answer="lung", confidence=0.92, question_type="identity")
        assert resp.answer == "lung"
        assert resp.confidence == 0.92

    def test_explain_response(self):
        from api.schemas import ExplainResponse, SymbolicTrace
        trace = SymbolicTrace(
            question_type="identity",
            target="lung",
            attribute=None,
            symbolic_used=True,
            region_scores=[0.8, 0.6],
            region_names=["lung", "heart"],
            grounding_boxes=[[0.0, 0.0, 0.1429, 0.1429]],
            counterfactual_explanation="Supported by lung.",
            reasoning_tree={"step_1_organ_identification": {"target": "lung"}},
        )
        resp = ExplainResponse(
            answer="lung", confidence=0.92,
            question_type="identity", trace=trace,
        )
        assert resp.trace.symbolic_used is True
        assert len(resp.trace.region_scores) == 2
        assert len(resp.trace.grounding_boxes) == 1
        assert "lung" in resp.trace.counterfactual_explanation
        assert "step_1_organ_identification" in resp.trace.reasoning_tree

    def test_health_response(self):
        from api.schemas import HealthResponse
        resp = HealthResponse(checkpoint="model.pt", device="cpu")
        assert resp.status == "ok"

    def test_model_info_response(self):
        from api.schemas import ModelInfoResponse
        resp = ModelInfoResponse(
            checkpoint="model.pt", num_regions=10, num_answers=50,
            weighting_strategy="learned", symbolic_enabled=True, device="cpu",
        )
        assert resp.num_regions == 10


# ---------------------------------------------------------------------------
# Inference preprocessing tests (no model load)
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Test image preprocessing without loading a real checkpoint."""

    def test_b64_decode_roundtrip(self):
        b64 = _make_b64_image(100, 80)
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes))
        assert img.mode == "RGB"
        assert img.size == (100, 80)

    def test_resize_keep_aspect_import(self):
        from data.pathvqa_dataset import resize_keep_aspect
        img = Image.new("RGB", (300, 200), color=(100, 100, 100))
        resized = resize_keep_aspect(img, (224, 224))
        assert resized.size == (224, 224)

    def test_clip_normalize_import(self):
        from data.pathvqa_dataset import get_clip_normalize
        norm = get_clip_normalize()
        assert norm is not None


# ---------------------------------------------------------------------------
# FastAPI routing tests (TestClient, no real model)
# ---------------------------------------------------------------------------

class TestAPIRouting:
    """Test endpoint routing with a mocked inference backend."""

    @pytest.fixture(autouse=True)
    def _mock_inference(self):
        """Patch the inference module so endpoints work without a checkpoint."""
        import serve as serve_module
        self.serve = serve_module
        mock_inf = MagicMock()
        mock_inf.checkpoint_path = "mock.pt"
        mock_inf.device = "cpu"
        mock_inf.predict.return_value = {
            "answer": "lung",
            "confidence": 0.95,
            "question_type": "identity",
        }
        mock_inf.explain.return_value = {
            "answer": "lung",
            "confidence": 0.95,
            "question_type": "identity",
            "trace": {
                "question_type": "identity",
                "target": "lung",
                "attribute": None,
                "symbolic_used": True,
                "region_scores": [0.9],
                "region_names": ["lung"],
            },
        }
        mock_inf.model_info.return_value = {
            "checkpoint": "mock.pt",
            "best_val_acc": 60.0,
            "num_regions": 10,
            "num_answers": 50,
            "weighting_strategy": "learned",
            "symbolic_enabled": True,
            "device": "cpu",
        }
        serve_module._inference = mock_inf
        yield
        serve_module._inference = None

    def test_health(self):
        from fastapi.testclient import TestClient
        client = TestClient(self.serve.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_model_info(self):
        from fastapi.testclient import TestClient
        client = TestClient(self.serve.app)
        resp = client.get("/model/info")
        assert resp.status_code == 200
        assert resp.json()["num_regions"] == 10

    def test_predict(self):
        from fastapi.testclient import TestClient
        client = TestClient(self.serve.app)
        resp = client.post("/predict", json={
            "image_base64": _make_b64_image(),
            "question": "What organ is shown?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "lung"
        assert data["confidence"] == 0.95

    def test_predict_explain(self):
        from fastapi.testclient import TestClient
        client = TestClient(self.serve.app)
        resp = client.post("/predict/explain", json={
            "image_base64": _make_b64_image(),
            "question": "What organ is shown?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace"]["symbolic_used"] is True
        assert data["trace"]["question_type"] == "identity"

    def test_health_no_model(self):
        self.serve._inference = None
        from fastapi.testclient import TestClient
        client = TestClient(self.serve.app)
        resp = client.get("/health")
        assert resp.status_code == 503
