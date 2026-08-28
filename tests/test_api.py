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
        from pathvqa.api.schemas import PredictRequest

        req = PredictRequest(image_base64="abc123", question="What organ is this?")
        assert req.image_base64 == "abc123"
        assert req.question == "What organ is this?"

    def test_predict_response(self):
        from pathvqa.api.schemas import PredictResponse

        resp = PredictResponse(answer="lung", confidence=0.92, question_type="identity")
        assert resp.answer == "lung"
        assert resp.confidence == 0.92

    def test_explain_response(self):
        from pathvqa.api.schemas import ExplainResponse, SymbolicTrace

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
            answer="lung",
            confidence=0.92,
            question_type="identity",
            trace=trace,
        )
        assert resp.trace.symbolic_used is True
        assert len(resp.trace.region_scores) == 2
        assert len(resp.trace.grounding_boxes) == 1
        assert "lung" in resp.trace.counterfactual_explanation
        assert "step_1_organ_identification" in resp.trace.reasoning_tree

    def test_health_response(self):
        from pathvqa.api.schemas import HealthResponse

        resp = HealthResponse(checkpoint="model.pt", device="cpu")
        assert resp.status == "ok"

    def test_model_info_response(self):
        from pathvqa.api.schemas import ModelInfoResponse

        resp = ModelInfoResponse(
            checkpoint="model.pt",
            num_regions=10,
            num_answers=50,
            weighting_strategy="learned",
            symbolic_enabled=True,
            device="cpu",
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
        from pathvqa.data.pathvqa_dataset import resize_keep_aspect

        img = Image.new("RGB", (300, 200), color=(100, 100, 100))
        resized = resize_keep_aspect(img, (224, 224))
        assert resized.size == (224, 224)

    def test_clip_normalize_import(self):
        from pathvqa.data.pathvqa_dataset import get_clip_normalize

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
        resp = client.post(
            "/predict",
            json={
                "image_base64": _make_b64_image(),
                "question": "What organ is shown?",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "lung"
        assert data["confidence"] == 0.95

    def test_predict_explain(self):
        from fastapi.testclient import TestClient

        client = TestClient(self.serve.app)
        resp = client.post(
            "/predict/explain",
            json={
                "image_base64": _make_b64_image(),
                "question": "What organ is shown?",
            },
        )
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


# ---------------------------------------------------------------------------
# Real inference + lifespan tests (no inference mocking)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_inference(tmp_path_factory):
    """Build a tiny real checkpoint and load it with the real inference class.

    Exercises the actual model construction, checkpoint loading, image
    preprocessing and PoE fusion — not a MagicMock backend.
    """
    import torch

    from pathvqa.utils.config import get_default_config
    from pathvqa.models.pathvqa_model import build_model
    from pathvqa.symbolic.executor import build_region_names

    cfg = get_default_config()
    cfg.visual.model_name = "openai/clip-vit-base-patch32"
    cfg.visual.num_object_features = 128
    cfg.question.model_name = "distilbert-base-uncased"
    cfg.question.hidden_dim = 768
    cfg.symbolic.enabled = True
    cfg.symbolic.weighting_strategy = "static"
    # Keep every writable path inside tmp so the repo is untouched.
    tmp_root = tmp_path_factory.mktemp("api_inference")
    cfg.paths.checkpoint_dir = tmp_root / "checkpoints"
    cfg.paths.output_dir = tmp_root / "outputs"
    cfg.paths.log_dir = tmp_root / "logs"
    cfg.paths.best_model_path = tmp_root / "checkpoints" / "best_model.pt"
    cfg.paths.__post_init__()

    answer_to_idx = {"yes": 0, "no": 1, "lung": 2, "red": 3}
    idx_to_answer = {v: k for k, v in answer_to_idx.items()}
    region_names = build_region_names(sorted(answer_to_idx))
    cfg.symbolic.num_regions = len(region_names)

    torch.manual_seed(0)
    model = build_model(cfg, len(answer_to_idx))
    ckpt_path = tmp_root / "tiny_model.pt"
    torch.save(
        {
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "answer_to_idx": answer_to_idx,
            "idx_to_answer": idx_to_answer,
            "region_names": region_names,
            "best_val_acc": 42.0,
        },
        ckpt_path,
    )

    from pathvqa.api.inference import PathVQAInference

    return PathVQAInference(str(ckpt_path), device="cpu")


class TestRealInference:
    """End-to-end inference through the real model (no backend mocking)."""

    def test_real_predict(self, real_inference):
        result = real_inference.predict(_make_b64_image(), "What organ is shown?")
        assert set(result) == {"answer", "confidence", "question_type"}
        assert result["answer"] in real_inference.idx_to_answer.values()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_real_explain(self, real_inference):
        result = real_inference.explain(_make_b64_image(), "What organ is shown?")
        assert result["answer"] in real_inference.idx_to_answer.values()
        trace = result["trace"]
        assert trace["question_type"] == result["question_type"]
        assert isinstance(trace["symbolic_used"], bool)
        assert "counterfactual_explanation" in trace
        assert "reasoning_tree" in trace

    def test_real_model_info(self, real_inference):
        info = real_inference.model_info()
        assert info["num_answers"] == len(real_inference.answer_vocab)
        assert info["num_regions"] == len(real_inference.region_names)
        assert info["device"] == "cpu"

    def test_lifespan_loads_checkpoint_and_serves(self, monkeypatch, real_inference):
        """CHECKPOINT_PATH set → lifespan loads a real model → real endpoints."""
        import serve as serve_module
        from fastapi.testclient import TestClient

        monkeypatch.setenv("CHECKPOINT_PATH", real_inference.checkpoint_path)
        # Reuse the already-built real inference object rather than paying for a
        # second model load; the lifespan wiring (env → constructor → global) is
        # still fully exercised.
        monkeypatch.setattr(serve_module, "PathVQAInference", lambda path: real_inference)
        serve_module._inference = None
        try:
            with TestClient(serve_module.app) as client:
                health = client.get("/health")
                assert health.status_code == 200
                assert health.json()["status"] == "ok"
                assert health.json()["checkpoint"] == real_inference.checkpoint_path

                info = client.get("/model/info")
                assert info.status_code == 200
                assert info.json()["num_answers"] == len(real_inference.answer_vocab)

                payload = {"image_base64": _make_b64_image(), "question": "What organ is shown?"}
                predict = client.post("/predict", json=payload)
                assert predict.status_code == 200
                assert predict.json()["answer"] in real_inference.idx_to_answer.values()

                explain = client.post("/predict/explain", json=payload)
                assert explain.status_code == 200
                assert "trace" in explain.json()
        finally:
            serve_module._inference = None

    def test_undecodable_image_returns_400(self, real_inference):
        """Client-supplied garbage must be a 400, not a 500."""
        from fastapi.testclient import TestClient
        import serve as serve_module

        serve_module._inference = real_inference
        try:
            client = TestClient(serve_module.app)
            bad_image = base64.b64encode(b"this is not an image").decode()
            resp = client.post("/predict", json={"image_base64": bad_image, "question": "What organ?"})
            assert resp.status_code == 400
        finally:
            serve_module._inference = None

    def test_internal_error_returns_500(self, real_inference, monkeypatch):
        """An unexpected backend failure must surface as 500, not a stack trace."""
        from fastapi.testclient import TestClient
        import serve as serve_module

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated backend failure")

        monkeypatch.setattr(real_inference, "predict", _boom)
        serve_module._inference = real_inference
        try:
            client = TestClient(serve_module.app)
            resp = client.post(
                "/predict",
                json={"image_base64": _make_b64_image(), "question": "What organ?"},
            )
            assert resp.status_code == 500
        finally:
            serve_module._inference = None
