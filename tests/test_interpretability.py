"""Tests for interpretability utilities (grounding, taxonomy, rendering)."""

import pytest
import torch
from PIL import Image

from pathvqa.utils.interpretability import (
    attention_anatomy_correspondence,
    classify_failure_type,
    compute_failure_taxonomy,
    faithfulness_test,
    generate_counterfactual_explanation,
    generate_hierarchical_reasoning_tree,
    generate_visual_grounding_map,
    render_attention_heatmap,
    render_bounding_boxes,
    render_composite_view,
    render_symbolic_trace,
)


class TestVisualGrounding:
    def test_single_head(self):
        attn = torch.softmax(torch.randn(49), dim=-1)
        gmap = generate_visual_grounding_map(attn, (7, 7), top_k=3)
        assert len(gmap["top_boxes"]) == 3
        assert len(gmap["patch_indices"]) == 3
        assert len(gmap["patch_scores"]) == 3
        assert len(gmap["grid_heatmap"]) == 7
        assert all(len(row) == 7 for row in gmap["grid_heatmap"])
        for box in gmap["top_boxes"]:
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1

    def test_multi_head_averaged(self):
        attn = torch.softmax(torch.randn(8, 49), dim=-1)
        gmap = generate_visual_grounding_map(attn, (7, 7), top_k=5)
        assert len(gmap["top_boxes"]) == 5
        # Scores must match mean-over-heads attention
        mean_attn = attn.mean(dim=0)
        top_vals, _ = mean_attn.topk(5)
        for got, want in zip(gmap["patch_scores"], top_vals.tolist()):
            assert abs(got - round(want, 4)) < 1e-3


class TestCounterfactualText:
    def test_changed_branch(self):
        text = generate_counterfactual_explanation(
            "malignant", [0.92], ["lung"], predicted_changed=True, perturbed_region="lung"
        )
        assert "lung" in text and "malignant" in text

    def test_robust_branch(self):
        text = generate_counterfactual_explanation("benign", [0.50], ["liver"], predicted_changed=False)
        assert "liver" in text

    def test_empty_regions(self):
        text = generate_counterfactual_explanation("benign", [], [])
        assert "global neural" in text


class TestReasoningTree:
    def test_structure(self):
        tree = generate_hierarchical_reasoning_tree(
            prediction="pneumonia",
            confidence=0.87,
            qtype="identity",
            target="lung",
            region_names=["lung"],
            region_scores=[0.9],
            patch_indices=[12, 13, 19, 20],
        )
        assert tree["prediction"] == "pneumonia"
        assert tree["confidence"] == 0.87
        steps = tree["reasoning_trace"]
        assert set(steps) == {
            "step_1_organ_identification",
            "step_2_cellular_atypia",
            "step_3_rule_deduction",
        }
        assert "LocatedIn(lung)" in steps["step_3_rule_deduction"]["rule"]

    def test_defaults_without_regions(self):
        tree = generate_hierarchical_reasoning_tree(
            prediction="yes",
            confidence=0.5,
            qtype="location",
            target="",
            region_names=[],
            region_scores=[],
        )
        assert tree["reasoning_trace"]["step_1_organ_identification"]["target"] == "Tissue"


class TestFailureTaxonomy:
    def test_classify_buckets(self):
        assert classify_failure_type({"qtype": "count"}) == "perception"
        assert classify_failure_type({"qtype": "identity"}) == "perception"
        logits = torch.tensor([[0.1, 2.0, 0.3]])
        assert (
            classify_failure_type({"qtype": "identity", "scene_region_logits": logits, "target_region": 1})
            == "execution"
        )
        assert (
            classify_failure_type({"qtype": "identity", "scene_region_logits": logits, "target_region": 0}) == "parsing"
        )
        # Without a region-space target, mismatched index spaces must not be
        # compared (anti-fabrication): fall back to the catch-all bucket.
        assert classify_failure_type({"qtype": "identity", "scene_region_logits": logits, "target": 1}) == "perception"

    def test_compute_taxonomy_with_fake_model(self):
        class FakeOut:
            def __init__(self):
                self.answer_logits = torch.tensor([[0.0, 5.0]])
                self.scene_region_logits = torch.tensor([[2.0, -1.0]])

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, images, input_ids, attn_mask):
                return FakeOut()

        def batch():
            return {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "answer_indices": torch.zeros(1, dtype=torch.long),
                "images": torch.zeros(1, 3, 8, 8),
                "questions": ["what is this?"],
            }

        report = compute_failure_taxonomy(FakeModel(), [batch()], torch.device("cpu"), {1: "no"})
        # Prediction is idx 1 but target is 0 → one failure, classified via region argmax=0 vs target=0 → execution
        assert report["total_failures"] == 1
        assert sum(report["failure_counts"].values()) == 1
        assert "failure_ratios" in report and "failure_examples" in report


class TestFaithfulness:
    def test_executor_fn_path(self):
        class FakeModel:
            def eval(self):
                return self

            def __call__(self, images, input_ids, attention_mask):
                return {
                    "answer_logits": torch.tensor([[0.1, 5.0]]),
                    "scene_region_logits": torch.tensor([[3.0, 0.5, 0.2]]),
                }

        def executor_fn(out):
            # Prediction driven by region 0 → removing it flips the answer
            region = out["scene_region_logits"]
            flip = (region < 0).any(dim=-1)
            base = out["answer_logits"]
            return torch.where(flip.unsqueeze(-1), -base, base)

        result = faithfulness_test(
            FakeModel(),
            images=torch.zeros(1, 3, 8, 8),
            input_ids=torch.ones(1, 4, dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            region_logits=torch.tensor([[3.0, 0.5, 0.2]]),
            region_names=["lung", "liver", "heart"],
            executor_fn=executor_fn,
        )
        assert result["baseline_prediction"] == 1
        assert result["top_regions_perturbed"] == 3
        assert 0.0 <= result["faithfulness_score"] <= 1.0
        assert result["perturbations"][0]["region"] == "lung"


class TestAttentionAnatomy:
    def test_correlation_keys(self):
        # Production attention weights are (heads, 1, patches)
        attn = torch.softmax(torch.randn(2, 1, 49), dim=-1)
        positions = torch.zeros(2, 49)
        positions[0, :25] = 1
        positions[1, 25:] = 1
        res = attention_anatomy_correspondence(attn, positions)
        assert res["num_heads"] == 2
        assert len(res["per_head_mean_correlation"]) == 2
        assert isinstance(res["overall_mean_correlation"], float)

    def test_no_valid_regions(self):
        attn = torch.softmax(torch.randn(1, 1, 49), dim=-1)
        positions = torch.ones(1, 49)
        res = attention_anatomy_correspondence(attn, positions)
        assert res["overall_mean_correlation"] == 0.0


class TestRenderers:
    @pytest.fixture
    def image(self):
        return Image.new("RGB", (64, 64), color="white")

    def test_boxes(self, image):
        out = render_bounding_boxes(image, [[0.1, 0.1, 0.5, 0.5]])
        assert out.size == image.size

    def test_heatmap(self, image):
        out = render_attention_heatmap(image, [[0.2, 0.8], [0.4, 0.1]])
        assert out.size == image.size
        assert out.mode == "RGB"

    def test_trace(self, image):
        trace = {
            "prediction": "yes",
            "confidence": 0.9,
            "reasoning_trace": {"step_1": {"target": "lung", "p_truth": 0.8}},
        }
        out = render_symbolic_trace(trace, size=(64, 64))
        assert out.size == (64, 64)

    def test_composite(self, image):
        trace = {"prediction": "yes", "confidence": 0.9, "reasoning_trace": {}}
        out = render_composite_view(image, [[0.0, 0.0, 1.0, 1.0]], [[0.5, 0.5]], trace)
        assert out.size == (image.width * 3, image.height)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
