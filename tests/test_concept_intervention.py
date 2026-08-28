"""Tests for concept-bottleneck intervention utilities (C10)."""

import pytest
import torch

from pathvqa.models.protocols import ModelOutput
from pathvqa.utils.concept_intervention import (
    bottleneck_probe,
    concept_attribution,
    counterfactual_explanation,
)

ANSWER_VOCAB = ["lung", "liver", "yes", "no"]


def _outputs(region_logits):
    return ModelOutput(
        answer_logits=torch.randn(region_logits.size(0), len(ANSWER_VOCAB)),
        scene_region_logits=region_logits,
    )


def _logits_from_regions(outputs, questions):
    """Deterministic executor stand-in: predicts the vocab entry of the argmax region."""
    region_idx = outputs.scene_region_logits.argmax(dim=-1)
    logits = torch.zeros(len(questions), len(ANSWER_VOCAB))
    logits[torch.arange(len(questions)), region_idx] = 5.0
    return logits, None


class TestBottleneckProbe:
    def test_probe_recovers_target(self):
        # Two samples; both questions target a region.
        questions = ["is this lung?", "is this liver?"]
        region_names = ["lung", "liver"]
        targets = torch.tensor([0, 1])
        outputs = _outputs(torch.randn(2, 2))

        report = bottleneck_probe(
            outputs=outputs,
            questions=questions,
            targets=targets,
            region_names=region_names,
            answer_vocab=ANSWER_VOCAB,
            compute_logits_fn=_logits_from_regions,
        )
        assert report["total"] == 2
        assert report["corrected_correct"] == 2
        # Original outputs must not be mutated by the probe
        assert not torch.equal(
            outputs.scene_region_logits,
            torch.where(torch.eye(2).bool(), torch.tensor(10.0), torch.tensor(-10.0)),
        )

    def test_probe_skips_unknown_targets(self):
        questions = ["what color is the tissue?"]
        targets = torch.tensor([3])
        outputs = _outputs(torch.randn(1, 2))
        report = bottleneck_probe(
            outputs=outputs,
            questions=questions,
            targets=targets,
            region_names=["lung", "liver"],
            answer_vocab=ANSWER_VOCAB,
            compute_logits_fn=_logits_from_regions,
        )
        assert report["baseline_correct"] == report["corrected_correct"]


class TestCounterfactualExplanation:
    def test_flip_and_robust_regions(self):
        questions = ["is this lung?"]
        # Region 0 dominates → removing it flips prediction to region 1's entry
        outputs = _outputs(torch.tensor([[5.0, 1.0, 0.5]]))
        explanations = counterfactual_explanation(
            outputs,
            questions,
            ["lung", "liver", "heart"],
            {k: v for k, v in enumerate(ANSWER_VOCAB)},
            _logits_from_regions,
        )
        assert len(explanations) == 3
        first = explanations[0]
        assert first["region"] == "lung"
        assert first["original_prediction"] == "lung"
        assert first["prediction_changed"] is True
        assert "absent" in first["explanation"]

    def test_none_region_logits(self):
        outputs = ModelOutput(answer_logits=torch.randn(1, 4))
        out = counterfactual_explanation(outputs, ["q"], ["lung"], {}, _logits_from_regions)
        assert out == []


class TestConceptAttribution:
    def test_attribution_gradients(self):
        """End-to-end attribution through a tiny fake neuro-symbolic model."""

        class TinySceneParser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(8, 3)
                self.tag = "scene_parser"

            def forward(self, x):
                from types import SimpleNamespace

                return SimpleNamespace(region_logits=self.lin(x))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scene_parser = TinySceneParser()
                self.head = torch.nn.Linear(3, 4)

            def forward(self, image, input_ids, attn_mask, qtype_onehot=None):
                # attended features simulated from the image
                pooled = image.mean(dim=(2, 3))
                scene = self.scene_parser(pooled)
                regions = scene.region_logits
                logits = self.head(regions)
                return ModelOutput(answer_logits=logits, scene_region_logits=regions)

        model = TinyModel()
        result = concept_attribution(
            model=model,
            image=torch.randn(1, 8, 4, 4),
            input_ids=torch.ones(1, 2, dtype=torch.long),
            attn_mask=torch.ones(1, 2, dtype=torch.long),
            qtype_onehot=torch.zeros(1, 5),
            answer_idx=1,
        )
        attr = result["region_attribution"]
        assert attr.shape == (3,)
        assert (attr >= 0).all()
        assert attr.sum() > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
