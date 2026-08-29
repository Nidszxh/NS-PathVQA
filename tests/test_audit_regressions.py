"""Regression tests for bugs found in the 2026-08 full-codebase audit."""

import json
import logging
import time
from types import SimpleNamespace

import pytest
import torch

from pathvqa.models.protocols import ModelOutput
from pathvqa.symbolic.dsl import DSLNode, DifferentiableDSLInterpreter, DSLProgramCompiler
from pathvqa.symbolic.ltn import MedicalLogicTensorNetwork
from pathvqa.utils.interpretability import generate_visual_grounding_map


class TestLegacySceneHeadMigration:
    """Regression for the removal of the never-consumed density head."""

    def test_strips_density_head_parameters(self):
        from pathvqa._compat import migrate_legacy_scene_head

        state = {
            "scene_parser.region_classifier.weight": torch.zeros(2, 2),
            "scene_parser.density_classifier.weight": torch.zeros(4, 2),
            "scene_parser.density_classifier.bias": torch.zeros(4),
            "classifier.0.weight": torch.zeros(3, 3),
        }
        filtered, removed = migrate_legacy_scene_head(state)
        assert "scene_parser.density_classifier.weight" not in filtered
        assert "scene_parser.density_classifier.bias" not in filtered
        assert "scene_parser.region_classifier.weight" in filtered
        assert set(removed) == {
            "scene_parser.density_classifier.weight",
            "scene_parser.density_classifier.bias",
        }

    def test_clean_state_dict_is_untouched(self):
        from pathvqa._compat import migrate_legacy_scene_head

        state = {"classifier.0.weight": torch.zeros(1)}
        filtered, removed = migrate_legacy_scene_head(state)
        assert filtered is state
        assert removed == []


class TestModelOutputSceneHeadDict:
    def test_roundtrips_all_heads(self):
        out = ModelOutput(
            answer_logits=torch.zeros(2, 5),
            scene_region_logits=torch.randn(2, 7),
            scene_object_presence=torch.sigmoid(torch.randn(2, 7)),
            scene_color_logits=torch.randn(2, 17),
            scene_shape_logits=torch.randn(2, 9),
            scene_size_logits=torch.randn(2, 11),
        )
        d = out.scene_head_dict()
        assert d["scene_region_logits"] is out.scene_region_logits
        assert d["scene_object_presence"] is out.scene_object_presence
        assert d["scene_color_logits"] is out.scene_color_logits
        assert d["scene_shape_logits"] is out.scene_shape_logits
        assert d["scene_size_logits"] is out.scene_size_logits

    def test_diff_executor_accepts_scene_head_dict(self):
        # Regression: DifferentiableExecutor crashed with AttributeError when
        # handed the ModelOutput dataclass instead of a dict.
        from pathvqa.symbolic.diff_executor import DifferentiableExecutor

        ex = DifferentiableExecutor({"yes": 0, "no": 1}, ["lung"])
        scene = {
            "scene_region_logits": torch.randn(2, 1),
            "scene_object_presence": torch.sigmoid(torch.randn(2, 1)),
            "scene_color_logits": None,
            "scene_shape_logits": None,
            "scene_size_logits": None,
        }
        soft_qtype = torch.softmax(torch.randn(2, 5), dim=-1)
        soft_target = torch.softmax(torch.randn(2, 1), dim=-1)
        logits = ex.execute(scene_logits=scene, soft_qtype=soft_qtype, soft_target=soft_target)
        assert logits.shape == (2, 2)


class TestLTNPartialDictLoss:
    def test_loss_non_negative_on_partial_heads(self):
        # Regression: summing placeholder constants over a smaller clause count
        # produced satisfaction > 1 and a negative loss.
        engine = MedicalLogicTensorNetwork(num_regions=1, region_names=["lung"])
        scene = {
            "scene_region_logits": torch.randn(2, 4),
            "scene_object_presence": torch.sigmoid(torch.randn(2, 4)),
            "scene_color_logits": None,
            "scene_shape_logits": None,
            "scene_size_logits": None,
        }
        qtype_onehot = torch.zeros(2, 5)
        qtype_onehot[:, 3] = 1.0  # attribute-type rows exercise attribute clauses
        loss, scores = engine.evaluate_clauses(scene, qtype_onehot=qtype_onehot)
        assert loss.item() >= 0.0
        assert 0.0 <= scores["overall_satisfaction"] <= 1.0


class TestDSLCompilerInterpreterContract:
    def test_compiler_count_anchored_like_parser(self):
        # Regression: unanchored "how many" matched mid-question phrases that
        # query_parser classified as identity.
        prog = DSLProgramCompiler.compile("tell me how many nuclei", target="nuclei", qtype="identity")
        assert prog.op == "Exist"
        anchored = DSLProgramCompiler.compile("how many nuclei are visible", target="nuclei")
        assert anchored.op == "Count"

    def test_interpreter_handles_all_emitted_ops(self):
        # Regression: Exist/Relate programs returned all-zero logits silently.
        interp = DifferentiableDSLInterpreter(visual_dim=8, hidden_dim=4)
        interp.eval()
        patches = torch.randn(2, 6, 8)
        scene = {"scene_color_logits": torch.randn(2, 17)}
        vocab = {"tumor": 0, "two": 1, "yes": 2, "no": 3}  # covers target/count/verify slots
        ops = {"Exist": "tumor", "Relate": "tumor", "Count": "two", "Verify": "yes"}
        for op, target in ops.items():
            node = DSLNode(op=op, args={"target": target}, child=DSLNode(op="Filter", args={"concept": "tumor"}))
            with torch.no_grad():
                logits = interp(node, patches, scene, vocab)
            assert logits.shape == (2, 4), f"op={op}"
            assert logits.abs().sum() > 0, f"{op} produced all-zero logits"


class TestGroundingMapPadding:
    def test_fewer_patches_than_grid_is_padded_not_crash(self):
        # Regression: .view(7, 7) crashed for non-49 patch counts.
        attn = torch.rand(30)
        result = generate_visual_grounding_map(attn, grid_size=(7, 7), top_k=3)
        assert len(result["grid_heatmap"]) == 7
        assert len(result["grid_heatmap"][0]) == 7
        assert len(result["top_boxes"]) == 3

    def test_more_patches_than_grid_truncates(self):
        attn = torch.rand(64)
        result = generate_visual_grounding_map(attn, grid_size=(7, 7), top_k=3)
        assert len(result["grid_heatmap"]) == 7


class TestDiffExecutorAutocast:
    def test_execute_under_autocast_does_not_crash(self):
        # Regression: scatter_add_ requires matching dtypes; under AMP the
        # scene heads emit fp16/bf16 while the executor buffers are fp32.
        from pathvqa.symbolic.diff_executor import DifferentiableExecutor

        ex = DifferentiableExecutor({"yes": 0, "no": 1}, ["lung"])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            scene = {
                "scene_region_logits": torch.randn(2, 1),
                "scene_object_presence": torch.sigmoid(torch.randn(2, 1)),
                "scene_color_logits": torch.randn(2, 17),
                "scene_shape_logits": None,
                "scene_size_logits": None,
            }
            soft_qtype = torch.softmax(torch.randn(2, 5), dim=-1)
            soft_target = torch.softmax(torch.randn(2, 1), dim=-1)
            logits = ex.execute(scene_logits=scene, soft_qtype=soft_qtype, soft_target=soft_target)
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()


class TestDiffExecutorFallback:
    def test_missing_region_logits_returns_zeros(self):
        from pathvqa.symbolic.diff_executor import DifferentiableExecutor

        ex = DifferentiableExecutor({"yes": 0, "no": 1}, ["lung"])
        out = ex.execute(
            scene_logits={},
            soft_qtype=torch.softmax(torch.randn(2, 5), dim=-1),
            soft_target=torch.softmax(torch.randn(2, 1), dim=-1),
        )
        assert out.shape == (2, 2)
        assert (out == 0).all()


class TestReasoningTreeNoFabricatedEvidence:
    def test_no_patch_indices_means_empty_grounding(self):
        # Regression: the tree used to fabricate placeholder indices
        # [12, 13, 19] / [19, 20] when no grounding was available — invented
        # evidence inside an explainability artifact.
        from pathvqa.utils.interpretability import generate_hierarchical_reasoning_tree

        tree = generate_hierarchical_reasoning_tree(
            prediction="yes",
            confidence=0.5,
            qtype="identity",
            target="",
            region_names=[],
            region_scores=[],
            patch_indices=None,
        )
        steps = tree["reasoning_trace"]
        assert steps["step_1_organ_identification"]["grounding_patch_indices"] == []
        assert steps["step_2_cellular_atypia"]["grounding_patch_indices"] == []

    def test_provided_indices_are_echoed(self):
        from pathvqa.utils.interpretability import generate_hierarchical_reasoning_tree

        tree = generate_hierarchical_reasoning_tree(
            prediction="yes",
            confidence=0.5,
            qtype="identity",
            target="",
            region_names=["lung"],
            region_scores=[0.9],
            patch_indices=[19, 12, 13],
        )
        steps = tree["reasoning_trace"]
        assert steps["step_1_organ_identification"]["grounding_patch_indices"] == [12, 13, 19]
        assert steps["step_2_cellular_atypia"]["grounding_patch_indices"] == []


class TestFailureTaxonomyInterfaces:
    def test_collateoutput_batches_supported(self):
        # Regression: compute_failure_taxonomy only accepted legacy dict
        # batches with an 'answer_indices' key; production collate_fn emits a
        # CollateOutput dataclass with 'targets'.
        from dataclasses import dataclass, field
        from typing import List as TList

        from pathvqa.utils.interpretability import compute_failure_taxonomy

        @dataclass
        class FakeCollate:
            images: torch.Tensor
            targets: torch.Tensor
            input_ids: torch.Tensor
            attention_mask: torch.Tensor
            questions: TList[str] = field(default_factory=list)

        class FakeOut:
            answer_logits = torch.tensor([[0.0, 5.0]])
            scene_region_logits = torch.tensor([[2.0, -1.0]])

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, images, input_ids, attn_mask):
                return FakeOut()

        batch = FakeCollate(
            images=torch.zeros(1, 3, 8, 8),
            targets=torch.zeros(1, dtype=torch.long),
            input_ids=torch.ones(1, 4, dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            questions=["what is this?"],
        )
        report = compute_failure_taxonomy(FakeModel(), [batch], torch.device("cpu"), {1: "no"})
        assert report["total_failures"] == 1

    def test_name_based_classification_when_region_names_given(self):
        # Regression: parsing/execution attribution compared region indices to
        # answer-vocab indices — different index spaces.
        import torch as t

        from pathvqa.utils.interpretability import classify_failure_type

        sample = {
            "qtype": "identity",
            "top_region_name": "lung",
            "target_answer": "lung",
            "scene_region_logits": t.tensor([2.0, -1.0]),
            "target": 999,  # deliberately wrong in index space; names must win
        }
        assert classify_failure_type(sample) == "execution"
        sample["target_answer"] = "liver"
        assert classify_failure_type(sample) == "parsing"


class TestPoEFusionEngine:
    def _engine_and_outputs(self):
        """Build a hard-executor fusion engine on a tiny synthetic setup."""
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(weighting_strategy="static", symbolic_weight=0.3)
        answer_to_idx = {"yes": 0, "no": 1, "lung": 2}
        region_names = ["lung"]

        outputs = ModelOutput(
            answer_logits=torch.randn(2, 3),
            scene_region_logits=torch.randn(2, 1),
            scene_object_presence=torch.sigmoid(torch.randn(2, 1)),
            scene_color_logits=torch.randn(2, 17),
            scene_shape_logits=torch.randn(2, 9),
            scene_size_logits=torch.randn(2, 11),
            patch_features=torch.randn(2, 4, 8),
            attention_weights=torch.softmax(torch.randn(2, 8, 1, 4), dim=-1),
        )
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx=answer_to_idx,
            region_names=region_names,
            device=torch.device("cpu"),
        )
        return engine, outputs, ["is there lung?", "what is this?"], answer_to_idx

    def test_matches_manual_poe_formula(self):
        engine, outputs, questions, answer_to_idx = self._engine_and_outputs()
        log_p, trace = engine(outputs, questions)
        assert trace["symbolic_used"][0] is True
        # Manual reference: hard executor + static-weight PoE
        from pathvqa.symbolic.executor import execute
        from pathvqa.symbolic.query_parser import parse_question

        queries = [parse_question(q, list(answer_to_idx)) for q in questions]
        exec_out = execute(
            scene_logits=outputs,
            queries=queries,
            region_names=["lung"],
            region_to_answer_idx=engine.region_to_answer_idx,
            attribute_mappings=engine.attribute_mappings,
            answer_to_idx=answer_to_idx,
            answer_vocab_size=3,
            neural_logits=outputs.answer_logits,
            ontology=None,
        )
        expected = (1 - 0.3) * torch.log_softmax(outputs.answer_logits, -1) + 0.3 * torch.log_softmax(
            exec_out.symbolic_logits + 1e-8, -1
        )
        expected = expected - torch.logsumexp(expected, dim=-1, keepdim=True)
        torch.testing.assert_close(log_p, expected)

    def test_soft_target_sized_by_region_names_not_stale_config(self):
        # Regression: the fallback soft_target was allocated with
        # config.symbolic.num_regions but indexed by region_names positions —
        # an IndexError whenever the two disagreed (stale checkpoint configs).
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.symbolic.diff_executor import DifferentiableExecutor
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(num_regions=1)  # stale: actual regions = 3
        answer_to_idx = {"yes": 0, "no": 1}
        region_names = ["lung", "liver", "heart"]
        ex = DifferentiableExecutor(answer_to_idx, region_names)
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx=answer_to_idx,
            region_names=region_names,
            device=torch.device("cpu"),
            diff_executor=ex,
        )
        outputs = ModelOutput(
            answer_logits=torch.randn(2, 2),
            scene_region_logits=torch.randn(2, 3),
            scene_object_presence=torch.sigmoid(torch.randn(2, 3)),
            patch_features=None,
        )
        log_p, _ = engine(outputs, ["is there lung?", "any finding?"])
        assert log_p.shape == (2, 2)

    def test_disabled_symbolic_returns_log_softmax(self):
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(enabled=False)
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx={"yes": 0, "no": 1},
            region_names=[],
            device=torch.device("cpu"),
        )
        outputs = ModelOutput(answer_logits=torch.randn(2, 2))
        log_p, trace = engine(outputs, ["q1", "q2"])
        torch.testing.assert_close(log_p, torch.log_softmax(outputs.answer_logits, -1))
        assert trace == {}

    def test_learned_gate_drives_per_sample_poe(self):
        """weighting_strategy="learned" must fuse with the *gate values*.

        Regression coverage for the primary contribution: the learned gate had
        no end-to-end fusion test, so a broken gate path (falling back to the
        static weight, or ignoring the per-sample gate) went undetected.
        """
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(weighting_strategy="learned", symbolic_weight=0.3)
        answer_to_idx = {"yes": 0, "no": 1, "lung": 2}
        region_names = ["lung"]
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx=answer_to_idx,
            region_names=region_names,
            device=torch.device("cpu"),
        )
        # Per-sample gates deliberately different so the two rows fuse
        # differently (row 0 favours neural, row 1 favours symbolic).
        gate_values = torch.tensor([[0.1], [0.9]])
        outputs = ModelOutput(
            answer_logits=torch.randn(2, 3),
            scene_region_logits=torch.randn(2, 1),
            scene_object_presence=torch.sigmoid(torch.randn(2, 1)),
            scene_color_logits=torch.randn(2, 17),
            scene_shape_logits=torch.randn(2, 9),
            scene_size_logits=torch.randn(2, 11),
            patch_features=torch.randn(2, 4, 8),
            gate_values=gate_values,
        )
        log_p, trace = engine(outputs, ["is there lung?", "any finding?"])
        assert log_p.shape == (2, 3)
        assert trace["symbolic_used"][0] is True

        # Manual reference: static executor + per-sample gate PoE.
        from pathvqa.symbolic.executor import execute
        from pathvqa.symbolic.query_parser import parse_question

        queries = [parse_question(q, list(answer_to_idx)) for q in ["is there lung?", "any finding?"]]
        exec_out = execute(
            scene_logits=outputs,
            queries=queries,
            region_names=region_names,
            region_to_answer_idx=engine.region_to_answer_idx,
            attribute_mappings=engine.attribute_mappings,
            answer_to_idx=answer_to_idx,
            answer_vocab_size=3,
            neural_logits=outputs.answer_logits,
            ontology=None,
        )
        log_n = torch.log_softmax(outputs.answer_logits, -1)
        log_s = torch.log_softmax(exec_out.symbolic_logits + 1e-8, -1)
        g = gate_values.squeeze(-1).unsqueeze(1)
        expected = (1 - g) * log_n + g * log_s
        expected = expected - torch.logsumexp(expected, dim=-1, keepdim=True)
        torch.testing.assert_close(log_p, expected)
        torch.testing.assert_close(torch.logsumexp(log_p, dim=-1), torch.zeros(2))

        # The two rows must not be identical: the gate is actually per-sample.
        assert not torch.allclose(log_p[0], log_p[1])

    def test_count_without_dsl_is_neural_only(self):
        """Rows the symbolic path cannot execute must not be blended.

        Regression: count rows were fused with ``log_softmax(zeros)`` (uniform)
        at weight 0.3, dragging every count prediction toward uniform even
        though the executor had produced no evidence.
        """
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(weighting_strategy="static", symbolic_weight=0.3)
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx={"yes": 0, "no": 1, "2": 2},
            region_names=["lung"],
            device=torch.device("cpu"),
            dsl_interpreter=None,
        )
        outputs = ModelOutput(
            answer_logits=torch.randn(1, 3),
            scene_region_logits=torch.randn(1, 1),
            scene_object_presence=torch.sigmoid(torch.randn(1, 1)),
            patch_features=torch.randn(1, 4, 8),
        )
        log_p, trace = engine(outputs, ["how many nuclei are present?"])
        assert trace["symbolic_used"] == [False]
        assert trace["qtypes"] == ["count"]
        torch.testing.assert_close(log_p, torch.log_softmax(outputs.answer_logits, -1))

    def test_dsl_count_evidence_is_fused(self):
        """Count rows use DSL logits when the interpreter emits evidence."""
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        answer_to_idx = {"yes": 0, "no": 1, "2": 2, "3": 3}
        row = torch.full((len(answer_to_idx),), -1e9)
        row[answer_to_idx["3"]] = float(torch.log(torch.tensor(0.9)))
        row[answer_to_idx["2"]] = float(torch.log(torch.tensor(0.1)))

        class StubDSL:
            def __call__(self, program, patch_features, scene_logits, vocab):
                return row.unsqueeze(0)

        cfg = SymbolicConfig(weighting_strategy="static", symbolic_weight=0.5, query_parser="regex")
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx=answer_to_idx,
            region_names=["lung"],
            device=torch.device("cpu"),
            dsl_interpreter=StubDSL(),
        )
        outputs = ModelOutput(
            answer_logits=torch.zeros(1, len(answer_to_idx)),
            scene_region_logits=torch.randn(1, 1),
            scene_object_presence=torch.sigmoid(torch.randn(1, 1)),
            patch_features=torch.randn(1, 4, 8),
        )
        log_p, trace = engine(outputs, ["how many nuclei are present?"])
        assert trace["symbolic_used"] == [True]
        assert trace["dsl_used"] == [True]
        probs = torch.softmax(log_p, dim=-1)
        assert probs[0, answer_to_idx["3"]] > 0.9 * probs[0, answer_to_idx["2"]]
        assert probs[0, answer_to_idx["yes"]] < 1e-5
        assert probs[0, answer_to_idx["no"]] < 1e-5

    def test_dsl_masked_evidence_stays_finite_under_fp16_outputs(self):
        """DSL count masks must not become negative infinity during AMP fusion."""
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.utils.config import SymbolicConfig

        answer_to_idx = {"yes": 0, "no": 1, "2": 2, "lung": 3}
        cfg = SymbolicConfig(query_parser="regex", executor="hard", symbolic_weight=0.3)
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx=answer_to_idx,
            region_names=["lung"],
            device=torch.device("cpu"),
            dsl_interpreter=DifferentiableDSLInterpreter(visual_dim=4, hidden_dim=4),
        )
        outputs = ModelOutput(
            answer_logits=torch.zeros(1, len(answer_to_idx), dtype=torch.float16),
            scene_region_logits=torch.zeros(1, 1, dtype=torch.float16),
            scene_object_presence=torch.full((1, 1), 0.5, dtype=torch.float16),
            patch_features=torch.randn(1, 3, 4),
        )

        fused, trace = engine(outputs, ["how many cells are present?"])

        assert trace["dsl_used"] == [True]
        assert torch.isfinite(fused).all()

    def test_dsl_zero_rows_fall_back_to_neural(self):
        """An interpreter that finds no candidate answer leaves the row neural-only."""
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        class ZeroDSL:
            def __call__(self, program, patch_features, scene_logits, vocab):
                return torch.zeros(1, len(vocab))

        cfg = SymbolicConfig(weighting_strategy="static", symbolic_weight=0.3, query_parser="regex")
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx={"yes": 0, "no": 1},
            region_names=["lung"],
            device=torch.device("cpu"),
            dsl_interpreter=ZeroDSL(),
        )
        outputs = ModelOutput(
            answer_logits=torch.randn(1, 2),
            scene_region_logits=torch.randn(1, 1),
            scene_object_presence=torch.sigmoid(torch.randn(1, 1)),
            patch_features=torch.randn(1, 4, 8),
        )
        log_p, trace = engine(outputs, ["how many nuclei are present?"])
        assert trace["symbolic_used"] == [False]
        assert trace["dsl_used"] == [False]
        torch.testing.assert_close(log_p, torch.log_softmax(outputs.answer_logits, -1))

    def test_unused_rows_keep_neural_distribution(self):
        """Fusion is decided per row, not per batch."""
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(weighting_strategy="static", symbolic_weight=0.3, query_parser="regex")
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx={"yes": 0, "no": 1, "lung": 2},
            region_names=["lung"],
            device=torch.device("cpu"),
        )
        outputs = ModelOutput(
            answer_logits=torch.randn(2, 3),
            scene_region_logits=torch.randn(2, 1),
            scene_object_presence=torch.sigmoid(torch.randn(2, 1)),
            patch_features=None,
        )
        log_p, trace = engine(outputs, ["what is this?", "how many nuclei?"])
        assert trace["symbolic_used"] == [True, False]
        log_n = torch.log_softmax(outputs.answer_logits, -1)
        assert not torch.allclose(log_p[0], log_n[0])
        torch.testing.assert_close(log_p[1], log_n[1])

    def test_learned_parser_drives_hard_executor(self):
        """With query_parser="learned" the hard executor must use the heads.

        Regression: the learned qtype/target heads only fed the differentiable
        executor, so the learned+hard ablation silently executed regex queries.
        """
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(weighting_strategy="static", query_parser="learned", executor="hard")
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx={"yes": 0, "no": 1, "lung": 2, "liver": 3},
            region_names=["lung", "liver"],
            device=torch.device("cpu"),
        )
        # Regex would call "what is this?" an identity query with no target;
        # the learned heads say yes/no about the liver.
        soft_qtype = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        soft_target = torch.tensor([[0.1, 0.9]])
        outputs = ModelOutput(
            answer_logits=torch.randn(1, 4),
            scene_region_logits=torch.randn(1, 2),
            scene_object_presence=torch.sigmoid(torch.randn(1, 2)),
            soft_qtype=soft_qtype,
            soft_target=soft_target,
        )
        from pathvqa.symbolic.query_parser import parse_question

        regex_query = parse_question("what is this?", ["yes", "no", "lung", "liver"])
        effective = engine._effective_queries(outputs, [regex_query], ["what is this?"])
        assert effective[0].qtype == "yes_no"
        assert effective[0].target == "liver"
        assert effective[0].program is not None and effective[0].program.op == "Verify"

        log_p, trace = engine(outputs, ["what is this?"])
        assert trace["qtypes"] == ["yes_no"]
        assert trace["symbolic_used"] == [True]

    def test_learned_target_head_can_abstain_on_non_region_yes_no(self):
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig
        from pathvqa.symbolic.query_parser import parse_question

        engine = SymbolicFusionEngine(
            config_symbolic=SymbolicConfig(query_parser="learned", executor="hard"),
            answer_to_idx={"yes": 0, "no": 1, "lung": 2},
            region_names=["lung"],
            device=torch.device("cpu"),
        )
        outputs = ModelOutput(
            answer_logits=torch.zeros(1, 3),
            scene_region_logits=torch.ones(1, 1),
            scene_object_presence=torch.tensor([[0.99]]),
            soft_qtype=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
            soft_target=torch.zeros(1, 1),
            target_null_probability=torch.tensor([0.9]),
        )

        queries = engine._effective_queries(
            outputs,
            [parse_question("is this malignant?", ["yes", "no", "lung"])],
            ["is this malignant?"],
        )

        assert queries[0].qtype == "yes_no"
        assert queries[0].target == ""

    def test_learned_without_gate_falls_back_to_static(self):
        """Missing gate values (e.g. warm-up) must fall back to the static weight."""
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.models.protocols import ModelOutput
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(weighting_strategy="learned", symbolic_weight=0.3)
        answer_to_idx = {"yes": 0, "no": 1, "lung": 2}
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx=answer_to_idx,
            region_names=["lung"],
            device=torch.device("cpu"),
        )
        outputs = ModelOutput(
            answer_logits=torch.randn(2, 3),
            scene_region_logits=torch.randn(2, 1),
            scene_object_presence=torch.sigmoid(torch.randn(2, 1)),
            scene_color_logits=torch.randn(2, 17),
            scene_shape_logits=torch.randn(2, 9),
            scene_size_logits=torch.randn(2, 11),
            patch_features=torch.randn(2, 4, 8),
            gate_values=None,
        )
        log_p, _ = engine(outputs, ["is there lung?", "any finding?"])
        from pathvqa.symbolic.executor import execute
        from pathvqa.symbolic.query_parser import parse_question

        queries = [parse_question(q, list(answer_to_idx)) for q in ["is there lung?", "any finding?"]]
        exec_out = execute(
            scene_logits=outputs,
            queries=queries,
            region_names=["lung"],
            region_to_answer_idx=engine.region_to_answer_idx,
            attribute_mappings=engine.attribute_mappings,
            answer_to_idx=answer_to_idx,
            answer_vocab_size=3,
            neural_logits=outputs.answer_logits,
            ontology=None,
        )
        log_n = torch.log_softmax(outputs.answer_logits, -1)
        log_s = torch.log_softmax(exec_out.symbolic_logits + 1e-8, -1)
        expected = 0.7 * log_n + 0.3 * log_s
        expected = expected - torch.logsumexp(expected, dim=-1, keepdim=True)
        torch.testing.assert_close(log_p, expected)


class TestBaselineArtifact:
    def test_disabled_symbolic_records_zero_regions(self, tmp_path):
        # Regression: the fallback ``... or config.symbolic.num_regions``
        # returned an int and ``len(int)`` raised TypeError for --no-symbolic.
        from train import Trainer
        from pathvqa.utils.config import Config

        cfg = Config()
        cfg.symbolic.enabled = False
        cfg.paths.output_dir = tmp_path
        cfg.paths.__post_init__()
        fake = SimpleNamespace(
            config=cfg,
            train_start_time=time.monotonic(),
            device=torch.device("cpu"),
            train_loader=None,
            val_loader=None,
            region_names=[],
            best_val_acc=0.0,
            best_val_epoch=-1,
            best_val_per_type={},
            current_epoch=0,
            epoch_times=[],
            amp_enabled=False,
            logger=logging.getLogger("test"),
        )
        Trainer._write_baseline(fake)
        payload = json.loads((tmp_path / f"{cfg.experiment_name}.json").read_text())
        assert payload["config"]["num_regions"] == 0

    def test_enabled_symbolic_records_region_count(self, tmp_path):
        from train import Trainer
        from pathvqa.utils.config import Config

        cfg = Config()
        cfg.paths.output_dir = tmp_path
        cfg.paths.__post_init__()
        fake = SimpleNamespace(
            config=cfg,
            train_start_time=time.monotonic(),
            device=torch.device("cpu"),
            train_loader=None,
            val_loader=None,
            region_names=["lung", "liver", "heart"],
            best_val_acc=0.0,
            best_val_epoch=-1,
            best_val_per_type={},
            current_epoch=0,
            epoch_times=[],
            amp_enabled=False,
            logger=logging.getLogger("test"),
        )
        Trainer._write_baseline(fake)
        payload = json.loads((tmp_path / f"{cfg.experiment_name}.json").read_text())
        assert payload["config"]["num_regions"] == 3


class TestResumeEMA:
    def test_missing_snapshot_reseeds_from_loaded_weights(self):
        # Regression: the shadow was seeded from the random model constructed
        # before load_state_dict, so a checkpoint without EMA validated on
        # random weights and persisted them.
        from pathvqa.models.ema import ModelEMA
        from train import Trainer

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        ema = ModelEMA(model)
        with torch.no_grad():
            model.weight.add_(3.0)
            model.bias.add_(2.0)
        fake = SimpleNamespace(
            ema=ema,
            model=model,
            resume_ckpt={"ema_state_dict": None},
            logger=logging.getLogger("test"),
        )
        Trainer._restore_ema_state(fake)
        torch.testing.assert_close(ema.shadow["weight"], model.weight.detach())
        torch.testing.assert_close(ema.shadow["bias"], model.bias.detach())

    def test_existing_snapshot_is_loaded_not_overwritten(self):
        from pathvqa.models.ema import ModelEMA
        from train import Trainer

        model = torch.nn.Linear(2, 2)
        ema = ModelEMA(model)
        snapshot = {name: torch.full_like(value, 5.0) for name, value in ema.shadow.items()}
        fake = SimpleNamespace(
            ema=ema,
            model=model,
            resume_ckpt={"ema_state_dict": snapshot},
            logger=logging.getLogger("test"),
        )
        Trainer._restore_ema_state(fake)
        torch.testing.assert_close(ema.shadow["weight"], snapshot["weight"])
        torch.testing.assert_close(ema.shadow["bias"], snapshot["bias"])


class TestDiffExecutorYesNoValidity:
    def test_non_region_yes_no_target_emits_no_evidence(self):
        from pathvqa.symbolic.diff_executor import DifferentiableExecutor

        ex = DifferentiableExecutor({"yes": 0, "no": 1, "malignant": 2}, ["lung"])
        scene = {
            "scene_region_logits": torch.tensor([[2.0], [2.0]]),
            "scene_object_presence": torch.tensor([[0.9], [0.9]]),
        }
        soft_qtype = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]])
        soft_target = torch.tensor([[1.0], [1.0]])
        out = ex.execute(
            scene_logits=scene,
            soft_qtype=soft_qtype,
            soft_target=soft_target,
            yes_no_valid=torch.tensor([True, False]),
        )
        assert torch.count_nonzero(out[0]).item() > 0
        # Proxy "any region present" evidence must not leak into the row.
        assert torch.count_nonzero(out[1]).item() == 0

    def test_uniform_target_without_mask_emits_no_evidence(self):
        from pathvqa.symbolic.diff_executor import DifferentiableExecutor

        ex = DifferentiableExecutor({"yes": 0, "no": 1}, ["lung", "liver"])
        scene = {
            "scene_region_logits": torch.tensor([[1.0, 1.0]]),
            "scene_object_presence": torch.tensor([[0.8, 0.8]]),
        }
        soft_qtype = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        uniform_target = torch.tensor([[0.5, 0.5]])
        out = ex.execute(scene_logits=scene, soft_qtype=soft_qtype, soft_target=uniform_target)
        assert torch.count_nonzero(out).item() == 0

    def test_poe_masks_non_region_yes_no_rows(self):
        from pathvqa.symbolic.diff_executor import DifferentiableExecutor
        from pathvqa.symbolic.poe import SymbolicFusionEngine
        from pathvqa.utils.config import SymbolicConfig

        cfg = SymbolicConfig(
            weighting_strategy="static", symbolic_weight=0.3, query_parser="regex", executor="differentiable"
        )
        answer_to_idx = {"yes": 0, "no": 1, "lung": 2, "malignant": 3}
        engine = SymbolicFusionEngine(
            config_symbolic=cfg,
            answer_to_idx=answer_to_idx,
            region_names=["lung"],
            device=torch.device("cpu"),
            diff_executor=DifferentiableExecutor(answer_to_idx, ["lung"]),
        )
        outputs = ModelOutput(
            answer_logits=torch.randn(2, 4),
            scene_region_logits=torch.randn(2, 1),
            scene_object_presence=torch.sigmoid(torch.randn(2, 1)),
        )
        log_p, trace = engine(outputs, ["is there lung?", "is this a malignant tumor?"])
        assert trace["qtypes"] == ["yes_no", "yes_no"]
        assert trace["symbolic_used"] == [True, False]
        torch.testing.assert_close(log_p[1], torch.log_softmax(outputs.answer_logits[1], -1))


class TestDSLQueryAttrBatchAlignment:
    def _interpreter_and_program(self):
        from pathvqa.symbolic.executor import build_attribute_mappings

        vocab = {"red": 0, "blue": 1, "large": 2, "small": 3}
        interp = DifferentiableDSLInterpreter(
            visual_dim=8, hidden_dim=4, attribute_mappings=build_attribute_mappings(vocab)
        )
        program = DSLProgramCompiler.compile("what color is the lesion?", target="lesion", qtype="attribute")
        return interp, program, vocab

    def test_matching_batch_executes(self):
        interp, program, vocab = self._interpreter_and_program()
        patches = torch.randn(2, 4, 8)
        scene = {"scene_color_logits": torch.randn(2, 17)}
        out = interp(program, patches, scene, vocab)
        assert out.shape == (2, len(vocab))

    def test_mismatched_scene_batch_raises(self):
        interp, program, vocab = self._interpreter_and_program()
        patches = torch.randn(1, 4, 8)
        scene = {"scene_color_logits": torch.randn(2, 17)}
        with pytest.raises(ValueError, match="batch"):
            interp(program, patches, scene, vocab)


class TestEvaluateGuards:
    def test_clustered_ci_rejects_empty_input(self):
        from evaluate import _clustered_accuracy_ci

        with pytest.raises(ValueError, match="zero examples"):
            _clustered_accuracy_ci([], [], bootstrap_samples=10)

    def test_architecture_fields_pin_module_shapes_and_normalization(self):
        from evaluate import _protect_checkpoint_architecture
        from pathvqa.utils.config import Config

        class Recorder:
            def __init__(self):
                self.messages = []

            def warning(self, message, *args):
                self.messages.append(message % args if args else message)

        ckpt = Config()
        override = Config()
        override.question.dropout = 0.5
        override.question.lora_target_modules = ("q_lin",)
        override.visual.lora_target_modules = ("q_proj",)
        override.data.norm = "imagenet"
        logger = Recorder()
        _protect_checkpoint_architecture(ckpt, override, logger)
        assert override.question.dropout == ckpt.question.dropout
        assert override.question.lora_target_modules == ckpt.question.lora_target_modules
        assert override.visual.lora_target_modules == ckpt.visual.lora_target_modules
        assert override.data.norm == ckpt.data.norm
        assert len(logger.messages) == 4


class TestNormalizationSelection:
    def test_imagenet_norm_matches_dataset_constant(self):
        from pathvqa.data.pathvqa_dataset import IMAGENET_MEAN, IMAGENET_STD, get_normalize

        transform = get_normalize("imagenet")
        assert list(transform.mean) == IMAGENET_MEAN
        assert list(transform.std) == IMAGENET_STD
