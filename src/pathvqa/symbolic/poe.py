"""Shared Product-of-Experts (PoE) fusion of neural and symbolic logits.

Single implementation used by ``train.py``, ``evaluate.py`` and the API layer;
keeping one copy avoids the device- and sizing-drift that three private copies
had already accumulated.

Fusion semantics (must stay log-space PoE — see DOCS.md):
    log p = (1 - g) * log p_neural + g * log p_symbolic

with a static weight ``w = symbolic.symbolic_weight`` when
``weighting_strategy="static"`` and per-sample gate values ``g`` when
``weighting_strategy="learned"``. Count queries are routed through the model's
DSL interpreter, and the differentiable executor replaces the hard executor
when the checkpoint config selects it.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from pathvqa.models.protocols import ExecutorOutput
from pathvqa.symbolic.executor import (
    build_attribute_mappings,
    build_region_mapping,
    execute,
)
from pathvqa.symbolic.query_parser import parse_question
from pathvqa.symbolic.routing import encode_question_types


class SymbolicFusionEngine:
    """Fuses neural answer logits with symbolic executor logits (PoE, log space).

    Holds every mapping the fusion needs (region→vocab, attribute maps,
    ontology, optional differentiable executor / DSL interpreter) so callers
    share one code path instead of maintaining private copies.
    """

    def __init__(
        self,
        config_symbolic: Any,
        answer_to_idx: Dict[str, int],
        region_names: List[str],
        device: torch.device,
        ontology: Optional[Any] = None,
        diff_executor: Optional[Any] = None,
        dsl_interpreter: Optional[Any] = None,
        region_to_answer_idx: Optional[torch.Tensor] = None,
        attribute_mappings: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Wire up every mapping the fusion needs, on ``device``.

        ``region_names`` must be the *actual* regions present in the checkpoint,
        not ``config.symbolic.num_regions``: ``soft_target`` is sized from it, so a
        stale count would crash the executor's writes.
        """
        self.cfg = config_symbolic
        self.answer_to_idx = dict(answer_to_idx)
        self.answer_vocab = list(self.answer_to_idx.keys())
        self.vocab_size = len(self.answer_to_idx)
        self.region_names = list(region_names)
        # Number of regions actually present — NOT a possibly-stale
        # config.symbolic.num_regions (mismatch would crash soft_target writes).
        self.num_regions = len(self.region_names)
        self.device = device
        self.ontology = ontology
        self.diff_executor = diff_executor
        self.dsl_interpreter = dsl_interpreter
        self.region_to_answer_idx = (
            region_to_answer_idx.to(device)
            if region_to_answer_idx is not None
            else build_region_mapping(self.region_names, self.answer_to_idx).to(device)
        )
        self.attribute_mappings = attribute_mappings or build_attribute_mappings(self.answer_to_idx)

    def __call__(self, outputs: Any, questions: List[str]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return ``(log_p_combined, trace)`` for one batch of model outputs."""
        if not self.cfg.enabled or outputs.scene_region_logits is None:
            return torch.log_softmax(outputs.answer_logits, dim=-1), {}
        queries = [parse_question(q, self.answer_vocab) for q in questions]

        # DSL routing for count queries
        dsl_logits: Optional[torch.Tensor] = None
        if outputs.patch_features is not None and self.dsl_interpreter is not None:
            dsl_logits = torch.zeros_like(outputs.answer_logits)
            for i, q in enumerate(queries):
                if q.qtype == "count" and q.program is not None:
                    dsl_out = self.dsl_interpreter(
                        q.program,
                        outputs.patch_features[i : i + 1],
                        outputs.scene_head_dict(),
                        self.answer_to_idx,
                    )
                    dsl_logits[i] = dsl_out[0]

        if self.diff_executor is not None:
            soft_qtype, soft_target = self._soft_routing(outputs, queries)
            diff_logits = self.diff_executor.execute(
                scene_logits=outputs.scene_head_dict(), soft_qtype=soft_qtype, soft_target=soft_target
            )
            # Count queries bypass symbolic execution (the neural path handles
            # counting), matching the hard executor's trace. Reporting True for
            # every query made symbolic-path accuracy equal overall accuracy and
            # incomparable with the hard executor.
            symbolic_used = [q.qtype != "count" for q in queries]
            exec_out = ExecutorOutput(
                symbolic_logits=diff_logits,
                region_logits=torch.zeros(0, device=outputs.answer_logits.device),
                trace={"symbolic_used": symbolic_used},
            )
        else:
            exec_out = execute(
                scene_logits=outputs,
                queries=queries,
                region_names=self.region_names,
                region_to_answer_idx=self.region_to_answer_idx,
                attribute_mappings=self.attribute_mappings,
                answer_to_idx=self.answer_to_idx,
                answer_vocab_size=self.vocab_size,
                neural_logits=outputs.answer_logits,
                ontology=self.ontology,
            )

        # For count queries, prefer DSL logits over the executor's zero logits
        if dsl_logits is not None:
            for i, q in enumerate(queries):
                if q.qtype == "count":
                    exec_out.symbolic_logits[i] = dsl_logits[i]

        return self._fuse(outputs, exec_out), exec_out.trace

    def _soft_routing(self, outputs: Any, queries: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft qtype/target routing for the differentiable executor.

        Uses the learned heads when present; otherwise falls back to regex
        parsing (Config E: Regex + Differentiable executor).
        """
        if outputs.soft_qtype is not None and outputs.soft_target is not None:
            return outputs.soft_qtype, outputs.soft_target

        soft_qtype = encode_question_types(queries).to(self.device)
        soft_target = torch.zeros(len(queries), self.num_regions, device=self.device)
        uniform_row = torch.full((self.num_regions,), 1.0 / max(self.num_regions, 1), device=self.device)
        for i, q in enumerate(queries):
            if q.target and q.target in self.region_names:
                soft_target[i, self.region_names.index(q.target)] = 1.0
            else:
                soft_target[i] = uniform_row
        return soft_qtype, soft_target

    def _fuse(self, outputs: Any, exec_out: ExecutorOutput) -> torch.Tensor:
        """Blend neural and symbolic log-probs: (1 - g)·log p_neural + g·log p_symbolic.

        Uses the model's per-sample ``gate_values`` under
        ``weighting_strategy="learned"``, else the fixed
        ``cfg.symbolic_weight``. Always log-space PoE.
        """
        log_p_neural: torch.Tensor = torch.log_softmax(outputs.answer_logits, dim=-1)
        log_p_symbolic: torch.Tensor = torch.log_softmax(exec_out.symbolic_logits + 1e-8, dim=-1)

        if self.cfg.weighting_strategy == "learned" and getattr(outputs, "gate_values", None) is not None:
            g = outputs.gate_values.squeeze(-1)  # (B,)
            fused: torch.Tensor = (1 - g).unsqueeze(1) * log_p_neural + g.unsqueeze(1) * log_p_symbolic
            return fused
        w = self.cfg.symbolic_weight
        fused_static: torch.Tensor = (1 - w) * log_p_neural + w * log_p_symbolic
        return fused_static
