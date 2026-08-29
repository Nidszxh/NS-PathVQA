"""Shared Product-of-Experts (PoE) fusion of neural and symbolic logits.

Single implementation used by ``train.py``, ``evaluate.py`` and the API layer;
keeping one copy avoids the device- and sizing-drift that three private copies
had already accumulated.

Fusion semantics (must stay log-space PoE — see CHANGELOG.md):

    log p = (1 - g) * log p_neural + g * log p_symbolic

with a static weight ``w = symbolic.symbolic_weight`` when
``weighting_strategy="static"`` and per-sample gate values ``g`` when
``weighting_strategy="learned"``.

A row is fused only when the symbolic path actually produced evidence. Rows
whose executor branch writes no evidence (count queries without a DSL
interpreter, identity queries whose target never appears as a region, ...) use
the neural distribution unchanged instead of being blended with a uniform or
all-zero symbolic distribution. ``trace["symbolic_used"]`` reports this
per-row decision.

Count queries are routed through the model's DSL interpreter; the
differentiable executor replaces the hard executor when the checkpoint config
selects it. When ``query_parser="learned"`` the hard executor consumes the
parser heads (argmax qtype, most probable region target) rather than silently
falling back to the regex parser.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from pathvqa.models.protocols import ExecutorOutput
from pathvqa.symbolic.dsl import DSLProgramCompiler
from pathvqa.symbolic.executor import (
    build_attribute_mappings,
    build_region_mapping,
    execute,
)
from pathvqa.symbolic.query_parser import Query, parse_question
from pathvqa.symbolic.routing import QTYPE_LIST, encode_question_types


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
        self.query_parser = getattr(self.cfg, "query_parser", "regex")
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

        batch_size = len(questions)
        regex_queries = [parse_question(q, self.answer_vocab) for q in questions]
        queries = self._effective_queries(outputs, regex_queries, questions)

        dsl_logits, dsl_used = self._run_dsl(outputs, queries)

        if self.diff_executor is not None:
            soft_qtype, soft_target = self._soft_routing(outputs, queries)
            diff_logits = self.diff_executor.execute(
                scene_logits=outputs.scene_head_dict(),
                soft_qtype=soft_qtype,
                soft_target=soft_target,
                yes_no_valid=self._yes_no_target_validity(queries),
            )
            exec_out = ExecutorOutput(
                symbolic_logits=diff_logits,
                region_logits=torch.zeros(0, device=outputs.answer_logits.device),
                # The differentiable executor always returns a dense blend;
                # the honest per-row decision is made below from the evidence.
                trace={"symbolic_used": [True] * batch_size},
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

        symbolic_logits = exec_out.symbolic_logits
        # Count queries are not executed by the hard executor (its count branch
        # is a no-op) and the differentiable executor's soft blend would leak
        # the other branches into them. Zero them first, then install DSL
        # evidence where the interpreter produced it.
        for i, query in enumerate(queries):
            if query.qtype == "count":
                symbolic_logits[i] = 0.0
                if dsl_used[i] and dsl_logits is not None:
                    symbolic_logits[i] = dsl_logits[i]

        symbolic_used = [bool(torch.count_nonzero(symbolic_logits[i]).item()) for i in range(batch_size)]
        fused = self._fuse(outputs, symbolic_logits, symbolic_used)

        trace: Dict[str, Any] = dict(exec_out.trace)
        trace["executor_used"] = list(exec_out.trace.get("symbolic_used", [False] * batch_size))
        trace["symbolic_used"] = symbolic_used
        trace["dsl_used"] = dsl_used
        trace["qtypes"] = [query.qtype for query in queries]
        return fused, trace

    # ------------------------------------------------------------------ queries

    def _effective_queries(
        self,
        outputs: Any,
        regex_queries: List[Query],
        questions: List[str],
    ) -> List[Query]:
        """Return the queries the symbolic path should actually execute.

        With ``query_parser="learned"`` the model's parser heads are
        authoritative: the interpreted qtype is the head's argmax and the
        target is its most probable region when that beats uniform chance.
        Attribute values have no learned head, so the regex attribute is kept
        (and a learned "attribute" decision without one falls back to the
        regex qtype instead of guessing "color"). The DSL program is
        recompiled so it matches the effective qtype rather than the regex one.
        """
        if self.query_parser != "learned":
            return regex_queries
        soft_qtype = getattr(outputs, "soft_qtype", None)
        if soft_qtype is None:
            return regex_queries

        soft_target = getattr(outputs, "soft_target", None)
        chance = 1.0 / max(self.num_regions, 1)
        effective: List[Query] = []
        for i, regex_query in enumerate(regex_queries):
            qtype = QTYPE_LIST[int(soft_qtype[i].argmax().item())]
            if qtype == "attribute" and not regex_query.attribute:
                qtype = regex_query.qtype
            target = regex_query.target
            if soft_target is not None and self.num_regions > 0:
                prob, region_idx = soft_target[i].max(dim=0)
                null_probability = getattr(outputs, "target_null_probability", None)
                if null_probability is not None:
                    if float(null_probability[i].item()) >= float(prob.item()):
                        target = ""
                    else:
                        target = self.region_names[int(region_idx.item())]
                elif float(prob.item()) > max(chance + 0.05, 0.5):
                    # Compatibility for older learned-target checkpoints with
                    # no explicit abstain class: accept only a confident target.
                    target = self.region_names[int(region_idx.item())]
                elif target.lower() not in {name.lower() for name in self.region_names}:
                    target = ""
            program = DSLProgramCompiler.compile(
                questions[i],
                target=target,
                attribute=regex_query.attribute,
                qtype=qtype,
            )
            effective.append(
                Query(
                    qtype=qtype,
                    target=target,
                    attribute=regex_query.attribute,
                    program=program,
                )
            )
        return effective

    def _run_dsl(self, outputs: Any, queries: List[Query]) -> Tuple[Optional[torch.Tensor], List[bool]]:
        """Execute count programs and report which rows got DSL evidence."""
        batch_size = len(queries)
        if outputs.patch_features is None or self.dsl_interpreter is None:
            return None, [False] * batch_size

        # Keep DSL log evidence in float32. In particular, the finite mask
        # floor used by the interpreter must not be cast to fp16 -1e9 (−inf),
        # which would make gold answers outside a DSL candidate set impossible
        # under NLL training.
        dsl_logits = torch.zeros(
            batch_size,
            self.vocab_size,
            device=outputs.answer_logits.device,
            dtype=torch.float32,
        )
        dsl_used = [False] * batch_size
        for i, query in enumerate(queries):
            if query.qtype != "count" or query.program is None:
                continue
            # Slice the scene heads to the same row as the patch features.
            # Passing the full batch while executing one row makes the
            # interpreter's attribute branch mix row i with rows 0..B-1.
            scene_row = {
                key: (value[i : i + 1] if value is not None else None)
                for key, value in outputs.scene_head_dict().items()
            }
            dsl_out = self.dsl_interpreter(
                query.program,
                outputs.patch_features[i : i + 1],
                scene_row,
                self.answer_to_idx,
            )
            row = dsl_out[0]
            # The interpreter returns all zeros when the program's target has no
            # answer-vocabulary slot; that row must stay neural-only rather than
            # being fused with a uniform distribution.
            if bool(torch.count_nonzero(row).item()):
                dsl_logits[i] = row
                dsl_used[i] = True
        return dsl_logits, dsl_used

    # ------------------------------------------------------------------ fusion

    def _yes_no_target_validity(self, queries: List[Query]) -> torch.Tensor:
        """Mask marking yes/no rows whose target names an actual region.

        The scene parser only has presence heads for its region vocabulary.
        A non-region target (e.g. "malignant") must stay neural-only rather
        than being answered with a proxy "is any organ present?" score.
        """
        regions = {name.lower() for name in self.region_names}
        flags = [bool(q.target) and q.target.lower() in regions for q in queries]
        return torch.tensor(flags, dtype=torch.bool, device=self.device)

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

    def _fuse(
        self,
        outputs: Any,
        symbolic_logits: torch.Tensor,
        symbolic_used: List[bool],
    ) -> torch.Tensor:
        """Blend neural and symbolic log-probs: (1 - g)·log p_neural + g·log p_symbolic.

        Uses the model's per-sample ``gate_values`` under
        ``weighting_strategy="learned"``, else the fixed
        ``cfg.symbolic_weight``. Always log-space PoE. Rows without symbolic
        evidence keep the neural distribution untouched.
        """
        log_p_neural: torch.Tensor = torch.log_softmax(outputs.answer_logits, dim=-1)
        if not any(symbolic_used):
            return log_p_neural

        log_p_symbolic: torch.Tensor = torch.log_softmax(symbolic_logits + 1e-8, dim=-1)

        if self.cfg.weighting_strategy == "learned" and getattr(outputs, "gate_values", None) is not None:
            g = outputs.gate_values.squeeze(-1)  # (B,)
        else:
            g = torch.full(
                (log_p_neural.size(0),),
                float(self.cfg.symbolic_weight),
                device=log_p_neural.device,
                dtype=log_p_neural.dtype,
            )

        fused = (1 - g).unsqueeze(1) * log_p_neural + g.unsqueeze(1) * log_p_symbolic
        used = torch.tensor(symbolic_used, dtype=torch.bool, device=log_p_neural.device).unsqueeze(1)
        # A weighted sum of normalized log-probabilities is an unnormalized
        # product of experts. Normalize its partition function before passing
        # the result to NLLLoss or exposing it as log probabilities.
        fused = fused - torch.logsumexp(fused, dim=-1, keepdim=True)
        return torch.where(used, fused, log_p_neural)
