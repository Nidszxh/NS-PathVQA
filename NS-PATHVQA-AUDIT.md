# NS-PathVQA: Neuro-Symbolic Architecture Audit & Research Roadmap

> **What this file owns:** the architecture critique — strengths, limitations, and code-level findings grounded in direct inspection — plus the ranked research proposal set (P1–P11), design alternatives that were considered, and the publication strategy. It supersedes two earlier standalone audit passes and folds in the design-alternatives/publication-context material that used to live in a separate `IMPLEMENT.md`. It does **not** hold the execution order (that's `TODO.md` — items here are *proposals*, not yet scheduled unless `TODO.md` says so), novelty claims (`NOVELTIES.md`), or the current-system reference (`PROJECT.md`). If this file's proposals conflict with what `TODO.md` has already scheduled, **`TODO.md` wins**.
>
> **Evaluation role:** Principal Research Scientist & Senior Systems Architect (Neuro-Symbolic AI & Multimodal Medical Vision). Findings reference specific files/line numbers where known; all proposals are actionable against the current codebase.

---

## 1. Executive Summary

### Current state

NS-PathVQA is a **functionally complete neuro-symbolic VQA system** for H&E histopathology. The training-evaluation-serving pipeline is end-to-end: data → PLIP + DistilBERT (both frozen + LoRA) → cross-modal attention → dual neural/symbolic inference paths → gated fusion → classification. The system trains on a single consumer GPU (8 GB VRAM, ~42 min/run), achieves **53.79% overall validation accuracy** at the Phase-1 baseline (82.4% yes/no, 66.0% location, 28.6% count, 19.3% identity) and **60.33% val / 59.87% test** after the full PLIP+DistilBERT+LoRA+gate upgrade, and ships a FastAPI server and Docker container.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     SYSTEM HEALTH SCORECARD                                        │
├────────────────────────────┬─────────────────────────────────┬────────────────────────────────────┤
│ Engineering & Pipeline     │ 9.0 / 10                        │ Clean caching, AMP, EMA, PyTest,   │
│ Production Readiness       │                                 │ FastAPI + Docker served.           │
├────────────────────────────┼─────────────────────────────────┼────────────────────────────────────┤
│ Computational Efficiency   │ 9.5 / 10                        │ Trains in ~0.38 GPU-h on single    │
│                            │                                 │ RTX 4060 (8GB VRAM) @ 59.87% test. │
├────────────────────────────┼─────────────────────────────────┼────────────────────────────────────┤
│ Neuro-Symbolic Integration │ 5.5 / 10                        │ Convex logit blending rather than  │
│ Mathematical Depth         │                                 │ differentiable formal reasoning.   │
├────────────────────────────┼─────────────────────────────────┼────────────────────────────────────┤
│ Domain Pathology Alignment │ 4.5 / 10                        │ Web-domain CLIP ViT-B/32; global   │
│                            │                                 │ 224px crop erases morphometry.     │
└────────────────────────────┴─────────────────────────────────┴────────────────────────────────────┘
```

### Genuine strengths

| Strength | Evidence |
|---|---|
| **Lightweight deployability** | 11.9M trainable params, 1×RTX 4060, batch 32, AMP+uint8 cache |
| **End-to-end pipeline & reproducibility** | Clean separation of concerns: data loading, model construction, training loop, evaluation, FastAPI serving |
| **Principled gated fusion (C1)** | `LearnedGate`: `g = σ(MLP([h_attn ‖ c_scene ‖ qtype_onehot]))` in `routing.py` |
| **Differentiable LTN constraints (C2)** | Product/Łukasiewicz t-norm clause satisfaction in `ltn.py` |
| **Ontology-grounded region discovery (C4)** | `AnatomicalOntology` DAG + SNOMED-CT hierarchy in `dataset_adapter.py` |
| **Architecture-grounded failure taxonomy (C7)** | Perception / Parsing / Execution buckets in `interpretability.py` |
| **Publication-grade evaluation** | ECE, temperature scaling, paired bootstrap, per-qtype breakdown |
| **Resource-constrained engineering** | Parameter-efficient LoRA (rank 16/32), gradient accumulation, FP16 mixed precision, EMA shadowing, atomic uint8 disk-caching |
| **Reproducibility discipline** | Fixed seeds, `--debug` fast path, smoke tests, pinned deps |

### Critical limitations (the honest story)

| Limitation | Root cause | Severity |
|---|---|---|
| **Symbolic path is post-hoc, not differentiable end-to-end** | `execute()` uses `scatter_add_` on static index mappings; SceneParser is not trained on symbolic supervision signals | **Critical** |
| **Counting is entirely neural (accuracy: 28.6%)** | `execute()` explicitly marks `count` as non-symbolic; `DifferentiableDSLInterpreter` is not wired into training | **High** |
| **SceneParser has no domain-specific initialization** | 5 linear heads over generic `attended_features` — no histopathology concept priors | **High** |
| **Question parser is purely lexical/regex** | `parse_question()` has no semantic understanding; e.g. "what do the cells in the image look like?" fails all patterns → fallback identity | **High** |
| **Perceptual domain mismatch** | General-domain OpenAI CLIP ViT-B/32, 224×224, discards nuclear atypia, chromatin patterns, mitotic counts, micro-architectural boundaries | **High** |
| **Yes/No logic (historical)** | Older executor versions boosted the raw predicate word (`"malignant"`) rather than binary `"yes"`/`"no"` classes. **Fixed** — the current `VectorizedSymbolicExecutor` does binary yes/no truth mapping (see `TODO.md`/`NOVELTIES.md` — this is the shipped state, kept here for historical context). |
| **Gate lacks a declared input** | `NOVELTIES.md` C1 claims conditioning on attention entropy `ent(attn)`, but `LearnedGate.forward()` takes only `[h_attn ‖ c_scene ‖ qtype_onehot]` — entropy is absent | **Medium** |
| **LTN is enabled by default** | `SymbolicConfig.ltn_enabled = True`; 2 clauses — no pathology-specific axioms yet | **Medium** |
| **~14% of val/test answers map to `<UNK>`** | Closed-vocabulary classifier; novel/rare histological terms are systematically wrong | **Medium** |
| **DSL interpreter is standalone / not wired** | `DifferentiableDSLInterpreter` in `dsl.py` is never called from `train.py` | **Medium** |
| **`QueryAttr` executor is broken** | `dsl.py`: `if ans in program.args.get("target", "")` — wrong substring logic | **Low–Medium** |
| **Faithfulness test is not faithful** | `faithfulness_test()` perturbs `region_logits` but calls `model()` without injecting the perturbation | **Medium** |
| **No calibration of symbolic logits** | `symbolic_weight=0.3` is hand-tuned; `scatter_add_` symbolic logit scale is unbounded | **Low–Medium** |
| **Single-block cross-attention** | `CrossModalTransformer` is one MHA block with no depth; CLS discarded from question encoder | **Low** |
| **Brittle question parsing** | Regex in `query_parser.py` cannot handle complex clinical queries, multi-clause conditionals, anatomical synonyms, or compound negation | **High** |

---

## 2. Architectural Map

### 2.1 Data flow (as implemented)

```
PathVQA HF Dataset
  └─ PIL Image → resize_keep_aspect(224,224) → uint8 cache → RandAugment (train) → CLIP normalize → float32 (3,224,224)
  └─ question string → DistilBERT tokenizer → (input_ids, attention_mask)
  └─ answer string → normalize_text() → lookup_answer_idx() → int target

FORWARD PASS
──────────────────────────────────────────────────────────────────────
CLIPViTEncoder(images)
  ├─ CLIPVisionModel [frozen] → last_hidden_state[:, 1:, :] → (B, 49, 768)
  ├─ LoRA adapters on {q,k,v,out_proj,fc1,fc2}             [trainable]
  └─ Linear(768→512)                                         [trainable]
  → visual["features"] (B, 49, 512), visual["mask"] (B, 49) all-ones

DistilBERTQuestionEncoder(input_ids, attention_mask)
  ├─ DistilBERT [frozen] → last_hidden_state[:,0,:] → (B, 768)
  └─ LoRA adapters on {q_lin,k_lin,v_lin,out_lin}           [trainable]
  → q["question_state"] (B, 768)

CrossModalTransformer(q_state, visual["features"], visual["mask"])
  ├─ Q = Linear(768→512)(q_state).view(B,1,8,64)
  ├─ K = Linear(512→512)(visual_feats).view(B,49,8,64)
  ├─ V = Linear(512→512)(visual_feats).view(B,49,8,64)
  ├─ scores = QK^T/√64; masked_fill(~mask, -inf); softmax
  ├─ attended = scores @ V → reshape(B,512) → out_proj → LayerNorm → + FFN
  → attended (B, 512)

NEURAL PATH
  fused = cat([attended, q_state], dim=-1)                  → (B, 1280)
  neural_logits = MLP(1280→512→ReLU→Dropout→answer_vocab)   → (B, V)

SYMBOLIC PATH [if symbolic_enabled]
  SceneParser(attended):                                     [trainable]
    region_logits   = Linear(512→N_reg)(attended)            → (B, N_reg)
    object_presence = sigmoid(Linear(512→N_reg)(attended))   → (B, N_reg)
    color_logits    = Linear(512→17)(attended)               → (B, 17)
    shape_logits    = Linear(512→9)(attended)                → (B, 9)
    size_logits     = Linear(512→11)(attended)                → (B, 11)

  QueryParser [per-question, CPU]:
    parse_question(q_str) → Query(qtype, target, attribute, program)

  Executor.execute() [vectorized, GPU]:
    qtype=identity/location → scatter_add(region_logits, reg_map, symbolic_logits)
    qtype=yes_no            → symbolic_logits[yes/no] = ±r_conf
    qtype=attribute         → symbolic_logits[attr_vocab] = attr_logits[attr_idx]
    qtype=count             → NO-OP (returns zeros)
    → symbolic_logits (B, V)

FUSION (train._compute_symbolic_logits)
  weighting_strategy="static":  combined = neural_logits + 0.3 * symbolic_logits
  weighting_strategy="learned":
    LearnedGate(attended, c_scene, qtype_onehot) → g (B,1)
    combined = (1-g) * neural_logits + g * symbolic_logits

LOSS
  L_ce  = CrossEntropyLoss(combined, targets)
  L_ltn = 1 - satisfaction(region_object_coherence, attribute_certainty)  [if ltn_enabled]
  L     = L_ce + ltn_weight * L_ltn
```

### 2.2 Critical interface point (neural → symbolic boundary)

```
attended (B, 512) ──→ SceneParser ──→ scene logits Dict[str, Tensor]
                                              │
                              (no gradient flows through executor)
                                              │
queries: List[Query] ──────────────────→ execute() ──→ symbolic_logits (B, V)
         (parsed on CPU per step)              │
                                      scatter_add_ on static
                                      precomputed index buffers
```

> **CRITICAL:** The neuro-symbolic interface is a **gradient dead end**. `execute()` applies index scatter operations over pre-computed `region_to_vocab` mappings (fixed integers derived from vocabulary overlap, not learned correspondences). No gradient flows from the symbolic output back through `execute()` to the `SceneParser`. This means the SceneParser is never trained *to be symbolic* — it is only trained via the `L_ce` signal on `combined`, which is dominated by the neural path. This is the root cause of the low identity and count accuracy figures. Fixing it is the entire motivation for **P1** below and for the planned differentiable pipeline (`NOVELTIES.md` C9).

### 2.3 Mathematical formulation of current fusion

$$\mathbf{z}_{\text{neural}} = \mathbf{W}_2 \operatorname{ReLU}\left(\mathbf{W}_1 [\mathbf{h}_{\text{attn}} \mathbin{\Vert} \mathbf{q}_{\text{state}}] + \mathbf{b}_1\right) + \mathbf{b}_2$$

$$\mathbf{z}_{\text{sym}}[k] = \begin{cases}
\mathbf{r}[j] & \text{if } \text{qtype} \in \{\text{identity}, \text{location}\} \land \operatorname{map}(j) = k \\
\max_j \mathbf{r}[j] & \text{if } \text{qtype} = \text{yes\_no} \land \operatorname{target\_idx} = k \\
\mathbf{a}_{\text{attr}}[m] & \text{if } \text{qtype} = \text{attribute} \land \operatorname{attr\_map}(m) = k \\
0 & \text{otherwise}
\end{cases}$$

$$g = \sigma\left(\mathbf{W}_g [\mathbf{h}_{\text{attn}} \mathbin{\Vert} \max_j c_j \mathbin{\Vert} \mathbf{e}_{\text{qtype}}] + \mathbf{b}_g\right)$$

$$\mathbf{z}_{\text{final}} = (1 - g) \cdot \mathbf{z}_{\text{neural}} + g \cdot \mathbf{z}_{\text{sym}}$$

---

## 3. Design Alternatives Considered

Before committing to the shipped design, the following alternatives were evaluated for each major subsystem. This table is the quick-reference; the ranked proposals in §4 give full mechanism detail for the "parked" rows that are still live candidates.

### 3.1 Fusion mechanisms

The fixed additive `neural + w·symbolic` was the original approach.

| Option | Description | Status |
|---|---|---|
| **A — Confidence-gated fusion** | `g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype)]))`, `final = (1-g)·neural + g·symbolic` | **Shipped** (C1, `TODO.md` Phase 5) |
| **B — Probabilistic logic fusion (Product-of-Experts)** | `P(a) ∝ P_neural(a)^(1-g) · P_symbolic(a)^g`, i.e. log-prob space blending | Parked — see **P5** below |
| **C — Differentiable program execution** | Compile parsed query into differentiable program over scene facts (NS-CL / Neural Module Networks style) | Parked — see **P1**, **P9**, and `NOVELTIES.md` C9 |

Option A was prioritized: cheapest to implement, neutralizes the most obvious reviewer weakness, and the additive mode is retained as the ablation baseline behind `symbolic.weighting_strategy`.

### 3.2 Region discovery

Two paths to formalize region discovery beyond prefix matching:

| Approach | Description | Status |
|---|---|---|
| **UMLS/RadLex linkage** | Map candidate region tokens to canonical anatomical ontology at build time | **Shipped** (C4, `TODO.md` Phase 8) |
| **Unsupervised clustering** | Cluster SceneParser region-logit embeddings, assign anatomical labels post-hoc | Parked (future work) |

### 3.3 Question parsing

The current regex-based `parse_question()` is deliberately simple. A formal DSL / functional-program parser (CLEVR/NSVQA-style) was scoped:

- Question → typed program tree (`Query(Filter(Scene, region="lung"), attr="color")`)
- DSL primitives: `Filter`, `Relate`, `Count`, `Compare`, `Query`, `Exist`
- Executor rewritten as program interpreter over scene graph

**Status:** Parked (see `NOVELTIES.md` C9, `NOVELTIES.md` C5 for the standalone DSL compiler that already exists). Count questions remain neural-only until real counting support is wired in (**P3** below).

### 3.4 Scene graph

A typed knowledge graph with `RegionNode`/`AttributeNode` nodes and `has_attribute`/`spatial_relation`/`part_of` edges was scoped as the substrate for a DSL executor.

**Status:** Parked — see **P9** (GAT scene graph) below. The pipeline currently passes tensors directly from SceneParser to Executor, with no explicit graph structure.

---

## 4. Ranked Technical Proposals

*Proposals are ranked by **Impact × Feasibility** (Impact: 1–5, Effort: 1 = low, 5 = high).*

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RESEARCH PROPOSALS IMPACT MATRIX                                    │
│                                                                                                       │
│   HIGH  │   [P6] Pathology-Specific Foundation Backbones   [P1] Symbolic Auxiliary Supervision        │
│         │   [P3] DSL Soft-Counting Wiring                  [P9] GAT Scene Graph / DSL Program Synth.  │
│ I M P A │                                                                                             │
│ C T     │                                                                                             │
│         │   [P7] Ontology-Conditioned Executor              [P5] Product-of-Experts Fusion            │
│   LOW   │   [P11] Cross-Dataset Symbolic Transfer            [P8] Conformal Prediction Routing         │
│         └─────────────────────────────────────────────────────────────────────────────────────────────┘
│           LOW                                  COMPLEXITY / EFFORT                                HIGH│
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### P1 — End-to-End Differentiable Symbolic Supervision [Impact: 5 | Effort: 3] — [x] DONE

**Problem:** SceneParser's 5 linear heads are trained only via the final CE loss on `combined` logits. The symbolic path contributes ~30% statically, so the gradient signal for SceneParser to learn meaningful anatomy is severely diluted.

**Proposal:** Introduce **symbolic auxiliary losses** providing direct supervision for each scene parser head.

**Mathematical formulation:**

For a training sample with answer `a`:
- `r_idx(a)` → the region index for `a` (via `region_to_vocab`)
- `color_idx(a)` → the color attribute index for `a` (via `attr_map_color`)

```
L_region = CrossEntropyLoss(region_logits, r_idx(a))        for identity/location questions
L_attr   = CrossEntropyLoss(attr_logits, attr_idx(a))       for attribute questions
L_yn     = BCELoss(object_presence[:, r_idx(target)], yn)   for yes/no questions

L_total  = L_ce + λ_r * L_region + λ_a * L_attr + λ_yn * L_yn + λ_ltn * L_ltn
```

**Implementation sketch:**

```python
# In train.py, train_epoch() — after _compute_symbolic_logits():
aux_losses = compute_symbolic_aux_losses(
    scene_logits=outputs,
    queries=queries,
    targets=targets,
    region_to_vocab=self.region_to_answer_idx,
    answer_to_idx=self.answer_to_idx,
)
loss = loss + 0.5 * aux_losses["region"] + 0.3 * aux_losses["attr"] + 0.2 * aux_losses["yn"]
```

**Expected impact:** SceneParser learns genuine anatomy-discriminative features. Symbolic path accuracy (tracked as `sym_acc` in `validate()`) should rise significantly, directly improving yes/no, location, and identity subtypes.

---

### P2 — Fix the Faithfulness Test (Critical Bug) [Impact: 4 | Effort: 1] — [x] DONE

**Problem:** `faithfulness_test()` in `interpretability.py` perturbs `region_logits` then calls `model(images, input_ids, attention_mask)` — the model re-computes scene logits from scratch. The perturbation is never seen by the model, making the faithfulness score always 0.

**Fix:** Pass perturbed scene logits directly into the executor without re-running the model:

```python
def faithfulness_test(model, images, input_ids, attention_mask,
                      region_names, executor, answer_to_idx, top_k=3):
    model.eval()
    with torch.no_grad():
        outputs = model(images, input_ids, attention_mask)
        baseline_logits, _ = compute_symbolic_logits(outputs, ...)
        baseline_pred = baseline_logits.argmax(-1)

    results = []
    for region_idx in outputs["scene_region_logits"][0].topk(top_k).indices:
        # Perturb the scene logits dict directly — do NOT re-run model
        perturbed = {k: v.clone() for k, v in outputs.items()}
        perturbed["scene_region_logits"][0, region_idx] = -1e9
        with torch.no_grad():
            perturbed_logits, _ = compute_symbolic_logits(perturbed, ...)
            perturbed_pred = perturbed_logits.argmax(-1)
        results.append({"changed": (perturbed_pred != baseline_pred).item()})
    return results
```

---

### P3 — Wire DSL Interpreter into Count Path [Impact: 4 | Effort: 2] — [x] DONE

**Problem:** Count questions score 28.6% — the worst subtype. `DifferentiableDSLInterpreter` with soft counting is implemented in `dsl.py` (`NOVELTIES.md` C5) but never called. `execute()` marks count as non-symbolic, returning zero logits.

**Key architectural constraint to fix first:** `DifferentiableDSLInterpreter.forward()` expects `patch_features (B, N, D)` but `CrossModalTransformer` outputs `attended (B, 512)` — a single aggregated vector. For soft counting, you must pass `visual["features"] (B, 49, 512)` directly, bypassing the attention aggregation for count queries.

**Fix:**

```python
# In NeuroSymbolicPathVQA.forward() — expose patch features in outputs:
result["patch_features"] = visual["features"]  # (B, 49, 512)

# In train._compute_symbolic_logits() — route count queries:
from symbolic.dsl import DifferentiableDSLInterpreter

for i, q in enumerate(queries):
    if q.qtype == "count" and q.program is not None:
        dsl_out = self.dsl_interpreter(
            q.program,
            outputs["patch_features"][i:i+1],  # (1, 49, 512)
            outputs,
            self.answer_to_idx
        )
        symbolic_logits[i] = dsl_out[0]
```

**Also fix `QueryAttr` broken logic:**
```python
# WRONG: if ans in program.args.get("target", ""):
# RIGHT: scatter attr_logits to matching vocab indices via attr_map_tensor
```

---

### P4 — Add Attention Entropy to Gate (Close Novelty Gap) [Impact: 3 | Effort: 1] — [x] DONE

**Problem:** `NOVELTIES.md` C1 claims the gate conditions on `ent(attn)`. The current `LearnedGate.forward()` takes only `[h_attn ‖ c_scene ‖ qtype_onehot]` — entropy is entirely absent. This is a paper-submission-blocking inconsistency.

**Implementation (two steps):**

Step 1 — Return attention weights from `CrossModalTransformer.forward()` (currently computed but discarded):

```python
# fusion.py
def forward(self, question_state, visual_features, mask):
    ...
    weights = torch.softmax(scores, dim=-1)   # (B, H, 1, 49)
    attended = ...
    return attended + self.ffn(attended), weights  # return weights
```

Step 2 — Add entropy as gate input in `routing.py`:

```python
def attn_entropy(weights: torch.Tensor) -> torch.Tensor:
    p = weights.mean(dim=1).squeeze(2)                    # (B, 49)
    return -(p * (p + 1e-8).log()).sum(-1, keepdim=True)  # (B, 1)

class LearnedGate(nn.Module):
    def __init__(self, visual_dim=512, num_regions=50, num_qtypes=5, dropout=0.1):
        gate_input_dim = visual_dim + num_regions + num_qtypes + 1  # +1 for entropy
        ...

    def forward(self, h_attn, c_scene, qtype_onehot, attn_weights=None):
        ent = attn_entropy(attn_weights) if attn_weights is not None \
              else torch.zeros(h_attn.size(0), 1, device=h_attn.device)
        gate_input = torch.cat([h_attn, c_scene, qtype_onehot, ent], dim=-1)
        return torch.sigmoid(self.gate_mlp(gate_input))
```

---

### P5 — Probabilistic Logic Fusion: Product-of-Experts [Impact: 4 | Effort: 2] — [x] DONE

**Problem:** Current fusion `neural + 0.3 * symbolic` is additive in logit space. This is incoherent: `scatter_add_` symbolic logits have arbitrary scale while neural logits are trained with cross-entropy. The two scales are incompatible; `symbolic_weight=0.3` has no principled justification.

**Proposal:** Implement fusion in **log-probability space (Product of Experts)** — this is "Option B" from §3.1:

```
P_final(a | I, q) ∝ P_neural(a | I, q)^(1-g) · P_symbolic(a | I, q)^g

log P_final = (1-g) · log P_neural + g · log P_symbolic + C
```

```python
# In _compute_symbolic_logits():
log_p_neural   = torch.log_softmax(neural_logits, dim=-1)
log_p_symbolic = torch.log_softmax(symbolic_logits + 1e-8, dim=-1)

if weighting_strategy == "learned":
    g = gate.squeeze(-1)                                    # (B,)
    log_p_combined = (1 - g).unsqueeze(1) * log_p_neural \
                   + g.unsqueeze(1) * log_p_symbolic
else:
    w = self.config.symbolic.symbolic_weight
    log_p_combined = (1 - w) * log_p_neural + w * log_p_symbolic

# NLL loss in log-prob space:
loss = F.nll_loss(log_p_combined, targets)
```

**Expected impact:** Eliminates logit scale mismatch. PoE is well-motivated theoretically (Hinton 2002) and proven in VQA fusion literature.

---

### P6 — Pathology-Specific PLIP Backbone + Nuclei-Aware Head [Impact: 5 | Effort: 3] — [x] DONE

**Problem:** CLIP ViT-B/32 was pretrained on natural image-text pairs. Histopathology images have fundamentally different texture statistics: nuclear morphology, staining variation (H&E), glandular architecture. CLIP is domain-mismatched — it lacks understanding of H&E staining characteristics, nuclear pleomorphism, and tissue histology.

**Candidate pathology foundation models:**
1. **CONCH** (Nature Medicine 2024): Vision-language foundation model trained on over 1.17M pathology image-text pairs.
2. **PLIP** (Pathology Language-Image Pretraining, Nature Medicine 2023): Open-access CLIP architecture trained on 208K medical pathology Twitter/OpenPath cases.
3. **UNI / Prov-GigaPath** (Nature Medicine 2024 / Nature 2024): Giant 1B-parameter vision encoder for gigapixel WSI representation.

**Expected impact:** 6–10% immediate boost on raw perception and zero-shot medical term alignment without altering the symbolic interface.

**Proposal (three stages):**

**Stage 1 — Default to PLIP:** The config already supports `backbone="plip"` (`VisualConfig.__post_init__` swaps `model_name` to `"vinid/plip"`). Simply change the default:

```python
# config.py — VisualConfig defaults:
backbone: str = "plip"
model_name: str = "vinid/plip"
```

PLIP is CLIP fine-tuned on 208K histopathology image-caption pairs. Requires no new code.

Drop-in integration blueprint for `visual_encoder.py` (any PLIP/CONCH-style HF model):

```python
from transformers import AutoModel, AutoProcessor

class PathologyVisionEncoder(nn.Module):
    """Specialized pathology visual encoder using PLIP/CONCH."""
    def __init__(self, model_id: str = "vinid/plip", num_object_features: int = 512):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        # Freeze backbone, train only LoRA adapters
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.vision_config.hidden_size # 768
        self.patch_projection = nn.Linear(hidden_dim, num_object_features)

    def forward(self, images: torch.Tensor) -> dict:
        vision_outputs = self.backbone.vision_model(pixel_values=images)
        # Extract spatial grid tokens (exclude CLS)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        features = self.patch_projection(patch_tokens)
        mask = torch.ones(images.size(0), features.size(1), dtype=torch.bool, device=images.device)
        return {"features": features, "mask": mask}
```

**Stage 2 — Nuclei density auxiliary head on SceneParser:**

```python
# scene_parser.py
NUCLEI_DENSITY_BINS = ["sparse", "moderate", "dense", "crowded"]

class SceneParser(nn.Module):
    def __init__(self, visual_dim, num_regions):
        ...
        self.nuclei_density = nn.Linear(visual_dim, len(NUCLEI_DENSITY_BINS))

    def forward(self, attended_features):
        return {
            ...,  # existing heads
            "nuclei_density_logits": self.nuclei_density(attended_features),
        }
```

Ground truth: discretize count answers from PathVQA count questions. This gives the visual encoder a domain-specific learning signal without external annotation.

**Stage 3 — Enable `MultiScaleVisualEncoder`:** implemented (`NOVELTIES.md` C6) but defaults to `use_multiscale=False`. Activate it. Quadrant crops at 224px emulate 2× zoom transitions (20×→40× magnification) critical for nuclei-level morphology questions.

**Multi-scale mechanism (target architecture):**

```
   Original High-Res H&E Slide
      ┌─────────────────────────┐
      │  [Global Downsample]    │ ───> Global Patch Tokens (49 × 512)
      │  [Local Micro-Crops]    │ ───> Local High-Res Tokens (4 × 49 × 512)
      └─────────────────────────┘                    │
                                       Cross-Scale Gated Attention
                                                     │
                                       Unified Morphometric Tokens (98 × 512)
```

---

### P7 — Medical Ontology Integration into Symbolic Inference [Impact: 4 | Effort: 3] — [x] DONE

**Problem:** `AnatomicalOntology` in `dataset_adapter.py` is used only for region vocabulary extraction (C4). It is **not used at inference time**. The executor does not know that "gastric" and "stomach" are synonymous, or that "glomerulus" is part of "renal cortex" which is part of "urinary system." Synonym failures are a major source of location question errors.

**Proposal: Ontology-conditioned executor with hierarchical match relaxation:**

```python
# executor.py — add ontology-aware fallback for identity/location queries
def _ontology_fallback(query, ontology, region_names, answer_to_idx):
    """If exact target not in region_names, try synonym expansion and parent traversal."""
    if not query.target:
        return None
    norm = ontology.normalize_term(query.target)
    # Check synonyms
    if norm in answer_to_idx:
        return answer_to_idx[norm]
    # Check parent systems
    for parent in ontology.get_parent_systems(norm):
        if parent in answer_to_idx:
            return answer_to_idx[parent]
    return None
```

**Knowledge graph embedding regularizer** (for training time):

```
L_onto = Σ_{(r1, r2) ∈ siblings} max(0, δ - cos(W_region[r1], W_region[r2]))
```

where `r1`, `r2` are region classifier weight vectors for organs in the same ontological system. This encourages `SceneParser` to cluster anatomically related regions together in its embedding space.

When the model answers `"Where is the lesion located?"`, predictions for sub-structures (e.g., `"renal cortex"`) should automatically propagate credit to higher-level concepts (`"kidney"`, `"urinary system"`) via transitive entailment: `"Glomerulonephritis" ⟹ IsA("Kidney Disease") ⟹ LocatedIn("Urinary System")`.

---

### P8 — Conformal Prediction for Uncertainty-Aware Routing [Impact: 4 | Effort: 4] — [x] DONE

**Problem:** `compute_uncertainty()` in `metrics.py` computes max-softmax and entropy, but these estimates are **never used for inference-time routing**. High-uncertainty symbolic predictions blindly override neural predictions.

**Proposal:** Implement **split conformal prediction** for coverage-guaranteed prediction sets:

```
Calibration phase (on val set):
  s_i = 1 - P(y_true | x_i)        # nonconformity score
  τ   = quantile(s_1,...,s_n, (1+1/n)(1-α))  # e.g., α=0.1 → 90% coverage

Inference:
  C(x) = {a : softmax(logits)[a] ≥ 1 - τ}
  |C(x)| = 1    → high confidence, return prediction
  |C(x)| > k    → uncertain; escalate to symbolic path or flag for review
  "yes" ∈ C(x) AND "no" ∈ C(x) → for yes/no, use symbolic object_presence sigmoid
```

This provides **formal coverage guarantees** (a clinically meaningful property) and costs only a scalar threshold computed on the val set.

---

### P9 — Structured Scene Graph with Typed Relational Edges [Impact: 5 | Effort: 5] — [ ] NOT STARTED

**Problem:** The current "scene graph" is 5 independent linear classifiers over a single aggregated vector. No *relationships* between entities are represented. "Is the lesion in the lung?" requires joint reasoning over the lesion entity and its spatial relationship to lung tissue. This explains the dominant failure mode for location questions.

**Proposal:** Replace `SceneParser` with a lightweight **Graph Attention Network (GAT)** scene graph:

```
Nodes: V_regions ∪ V_attributes ∪ V_objects
Node features: k-means cluster centroids of visual patch tokens (B, 49, 512)

Edge types:
  has_attribute(region, color/shape/size)   ← from attribute logit correlation
  spatially_adjacent(r1, r2)                ← from patch grid topology
  part_of(organ, system)                    ← from AnatomicalOntology (hard, no params)
  co_occurs(e1, e2)                         ← from training co-occurrence statistics

Message passing: 2-layer GAT
Output: enriched node embeddings → symbolic logits with relational context
```

Ontology-defined `part_of` edges are hard constraints (no learnable parameters), making the graph structure interpretable by construction. The GAT weights learn which relational patterns predict correct answers.

---

### P10 — DeepProbLog-Style Probabilistic Logic Programs [Impact: 5 | Effort: 5] — [ ] NOT STARTED

**Problem:** The symbolic engine is a deterministic rule applier with no uncertainty representation. It cannot express "lung OR bronchus → respiratory system" as a probabilistic disjunction, nor handle noisy neural predicate outputs gracefully.

**Proposal:** Encode executor rules as **probabilistic logic programs** with neural predicate weights:

```prolog
% Neural predicate: scene parser output as soft fact
nn(region_clf, [X], Region, [lung, liver, heart, ...]) :: region(X, Region).

% Deterministic rules (from ontology)
system(X, respiratory) :- region(X, lung).
system(X, respiratory) :- region(X, bronchus).
system(X, hepatobiliary) :- region(X, liver).

% Yes/no: existential verification
answer(X, yes) :- yes_no_question(X), system(X, _Target).
answer(X, no)  :- yes_no_question(X), \+ system(X, _Target).

% Identity grounding
answer(X, Sys) :- identity_question(X), system(X, Sys).
```

Gradients flow via the semiring (WMC) computation. Implementation path: use the `problog` Python library or implement a mini semiring over the relevant rule tree.

**Start small:** implement only the yes/no and identity rules (2 rule families) as a proof-of-concept. The architecture already provides the neural predicate values via `SceneParser`.

---

### P11 — Cross-Dataset Symbolic Transfer (VQA-RAD, SLAKE) [Impact: 4 | Effort: 3] — [ ] NOT STARTED

**Problem:** `VQARADAdapter` and `SLAKEAdapter` are implemented in `dataset_adapter.py` (C4) but neither has been trained or evaluated. The claimed "cross-dataset transfer" has no experimental evidence.

**Proposal — Symbolic knowledge transfer protocol:**

```
Phase 1: Train NS-PathVQA on PathVQA to convergence (full symbolic pipeline)
Phase 2: Freeze [SceneParser weights + Executor index buffers]
         Fine-tune only [CLIPViTEncoder LoRA + CrossModalTransformer + gate]
         on VQA-RAD (radiology) and SLAKE (bilingual)
Phase 3: Report per-dataset accuracy and symbolic path utilization rate
```

**Why this works:** The `AnatomicalOntology` is a superset of VQA-RAD's anatomy terms. The symbolic execution logic (scatter_add, yes/no mapping) is dataset-independent. Only the perceptual features need adaptation.

**Expected contribution:** Demonstrate that **symbolic knowledge transfers across imaging modalities while neural perception adapts** — this transfer narrative is currently unverified.

---

## 5. Execution Matrix

### Tier 0 — Bug fixes (immediate, < 1 day each)

| Status | ID | Action | File | Impact |
|---|---|---|---|---|
| [x] | B1 | Fix `faithfulness_test()` — inject perturbation into executor, not model re-run | `interpretability.py` | Correctness: metric is currently always 0 |
| [x] | B2 | Fix `QueryAttr` executor in `DifferentiableDSLInterpreter` — replace substring check with proper vocab scatter | `dsl.py` | Correctness |
| [x] | B3 | Add attention entropy to `LearnedGate` + return attn weights from fusion | `routing.py`, `fusion.py` | Novelty integrity: closes C1 claim gap |
| [x] | B4 | Enable LTN by default + add 2–3 pathology-specific clauses | `config.py` | Utilizes implemented C2 contribution |
| [x] | B5 | Remove dead code in `paired_bootstrap_test` (`int(acc1*n1)` unused) | `metrics.py` | Code quality |

> **B3 note:** Gate entropy + transformer weights return are implemented (`routing.py:30-81`, `fusion.py:56`). `pathvqa_model.py:91` now unpacks the `(attended, weights)` tuple and passes `attn_weights` to `self.gate()` on line 108. Additionally fixed `LearnedGate.__init__` gate input dim (`c_scene` is always `(B, 1)` from `.max()`, not `(B, num_regions)`). Wiring bug resolved.

### Tier 1 — Architectural fixes (1–2 weeks, high payoff)

| Status | Priority | Proposal | Expected accuracy Δ | Ablation evidence |
|---|---|---|---|---|
| [x] | **1** | P1 — Symbolic auxiliary losses (region/attr/yn supervision) | **+5–8% overall** (identity, location) | w/ vs. w/o aux loss |
| [x] | **2** | P5 — Product-of-experts fusion in log-prob space | +1–3% overall (fixes scale mismatch) | additive vs. PoE |
| [x] | **3** | P3 — Wire DSL soft counting into count path | **+5–10% count subtype** (from 28.6%) | per-type accuracy |
| [x] | **4** | P4 — Add entropy to gate (B3) + re-run gate ablation | Minor Δ accuracy; strengthens C1 | gate value heatmap by qtype |
| [x] | **5** | P6 — Switch default backbone to PLIP | +2–5% overall (domain specialization) | backbone ablation: CLIP vs. PLIP |

**Implementation order rationale:** P1 first — it provides the SceneParser with learning signal that all downstream proposals depend on. P5 before ablations — fixes the fusion scale problem making static vs. learned gate comparisons currently misleading. P3 directly addresses the worst-performing subtype with already-implemented tooling.

### Tier 2 — Research features (2–8 weeks, publication-grade)

| Status | Phase | Proposal | Target venue | Contribution type |
|---|---|---|---|---|
| [x] | A | P7 — Ontology-conditioned executor with hierarchy relaxation | MICCAI 2026 | C4 evidence (ontology in inference loop, not just vocabulary) |
| [x] | A | P8 — Conformal prediction uncertainty routing | NeurIPS 2026 Workshop | Novel clinical safety contribution; formal guarantees |
| [~] | B | P6 Stage 2/3 — Nuclei density head (done) + multi-scale activation (parked) | MICCAI 2026 | Pathology-specific vision contribution |
| [ ] | B | P9 — GAT scene graph with relational edges | NeurIPS 2026 | Core NS-AI architectural contribution |
| [ ] | C | P10 — DeepProbLog probabilistic rules | NeurIPS 2026 | Strong NS-AI novelty (probabilistic, not deterministic, execution) |
| [ ] | C | P11 — Cross-dataset symbolic transfer | CVPR 2026 | Generalization claim with experimental evidence |

### Tier 3 — Next-generation research directions

#### 3.1 Differentiable program induction from questions

Replace regex `parse_question()` with a seq2seq model that generates DSL program trees. Use the symbolic executor as a differentiable reward signal:

```
L = -log P_parser(prog | q) · R(prog, a)

R(prog, a) = 1   if execute(prog, scene) = a
           = 0   otherwise
```

This enables learning pathology-specific question parsings (e.g., "demonstrate the cellular architecture") without manual pattern engineering, and provides gradient signal even for programs that produce wrong answers. Related, more incrementally-scoped work: `NOVELTIES.md` C9's `DifferentiableQueryClassifier`, which learns qtype directly rather than full program trees.

#### 3.2 SNOMED CT concept embeddings as region-classifier priors

Initialize `SceneParser.region_classifier` weight matrix using SNOMED CT embeddings (via SAPBERT or BioBERT) rather than random initialization:

```
W_region[i, :] ← SAPBERT(region_name[i])
```

This gives the classifier a structured prior over concept similarity before any training. Anatomically related regions (lung, bronchus) start near each other in embedding space. Fine-tune from this initialization.

#### 3.3 Mamba state space models for WSI-level context

PathVQA operates on isolated image tiles. Whole-slide images have spatial coherence across tiles. A Mamba SSM could encode tile sequences (spatial trajectory over a WSI grid) providing spatially-aware context to the symbolic scene graph, enabling VQA at the WSI level — a scope expansion positioning the system in the TCGA/WSI-VQA benchmark space.

#### 3.4 Multi-agent symbolic debate for uncertain predictions

For samples with gate value `g ≈ 0.5` (uncertain neural/symbolic boundary):
- **Neural agent:** top prediction + softmax confidence
- **Symbolic agent:** top prediction + LTN clause satisfaction score

Resolution: lightweight MLP trained on validation set. Provides an interpretable, auditable uncertainty resolution mechanism — clinically meaningful for triage scenarios.

---

## 6. Publication Strategy

### 6.1 Positioning

To publish in a premier venue (**MICCAI, NeurIPS, AAAI, CVPR Medical Workshop**), the paper should position itself around a definitive paradigm:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PUBLICATION POSITIONING STRATEGY                                 │
├─────────────────────────┬────────────────────────────────────────────────────────────────────────┤
│ Title Pitch             │ "Differentiable Neuro-Symbolic Reasoning for Interpretable,             │
│                         │  Resource-Efficient Pathology VQA" (see `NOVELTIES.md` for the           │
│                         │  concept-bottleneck variant title once C9/C10 land)                       │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Core Novelty Pillars    │ 1. Learned confidence gate + differentiable LTN clause loss (C1, C2).   │
│                         │ 2. Ontology-grounded region discovery with cross-dataset story (C4).     │
│                         │ 3. Certified diagnostic failure taxonomy & counterfactual traces (C7).   │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Pareto Claim            │ Competitive with 10B+ VLMs (LLaVA-Med) on clinical accuracy trend while  │
│                         │ training on consumer GPUs (1× RTX 4060) in < 1 GPU-hour with auditable   │
│                         │ decision provenance.                                                     │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

> **IMPORTANT:** Do not claim accuracy parity with LLM-based systems. Frame as a **Pareto trade-off**: interpretable, auditable, ~100× smaller, clinically deployable, training-free at inference.

### 6.2 Current accuracy positioning

| System | Open-Ended | Yes/No | Params | Hardware |
|---|---|---|---|---|
| LLaVA-Med (7B) | ~65% | ~90% | 7B | Multi-A100 |
| PathChat (7B) | ~58% | ~88% | 7B | Multi-A100 |
| **NS-PathVQA (Tier 0+1 target)** | ~62%* | ~84% | **11.9M** | **1×RTX 4060** |
| **NS-PathVQA (current)** | **53.8%** | **82.4%** | 11.9M | 1×RTX 4060 |
| CLIP+head (baseline) | ~32% | ~78% | ~7M | 1×RTX 4060 |

*Projected after implementing P1+P3+P5+P6. The accuracy gap narrows significantly.

### 6.3 Recommended experimental benchmark matrix

| Benchmark dataset | Domain | Samples | Key metric | Target SOTA baseline |
|---|---|---|---|---|
| **PathVQA** (current) | Pathology / Microscopy | 32,632 | Accuracy / ECE / Macro-F1 | LLaVA-Med (67.2%), MedCLIP (63.1%) |
| **VQA-RAD** | Radiology (CT/MRI/X-Ray) | 3,515 | Accuracy / Closed vs Open | MEVF, BAN, BioViL |
| **SLAKE** | Bilingual Multimodal Med | 14,028 | Semantic Entity Recall | Med-VInT, MCAL |
| **PMC-VQA** | Broad Biomedical Literature | 227k | Zero-shot transfer | BiomedCLIP, LLaVA-Med |

### 6.4 Baselines needed for the paper

Published numbers (not retrained) are sufficient; compute constraints justify citation over reproduction. See `outputs/baselines.md`.

| Category | Examples |
|---|---|
| Classic medical VQA | MEVF, MMQ, BAN-style bilinear attention |
| Medical vision-language pretraining | BioViL / BioViL-T, MedCLIP |
| Medical LLM/VLM | LLaVA-Med, MedVInT |
| Generic VL transformer | ViLT, CLIP+classifier head |

### 6.5 Ablation design for submission

Minimum viable ablation table (all runs executable with existing `--no-symbolic`, `--experiment`, and config flags; results write to `outputs/<experiment>.json` automatically):

```
A0: Neural-only baseline (--no-symbolic)                         → base
A1: + SceneParser (no executor)                                  → visual symbol grounding value
A2: + Executor, static fusion (weighting_strategy=static)        → executor value = A2 - A1
A3: + Executor, learned gate                                     → gate value = A3 - A2
A4: + Attention entropy in gate (fix B3)                         → entropy feature = A4 - A3
A5: + LTN consistency loss (ltn_enabled=True)                    → LTN value = A5 - A4
A6: + Symbolic auxiliary supervision (P1)                        → direct supervision = A6 - A5
A7: + Product-of-experts fusion (P5)                              → PoE value = A7 - A6
```

Beyond this headline table, the fuller ablation design for the paper also covers:

1. **Symbolic component** — neural-only, +scene parser only, +executor fixed rules, +executor learned gate (headline ablation, same as A0–A3 above)
2. **Top-k proposals** — sweep k against accuracy + latency
3. **Region count** — discovered vs. fixed smaller/larger counts
4. **Cross-modal attention** — attention vs. mean-pooled features
5. **Per-question-type breakdown** — yes/no, identity, location, attribute, count

### 6.6 PathVQA-Hard benchmark opportunity

Current PathVQA has a 54%/46% yes/no imbalance and many trivially-answered visual questions. Curate **PathVQA-Hard** by filtering:

1. Questions requiring compositional reasoning (>3 entity mentions)
2. Attribute questions requiring multi-attribute reasoning ("what color and shape is the lesion?")
3. Count questions with answers > 3
4. Questions where yes/no baseline (always predict "yes") fails

Contributes as a workshop paper (MICCAI/MIDL) or appendix benchmark. Tests symbolic reasoning specifically under conditions where pure statistical baselines fail.

### 6.7 Venue targeting

| Venue | Target | Primary selling point |
|---|---|---|
| **MICCAI 2026** | Full paper | Interpretable NS-VQA; clinical reasoning traces; resource efficiency |
| **MIDL 2026** | Full paper | Medical image VQA; explicit symbolic audit trail for pathologists |
| **NeurIPS 2026** | Workshop (NS-AI) | P9/P10 — relational scene graph + probabilistic logic execution |
| **CVPR 2026** | Full paper (post P11) | Cross-dataset symbolic transfer across imaging modalities |

**MICCAI 2026 is the primary target.** The combination of (a) histopathology domain, (b) lightweight deployment, (c) auditable reasoning traces, and (d) calibrated uncertainty maps directly to MICCAI reviewer priorities. The failure taxonomy (perception/parsing/execution) is a strong explainability contribution.

### 6.8 Failure taxonomy (reference)

Errors map one-to-one onto architecture stages (implemented as C7 — see `NOVELTIES.md`):

| Failure type | Definition | Diagnostic signal |
|---|---|---|
| **Perception** | Correct parse + logic, wrong visual features | Scene parser region disagrees with GT |
| **Parsing** | Wrong question type or target | Hand-labeled parse-accuracy set |
| **Execution** | Correct facts + parse, wrong executor output | Gold-fact isolation test |

### 6.9 Interpretability metrics (reference)

- **Region-alignment accuracy**: IoU between top-attended proposals and GT anatomical regions
- **Attention–anatomy correspondence**: Spearman rank correlation between attention weights and region-presence sigmoids
- **Faithfulness test**: perturb top symbolic fact → measure prediction change (pending the B1 fix)
- **Human expert rating**: blinded Likert-scale evaluation of reasoning traces (2–3 pathologists, 50–100 cases) — out of scope for the automated pipeline, optional post-submission addition

### 6.10 Hierarchical reasoning trace schema

**Clinical motivation:** Pathologists do not trust raw confidence numbers. They require verifiable visual evidence and clinical reasoning steps. Target structured, verifiable JSON execution traces (feeds the FastAPI `/predict/explain` endpoint):

```json
{
  "prediction": "Colonic Adenocarcinoma",
  "confidence": 0.892,
  "reasoning_trace": {
    "step_1_organ_identification": {
      "target": "Colon",
      "p_truth": 0.941,
      "grounding_patch_indices": [12, 13, 19, 20],
      "morphology": "Crypt architecture with severe architectural distortion"
    },
    "step_2_cellular_atypia": {
      "target": "High-Grade Dysplasia",
      "p_truth": 0.876,
      "grounding_patch_indices": [19, 20, 27]
    },
    "step_3_rule_deduction": {
      "rule": "LocatedIn(Colon) ∧ SevereGlandularDistortion ∧ CellularAtypia → Colonic Adenocarcinoma",
      "rule_satisfaction": 0.892
    }
  }
}
```

Extend the faithfulness test (once P2/B1 is fixed) to generate **counterfactual clinical hypotheses**: *"What minimal change in visual/symbolic features would flip the diagnosis from Malignant to Benign?"* — invert the top predicate `argmin_Δr ‖Δr‖₂` such that `ŷ ≠ y`. See `NOVELTIES.md` C10 (Module C) for the concrete implementation of this as `counterfactual_explanation()`.

### 6.11 Narrative framing

Strongest differentiators for publication:

- **Resource efficiency as first-class contribution**: training time, GPU-hours, peak VRAM vs. A100-class VLM costs
- **Deterministic, auditable reasoning traces**: every prediction traceable through SceneParser → QueryParser → Executor
- **Failure mode legibility**: precisely where the system fails (perception vs. parsing vs. execution)

Frame as Pareto trade-off (interpretability + efficiency for comparable accuracy), not raw accuracy win.

### 6.12 Manuscript outline

1. Introduction — interpretability/efficiency gap in medical VQA
2. Related Work — medical VQA, neuro-symbolic VQA, medical VLMs
3. Method — architecture, gated fusion, ontology-grounded regions
4. Experimental Setup — dataset, baselines, training details, hardware
5. Results — accuracy table, ablations, transfer study
6. Interpretability Analysis — alignment metrics, human eval, failure taxonomy
7. Discussion & Limitations — partial transfer, count neural-only, no clinical claims
8. Conclusion

**Key figures:** architecture diagram, reasoning trace visualization, failure taxonomy breakdown, efficiency scatter — plus the C9/C10-specific figures listed in `NOVELTIES.md` (soft qtype heatmap, concept attribution bar chart, intervention accuracy delta) once those phases land.

---

## 7. Suggested Next Steps

1. **Tier 0 execution:** Apply the bug fixes (B1–B5) — cheapest, highest-integrity wins before anything else.
2. **Pathology backbone benchmark:** Run an ablation training comparing `openai/clip-vit-base-patch32` vs `vinid/plip` using `--debug` and full dataset modes.
3. **Tier 1 architectural fixes:** P1 (symbolic auxiliary supervision) first, since every other proposal benefits from a better-trained SceneParser.
4. **Drafting paper experiments:** Execute the benchmark suite against VQA-RAD and SLAKE (P11) to complete the transferability story once Tier 1 is done.

---

## 8. Appendix: Code-Level Findings Index

| Issue | File | Line(s) | Type |
|---|---|---|---|
| Gate entropy claim — implemented via `LearnedGate.forward()` with `attn_weights` param | `routing.py` | ~48–82 | Fixed (B3 done) |
| `faithfulness_test` — fixed to inject perturbation into executor directly | `interpretability.py` | ~144–146 | Fixed (B1 done) |
| `QueryAttr` — fixed to use `mapping[valid_attr_mask]` scatter instead of substring | `dsl.py` | ~130–131 | Fixed (B2 done) |
| `DifferentiableDSLInterpreter` wired into `train.py` for count queries | `dsl.py`, `train.py` | ~64, ~270–300 | Fixed (P3 done) |
| Count qtype — DSL interpreter now handles count queries | `executor.py` | ~237 | Fixed (P3 done) |
| LTN enabled by default (`ltn_enabled=True`) | `config.py` | ~99 | Fixed (B4 done) |
| SceneParser — has direct symbolic supervision via auxiliary losses (P1) | `scene_parser.py` | ~21–51 | Fixed (P1 done) |
| PLIP is now the default backbone (was CLIP ViT-B/32) | `config.py`, `visual_encoder.py` | ~27, ~68 | Fixed (P6 done) |
| `MultiScaleVisualEncoder` not activated (`use_multiscale=False`) | `visual_encoder.py` | ~100 | Under-utilization (P6 Stage 3 — parked) |
| Single-block cross-attention (no depth scaling) | `fusion.py` | ~7–56 | Capacity limitation |
| `paired_bootstrap_test` dead code — removed (B5 done) | `metrics.py` | ~87–88 | Fixed (B5 done) |
| `_extract_target()` handles multi-word phrases (up to 4 words) | `query_parser.py` | ~50–62 | Fixed — was incorrectly listed as single-word only |
| `AnatomicalOntology` not invoked at inference time | `dataset_adapter.py` | ~109+ | Untapped resource (P7) |
| `CrossModalTransformer` now returns `(attended, weights)` tuple | `fusion.py` | ~56 | Fixed (P4 done) |
| `VQARADAdapter` / `SLAKEAdapter` implemented, never trained/evaluated | `dataset_adapter.py` | — | Unverified claim (P11) |

---

*Document consolidated 2026-08-19 from two prior audit passes (`AUDIT.md`, `PATHVQA_NEURO_SYMBOLIC_AUDIT_ROADMAP.md`) plus the design-alternatives/publication-context sections of a retired `IMPLEMENT.md`. Reviewed against: all source files in `src/`, `train.py`, `evaluate.py`, and the sibling docs `TODO.md`, `NOVELTIES.md`, `PROJECT.md`. Impact rankings assume the published PathVQA dataset split from `flaviagiammarino/path-vqa` with the current training vocabulary (~4,700 classes + `<UNK>`).*
