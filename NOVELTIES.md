# PathVQA — Novelty Register

> **What this file owns:** the single source of truth for **what is novel** in this project, **what is implemented vs. planned**, **how each claim gets verified**, and the **full implementation spec** for planned claims. Backtrack here when writing the paper, responding to reviewers, or deciding whether a change strengthens or dilutes the story. It does **not** hold the execution order (see `TODO.md`), the current-system reference (`PROJECT.md`), or the architecture critique / broader research proposal set (`NS-PATHVQA-AUDIT.md`).

Status markers: `[x]` implemented in current code · `[~]` partially implemented · `[ ]` planned.

---

## Primary contributions (the paper story)

### C1 — Learned per-sample confidence gate for neuro-symbolic VQA fusion

**What:** A gating function `g = σ(MLP([h_attn ‖ c_scene ‖ onehot(qtype) ‖ ent(attn)]))` that blends neural and symbolic logits sample-by-sample: `final = (1−g)·neural + g·symbolic`. The gate conditions on visual attention state, parsed scene confidence, question type, and attention entropy — making symbolic trust explicit, question-type-conditional, and instance-adaptive.

**Status:** [x] Implemented — `src/symbolic/routing.py`, wired into `train.py` + `evaluate.py`. Ablation baseline retained behind `weighting_strategy ∈ {static, learned}`. Attention entropy `ent(attn)` is wired in as gate input (Tier 1 P4/B3 complete).

**Prior work / distinction:**
- Bao et al. (2021) propose a confidence-based neuro-symbolic VQA approach, but their gate is fixed (not learned end-to-end) and does not condition on question type or attention entropy.
- AOR (2025) uses anatomical ontology grounding for CXR interpretation via LLM, not trained symbolic modules; the routing is LLM-mediated, not a lightweight learned gate.
- Typical neuro-symbolic VQA fusion uses fixed weighted sums (e.g., `neural + w·symbolic`). Our gate is the first to combine scene confidence, parsed question type, and attention entropy into a learned sigmoid blend for medical VQA.

**Key differentiator:** Sample-level, learned, conditioned on 4 heterogeneous signals; retains static mode as ablation baseline. Lightweight (128-d hidden, ~66K params for the gate alone).

---

### C2 — Differentiable Logic Tensor Network as auxiliary loss for medical VQA

**What:** `MedicalLogicTensorNetwork` in `src/symbolic/ltn.py` enforces domain-consistency constraints as differentiable fuzzy logic clause satisfaction:
- **Region-Object Coherence:** `∀r (Object(r) ↔ Region(r))` — bidirectional implication between scene parser object-presence and region logits.
- **Attribute Certainty:** sharpened softmax over color logits as a regularizer.

The clause loss `L_sat = 1 − satisfaction` is added to the main cross-entropy loss.

**Status:** [x] Implemented — `src/symbolic/ltn.py` (195 lines), integrated into `train.py` loss computation. **Caveat:** `ltn_enabled = True` by default (B4 done) — the loss is implemented and active with 5 clauses (region-object coherence, attribute certainty, morphology-region coherence, attribute confidence union, region sparsity), but could benefit from additional pathology-specific axioms.

**Prior work / distinction:**
- Bergamin et al. (2025) use LTNs for medical *semantic segmentation* (hippocampus MRI with SwinUNETR), not VQA. Their constraints are geometric (shape, connectedness, volume similarity) — different domain, different task, different constraint semantics.
- LTNs have been applied to knowledge graph completion and image classification, but **not** to medical VQA clause satisfaction losses.
- Our clauses are specific to the neuro-symbolic VQA pipeline (scene parser → executor coherence), not general geometric priors.

**Key differentiator:** First application of LTN fuzzy logic clause satisfaction as an auxiliary loss in medical VQA. Domain-specific clauses tie scene parser output consistency to answer quality. Computationally negligible (~0 overhead, uses existing scene logits).

---

### C3 — Resource-efficient neuro-symbolic pathology VQA (RTX 4060–feasible)

**What:** The full system trains end-to-end on a single consumer RTX 4060 (8 GB VRAM) using:
- Frozen PLIP + DistilBERT with LoRA (rank 16/32), ~11.9M trainable parameters (4.4%)
- AMP, gradient accumulation, gradient checkpointing, uint8 tensor cache
- Zero dependency on LLMs, GPT-4V, or cloud inference

**Status:** [x] Implemented — `train.py` fits 8GB at batch 16–32; `outputs/probe_31.json` records VRAM headroom.

**Prior work / distinction (PathVQA SOTA, 2024–2026):**

| System | Approach | Params | Hardware |
|---|---|---|---|
| LLaDA-MedV (2025) | Closed-form, LLM-based | >7B | Multi-A100 |
| PeFoMed (2025) | Few-shot prompting | >7B | Multi-GPU |
| BioVLM (2025) | Large VLM | >7B | Multi-GPU |
| PathChat (2024) | Generative, zero-shot | >7B | Multi-GPU |
| PA-LLaVA (2024) | PLIP + LLaVA | >7B | 16×A100 |
| WSI-VQA (2024) | Co-attention transformer | 25.6M (ResNet) + decoder | Multi-GPU |
| **Ours** | Neuro-symbolic + LoRA | **11.9M trainable** | **1×RTX 4060** |

**Key differentiator:** Achieves interpretable, auditable predictions at ~100× fewer parameters than LLM-based SOTA, with no cloud dependency. Positioned as Pareto trade-off (interpretability + efficiency), not raw accuracy.

---

### C4 — Ontology-grounded region discovery with cross-dataset transfer story

**What:** `DatasetAdapter.discover_regions()` in `src/data/dataset_adapter.py` links answer vocabulary terms to UMLS/RadLex via `AnatomicalOntology` (SNOMED-CT/RadLex organ-system hierarchy, 10 systems, 140+ organ terms, synonym normalization). Adapters for PathVQA, VQA-RAD, and SLAKE are implemented with coverage reporting.

**Status:** [x] Implemented — `src/data/dataset_adapter.py` (232 lines). `PathVQAAdapter` with UMLS mapping support, `VQARADAdapter`, `SLAKEAdapter`. **Caveat:** the VQA-RAD/SLAKE adapters are implemented but neither has been trained or evaluated — see "What could dilute the story" below and `NS-PATHVQA-AUDIT.md` P11 for the transfer protocol that would produce this evidence.

**Prior work / distinction:**
- AOR (2025) uses anatomical ontology for *interpretability* of LLM outputs in CXR — post-hoc, not trained into the model.
- Domain-specific VQA systems (PA-LLaVA, PathChat) use VLMs and do not ground their symbolic modules in medical ontologies.
- Existing neuro-symbolic VQA uses hand-crafted region lists, not ontology-grounded discovery from the answer vocabulary.

**Key differentiator:** Region discovery is automated (extracted from training answer vocab), ontology-linked, and dataset-portable via the adapter pattern. Enables transfer narrative across PathVQA → VQA-RAD → SLAKE with the same symbolic pipeline.

---

## Secondary contributions (supporting evidence)

### C5 — Differentiable DSL program synthesis with soft counting

**What:** `DSLProgramCompiler` compiles questions into typed AST trees (`Filter`, `Count`, `Verify`, `QueryAttr`, `Relate`, `Exist`). `DifferentiableDSLInterpreter` executes these over patch tokens with differentiable soft counting: `soft_count = Σ σ(W v_i)`, mapped to discrete count answers via Gaussian kernels.

**Status:** [x] Implemented — `src/symbolic/dsl.py`. Wired into training pipeline for count queries (Tier 1 P3). QueryAttr bug fixed (Tier 0 B2).

**Prior work / distinction:**
- PIPS (NeurIPS 2025) and NSA (2025, ARC) use LLM-guided program synthesis for abstract reasoning — not differentiable, not for VQA.
- NS-CL / Neural Module Networks compile questions into differentiable programs but operate on natural images, not pathology.
- Our soft-counting head with Gaussian kernel mapping is specific to medical VQA count questions, where counts are small integers ("1", "2", "few", "multiple").

**Key differentiator:** End-to-end differentiable AST execution (no LLM in the loop) with Gaussian kernel count discretization, designed for medical VQA where counts are small and noisy.

---

### C6 — Multi-scale morphological visual encoding

**What:** `MultiScaleVisualEncoder` in `src/models/visual/visual_encoder.py` encodes the global 224×224 image and 4 high-res quadrant crops, fusing via a learned per-token gate: `fused = MLP([global ‖ local])`.

**Status:** [x] Implemented — `src/models/visual/visual_encoder.py` (139 lines). Extends `CLIPViTEncoder`, usable behind `use_multiscale` flag (off by default).

**Prior work / distinction:**
- MR-PLIP (CVPR 2025) uses multi-resolution *pre-training* (different magnifications of WSI patches), not multi-scale *architectural encoding* at inference time.
- ViLa-MIL (CVPR 2024) uses dual-scale (5× and 10× WSI patches) for MIL classification, not for VQA.
- PA-LLaVA uses scale-invariant connectors to *preserve* original resolution, not to *augment* with high-res crops.

**Key differentiator:** Architectural multi-scale encoding (global + local crops) with learned fusion gate, at the ViT-B/32 image level. Lightweight extension (~200K extra params), not a pre-training paradigm.

---

### C7 — Architecture-grounded failure taxonomy for medical VQA

**What:** `classify_failure_type()` in `src/utils/interpretability.py` maps every error to one of three architecture stages:
- **Perception** — correct parse/logic, wrong visual features (scene parser region disagrees with GT)
- **Parsing** — wrong question type or target (hand-labeled parse-accuracy set)
- **Execution** — correct facts + parse, wrong executor output (gold-fact isolation test)

Plus: faithfulness test (`faithfulness_test()`), attention–anatomy correspondence (`attention_anatomy_correspondence()`), visual grounding maps, counterfactual explanations.

**Status:** [x] Implemented — `src/utils/interpretability.py` (298 lines). Metrics integrated into evaluation pipeline. Faithfulness test bug (B1) fixed — `executor_fn` parameter injects perturbations directly into the executor without re-running the full model.

**Prior work / distinction:**
- ERVQA (EMNLP 2024) defines a hospital-setting error taxonomy — task-specific, not architecture-linked.
- GPT-4 USMLE error taxonomy (2024) is a post-hoc annotation scheme for LLM failures — not tied to model components.
- MedThink (NAACL 2025) focuses on rationale generation, not failure localization.
- GEMeX (ICCV 2025) focuses on visual/textual grounding for answer explanations, not architecture-mapped failure diagnosis.

**Key differentiator:** The taxonomy is *prescriptive* (maps to specific model components, enabling targeted fixes) rather than *descriptive* (post-hoc error labeling). Faithfulness testing measures whether symbolic facts causally influence predictions (pending the bug fix above).

---

### C8 — Calibrated evaluation with statistical rigor

**What:** Evaluation pipeline includes:
- Expected Calibration Error (ECE) with temperature scaling
- Max-softmax uncertainty estimation
- Paired bootstrap significance tests for baseline comparisons
- BLEU/ROUGE + entity precision/recall for free-form answers
- Per-question-type accuracy breakdown

**Status:** [x] Implemented — `src/utils/metrics.py`, `evaluate.py`. Dead code in `paired_bootstrap_test` removed (B5 done).

**Prior work / distinction:**
- Most medical VQA papers report only accuracy; calibration and uncertainty are rarely included.
- SURE-VQA (2024) highlights that medical VLM robustness evaluations are systematically flawed — our calibrated metrics address this gap.
- Paired bootstrap tests provide statistical rigor absent from most medical VQA comparisons.

**Key differentiator:** Publication-grade evaluation suite with calibration, uncertainty, and significance testing — not just accuracy numbers.

---

## What is NOT novel (honest accounting)

The following components use established techniques without claiming novelty:

| Component | Status | What it is |
|---|---|---|
| CLIP ViT-B/32 backbone | Standard 2024–25 recipe | Frozen + LoRA adapter fine-tuning |
| DistilBERT question encoder | Standard | Frozen + LoRA, HF tokenizer |
| Cross-attention transformer | Standard | Multi-head cross-attention (Q from question, K/V from patches) |
| LoRA adapters | Standard | `peft` library, rank 16/32 |
| AdamW + cosine + warmup | Standard | Replaces Adam + ReduceLROnPlateau |
| EMA of weights | Standard | `src/models/ema.py`, `beta=0.999` |
| RandAugment | Standard | `torchvision.transforms.RandAugment` |
| uint8 tensor cache | Engineering | Atomic writes, keyed by `(image_size, norm)` |
| FastAPI + Docker | Engineering | Standard deployment |

These are listed to avoid overclaiming; they are engineering contributions, not research contributions.

---

## Novelty strength assessment (for reviewer anticipation)

| Claim | Strength | Risk | Mitigation |
|---|---|---|---|
| **C9** — Differentiable pipeline | **Very strong** | Complex to implement; may not improve accuracy | Warm-up ensures it matches hard pipeline; ablation shows value of each component. |
| **C10** — Concept intervention | **Strong** | Interventions may show no accuracy improvement | Negative result is still valid ("concepts are not causally used" is a finding). |
| **C1** — Learned gate | **Strong** | Reviewer may cite Bao et al. (2021) | Distinguish: our gate is end-to-end learned, conditioned on 4 signals, sample-adaptive; Bao's is fixed. |
| **C2** — LTN loss | **Strong** | Bergamin et al. (2025) uses LTN in medical imaging | Distinguish: their task is segmentation, not VQA; different constraints, different domain application. |
| **C3** — Resource efficiency | **Strong** | May be dismissed as "engineering" | Frame as Pareto contribution: interpretability + efficiency at comparable accuracy; cite compute costs of SOTA. |
| **C4** — Ontology grounding | **Moderate** | Ontology use in medical AI is not new | Distinguish: ontology-grounded *region discovery* in a neuro-symbolic VQA pipeline, not post-hoc interpretation. |
| **C5** — DSL + soft counting | **Moderate** | Wired into training for count queries; standalone contribution | Position as supporting evidence for differentiable pipeline story. |
| **C6** — Multi-scale encoding | **Moderate** | MR-PLIP, ViLa-MIL use multi-scale | Distinguish: architectural encoding (not pre-training); image-level crops (not WSI patches); VQA-specific. |
| **C7** — Failure taxonomy | **Moderate** | Error taxonomies exist in medical NLP | Distinguish: architecture-grounded, prescriptive, not descriptive; maps to specific model components. |
| **C8** — Calibrated evaluation | **Weak–Moderate** | Standard metrics, rarely applied together | Position as contribution to medical VQA methodology, not a primary claim. |

---

## Paper framing (updated with C9/C10)

**Title direction:** "Differentiable Neuro-Symbolic Pathology VQA with Concept Bottleneck Interventions"

**Core story:** We build a neuro-symbolic system where the entire reasoning chain is differentiable and the intermediate concepts are provably causally relevant to the final answer. Positioned as a Pareto-optimal alternative to billion-parameter VLMs when interpretability and deployment cost matter.

**Contributions (ranked):**

1. **Fully differentiable neuro-symbolic pipeline** (C9) — first differentiable symbolic executor for medical VQA
2. **Learned per-sample confidence gate** (C1) — 4-signal adaptive fusion
3. **LTN auxiliary loss** (C2) — domain-consistency regularization
4. **Concept bottleneck intervention testing** (C10) — proving causal use of intermediate concepts
5. **Resource-efficient design** (C3) — RTX 4060–feasible (supporting evidence)

**Key figures:**
1. Architecture diagram (neural + symbolic paths + differentiable pipeline)
2. Soft qtype distribution heatmap (shows classifier learning question types)
3. Concept attribution bar chart (which scene concepts drive each answer)
4. Intervention accuracy delta per question type (proves concepts are causally used)
5. Failure taxonomy breakdown (perception / parsing / execution)
6. Efficiency scatter (accuracy vs. compute / parameters)

---

## Planned contributions — full implementation spec

Two directions to strengthen the paper's research contribution beyond engineering. **Direction 1 (C9) is the core novelty; Direction 2 (C10) pairs with it for a stronger story.** Scheduling of this work is tracked in `TODO.md` ("Planned novelty phases").

### C9 — Fully differentiable neuro-symbolic pipeline

**What:** Replace the non-differentiable regex query parser and hard-coded executor with learned, differentiable modules:
- `DifferentiableQueryClassifier`: learned qtype prediction from question embedding (replaces regex)
- `DifferentiableTargetExtractor`: cross-attention soft alignment between question and anatomical regions (replaces `_extract_target()`)
- `DifferentiableExecutor`: weighted blend of per-type execution strategies, all differentiable (replaces hard `if/else` branches)

**Status:** [ ] Planned — full spec below.

**Why it's the strongest novelty:** NS-CL and Neural Module Networks do differentiable program execution on CLEVR/natural images. Nobody has done it for medical VQA with anatomical ontology grounding. The current non-differentiable break (documented in `NS-PATHVQA-AUDIT.md` §2.2 as a "gradient dead end") is the single biggest gap in the system.

**Expected impact:** End-to-end training of the symbolic chain; the qtype classifier and target extractor learn from the task loss, not just regex pseudo-labels.

#### The problem

The symbolic path currently has a **non-differentiable break**:

```
attended_features ──→ SceneParser ──→ region/object/attr logits  [differentiable ✓]
                                      ↓
question_string ────→ regex parse  ──→ discrete qtype + target    [NOT differentiable ✗]
                                      ↓
scene_logits + query → Executor    ──→ symbolic_logits            [NOT differentiable ✗]
```

Gradients die at the regex step. The symbolic reasoning chain cannot be trained end-to-end.

#### Target architecture

```
attended_features ──→ SceneParser ──→ region/object/attr logits  [differentiable ✓]
                                      ↓
question_embedding ─→ QTypeClassifier → soft qtype weights (B, 5) [differentiable ✓]
question_embedding ─→ TargetExtractor → soft target alignment      [differentiable ✓]
                                      ↓
scene_logits + soft_qtype + soft_target → DiffExecutor → symbolic_logits [differentiable ✓]
```

#### Module 1: `src/symbolic/diff_query_classifier.py` (NEW)

**Purpose:** Replace regex-based `parse_question()` with a learned, differentiable question type classifier.

**Interface:**

```
Input:  question_state (B, 768)  — from DistilBERT [CLS]
Output: qtype_weights (B, 5)    — soft distribution over [identity, location, yes_no, attribute, count]
```

**Architecture:**

```
Linear(768, 256) → ReLU → Dropout(0.1) → Linear(256, 5) → Softmax(dim=-1)
```

**Training strategy:**

- Phase 1 (warm-up): train with cross-entropy against regex-derived pseudo-labels from `parse_question()`.
- Phase 2 (end-to-end): freeze regex labels, train the full pipeline end-to-end. The classifier learns from the downstream task loss flowing back through the executor.
- The soft output means every execution strategy gets a non-zero weight. Gradients flow through all strategies, but the classifier learns to concentrate weight on the correct one.

**Parameters:** ~200K (768×256 + 256×5 + biases). Negligible.

#### Module 2: `src/symbolic/diff_target_extractor.py` (NEW)

**Purpose:** Replace `_extract_target()` (regex + vocab match) with a differentiable soft alignment between question tokens and anatomical region concepts.

**Interface:**

```
Input:  question_state (B, 768), region_embeddings (N_regions, 512) — from answer vocab
Output: soft_target_weights (B, N_regions) — attention over regions
```

**Architecture:**

```
region_proj = Linear(512, 768)                         # project regions to question space
query = question_state.unsqueeze(1)                    # (B, 1, 768)
keys = region_proj(region_embeddings).unsqueeze(0)     # (1, N_regions, 768) — broadcast
attention = softmax(query @ keys^T / √768)             # (B, 1, N_regions)
soft_target = attention.squeeze(1)                     # (B, N_regions)
```

**Key design choice:** Single-head cross-attention from question to anatomical regions. The output is a soft probability distribution over regions, replacing the hard `_extract_target()` lookup. The `region_embeddings` are learned embeddings for each region name (initialized from the scene parser's region classifier weights).

**Parameters:** ~400K (512×768 projection + 768×768 attention). Small.

#### Module 3: `src/symbolic/diff_executor.py` (NEW)

**Purpose:** Replace the hard-coded `execute()` function with a differentiable weighted blend of per-type execution strategies.

**Interface:**

```
Input:  scene_logits (dict of (B, N) tensors), soft_qtype (B, 5), soft_target (B, N_regions),
        patch_features (B, 49, 512), attribute_mappings, answer_to_idx
Output: symbolic_logits (B, vocab_size)
```

**Core logic:**

```python
def diff_execute(scene_logits, soft_qtype, soft_target, patch_features, ...):
    region_logits = scene_logits["scene_region_logits"]      # (B, N_reg)
    color_logits  = scene_logits.get("scene_color_logits")   # (B, 17)

    # --- Strategy 1: Identity ---
    # Boost each region's answer entry weighted by soft_target
    # soft_target @ region_to_vocab_mapping → (B, vocab)
    strat_identity = (soft_target.unsqueeze(-1) * region_to_vocab_matrix).sum(dim=1)

    # --- Strategy 2: Location ---
    # Same as identity (location maps regions to answers)
    strat_location = strat_identity  # reuse

    # --- Strategy 3: Yes/No ---
    # Scalar confidence from soft_target @ region_logits → yes/no logits
    r_conf = (soft_target * region_logits).sum(dim=-1, keepdim=True)   # (B, 1)
    strat_yesno = torch.zeros(B, vocab, device=device)
    strat_yesno[:, yes_idx] = r_conf.squeeze(-1)
    strat_yesno[:, no_idx]  = -r_conf.squeeze(-1)

    # --- Strategy 4: Attribute ---
    # Weighted color/shape/size logits → attribute answers
    attr_weights = soft_target.unsqueeze(-1)   # (B, N_reg, 1)
    strat_attribute = attribute_strategy(color_logits, attr_weights, ...)

    # --- Strategy 5: Count ---
    # Soft counting from DSL (already implemented in dsl.py)
    strat_count = soft_count_strategy(patch_features, ...)

    # --- Blend all strategies by soft qtype ---
    strategies = torch.stack([strat_identity, strat_location,
                              strat_yesno, strat_attribute, strat_count], dim=1)
    # strategies: (B, 5, vocab)
    symbolic_logits = (soft_qtype.unsqueeze(-1) * strategies).sum(dim=1)
    # symbolic_logits: (B, vocab)

    return symbolic_logits
```

**Why this works:** The hard branching (`if qtype == "identity"`) is replaced by a weighted sum. During training, gradients flow through all strategies, but the qtype classifier learns to concentrate weight on the correct one. At inference, you can optionally take `argmax(qtype)` and use only that strategy (hard routing) for interpretability.

**Parameters:** ~50K (small per-strategy heads). The executor itself is mostly fixed operations, not learned.

#### Integration into existing code

**`src/models/pathvqa_model.py` — modify `forward()`:**

```python
def forward(self, images, input_ids, attention_mask, ...):
    # ... existing encode + cross-attention ...

    if self.symbolic_enabled:
        scene = self.scene_parser(attended)

        if self.differentiable_pipeline:
            # NEW: differentiable path
            soft_qtype = self.diff_qtype_classifier(q_state)
            soft_target = self.diff_target_extractor(q_state, self.region_embeddings)
            symbolic_logits = self.diff_executor(scene, soft_qtype, soft_target, ...)
            result["soft_qtype"] = soft_qtype           # for logging
            result["soft_target"] = soft_target          # for interpretability
        else:
            # EXISTING: hard path (for ablation)
            # ... existing executor logic unchanged ...

        # Gate fusion (same as before)
        if self.weighting_strategy == "learned":
            gate = self.gate(attended, c_scene, qtype_onehot)
            final = (1 - gate) * answer_logits + gate * symbolic_logits
```

**`src/utils/config.py` — add config field:**

```python
@dataclass
class SymbolicConfig:
    ...
    differentiable_pipeline: bool = True   # False = hard regex + hard executor (ablation)
    qtype_warmup_epochs: int = 3           # epochs to pre-train classifier on regex labels
```

**`train.py` — training loop modification:**

```python
# Phase 1: warm-up — train qtype classifier on regex labels
if epoch < config.symbolic.qtype_warmup_epochs:
    regex_labels = [parse_question(q).qtype for q in questions]
    qtype_loss = cross_entropy(model.diff_qtype_classifier(q_state), regex_labels)
    loss = loss + 0.1 * qtype_loss   # auxiliary warm-up loss

# Phase 2: end-to-end — gradients flow through full pipeline
# (no change needed — the soft pipeline is always differentiable)
```

#### Ablation matrix (paper table)

| Run | Query Parser | Executor | Gate | What it tests |
|---|---|---|---|---|
| A (baseline) | Regex | Hard-coded | Static | Original system |
| B (current SOTA) | Regex | Hard-coded | Learned | Current shipped system |
| C (differentiable) | Learned | Differentiable | Learned | Full differentiable pipeline |
| D (no gate) | Learned | Differentiable | None | Value of gate vs. differentiable-only |
| E (regex + diff exec) | Regex | Differentiable | Learned | Value of learned qtype classifier |

#### Expected outcomes

- **C > B:** Differentiable pipeline should outperform hard pipeline because the executor can be trained end-to-end (gradients from answer loss refine the qtype classifier and target extractor).
- **B > A:** Learned gate should outperform static (already verified).
- **E > B:** Even with regex qtype, a differentiable executor should help (gradients flow through the soft target extractor).
- **C > E:** Learned qtype classifier should outperform regex (captures patterns regex misses).

---

### C10 — Concept bottleneck intervention testing

**What:** Test whether intermediate scene concepts (region, color, shape) are *causally* used in predictions:
- `bottleneck_probe()`: override scene concepts → measure accuracy recovery
- `concept_attribution()`: gradient-based attribution of answer logits to scene concepts
- `counterfactual_explanation()`: "if region X were absent, prediction would change from A to B"

**Status:** [ ] Planned — full spec below.

**Why it's novel:** HEAL-MedVQA (2025) tests localization in LLMs. GEMeX (ICCV 2025) tests visual grounding. But nobody has tested *concept-level intervention* in a neuro-symbolic medical VQA system — proving intermediate representations are causally relevant, not just decorative.

**Expected impact:** Moves from "interpretable by design" to "interpretable by evidence."

#### The problem

Even neuro-symbolic systems don't test whether intermediate concepts (region, color, shape) are actually causally used in the final prediction. We can claim "interpretable reasoning" but haven't proven the concepts matter.

#### What to build

Three evaluation modules in `src/utils/concept_intervention.py` (NEW file).

##### Module A: Bottleneck Probe

**Idea:** Force predictions through the scene parser bottleneck, allow overrides, measure accuracy recovery.

```python
def bottleneck_probe(model, dataloader, device, override_fn):
    """
    For each sample:
      1. Run scene parser → get predicted region/color/shape
      2. Apply override_fn to "correct" the top concept
      3. Re-run executor with corrected concept
      4. Measure: does accuracy improve?

    override_fn: (scene_logits, sample) → corrected_scene_logits
                 e.g., flip top-1 region to GT region
    """
    results = {"baseline_acc": 0, "post_override_acc": 0, "samples": []}

    for batch in dataloader:
        # Baseline prediction
        outputs = model(images, input_ids, attn_mask)
        baseline_pred = outputs["answer_logits"].argmax(dim=-1)

        # Override scene concept
        corrected_scene = override_fn(outputs["scene_region_logits"], gt_regions)
        # Re-run executor only (not full forward)
        corrected_logits = executor.execute(corrected_scene, queries)
        corrected_pred = corrected_logits.argmax(dim=-1)

        # Measure improvement
        results["baseline_acc"] += (baseline_pred == targets).sum()
        results["post_override_acc"] += (corrected_pred == targets).sum()

    return results
```

**Metrics:**
- `intervention_accuracy_delta` = post_override_acc − baseline_acc
- `concept_fidelity` = % of samples where overriding the correct concept flips the prediction correctly

##### Module B: Gradient-Based Concept Attribution

**Idea:** Use gradients to measure which scene concepts drive each answer.

```python
def concept_attribution(model, image, question, answer_idx):
    """
    Compute ∂answer_logit[answer_idx] / ∂scene_concept for each concept.

    Returns:
      region_attribution: (N_regions,) — which regions matter
      color_attribution:  (17,)        — which colors matter
      shape_attribution:  (9,)         — which shapes matter
    """
    outputs = model(image.unsqueeze(0), input_ids, attn_mask)
    answer_logit = outputs["answer_logits"][0, answer_idx]

    # Backprop to scene concepts
    region_grads = torch.autograd.grad(
        answer_logit, outputs["scene_region_logits"],
        retain_graph=True, create_graph=False
    )[0]

    # Attribution = input × gradient (saliency)
    region_attr = (outputs["scene_region_logits"][0].detach() * region_grads[0]).abs()
    return region_attr
```

**Metrics:**
- Per-question-type attribution patterns (do yes/no questions attend to different regions than identity questions?)
- Attribution sparsity (L1 norm of attributions — are predictions driven by few or many concepts?)

##### Module C: Counterfactual Reasoning

**Idea:** Use the symbolic path to generate counterfactual explanations.

```python
def counterfactual_explanation(model, image, question, region_names):
    """
    For the top-K predicted regions:
      1. Zero out that region's logit
      2. Re-predict
      3. Record: "If {region} were absent, prediction would change from {A} to {B}"

    Returns structured explanation trace.
    """
    outputs = model(image, input_ids, attn_mask)
    baseline_pred = idx_to_answer[outputs["answer_logits"].argmax().item()]
    region_logits = outputs["scene_region_logits"][0]

    top_regions = region_logits.topk(3).indices
    explanations = []

    for r_idx in top_regions:
        # Perturb: zero out this region
        perturbed_scene = region_logits.clone()
        perturbed_scene[r_idx] = -1e9  # effectively remove
        # Re-run executor
        perturbed_logits = executor.execute(perturbed_scene, queries)
        perturbed_pred = idx_to_answer[perturbed_logits.argmax().item()]

        changed = perturbed_pred != baseline_pred
        explanations.append({
            "region": region_names[r_idx],
            "original_prediction": baseline_pred,
            "counterfactual_prediction": perturbed_pred,
            "prediction_changed": changed,
            "explanation": (
                f"If '{region_names[r_idx]}' were absent, "
                f"prediction would change from '{baseline_pred}' to '{perturbed_pred}'"
                if changed else
                f"Prediction '{baseline_pred}' is robust to removal of '{region_names[r_idx]}'"
            ),
        })

    return explanations
```

#### Integration into evaluation pipeline

**`evaluate.py` — add intervention analysis section:**

```python
# After standard evaluation...
if config.eval.concept_intervention:
    intervention_results = bottleneck_probe(model, val_loader, device, override_fn=top1_to_gt)
    attribution_results = concept_attribution_analysis(model, val_loader, device)
    counterfactual_results = counterfactual_analysis(model, val_loader, device, region_names)

    report["concept_intervention"] = intervention_results
    report["concept_attribution"] = attribution_results
    report["counterfactual_analysis"] = counterfactual_results
```

**Paper figure:** Bar chart showing `intervention_accuracy_delta` per question type — demonstrates which concepts are causally used.

---

## Risk assessment (C9/C10)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Differentiable pipeline doesn't improve over hard pipeline | Medium | The warm-up phase ensures it matches hard pipeline; end-to-end should only help. If not, the ablation still tells a story. |
| Soft qtype classifier collapses to always predict "identity" | Low | The warm-up loss prevents this; the diversity of question types in PathVQA provides natural supervision. |
| Concept bottleneck interventions show no accuracy improvement | Medium | This would actually be an interesting negative result — "concepts are not causally used" is a valid finding. |
| Training instability from soft routing | Medium | Start with high warm-up weight, decay it; gradient clipping already in place. |
| 10-day timeline slips | Medium | Phase A-C are independent modules; can be tested individually before integration. |

---

## What could dilute the story (watch out)

1. **Overclaiming DSL novelty:** The DSL is wired into training for count queries (P3 done), but the full differentiable pipeline (C9) is not yet implemented. Frame DSL as supporting evidence, not a core contribution.
2. **Accuracy gap with SOTA:** LLM-based methods (95%+) are far ahead; do not claim accuracy parity. Frame as Pareto trade-off.
3. **Count questions:** DSL soft counting now provides symbolic logits for count queries via the PoE fusion path, but accuracy may still lag other subtypes. Acknowledge as area for improvement.
4. **VQA-RAD / SLAKE adapters:** Implemented but not trained/evaluated; claim transfer *potential*, not demonstrated transfer.
5. **If differentiable pipeline doesn't beat hard pipeline:** The ablation is still valuable (shows when differentiability matters). Frame as analysis, not failure.
6. **C1 gate entropy:** `ent(attn)` is wired as a gate input (P4/B3 done). Safe to claim in the paper.
7. **C7 faithfulness metric:** `faithfulness_test()` bug is fixed (B1 done). Faithfulness numbers are valid to report.
