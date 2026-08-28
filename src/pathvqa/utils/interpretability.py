"""Interpretability: faithfulness, attention-anatomy correspondence, visual grounding, counterfactuals."""

from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PIL.Image


def generate_visual_grounding_map(
    attention_weights: torch.Tensor,
    grid_size: Tuple[int, int] = (7, 7),
    top_k: int = 3,
) -> Dict[str, Any]:
    """Generate spatial visual grounding bounding boxes and heatmaps from attention weights.

    Args:
        attention_weights: (N_patches,) or (num_heads, N_patches) attention weights
        grid_size: (H_grid, W_grid) spatial patch grid dimensions (default 7x7 for ViT-B/32)
        top_k: Number of highest-attended patch boxes to return

    Returns:
        Dict containing:
          - 'top_boxes': List of [x1, y1, x2, y2] normalized bounding coordinates in [0, 1]
          - 'patch_indices': List of top patch indices (0 .. grid_size product)
          - 'patch_scores': List of attention weights for top patches
          - 'grid_heatmap': 2D list representation of the (H_grid, W_grid) spatial attention
    """
    if attention_weights.ndim > 1:
        attn = attention_weights.mean(dim=0)
    else:
        attn = attention_weights

    # Collapse any residual sequence dim so we always hold one score per patch.
    if attn.ndim > 1:
        attn = attn.mean(dim=0)

    attn_flat = attn.detach().cpu()
    n_attn = attn_flat.shape[0]
    h_g, w_g = grid_size
    n_patches = min(h_g * w_g, n_attn)
    if n_patches < h_g * w_g:
        # Fewer patches than the grid holds (e.g. non-default num_objects):
        # zero-pad so the reshape stays valid; missing cells contribute nothing.
        scores = torch.zeros(h_g * w_g)
        scores[:n_patches] = attn_flat[:n_patches]
    else:
        scores = attn_flat[:n_patches]

    top_values, top_indices = scores.topk(min(top_k, scores.shape[0]))

    box_list = []
    for idx in top_indices:
        r = idx.item() // w_g
        c = idx.item() % w_g
        x1, y1 = c / w_g, r / h_g
        x2, y2 = (c + 1) / w_g, (r + 1) / h_g
        box_list.append([round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)])

    heatmap = scores.view(h_g, w_g).numpy().tolist()
    return {
        "top_boxes": box_list,
        "patch_indices": top_indices.tolist(),
        "patch_scores": [round(v.item(), 4) for v in top_values],
        "grid_heatmap": [[round(val, 4) for val in row] for row in heatmap],
    }


def generate_counterfactual_explanation(
    baseline_prediction_str: str,
    region_scores: List[float],
    region_names: List[str],
    predicted_changed: bool = False,
    perturbed_region: Optional[str] = None,
) -> str:
    """Construct a counterfactual explanation sentence."""
    if not region_names:
        return f"Prediction '{baseline_prediction_str}' was driven primarily by global neural visual representations."

    top_region = perturbed_region or region_names[0]
    top_score = region_scores[0] if region_scores else 0.0

    if predicted_changed:
        return (
            f"The model predicts '{baseline_prediction_str}' grounded in high confidence for '{top_region}' "
            f"({top_score:.2f}). Perturbing or masking this anatomical feature flips the predicted diagnosis."
        )
    return (
        f"The prediction '{baseline_prediction_str}' is supported by '{top_region}' with confidence {top_score:.2f}. "
        "The model maintains confidence across minor variations in surrounding anatomical context."
    )


def generate_hierarchical_reasoning_tree(
    prediction: str,
    confidence: float,
    qtype: str,
    target: str,
    region_names: List[str],
    region_scores: List[float],
    patch_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Construct a multi-step clinical reasoning tree with grounding and logical rule.

    Grounding patch indices are echoed only when actually provided (e.g. from
    attention grounding). No placeholder indices are ever fabricated: an
    explainability artifact must not invent evidence.
    """
    top_organ = region_names[0] if region_names else (target or "Tissue")
    top_organ_score = region_scores[0] if region_scores else confidence

    step1_grounding = sorted(set(patch_indices[:3])) if patch_indices else []
    step2_grounding = sorted(set(patch_indices[1:4])) if patch_indices and len(patch_indices) > 3 else []

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "reasoning_trace": {
            "step_1_organ_identification": {
                "target": top_organ,
                "p_truth": round(top_organ_score, 4),
                "grounding_patch_indices": step1_grounding,
                "morphology": f"Anatomical structure consistent with {top_organ}",
            },
            "step_2_cellular_atypia": {
                "target": prediction if qtype != "location" else "Normal / Abnormal architecture",
                "p_truth": round(confidence, 4),
                "grounding_patch_indices": step2_grounding,
            },
            "step_3_rule_deduction": {
                "rule": f"LocatedIn({top_organ}) ∧ Feature({target or prediction}) → {prediction}",
                "rule_satisfaction": round(min(confidence, top_organ_score), 4),
            },
        },
    }


def faithfulness_test(
    model: torch.nn.Module,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    region_logits: torch.Tensor,
    region_names: List[str],
    executor_fn: Any,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Measure how much perturbing the top regions changes the prediction.

    ``executor_fn(outputs)`` re-decodes a scene into answer-space logits; the top
    ``top_k`` regions are each set to -1e9 and re-decoded so every comparison
    stays in answer space.
    """
    model.eval()
    with torch.no_grad():
        baseline_out = model(images, input_ids, attention_mask)
        baseline_logits = executor_fn(baseline_out)
        baseline_pred = (
            baseline_logits.argmax(dim=-1).item()
            if baseline_logits.ndim <= 1 or (baseline_logits.ndim == 2 and baseline_logits.size(0) == 1)
            else baseline_logits.argmax(dim=-1)[0].item()
        )

    k_actual = min(top_k, region_logits.size(-1))
    top_regions = region_logits[0].topk(k_actual).indices
    perturbed_predictions = []

    for region_idx in top_regions:
        perturbed_region_logits = region_logits.clone()
        perturbed_region_logits[0, region_idx] = -1e9
        if isinstance(baseline_out, dict):
            perturbed_out_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in baseline_out.items()}
            perturbed_out_dict["scene_region_logits"] = perturbed_region_logits
        else:
            perturbed_out_dict = replace(baseline_out, scene_region_logits=perturbed_region_logits)
        with torch.no_grad():
            perturbed_logits_out = executor_fn(perturbed_out_dict)
            perturbed_pred = (
                perturbed_logits_out.argmax(dim=-1).item()
                if perturbed_logits_out.ndim <= 1
                or (perturbed_logits_out.ndim == 2 and perturbed_logits_out.size(0) == 1)
                else perturbed_logits_out.argmax(dim=-1)[0].item()
            )

        perturbed_predictions.append(
            {
                "region": region_names[region_idx.item()]
                if region_idx.item() < len(region_names)
                else f"region_{region_idx.item()}",
                "original_logit": region_logits[0, region_idx].item(),
                "predicted_change": perturbed_pred != baseline_pred,
            }
        )

    n_changed = sum(1 for p in perturbed_predictions if p["predicted_change"])
    return {
        "baseline_prediction": baseline_pred,
        "top_regions_perturbed": len(top_regions),
        "predictions_changed": n_changed,
        "faithfulness_score": n_changed / max(1, len(top_regions)),
        "perturbations": perturbed_predictions,
    }


def attention_anatomy_correspondence(
    attention_weights: torch.Tensor,
    region_positions: torch.Tensor,
) -> Dict[str, Any]:
    """Compute Spearman correlation between attention and anatomical regions."""
    from scipy.stats import spearmanr

    results: Dict[str, Any] = {}
    per_head_correlations = []
    for head_idx in range(attention_weights.shape[0]):
        head_attn = attention_weights[head_idx].mean(dim=0)
        correlations = []
        for region_idx in range(region_positions.shape[0]):
            mask = region_positions[region_idx].bool()
            if mask.any() and (~mask).any():
                region_signal = mask.float().cpu().numpy()
                corr, _ = spearmanr(head_attn.cpu().numpy(), region_signal)
                if not np.isnan(corr):
                    correlations.append(corr)
        if correlations:
            per_head_correlations.append(float(np.mean(correlations)))

    results["per_head_mean_correlation"] = per_head_correlations
    results["overall_mean_correlation"] = float(np.mean(per_head_correlations)) if per_head_correlations else 0.0
    results["num_heads"] = len(per_head_correlations)
    return results


def classify_failure_type(error_sample: Dict[str, Any]) -> str:
    """Classify a failure sample into perception/parsing/execution buckets.

    Preferred input resolves index-space conflation by comparing *names*:
    provide ``top_region_name`` (str) and ``target_answer`` (str). When only
    legacy integer fields are present, ``scene_region_logits`` argmax is
    compared against ``target_region`` — a region-space index, valid only when
    the caller supplies it explicitly.
    """
    qtype = error_sample.get("qtype", "identity")
    if qtype == "count":
        return "perception"

    top_region_name = error_sample.get("top_region_name")
    target_answer = error_sample.get("target_answer")
    if top_region_name is not None and target_answer is not None:
        # Name-space comparison: same anatomical structure predicted as the
        # answer → scene parsing succeeded, so blame execution.
        return "execution" if str(top_region_name).lower() == str(target_answer).lower() else "parsing"

    region_logits = error_sample.get("scene_region_logits")
    if region_logits is None:
        return "perception"

    # Legacy integer fallback: region-space and answer-vocab indices are
    # different spaces, so this is only meaningful when the caller supplies a
    # region-space target explicitly (``target_region``). Without it we cannot
    # attribute the failure and fall through to the catch-all bucket.
    target_region = error_sample.get("target_region")
    if target_region is None:
        return "perception"

    top_region = int(region_logits.argmax(dim=-1).item())
    if top_region == target_region:
        return "execution"
    else:
        return "parsing"


def _batch_field(batch: Any, name: str, legacy: Optional[str] = None) -> Any:
    """Read a field from a BatchData/CollateOutput dataclass or a legacy dict."""
    if isinstance(batch, dict):
        return batch.get(name, batch.get(legacy) if legacy else None)
    return getattr(batch, name, None)


def compute_failure_taxonomy(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    idx_to_answer: Dict[int, str],
    region_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute the failure taxonomy over a dataset split.

    Attribution is name-based: with ``region_names`` the top region name is
    compared against the target answer to separate parsing from execution;
    otherwise every failure is reported as ``perception``.
    """
    from pathvqa.symbolic.query_parser import parse_question

    failure_counts: Dict[str, int] = defaultdict(int)
    failure_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = _batch_field(batch, "images").to(device)
            targets = _batch_field(batch, "targets", legacy="answer_indices").to(device)
            questions = _batch_field(batch, "questions") or []
            prepared_input_ids = _batch_field(batch, "input_ids")
            prepared_mask = _batch_field(batch, "attention_mask")

            if prepared_input_ids is not None:
                input_ids = prepared_input_ids.to(device)
                attn_mask = prepared_mask.to(device)
            else:
                # CollateOutput has no tokenized text; tokenize on the fly via
                # the model's tokenizer when available.
                tokenizer = getattr(model, "tokenizer", None)
                if tokenizer is None:
                    raise ValueError(
                        "compute_failure_taxonomy needs tokenized batches (BatchData) "
                        "or a model exposing a .tokenizer attribute"
                    )
                enc = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)

            outputs = model(images, input_ids, attn_mask)
            preds = outputs.answer_logits.argmax(dim=-1)

            for i in range(len(questions)):
                if preds[i] != targets[i]:
                    qtype = parse_question(questions[i]).qtype
                    region_logits_i = (
                        outputs.scene_region_logits[i] if outputs.scene_region_logits is not None else None
                    )
                    error_sample: Dict[str, Any] = {
                        "qtype": qtype,
                        "predicted": preds[i].item(),
                        "target": targets[i].item(),
                        "scene_region_logits": region_logits_i,
                    }
                    if region_names is not None and region_logits_i is not None:
                        top_r = region_logits_i.argmax().item()
                        error_sample["top_region_name"] = (
                            region_names[top_r] if top_r < len(region_names) else f"region_{top_r}"
                        )
                        error_sample["target_answer"] = idx_to_answer.get(targets[i].item(), "")
                    failure_type = classify_failure_type(error_sample)
                    failure_counts[failure_type] += 1
                    if len(failure_examples[failure_type]) < 10:
                        failure_examples[failure_type].append(
                            {
                                "question": questions[i],
                                "predicted": idx_to_answer.get(preds[i].item(), f"idx_{preds[i].item()}"),
                                "target": idx_to_answer.get(targets[i].item(), f"idx_{targets[i].item()}"),
                                "qtype": qtype,
                            }
                        )

    total_failures = sum(failure_counts.values())
    return {
        "total_failures": total_failures,
        "failure_counts": dict(failure_counts),
        "failure_ratios": {k: v / max(1, total_failures) for k, v in failure_counts.items()},
        "failure_examples": dict(failure_examples),
    }


def render_bounding_boxes(
    image: "PIL.Image.Image", boxes: List[List[float]], color: str = "red", width: int = 3
) -> "PIL.Image.Image":
    """Draw top-k bounding boxes on the image."""
    from PIL import ImageDraw

    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=color, width=width)
    return out


def render_attention_heatmap(
    image: "PIL.Image.Image", heatmap_2d: List[List[float]], alpha: float = 0.5
) -> "PIL.Image.Image":
    """Overlay a spatial attention heatmap on the image."""
    from PIL import Image

    h_g = len(heatmap_2d)
    w_g = len(heatmap_2d[0]) if h_g > 0 else 0
    flat = [val for row in heatmap_2d for val in row]
    max_val = max(flat) if flat else 1.0
    max_val = max_val if max_val > 0 else 1.0

    heatmap_img = Image.new("L", (w_g, h_g))
    heatmap_img.putdata([int((val / max_val) * 255) for val in flat])
    heatmap_img = heatmap_img.resize(image.size, Image.Resampling.BILINEAR)

    red_overlay = Image.new("RGB", image.size, (255, 0, 0))
    mask = heatmap_img.point(lambda p: int(p * alpha))
    out = Image.composite(red_overlay, image.convert("RGB"), mask)
    return out


def render_symbolic_trace(trace: Dict[str, Any], size: Tuple[int, int]) -> "PIL.Image.Image":
    """Render the symbolic trace dictionary as a text image."""
    from PIL import Image, ImageDraw

    out = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(out)
    y = 10
    draw.text((10, y), f"Prediction: {trace.get('prediction', '')}", fill="black")
    y += 20
    draw.text((10, y), f"Confidence: {trace.get('confidence', 0.0)}", fill="black")
    y += 20
    for k, v in trace.get("reasoning_trace", {}).items():
        draw.text((10, y), f"{k}:", fill="blue")
        y += 20
        for sub_k, sub_v in v.items():
            draw.text((20, y), f"{sub_k}: {sub_v}", fill="black")
            y += 20
    return out


def render_composite_view(
    image: "PIL.Image.Image", boxes: List[List[float]], heatmap_2d: List[List[float]], trace: Dict[str, Any]
) -> "PIL.Image.Image":
    """Create a side-by-side composite: original + boxes, heatmap, and trace."""
    from PIL import Image

    box_img = render_bounding_boxes(image, boxes)
    heat_img = render_attention_heatmap(image, heatmap_2d)
    trace_img = render_symbolic_trace(trace, size=(image.width, image.height))

    composite = Image.new("RGB", (image.width * 3, image.height))
    composite.paste(box_img, (0, 0))
    composite.paste(heat_img, (image.width, 0))
    composite.paste(trace_img, (image.width * 2, 0))
    return composite


if __name__ == "__main__":
    print("Testing visual grounding map...")
    dummy_attn = torch.softmax(torch.randn(8, 49), dim=-1)
    g_map = generate_visual_grounding_map(dummy_attn, (7, 7), top_k=3)
    print(f"Top boxes: {g_map['top_boxes']}")
    print(f"Top patch indices: {g_map['patch_indices']}")

    cf = generate_counterfactual_explanation("malignant", [0.92], ["lung"], True, "lung")
    print(f"Counterfactual: {cf}")
    print("Interpretability test passed!")
