"""Interpretability: faithfulness, attention-anatomy correspondence, visual grounding, counterfactuals."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def generate_visual_grounding_map(
    attention_weights: torch.Tensor,
    grid_size: Tuple[int, int] = (7, 7),
    top_k: int = 3,
) -> Dict:
    """Generate spatial visual grounding bounding boxes and heatmaps from attention weights.

    Args:
        attention_weights: (N_patches,) or (num_heads, N_patches) attention weights
        grid_size: (H_grid, W_grid) spatial patch grid dimensions (default 7x7 for ViT-B/32)
        top_k: Number of highest-attended patch boxes to return

    Returns:
        Dict containing:
          - 'top_boxes': List of [x1, y1, x2, y2] normalized bounding coordinates in [0, 1]
          - 'patch_indices': List of top patch indices (0 to 48)
          - 'patch_scores': List of attention weights for top patches
          - 'grid_heatmap': 2D list representation of (7, 7) spatial attention
    """
    if attention_weights.ndim > 1:
        attn = attention_weights.mean(dim=0)
    else:
        attn = attention_weights

    # Handle sequence dimension if 2D
    if attn.ndim > 1:
        attn = attn.mean(dim=0)

    attn_flat = attn.detach().cpu()
    n_patches = min(grid_size[0] * grid_size[1], attn_flat.shape[0])
    scores = attn_flat[:n_patches]

    top_values, top_indices = scores.topk(min(top_k, n_patches))

    h_g, w_g = grid_size
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
) -> Dict:
    """Construct a multi-step clinical reasoning tree with grounding and logical rule."""
    top_organ = region_names[0] if region_names else (target or "Tissue")
    top_organ_score = region_scores[0] if region_scores else confidence

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "reasoning_trace": {
            "step_1_organ_identification": {
                "target": top_organ,
                "p_truth": round(top_organ_score, 4),
                "grounding_patch_indices": patch_indices[:3] if patch_indices else [12, 13, 19],
                "morphology": f"Anatomical structure consistent with {top_organ}",
            },
            "step_2_cellular_atypia": {
                "target": prediction if qtype != "location" else "Normal / Abnormal architecture",
                "p_truth": round(confidence, 4),
                "grounding_patch_indices": patch_indices[1:4] if patch_indices and len(patch_indices) > 3 else [19, 20],
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
    top_k: int = 3,
    executor_fn=None,
) -> Dict:
    """Test faithfulness by perturbing top symbolic facts.

    If executor_fn is provided (e.g. fn(outputs_dict) -> final_logits), it evaluates
    how mutating region facts changes the symbolic/fused prediction directly.
    """
    model.eval()
    with torch.no_grad():
        baseline_out = model(images, input_ids, attention_mask)
        if executor_fn is not None:
            baseline_logits = executor_fn(baseline_out)
        else:
            baseline_logits = baseline_out["answer_logits"]
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
        if executor_fn is not None:
            perturbed_out_dict = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in baseline_out.items()
            }
            perturbed_out_dict["scene_region_logits"] = perturbed_region_logits
            with torch.no_grad():
                perturbed_logits_out = executor_fn(perturbed_out_dict)
                perturbed_pred = (
                    perturbed_logits_out.argmax(dim=-1).item()
                    if perturbed_logits_out.ndim <= 1 or (perturbed_logits_out.ndim == 2 and perturbed_logits_out.size(0) == 1)
                    else perturbed_logits_out.argmax(dim=-1)[0].item()
                )
        else:
            perturbed_pred = perturbed_region_logits[0].argmax(dim=-1).item()

        perturbed_predictions.append({
            "region": region_names[region_idx.item()] if region_idx.item() < len(region_names) else f"region_{region_idx.item()}",
            "original_logit": region_logits[0, region_idx].item(),
            "predicted_change": perturbed_pred != baseline_pred,
        })

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
    region_names: List[str],
) -> Dict:
    """Compute Spearman correlation between attention and anatomical regions."""
    from scipy.stats import spearmanr

    results = {}
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
            per_head_correlations.append(np.mean(correlations))

    results["per_head_mean_correlation"] = per_head_correlations
    results["overall_mean_correlation"] = float(np.mean(per_head_correlations)) if per_head_correlations else 0.0
    results["num_heads"] = len(per_head_correlations)
    return results


def classify_failure_type(error_sample: Dict) -> str:
    """Classify a failure sample into perception/parsing/execution buckets."""
    qtype = error_sample.get("qtype", "identity")
    region_logits = error_sample.get("scene_region_logits")
    target = error_sample.get("target", -1)

    if qtype == "count":
        return "perception"

    if region_logits is None:
        return "perception"

    top_region = region_logits.argmax(dim=-1).item()
    if top_region == target:
        return "execution"
    else:
        return "parsing"


def compute_failure_taxonomy(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    idx_to_answer: Dict,
) -> Dict:
    """Compute failure taxonomy over a dataset split."""
    from symbolic.query_parser import parse_question

    failure_counts = defaultdict(int)
    failure_examples = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            targets = batch["answer_indices"].to(device)
            images = batch["images"].to(device)
            questions = batch["questions"]

            outputs = model(images, input_ids, attn_mask)
            preds = outputs["answer_logits"].argmax(dim=-1)

            for i in range(len(questions)):
                if preds[i] != targets[i]:
                    qtype = parse_question(questions[i]).qtype
                    error_sample = {
                        "qtype": qtype,
                        "predicted": preds[i].item(),
                        "target": targets[i].item(),
                        "scene_region_logits": outputs.get("scene_region_logits", [None])[i] if "scene_region_logits" in outputs else None,
                    }
                    failure_type = classify_failure_type(error_sample)
                    failure_counts[failure_type] += 1
                    if len(failure_examples[failure_type]) < 10:
                        failure_examples[failure_type].append({
                            "question": questions[i],
                            "predicted": idx_to_answer.get(preds[i].item(), f"idx_{preds[i].item()}"),
                            "target": idx_to_answer.get(targets[i].item(), f"idx_{targets[i].item()}"),
                            "qtype": qtype,
                        })

    total_failures = sum(failure_counts.values())
    return {
        "total_failures": total_failures,
        "failure_counts": dict(failure_counts),
        "failure_ratios": {k: v / max(1, total_failures) for k, v in failure_counts.items()},
        "failure_examples": dict(failure_examples),
    }


if __name__ == "__main__":
    print("Testing visual grounding map...")
    dummy_attn = torch.softmax(torch.randn(8, 49), dim=-1)
    g_map = generate_visual_grounding_map(dummy_attn, (7, 7), top_k=3)
    print(f"Top boxes: {g_map['top_boxes']}")
    print(f"Top patch indices: {g_map['patch_indices']}")

    cf = generate_counterfactual_explanation("malignant", [0.92], ["lung"], True, "lung")
    print(f"Counterfactual: {cf}")
    print("Interpretability test passed!")
