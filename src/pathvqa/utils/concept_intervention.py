import torch
import torch.nn as nn
from typing import Dict, List, Callable
from pathvqa.symbolic.query_parser import parse_question

def bottleneck_probe(
    model: nn.Module, 
    outputs,
    questions: List[str],
    targets: torch.Tensor,
    region_names: List[str],
    answer_vocab: List[str],
    compute_logits_fn: Callable,
) -> Dict:
    """
    For a batch:
      1. We have the baseline `outputs` and we can compute baseline preds.
      2. Apply override to `outputs.scene_region_logits` based on ground truth.
      3. Re-run executor with corrected concept.
      4. Measure accuracy recovery.
    """
    # Baseline
    with torch.no_grad():
        baseline_logits, _ = compute_logits_fn(outputs, questions)
        baseline_preds = baseline_logits.argmax(dim=-1)
        baseline_correct = (baseline_preds == targets).sum().item()

        # Override
        from dataclasses import replace
        perturbed_outputs = replace(outputs)
        perturbed_outputs.scene_region_logits = outputs.scene_region_logits.clone()

        queries = [parse_question(q, answer_vocab) for q in questions]
        
        for i, q in enumerate(queries):
            if q.target:
                reg_idx = region_names.index(q.target) if q.target in region_names else -1
                if reg_idx >= 0:
                    # Force the model to "perfectly" detect the target region
                    perturbed_outputs.scene_region_logits[i, :] = -10.0 # Suppress others
                    perturbed_outputs.scene_region_logits[i, reg_idx] = 10.0 # Boost target

        corrected_logits, _ = compute_logits_fn(perturbed_outputs, questions)
        corrected_preds = corrected_logits.argmax(dim=-1)
        corrected_correct = (corrected_preds == targets).sum().item()

    return {
        "baseline_correct": baseline_correct,
        "corrected_correct": corrected_correct,
        "total": targets.size(0)
    }

def concept_attribution(
    model: nn.Module, 
    image: torch.Tensor, 
    input_ids: torch.Tensor, 
    attn_mask: torch.Tensor,
    qtype_onehot: torch.Tensor,
    answer_idx: int
) -> Dict[str, torch.Tensor]:
    """
    Compute d(answer_logit[answer_idx]) / d(scene_concept) for each concept.
    """
    model.eval()
    
    region_logits_cache = []
    def hook(module, inp, out):
        region_logits_cache.append(out.region_logits)
        out.region_logits.retain_grad()

    handle = model.scene_parser.register_forward_hook(hook)
    
    # We must ensure gradients can flow
    for param in model.parameters():
        param.requires_grad_(False)
        
    # Forward pass requiring grad
    outputs = model(image, input_ids, attn_mask, qtype_onehot)
    answer_logit = outputs.answer_logits[0, answer_idx]
    
    model.zero_grad()
    answer_logit.backward(retain_graph=True)
    
    handle.remove()
    
    if len(region_logits_cache) > 0 and region_logits_cache[0].grad is not None:
        region_grads = region_logits_cache[0].grad[0]
        region_attr = (region_logits_cache[0][0].detach() * region_grads).abs()
    else:
        # Fallback if graph is detached
        region_attr = torch.zeros_like(outputs.scene_region_logits[0] if outputs.scene_region_logits is not None else torch.zeros(1, 1)[0])
        
    return {"region_attribution": region_attr}

def counterfactual_explanation(
    outputs,
    questions: List[str],
    region_names: List[str],
    idx_to_answer: Dict[int, str],
    compute_logits_fn: Callable
) -> List[Dict]:
    """
    For the top-K predicted regions:
      1. Zero out that region's logit
      2. Re-predict
      3. Record explanations.
    """
    if outputs.scene_region_logits is None:
        return []
        
    with torch.no_grad():
        baseline_logits, _ = compute_logits_fn(outputs, questions)
        baseline_pred_idx = baseline_logits[0].argmax(dim=-1).item()
        baseline_pred = idx_to_answer.get(baseline_pred_idx, str(baseline_pred_idx))
        region_logits = outputs.scene_region_logits[0]
        
        top_regions = region_logits.topk(3).indices
        explanations = []
        
        from dataclasses import replace
        for r_idx in top_regions:
            perturbed_outputs = replace(outputs)
            perturbed_outputs.scene_region_logits = outputs.scene_region_logits.clone()
            perturbed_outputs.scene_region_logits[0, r_idx] = -1e9  # Remove region
            
            perturbed_logits, _ = compute_logits_fn(perturbed_outputs, questions)
            perturbed_pred_idx = perturbed_logits[0].argmax(dim=-1).item()
            perturbed_pred = idx_to_answer.get(perturbed_pred_idx, str(perturbed_pred_idx))
            
            changed = perturbed_pred != baseline_pred
            r_name = region_names[r_idx.item()] if r_idx.item() < len(region_names) else str(r_idx.item())
            
            explanations.append({
                "region": r_name,
                "original_prediction": baseline_pred,
                "counterfactual_prediction": perturbed_pred,
                "prediction_changed": changed,
                "explanation": (
                    f"If '{r_name}' were absent, prediction would change from '{baseline_pred}' to '{perturbed_pred}'"
                    if changed else
                    f"Prediction '{baseline_pred}' is robust to removal of '{r_name}'"
                )
            })
            
    return explanations
