"""Publication figures from *real* evaluation artifacts.

Every figure is derived from fields actually present in an ``evaluate.py``
output JSON. Nothing is fabricated: if a required field is missing the figure
is skipped with a message rather than filled with synthetic numbers.

Use ``--mock`` only to smoke-test the plotting pipeline; mock figures are
watermarked and must never be used as empirical evidence.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    """Setup matplotlib style for publication-ready figures."""
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def generate_mock_data() -> Dict[str, Any]:
    """Synthetic data with the same shape as a real eval artifact (--mock only)."""
    return {
        "accuracy": 82.4,
        "qtype_confusion": {
            "matrix": [
                [92, 2, 1, 4, 1],
                [5, 88, 3, 2, 2],
                [1, 4, 95, 0, 0],
                [10, 5, 2, 80, 3],
                [2, 2, 1, 0, 95],
            ],
            "labels": ["identity", "location", "yes_no", "attribute", "count"],
        },
        "per_type_accuracy": {
            "identity": {"accuracy": 92.0, "correct": 92, "total": 100},
            "location": {"accuracy": 88.0, "correct": 88, "total": 100},
            "yes_no": {"accuracy": 95.0, "correct": 95, "total": 100},
            "attribute": {"accuracy": 80.0, "correct": 80, "total": 100},
            "count": {"accuracy": 95.0, "correct": 95, "total": 100},
        },
        "calibration": {
            "ece": 0.04,
            "temperature": 1.1,
            "reliability_bins": [
                {
                    "bin_lower": i / 10,
                    "bin_upper": (i + 1) / 10,
                    "accuracy": min(1.0, i / 10 + 0.05),
                    "confidence": i / 10 + 0.03,
                    "count": 100,
                }
                for i in range(10)
            ],
        },
        "failure_taxonomy": {
            "failure_counts": {"perception": 45, "parsing": 15, "execution": 25},
        },
        "concept_interventions": {"probe_accuracy_delta": 4.2},
    }


def _annotate_mock(fig) -> None:
    fig.text(0.5, 0.01, "SYNTHETIC (--mock) — NOT EMPIRICAL EVIDENCE", ha="center", color="red", fontsize=10)


def plot_qtype_heatmap(eval_data: Dict[str, Any], out_dir: Path, mock: bool) -> bool:
    """Row-normalized q-type confusion matrix (soft classifier vs regex labels)."""
    conf = eval_data.get("qtype_confusion")
    if not conf:
        return False
    matrix = np.asarray(conf["matrix"], dtype=float)
    labels = conf.get("labels", [str(i) for i in range(matrix.shape[0])])
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.set_title("Differentiable Query Classifier Confusion")
    ax.set_xlabel("Predicted Soft Q-Type")
    ax.set_ylabel("True Regex Q-Type")
    if mock:
        _annotate_mock(fig)
    fig.savefig(out_dir / "fig2_qtype_heatmap.png")
    plt.close(fig)
    return True


def plot_per_type_accuracy(eval_data: Dict[str, Any], out_dir: Path, mock: bool) -> bool:
    """Per-question-type accuracy from the evaluation split."""
    per_type = eval_data.get("per_type_accuracy")
    if not per_type:
        return False
    qtypes = sorted(per_type)
    accs = [per_type[q]["accuracy"] for q in qtypes]

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(qtypes))
    ax.bar(x_pos, accs, color=plt.cm.viridis(np.linspace(0, 1, len(qtypes))))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(qtypes, rotation=45)
    ax.set_ylim(0, 100)
    ax.set_title("Per-Question-Type Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Question Type")
    if mock:
        _annotate_mock(fig)
    fig.savefig(out_dir / "fig3_per_type_accuracy.png")
    plt.close(fig)
    return True


def plot_intervention_delta(eval_data: Dict[str, Any], out_dir: Path, mock: bool) -> bool:
    """Accuracy delta from concept-probe interventions (real runs only)."""
    interventions = eval_data.get("concept_interventions")
    if not interventions or "probe_accuracy_delta" not in interventions:
        return False
    delta = float(interventions["probe_accuracy_delta"])

    fig, ax = plt.subplots(figsize=(5, 4))
    bar = ax.bar(["Concept probe"], [delta], color="#4c72b0")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy Delta (%)")
    ax.set_title("Accuracy Recovery via Concept Interventions")
    ax.annotate(
        f"{delta:+.2f}%",
        xy=(bar[0].get_x() + bar[0].get_width() / 2, delta),
        xytext=(0, 3 if delta >= 0 else -12),
        textcoords="offset points",
        ha="center",
        fontweight="bold",
    )
    if mock:
        _annotate_mock(fig)
    fig.savefig(out_dir / "fig4_intervention_delta.png")
    plt.close(fig)
    return True


def plot_failure_taxonomy(eval_data: Dict[str, Any], out_dir: Path, mock: bool) -> bool:
    """Failure-taxonomy breakdown from measured failure counts."""
    tax = eval_data.get("failure_taxonomy")
    if not tax:
        return False
    counts = tax.get("failure_counts") or {}
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        return False

    fig, ax = plt.subplots(figsize=(6, 6))
    labels = list(counts)
    values = [counts[k] for k in labels]
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
    ax.pie(values, labels=labels, colors=colors, autopct="%.1f%%", startangle=140, wedgeprops={"edgecolor": "white"})
    ax.set_title("Error Taxonomy Breakdown")
    if mock:
        _annotate_mock(fig)
    fig.savefig(out_dir / "fig5_failure_taxonomy.png")
    plt.close(fig)
    return True


def plot_calibration(eval_data: Dict[str, Any], out_dir: Path, mock: bool) -> bool:
    """Reliability diagram from measured calibration bins."""
    calib = eval_data.get("calibration")
    bins: Optional[List[Dict[str, Any]]] = calib.get("reliability_bins") if calib else None
    if not bins:
        return False
    conf = [b["confidence"] for b in bins if b.get("count", 0) > 0]
    acc = [b["accuracy"] for b in bins if b.get("count", 0) > 0]
    if not conf:
        return False

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(conf, acc, marker="o", color="#c44e52", label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ece = calib.get("ece")
    title = "Calibration Reliability Diagram" + (f" (ECE={ece:.3f})" if ece is not None else "")
    ax.set_title(title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="upper left")
    if mock:
        _annotate_mock(fig)
    fig.savefig(out_dir / "fig6_calibration.png")
    plt.close(fig)
    return True


PLOTTERS = [
    plot_qtype_heatmap,
    plot_per_type_accuracy,
    plot_intervention_delta,
    plot_failure_taxonomy,
    plot_calibration,
]


def load_eval_data(eval_path: str) -> Dict[str, Any]:
    """Load a real evaluation artifact."""
    with open(eval_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures for Neuro-Symbolic PathVQA.")
    parser.add_argument("--eval-results", type=str, help="Path to an evaluate.py output JSON (required unless --mock)")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate clearly-watermarked synthetic figures to smoke-test plotting (never empirical)",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/figures", help="Directory to save output figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    if args.mock:
        print("WARNING: generating SYNTHETIC figures (--mock). Not empirical evidence.")
        data = generate_mock_data()
    else:
        if not args.eval_results:
            parser.error("--eval-results is required unless --mock is given")
        if not Path(args.eval_results).exists():
            parser.error(f"eval results not found: {args.eval_results}")
        print(f"Parsing real results from {args.eval_results}...")
        data = load_eval_data(args.eval_results)

    generated = 0
    for plotter in PLOTTERS:
        if plotter(data, out_dir, args.mock):
            generated += 1
        else:
            print(f"  skipped {plotter.__name__}: required fields not present in eval data")
    print(f"Done. {generated}/{len(PLOTTERS)} figures written to {out_dir}.")


if __name__ == "__main__":
    main()
