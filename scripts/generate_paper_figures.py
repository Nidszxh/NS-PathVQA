import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def setup_style():
    """Setup matplotlib style for publication-ready figures."""
    plt.style.use('ggplot')
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

def generate_mock_data():
    """Generates synthetic data mirroring the expected evaluation outputs."""
    return {
        "qtype_heatmap": {
            "matrix": np.array([
                [0.92, 0.02, 0.01, 0.04, 0.01],
                [0.05, 0.88, 0.03, 0.02, 0.02],
                [0.01, 0.04, 0.95, 0.00, 0.00],
                [0.10, 0.05, 0.02, 0.80, 0.03],
                [0.02, 0.02, 0.01, 0.00, 0.95]
            ]),
            "labels": ["Identity", "Location", "Yes/No", "Attribute", "Count"]
        },
        "concept_attribution": {
            "concepts": ["nucleus", "stroma", "lymphocyte", "gland", "mass", "necrosis", "mitosis"],
            "attribution_scores": [0.35, 0.15, 0.20, 0.12, 0.10, 0.05, 0.03]
        },
        "intervention_delta": {
            "qtypes": ["Identity", "Location", "Yes/No", "Attribute", "Count"],
            "deltas": [4.2, 5.8, 2.1, 6.4, 3.5]
        },
        "failure_taxonomy": {
            "categories": ["Visual Perception", "Query Parsing", "Symbolic Execution", "Ambiguous Ground Truth"],
            "percentages": [45, 15, 25, 15]
        },
        "efficiency_scatter": {
            "models": ["Neuro-Symbolic PathVQA", "LLaVA-Med", "Med-Flamingo", "PathVQA-Base"],
            "accuracy": [82.4, 84.1, 85.5, 76.2],
            "parameters_b": [0.15, 7.0, 9.0, 0.11]  # In billions
        }
    }

def plot_qtype_heatmap(data, out_dir):
    """Figure 2: Soft qtype distribution heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data["matrix"], cmap="Blues")
    plt.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(data["labels"])))
    ax.set_yticks(np.arange(len(data["labels"])))
    ax.set_xticklabels(data["labels"])
    ax.set_yticklabels(data["labels"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(data["labels"])):
        for j in range(len(data["labels"])):
            ax.text(j, i, f"{data['matrix'][i, j]:.2f}", ha="center", va="center", 
                    color="white" if data["matrix"][i, j] > 0.5 else "black")
                    
    ax.set_title("Differentiable Query Classifier Confusion")
    ax.set_xlabel("Predicted Soft Q-Type")
    ax.set_ylabel("True Regex Q-Type")
    fig.savefig(out_dir / "fig2_qtype_heatmap.png")
    plt.close(fig)

def plot_concept_attribution(data, out_dir):
    """Figure 3: Concept attribution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(data["concepts"]))
    ax.bar(x_pos, data["attribution_scores"], color=plt.cm.viridis(np.linspace(0, 1, len(data["concepts"]))))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(data["concepts"], rotation=45)
    ax.set_title("Average Concept Attribution Saliency")
    ax.set_ylabel("Normalized Attribution Score")
    ax.set_xlabel("Scene Concepts (Regions)")
    fig.savefig(out_dir / "fig3_concept_attribution.png")
    plt.close(fig)

def plot_intervention_delta(data, out_dir):
    """Figure 4: Intervention accuracy delta per question type."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos = np.arange(len(data["qtypes"]))
    bars = ax.bar(x_pos, data["deltas"], color=plt.cm.magma(np.linspace(0.2, 0.8, len(data["qtypes"]))))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(data["qtypes"])
    ax.set_title("Accuracy Recovery via Concept Interventions")
    ax.set_ylabel("Accuracy Delta (%)")
    ax.set_xlabel("Question Type")
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'+{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
                    
    fig.savefig(out_dir / "fig4_intervention_delta.png")
    plt.close(fig)

def plot_failure_taxonomy(data, out_dir):
    """Figure 5: Failure taxonomy breakdown pie chart."""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(data["categories"])))
    ax.pie(data["percentages"], labels=data["categories"], colors=colors, autopct='%.1f%%',
           startangle=140, wedgeprops={'edgecolor': 'white'})
    ax.set_title("Error Taxonomy Breakdown (C7)")
    fig.savefig(out_dir / "fig5_failure_taxonomy.png")
    plt.close(fig)

def plot_efficiency_scatter(data, out_dir):
    """Figure 6: Efficiency scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    markers = ["*", "o", "s", "X"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(data["models"])))
    
    for i, model in enumerate(data["models"]):
        ax.scatter(data["parameters_b"][i], data["accuracy"][i], s=150, 
                   label=model, marker=markers[i], color=colors[i])
                    
    ax.set_xscale("log")
    ax.set_title("Pareto Trade-off: Accuracy vs. Model Size")
    ax.set_xlabel("Parameters (Billions) - Log Scale")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.savefig(out_dir / "fig6_efficiency_scatter.png", bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures for Neuro-Symbolic PathVQA.")
    parser.add_argument("--eval-results", type=str, help="Path to actual eval_results.json")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic data (useful before full intervention runs)")
    parser.add_argument("--out-dir", type=str, default="outputs/figures", help="Directory to save output figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    if args.mock:
        print("Using synthetic mock data to generate figures...")
        data = generate_mock_data()
    else:
        if not args.eval_results or not os.path.exists(args.eval_results):
            print("Error: Must provide --eval-results path or use --mock.")
            return
            
        print(f"Parsing actual results from {args.eval_results}...")
        # Fall back to mock data for demonstration
        print("Real data parsing is a placeholder. Falling back to mock data for demonstration.")
        data = generate_mock_data()

    print(f"Saving figures to {out_dir}...")
    plot_qtype_heatmap(data["qtype_heatmap"], out_dir)
    plot_concept_attribution(data["concept_attribution"], out_dir)
    plot_intervention_delta(data["intervention_delta"], out_dir)
    plot_failure_taxonomy(data["failure_taxonomy"], out_dir)
    plot_efficiency_scatter(data["efficiency_scatter"], out_dir)
    print("Done! All 5 figures generated successfully.")

if __name__ == "__main__":
    main()
