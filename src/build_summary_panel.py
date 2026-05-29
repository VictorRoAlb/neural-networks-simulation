from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


PANELS = [
    (
        "class_distribution_and_weights.png",
        "Class imbalance",
        "Distribution and class weights used to compensate the minority class.",
    ),
    (
        "training_history_reference.png",
        "Learning curves",
        "Accuracy, loss and Macro F1 trajectories for the selected network.",
    ),
    (
        "model_comparison_summary.png",
        "Model comparison",
        "Validation and test Macro F1 comparison across all published candidates.",
    ),
    (
        "final_confusion_matrix.png",
        "Final confusion matrix",
        "Class-wise test performance for the selected burnout classifier.",
    ),
]


def load_image(name: str):
    return mpimg.imread(FIGURES / name)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="white")

    for ax, (filename, title, subtitle) in zip(axes.flat, PANELS):
        ax.imshow(load_image(filename))
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.text(
            0.5,
            -0.07,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#555555",
        )
        ax.axis("off")

    fig.suptitle(
        "Burnout multiclass neural-network project summary",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "Public summary panel combining the four figures used to explain the final coursework narrative.",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08, wspace=0.08, hspace=0.24)

    out_path = FIGURES / "neural_project_summary_panel.png"
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
