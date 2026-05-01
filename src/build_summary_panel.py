from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def load_image(name: str):
    return mpimg.imread(FIGURES / name)


def main() -> None:
    training = load_image("training_history_reference.png")
    confusion = load_image("final_confusion_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), facecolor="white")
    panels = [
        (training, "Training behaviour", "Representative learning curves from the coursework notebook."),
        (confusion, "Final confusion matrix", "Class-by-class evaluation view from the final classification stage."),
    ]

    for ax, (img, title, subtitle) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#555555",
        )
        ax.axis("off")

    fig.suptitle("Neural networks project summary", fontsize=21, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.03,
        "Clean public snapshot combining the representative training dynamics and the final classification view.",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.14, wspace=0.08)
    out_path = FIGURES / "neural_project_summary_panel.png"
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
