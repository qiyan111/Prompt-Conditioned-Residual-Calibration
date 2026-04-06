import argparse

import matplotlib.pyplot as plt
import numpy as np


LABELS = [
    "Full",
    "w/o\nfunnel",
    "w/o\nPromptMHA",
    "w/o residual\nlearning",
    "Frozen\nlinear probe",
]

AIGCIQA2023 = {
    "C-SROCC": [0.8356, 0.8159, 0.8155, 0.8046, 0.7030],
    "C-PLCC": [0.8134, 0.8000, 0.7993, 0.7874, 0.6888],
    "Q-SROCC": [0.8629, 0.8482, 0.8439, 0.8541, 0.7914],
    "Q-PLCC": [0.8741, 0.8602, 0.8623, 0.8679, 0.8014],
}


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0012,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4d4d4d",
        )


def parse_args():
    parser = argparse.ArgumentParser("make_fig3_ablation_bars")
    parser.add_argument("--output_pdf", default="Fig3_ablation_bars.pdf")
    parser.add_argument("--output_png", default="Fig3_ablation_bars.png")
    return parser.parse_args()


def main():
    args = parse_args()
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    x = np.arange(len(LABELS))
    width = 0.34

    teal = "#5A9EA6"
    orange = "#E9A047"
    edge = "#404040"

    alignment_srocc = AIGCIQA2023["C-SROCC"]
    alignment_plcc = AIGCIQA2023["C-PLCC"]
    quality_srocc = AIGCIQA2023["Q-SROCC"]
    quality_plcc = AIGCIQA2023["Q-PLCC"]

    ax = axes[0]
    bars1 = ax.bar(x - width / 2, alignment_srocc, width, label="C-SROCC", color=teal, edgecolor=edge)
    bars2 = ax.bar(x + width / 2, alignment_plcc, width, label="C-PLCC", color=orange, edgecolor=edge)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_title("(a) Alignment Metrics on AIGCIQA2023", fontsize=16)
    ax.set_ylabel("Correlation", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=12)
    ax.set_ylim(0.67, 0.845)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=12, frameon=True)

    ax = axes[1]
    bars3 = ax.bar(x - width / 2, quality_srocc, width, label="Q-SROCC", color=teal, edgecolor=edge)
    bars4 = ax.bar(x + width / 2, quality_plcc, width, label="Q-PLCC", color=orange, edgecolor=edge)
    annotate_bars(ax, bars3)
    annotate_bars(ax, bars4)
    ax.set_title("(b) Quality Metrics on AIGCIQA2023", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=12)
    ax.set_ylim(0.78, 0.878)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=12, frameon=True)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(edge)
            spine.set_linewidth(1.0)
        ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    fig.savefig(args.output_png, dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
