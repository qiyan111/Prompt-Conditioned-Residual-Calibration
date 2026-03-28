from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


CSV_PATH = Path(r"D:\Download\val_preds (2).csv")
OUTPUT_PDF = Path("Fig4_residual_scatter.pdf")
OUTPUT_PNG = Path("Fig4_residual_scatter.png")


def main():
    df = pd.read_csv(CSV_PATH)
    target = df["target_c"].astype(float).to_numpy()
    coarse = df["pred_c_coarse"].astype(float).to_numpy()
    final = df["pred_c"].astype(float).to_numpy()

    coarse_srocc = spearmanr(target, coarse).statistic
    coarse_plcc = pearsonr(target, coarse).statistic
    final_srocc = spearmanr(target, final).statistic
    final_plcc = pearsonr(target, final).statistic

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    base_color = "#E9A047"
    final_color = "#5A9EA6"
    edge = "#404040"

    ax.scatter(
        target,
        coarse,
        s=22,
        alpha=0.35,
        color=base_color,
        edgecolors="none",
        label=f"Base consistency head\nSROCC={coarse_srocc:.3f}, PLCC={coarse_plcc:.3f}",
    )
    ax.scatter(
        target,
        final,
        s=22,
        alpha=0.40,
        color=final_color,
        edgecolors="none",
        label=f"Final calibrated score\nSROCC={final_srocc:.3f}, PLCC={final_plcc:.3f}",
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#808080", linewidth=1.2, label="Ideal diagonal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("GT alignment MOS (normalized)", fontsize=13)
    ax.set_ylabel("Predicted alignment score", fontsize=13)
    ax.set_title("Residual Calibration Effect on AGIQA-3K", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    for spine in ax.spines.values():
        spine.set_color(edge)
        spine.set_linewidth(1.0)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")

    print(f"coarse_srocc={coarse_srocc:.6f}")
    print(f"coarse_plcc={coarse_plcc:.6f}")
    print(f"final_srocc={final_srocc:.6f}")
    print(f"final_plcc={final_plcc:.6f}")


if __name__ == "__main__":
    main()
