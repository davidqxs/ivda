"""
Regenerate the manuscript's data figures from real experiment results
(results/*.csv), following the dataviz skill's static-figure adaptation:
fixed categorical hue order, single-hue sequential for magnitude, no
dual-axis (small multiples instead), thin marks, muted gridlines, direct
labels on bars.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---- palette (dataviz skill reference palette, light mode) ----
CAT = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
    "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "magenta": "#e87ba4", "orange": "#eb6834",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10,
})


def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)


# ---------------------------------------------------------------------
# Figure 3 (regenerated): coverage sweep - DR, FPR, Acc, #detectors vs C_target
# ---------------------------------------------------------------------
def fig3_coverage_sweep():
    df = pd.read_csv("results/nslkdd_coverage_sweep.csv")
    df = df.sort_values("c_target")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4))
    panels = [
        ("DR", "(a) Detection rate (%)", 100),
        ("FPR", "(b) False positive rate (%)", 100),
        ("Acc", "(c) Accuracy (%)", 100),
        ("n_detectors", "(d) Number of detectors", 1),
    ]
    for ax, (col, title, scale) in zip(axes.flat, panels):
        y = df[col] * scale
        ax.plot(df["c_target"], y, color=CAT["blue"], linewidth=2,
                marker="o", markersize=7, markerfacecolor=CAT["blue"],
                markeredgecolor=SURFACE, markeredgewidth=1)
        for x, yy in zip(df["c_target"], y):
            ax.annotate(f"{yy:.2f}" if scale == 100 else f"{int(yy)}", (x, yy),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
        ax.set_xlabel("Target coverage C_target")
        style_axes(ax)
    fig.suptitle("Figure 3. Detection result under different target coverage "
                  "(full IVDA, binary, NSL-KDD, regenerated from a re-run sweep)",
                  fontsize=10, color=INK_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/figure3_coverage_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# New figure: ablation comparison (V0-V4)
# ---------------------------------------------------------------------
def fig_ablation():
    df = pd.read_csv("results/nslkdd_ablation_summary.csv")
    variants = ["V0_classical", "V1_stat_term", "V2_boundary", "V3_full_binary"]
    labels = ["V0\nclassical", "V1\n+stat.", "V2\n+boundary", "V3\nfull IVDA"]
    colors = [CAT["red"], CAT["yellow"], CAT["aqua"], CAT["blue"]]

    def get(metric):
        return [df[(df.variant == v) & (df.metric == metric)]["mean"].iloc[0] for v in variants]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    metrics = [("DR", "Detection rate (%)", 100), ("FPR", "False positive rate (%)", 100),
               ("n_detectors", "# Detectors", 1)]
    for ax, (col, title, scale) in zip(axes, metrics):
        vals = np.array(get(col)) * scale
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}" if scale == 100 else f"{int(v)}",
                        (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        style_axes(ax)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Ablation study on NSL-KDD (10-fold CV mean)", fontsize=11, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig("figures/figure_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# New figure: per-class recall/F1 (multi-class IVDA)
# ---------------------------------------------------------------------
def fig_per_class():
    df = pd.read_csv("results/nslkdd_per_class_folds.csv")
    agg = df.groupby("cls")[["precision", "recall", "f1"]].mean()
    order = ["DoS", "Probe", "normal", "R2L", "U2R"]
    agg = agg.reindex(order)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(order))
    width = 0.25
    metric_colors = [CAT["blue"], CAT["aqua"], CAT["violet"]]
    for i, (m, c) in enumerate(zip(["precision", "recall", "f1"], metric_colors)):
        vals = agg[m] * 100
        ax.bar(x + (i - 1) * width, vals, width=width, label=m.capitalize(), color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("%")
    ax.set_title("Per-class precision / recall / F1, multi-class IVDA (NSL-KDD, 10-fold CV mean)",
                 fontsize=10, color=INK_PRIMARY)
    ax.legend(frameon=False, loc="lower left")
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("figures/figure_per_class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# New figure: UNSW-NB15 generalization (classical vs full IVDA)
# ---------------------------------------------------------------------
def fig_unsw():
    df = pd.read_csv("results/unsw_binary.csv")
    variants = ["V0_classical", "V3_full_binary"]
    labels = ["V0 classical", "V3 full IVDA"]
    colors = [CAT["red"], CAT["blue"]]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    metrics = [("DR", "Detection rate (%)", 100), ("FPR", "False positive rate (%)", 100),
               ("n_detectors", "# Detectors", 1)]
    for ax, (col, title, scale) in zip(axes, metrics):
        vals = [df[df.variant == v][col].iloc[0] * scale for v in variants]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}" if scale == 100 else f"{int(v)}",
                        (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        style_axes(ax)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Generalization to UNSW-NB15 (single 80/20 split)", fontsize=11, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig("figures/figure_unsw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig3_coverage_sweep()
    fig_ablation()
    fig_per_class()
    fig_unsw()
    print("Figures written to figures/")
