"""
Figures for the spiking implementation (paper Figures 3 and 8).

Same palette and rcParams as generate_figures.py (dataviz reference palette, light mode).
Per-panel palettes were validated with the skill's checks (Python port, all-pairs):
  Figure 8a  blue / aqua / yellow  - CVD and normal-vision PASS; aqua and yellow fall below
             3:1 contrast, so every series also carries a distinct marker, a legend entry and a
             direct label, and all values are tabulated in the paper.
  Figure 8b  blue / red            - all checks PASS.
Identity is never carried by colour alone.

Figure 3 is a schematic of the compilation (how negative selection maps onto a spiking
network). Figure 8 is data: (a) the detection/false-alarm frontier of the exact detector
against the spiking network with and without self-tolerance restoration, (b) energy per
inference against time steps. Data come from the pooled ten folds via the detector sets cached
by run_snn_guard.py and results/snn_guard_summary.csv.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
# Same style as the paper's other figures:
# Times New Roman with STIX maths, Okabe-Ito colours, IVDA blue and classical orange throughout.
BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#6A3D9A"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#202124", "#4D4D4D", "#7F7F7F"
GRID, SURFACE, PALE_GREEN, SELF_GREEN = "#D9D9D9", "#FFFFFF", "#E7F3E8", "#2E7D32"
RED, AQUA, YELLOW = ORANGE, GREEN, PURPLE          # entity roles: classical / self-tolerant / unguarded
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "axes.edgecolor": INK_MUTED, "savefig.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=11)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def blank(ax, title):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left", pad=6, fontweight="bold")


# --------------------------------------------------------------------------- Figure 3
def figure3():
    """Figure 3: compilation of a detector set into a spiking network (PNG, PDF, SVG)."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    (a, b), (c, d) = axes

    # (a) detectors tangent to the self region; dashed = self-tolerant radius
    blank(a, "(a) Detectors")
    rng = np.random.default_rng(3)
    ex, ey, ew, eh = 0.40, 0.46, 0.40, 0.33
    from matplotlib.patches import Ellipse
    a.add_patch(Ellipse((ex, ey), ew, eh, fc=PALE_GREEN, ec="#55A868", lw=0.8, zorder=1))
    pts = rng.normal([ex, ey], [0.06, 0.05], size=(400, 2))
    inside = ((pts[:, 0] - ex) / (ew / 2)) ** 2 + ((pts[:, 1] - ey) / (eh / 2)) ** 2 < 0.85
    self_pts = pts[inside][:90]
    a.scatter(self_pts[:, 0], self_pts[:, 1], s=7, color=SELF_GREEN, zorder=2, linewidths=0)
    tt = np.linspace(0, 2 * np.pi, 2000)
    rim = np.c_[ex + ew / 2 * np.cos(tt), ey + eh / 2 * np.sin(tt)]
    centres = [(0.78, 0.52), (0.40, 0.82), (0.66, 0.15), (0.06, 0.40)]
    radii = [float(np.min(np.linalg.norm(rim - np.array(cc), axis=1))) for cc in centres]
    for (cx, cy), r in zip(centres, radii):
        a.add_patch(Circle((cx, cy), r, fill=False, ec=BLUE, lw=1.6, zorder=3))
        a.plot(cx, cy, "o", ms=3.5, color=BLUE, zorder=4)
    (cx, cy), r = centres[0], radii[0]
    a.add_patch(Circle((cx, cy), r * 0.82, fill=False, ec=AQUA, lw=1.4, ls=(0, (3, 2)), zorder=3))
    a.annotate("", xy=(cx, cy - r), xytext=(cx, cy),
               arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=0.9))
    a.text(cx + 0.012, cy - r * 0.55, r"$r_k$", fontsize=10.7, color=INK_PRIMARY)
    a.text(cx + 0.015, cy + 0.012, r"$c_k$", fontsize=10.7, color=INK_PRIMARY)
    a.annotate(r"$r_k'$", xy=(cx + r * 0.82 * 0.71, cy - r * 0.82 * 0.71), xytext=(cx + r * 1.05, cy - r * 1.02),
               fontsize=10.7, color=INK_PRIMARY,
               arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7))
    a.plot(cx - r * 0.35, cy + r * 0.45, marker="*", ms=11, color=INK_PRIMARY, zorder=5)
    a.text(cx - r * 0.35 + 0.05, cy + r * 0.45 + 0.012, r"$x$", fontsize=10.7, color=INK_PRIMARY)
    a.text(0.25, ey - eh / 2 - 0.06, r"$S$", fontsize=11.5, color=SELF_GREEN, ha="center")
    a.set_xlim(-0.12, 1.02); a.set_ylim(-0.12, 1.06); a.set_aspect("equal")

    # (b) one spike per input, graded norm value
    blank(b, "(b) Spike encoding")
    T = 16
    vals = [0.80, -0.30, 0.55, -0.70]
    rows = []
    for i, v in enumerate(vals, 1):
        rows.append((rf"$v_{i}^{{+}}$", v if v > 0 else 0.0))
        rows.append((rf"$v_{i}^{{-}}$", -v if v < 0 else 0.0))
    y0 = len(rows) + 1
    for j, (lab, mag) in enumerate(rows):
        yy = y0 - j
        b.plot([0, T], [yy, yy], color=GRID, lw=0.8, zorder=1)
        b.text(-0.6, yy, lab, ha="right", va="center", fontsize=10.2, color=INK_SECONDARY)
        if mag > 0:
            t = (1 - round(mag * T) / T) * T
            b.plot([t, t], [yy - 0.32, yy + 0.32], color=INK_PRIMARY, lw=1.8, zorder=3)
    b.text(-1.6, y0 - len(rows) + 0.45, r"$\vdots$", ha="center", va="center", fontsize=10.7,
           color=INK_SECONDARY)
    yy = y0 - len(rows) - 0.4
    b.plot([0, T], [yy, yy], color=GRID, lw=0.8)
    b.text(-0.6, yy, r"$\|\hat{x}\|^2$", ha="right", va="center", fontsize=10.2, color=INK_SECONDARY)
    b.add_patch(FancyBboxPatch((0.2, yy - 0.3), 0.8, 0.6, boxstyle="round,pad=0.02",
                               fc=BLUE, ec="none", zorder=3))
    b.annotate("", xy=(T + 0.3, 0.15), xytext=(0, 0.15),
               arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9))
    b.text(0, -0.45, "0", ha="center", fontsize=9.7, color=INK_SECONDARY)
    b.text(T, -0.45, r"$T$", ha="center", fontsize=10.2, color=INK_SECONDARY)
    b.text(T / 2, -0.55, r"$t$", ha="center", fontsize=10.7, color=INK_SECONDARY)
    b.set_xlim(-3.5, T + 0.8); b.set_ylim(-0.9, y0 + 0.8)

    # (c) one integrate-and-fire neuron per detector
    blank(c, "(c) Integrate-and-fire layer")
    ins = np.linspace(0.86, 0.14, 6)
    outs = np.linspace(0.80, 0.20, 4)
    for yi in ins:
        for yo in outs:
            c.plot([0.18, 0.62], [yi, yo], color=GRID, lw=0.7, zorder=1)
    c.scatter(np.full(len(ins), 0.15), ins, s=170, facecolor=SURFACE, edgecolor=INK_SECONDARY,
              linewidths=1, zorder=3)
    c.text(0.15, 0.94, r"$\tilde{x}$", ha="center", va="bottom", fontsize=11.7, color=INK_PRIMARY)
    c.text(0.40, 0.94, r"$w_k$", ha="center", va="bottom", fontsize=11.7, color=INK_PRIMARY)
    c.scatter(np.full(len(outs), 0.66), outs, s=420, facecolor=SURFACE, edgecolor=BLUE,
              linewidths=1.6, zorder=3)
    for k, yo in enumerate(outs):
        c.text(0.66, yo, f"{k + 1}" if k < 3 else r"$K$", ha="center", va="center", fontsize=9.7,
               color=INK_PRIMARY, zorder=4)
    ins_ax = c.inset_axes([0.76, 0.38, 0.23, 0.32])
    tt2 = np.arange(0, 17)
    vm = np.minimum(np.cumsum(np.r_[0, np.full(16, 0.085)]), 1.25)
    ins_ax.plot(tt2, vm, color=BLUE, lw=1.4)
    ins_ax.axhline(1.0, color=INK_MUTED, lw=0.9, ls=(0, (3, 2)))
    ins_ax.text(0.5, 1.07, r"$\theta_k$", fontsize=9.7, color=INK_SECONDARY)
    ins_ax.plot([12], [1.02], marker="v", ms=5, color=INK_PRIMARY)
    ins_ax.set_xticks([]); ins_ax.set_yticks([])
    for sp in ins_ax.spines.values():
        sp.set_color(INK_MUTED)
    ins_ax.set_title(r"$V_k$", fontsize=10.2, color=INK_SECONDARY, pad=2)
    c.set_xlim(0, 1); c.set_ylim(0.0, 1.08)

    # (d) readout as a symbolic flow
    blank(d, "(d) Readout")
    box = dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=INK_MUTED, lw=0.9)
    def node(x, y, s, fs=11.2):
        return d.text(x, y, s, fontsize=fs, color=INK_PRIMARY, bbox=box, ha="center", va="center")
    def link(t0, t1):
        d.annotate("", xy=(0, 0.5), xycoords=t1, xytext=(1, 0.5), textcoords=t0,
                   arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=0.9,
                                   shrinkA=4, shrinkB=4))
    d.text(0.02, 0.84, "binary", fontsize=9.7, color=INK_SECONDARY, style="italic")
    b1 = node(0.17, 0.66, r"$s_1,\ldots,s_K$")
    b2 = node(0.52, 0.66, "OR")
    b3 = node(0.83, 0.66, "non-self")
    link(b1, b2); link(b2, b3)
    d.text(0.02, 0.40, "multi-class", fontsize=9.7, color=INK_SECONDARY, style="italic")
    m1 = node(0.20, 0.22, r"$\Phi_1(x),\ldots,\Phi_K(x)$")
    m2 = node(0.66, 0.22, r"$\arg\min_k\,\Phi_k$")
    m3 = node(0.93, 0.22, r"$\hat{y}$", 11.7)
    link(m1, m2); link(m2, m3)
    d.set_xlim(0, 1); d.set_ylim(0, 1)

    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"figures/figure3_snn_compilation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/figure3_snn_compilation.{png,pdf,svg}")

# --------------------------------------------------------------------------- Figure 8
def exact_curve():
    """Mean DR/FPR of the exact IVDA detector with radii scaled by alpha, pooled ten folds."""
    from data_pipeline import load_nslkdd, build_nslkdd_features
    from run_snn_encodings import decide
    from run_snn_guard import CACHE
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X, y, _, _, _, _ = build_nslkdd_features(full, full.iloc[:1], pca_dim=15, random_state=0)
    cache = dict(np.load(CACHE))
    # fine near alpha = 1, where the whole trade-off happens (run_snn_rigor.py showed a coarse
    # grid misplaces the SNN relative to this curve)
    alphas = np.round(np.r_[np.arange(0.95, 0.98, 0.005), np.arange(0.98, 1.0 + 1e-9, 0.0005)], 4)
    acc = np.zeros((len(alphas), 2))
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X, y))
    for f, (tr, te) in enumerate(folds):
        S = X[tr][y[tr] == "normal"]
        mu = S.mean(axis=0)
        C = cache[f"V3_ivda_f{f}_c"] - mu
        r = cache[f"V3_ivda_f{f}_r"]
        Xc = X[te] - mu
        Nc = (Xc ** 2).sum(axis=1)
        att, nor = y[te] != "normal", y[te] == "normal"
        for i, a_ in enumerate(alphas):
            p = decide(Xc, Nc, C, (C ** 2).sum(axis=1) - (a_ * r) ** 2)
            acc[i] += [p[att].mean(), p[nor].mean()]
    acc /= len(folds)
    return alphas, acc


def figure8():
    g = pd.read_csv("results/snn_guard_summary.csv")
    _, curve = exact_curve()
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (a) frontier
    style(a)
    a.plot(curve[:, 1] * 100, curve[:, 0] * 100, color=INK_MUTED, lw=1.6, zorder=2,
           label=r"exact detector, radii $\times\alpha$")
    ex = g[(g.variant == "V3_ivda") & (g.guard == "none") & (g["T"] == 16)].iloc[0]
    a.plot(ex.FPR_exact * 100, ex.DR_exact * 100, "o", ms=9, color=BLUE, mec=SURFACE, mew=1.5,
           zorder=5, label="exact detector")
    # left of the point: the unguarded T=64 label sits to its right at page size
    a.annotate("exact", (ex.FPR_exact * 100, ex.DR_exact * 100), xytext=(-8, 5), ha="right",
               textcoords="offset points", fontsize=11.5, color=INK_SECONDARY)
    for guard, col, mk, lab in [("none", YELLOW, "s", "SNN, unguarded"),
                                ("self_tolerant", AQUA, "^", "SNN, self-tolerant")]:
        s = g[(g.variant == "V3_ivda") & (g.guard == guard) & (g["T"].isin([16, 32, 64]))]
        s = s.sort_values("T")
        a.plot(s.FPR * 100, s.DR * 100, ls=(0, (2, 2)), color=col, lw=1.2, zorder=3)
        a.plot(s.FPR * 100, s.DR * 100, mk, ms=8, color=col, mec=SURFACE, mew=1.5, zorder=4,
               label=lab)
        for _, row in s.iterrows():
            T_ = int(row["T"])
            if guard == "self_tolerant":
                off, ha = (5, -10), "left"
            else:                                   # T=64 and T=32 sit close together
                off, ha = {64: ((0, 9), "center"), 32: ((9, 7), "left"), 16: ((5, 5), "left")}[T_]
            a.annotate(f"T={T_}", (row.FPR * 100, row.DR * 100), xytext=off, ha=ha,
                       textcoords="offset points", fontsize=11, color=INK_SECONDARY)
    a.set_xlabel("False positive rate (%)", fontsize=12.5, color=INK_SECONDARY)
    a.set_ylabel("Detection rate (%)", fontsize=12.5, color=INK_SECONDARY)
    a.set_xlim(-0.05, 2.0); a.set_ylim(84, 93.5)
    a.set_title("(a) Detection / false-alarm frontier (IVDA)", fontsize=13.5, loc="left",
                color=INK_PRIMARY, fontweight="bold")
    a.legend(fontsize=11, frameon=False, loc="lower right")

    # (b) energy per inference vs T
    style(b)
    for v, col, mk, lab in [("V3_ivda", BLUE, "o", "IVDA (134 neurons)"),
                            ("V0_classical", RED, "s", "classical (1500 neurons)")]:
        s = g[(g.variant == v) & (g.guard == "self_tolerant")].sort_values("T")
        b.plot(s["T"], s.E_total_pJ / 1000, color=col, lw=2, marker=mk, ms=7, mec=SURFACE,
               mew=1.2, label=f"{lab}, SNN")
        b.axhline(s.E_dense_pJ.iloc[0] / 1000, color=col, lw=1.2, ls=(0, (4, 3)))
        b.text(8.4, s.E_dense_pJ.iloc[0] / 1000 * 1.10, "dense evaluation", fontsize=11,
               color=INK_SECONDARY, ha="left")
    b.set_yscale("log")
    b.set_xticks([8, 16, 32, 64])
    b.set_xlabel("Time steps T", fontsize=12.5, color=INK_SECONDARY)
    b.set_ylabel("Energy per inference (nJ, 45 nm est.)", fontsize=12.5, color=INK_SECONDARY)
    b.set_title("(b) Energy per inference", fontsize=13.5, loc="left", color=INK_PRIMARY, fontweight="bold")
    b.set_ylim(1.5, 220)
    b.legend(fontsize=11, frameon=False, loc="lower right")

    fig.tight_layout(w_pad=2.0)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"figures/figure8_snn_frontier_energy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/figure8_snn_frontier_energy.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["3", "8"]
    if "3" in which:
        figure3()
    if "8" in which:
        figure8()
