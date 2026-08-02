"""
fig3_paradox.py — Figure 3: The Directional Paradox

比较合成协方差注入（各向同性/各向异性）与真实 TTA 轨迹的 AUC vs D_Σ 曲线。
数据集从 causal_*.csv 实际数据中动态读取，避免硬编码。

读取:
  causal_isotropic.csv, causal_anisotropic.csv (exp_causal_intervention.py)
  causal_real_tta_trajectory.csv
输出:
  fig3_paradox.pdf      矢量图
  fig3_paradox.png      光栅图（备选）
"""

import os, sys, numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, figure
from style import set_style, C, set_ax_grid, save_fig

set_style()

# 曲线颜色（来自统一调色板）
CURVE_C = {
    "iso":   C["iso"],
    "aniso": C["aniso"],
    "real":  C["real"],
}

df_iso   = pd.read_csv(results_csv("causal_isotropic.csv"))
df_aniso = pd.read_csv(results_csv("causal_anisotropic.csv"))
df_real  = pd.read_csv(results_csv("causal_real_tta_trajectory.csv"))

# 动态读取实际存在的数据集
DATASETS = sorted(df_iso["dataset"].unique())

# 动态计算每个数据集的 GAP 和 YLIM
TITLES, GAP, GAP_COLOR, YLIM = {}, {}, {}, {}
for ds in DATASETS:
    iso   = df_iso[df_iso.dataset == ds].sort_values("D_Sigma")
    aniso = df_aniso[df_aniso.dataset == ds].sort_values("D_Sigma")
    real  = df_real[df_real.dataset == ds]
    TITLES[ds] = ds

    aniso_at_1 = aniso[np.isclose(aniso.delta, 1.0, atol=0.01)]
    if len(aniso_at_1) > 0 and len(real) > 0:
        auc_synth = aniso_at_1["AUC"].values[0]
        auc_real  = real["AUC"].values[-1]
        gap = auc_real - auc_synth
    else:
        gap = 0.0
    GAP[ds] = gap
    GAP_COLOR[ds] = C["as_0.3"] if gap > 0 else "#d62728"

    all_auc = pd.concat([iso["AUC"], aniso["AUC"], real["AUC"]])
    lo, hi = all_auc.min(), all_auc.max()
    margin = 0.05 * (hi - lo) if hi > lo else 0.1
    YLIM[ds] = (lo - margin, hi + margin)

fig, axes = plt.subplots(1, len(DATASETS), figsize=(7.0, 2.8),
                          gridspec_kw={"wspace": 0.42})
if len(DATASETS) == 1:
    axes = [axes]

for ax, ds in zip(axes, DATASETS):
    iso   = df_iso[df_iso.dataset == ds].sort_values("D_Sigma")
    aniso = df_aniso[df_aniso.dataset == ds].sort_values("D_Sigma")
    real  = df_real[df_real.dataset == ds]

    ax.scatter(real["D_Sigma"], real["AUC"],
               color=CURVE_C["real"], s=7, alpha=0.6, zorder=4)
    ax.plot(iso["D_Sigma"], iso["AUC"],
            color=CURVE_C["iso"], lw=1.3, ls="--", zorder=2)
    ax.plot(aniso["D_Sigma"], aniso["AUC"],
            color=CURVE_C["aniso"], lw=1.8, ls="-", zorder=3)

    # δ=1 star
    d1 = aniso[np.isclose(aniso.delta, 1.0, atol=0.01)]
    if len(d1) > 0:
        ax.scatter(d1["D_Sigma"], d1["AUC"],
                   marker="*", s=160, color="gold",
                   edgecolors="black", linewidths=0.4, zorder=5)

    # Gap annotation box
    gap, gc = GAP[ds], GAP_COLOR[ds]
    ax.text(0.96, 0.04, f"Gap = {gap:+.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=gc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=gc, alpha=0.9, linewidth=1.0))

    # Paradox direction annotation (only for negative gap)
    if gap < 0:
        ax.text(0.96, 0.96,
                "← synthetic ↑\n  real TTA ↓",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=CURVE_C["aniso"],
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=CURVE_C["aniso"], alpha=0.85, linewidth=0.8))

    ax.set_ylim(YLIM[ds])
    ax.set_xlabel("$D_\\Sigma$")
    ax.set_ylabel("AUC")
    ax.set_title(TITLES[ds], fontsize=9, pad=5)
    set_ax_grid(ax)

# Shared legend below all panels
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=CURVE_C["iso"], lw=1.3, ls="--",
           label="Isotropic injection (synthetic)"),
    Line2D([0], [0], color=CURVE_C["aniso"], lw=1.8, ls="-",
           label="Anisotropic injection (synthetic)"),
    Line2D([0], [0], color=CURVE_C["real"], marker="o", lw=0, markersize=4,
           alpha=0.7, label="Real TTA trajectory (80 steps)"),
    Line2D([0], [0], color="black", marker="*", lw=0, markersize=8,
           markerfacecolor="gold",
           label="δ = 1 (matches real TTA covariance endpoint)"),
]
fig.legend(handles=legend_elements, loc="upper center",
           ncol=2, fontsize=7, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.10), columnspacing=1.0)

plt.tight_layout(rect=[0, 0.06, 1, 0.94])
save_fig(fig, "fig3_paradox", png=True)