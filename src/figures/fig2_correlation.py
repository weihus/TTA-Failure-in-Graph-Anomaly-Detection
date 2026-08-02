"""
fig2_correlation.py — Figure 2: Correlational Evidence for D_Σ → M²_A → AUC

Panel (a): Stratified scatter log(D_Σ) vs M²_A, single-hue gradient per anomaly str
Panel (b): M²_A vs AUC, distinct shapes & colors per dataset
Panel (c): M²_N self-consistency with merged legend in upper-right

读取:
  multi_shift_results.csv    (exp_multi_shift.py — Cora 54 条件)
  real_dataset_results.csv   (exp_real_datasets_v3.py — questions/reddit)
输出:
  fig2_correlation.pdf       矢量图
"""

import os, sys, numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, figure
from style import set_style, C, set_ax_grid, save_fig
import seaborn as sns

set_style()

# ── Load ──────────────────────────────────────────────────────────────────────
cora = pd.read_csv(results_csv("multi_shift_results.csv"))
real = pd.read_csv(results_csv("real_dataset_results.csv"))

cora["source"] = "cora"
real["source"] = real["dataset"]
real["anomaly_str"] = None

d = 64  # hidden dimension

# ── Academic color palette ─────────────────────────────────────────────────
BLUES = sns.color_palette("Blues", 5)
AS_COLORS = {0.3: BLUES[1], 0.5: BLUES[2], 0.8: BLUES[3]}
AS_LABELS = {0.3: "anomaly str = 0.3", 0.5: "anomaly str = 0.5", 0.8: "anomaly str = 0.8"}

DS_MARKER = {"questions": "s", "reddit": "^", "tolokers": "D", "weibo": "v"}
DS_COLOR  = {
    "questions": "#005AB5",
    "reddit":    "#DC3220",
    "tolokers":  "#009E73",
    "weibo":     "#6A3D9A",
}
COREA_COLOR = "#888888"

# ── M²_N data ───────────────────────────────────────────────────────────────
all_m2n = list(cora["M2_N_T"].values)
for ds in ["questions", "reddit"]:
    sub = real[real.dataset == ds]
    if "M2_N_T" in sub.columns and len(sub) > 0:
        all_m2n += list(sub["M2_N_T"].values)

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
ax_a, ax_b, ax_c = axes

# ═══════════════════════════════════════════════════════════════════════════
# Panel (a): Stratified D_Σ → M²_A (single-hue gradient)
# ═══════════════════════════════════════════════════════════════════════════
for as_ in [0.3, 0.5, 0.8]:
    sub = cora[cora.anomaly_str == as_]
    logD = np.log(sub["D_Sigma_T"].clip(lower=1))
    M    = sub["M2_A_T"].values
    r_s, p_s = ss.spearmanr(logD, M)
    ax_a.scatter(logD, M, color=AS_COLORS[as_], s=28, alpha=0.78,
                 edgecolors="white", linewidth=0.4, label=AS_LABELS[as_])
    reg   = LinearRegression().fit(logD.values.reshape(-1, 1), M)
    x_fit = np.linspace(logD.min(), logD.max(), 80)
    y_fit = np.clip(reg.predict(x_fit.reshape(-1, 1)), 0, None)
    ax_a.plot(x_fit, y_fit, color=AS_COLORS[as_], lw=1.6, alpha=0.9)
    ax_a.text(x_fit[0], y_fit[0] + 80, f"$r_S$={r_s:.2f}",
              fontsize=8, color=AS_COLORS[as_], ha="left", va="bottom",
              fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))

y_lo_a, y_hi_a = ax_a.get_ylim()
ax_a.set_ylim(y_lo_a, y_hi_a + 0.05 * (y_hi_a - y_lo_a))
ax_a.set_xlabel("$\\log(D_\\Sigma)$")
ax_a.set_ylabel("$M^2_{A,t}$")
ax_a.set_title("(a) $D_\\Sigma \\rightarrow M^2_A$ (stratified)", fontsize=10, pad=5)
ax_a.legend(fontsize=7, loc="upper right", framealpha=0.88)
set_ax_grid(ax_a)

# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): M²_A → AUC (all datasets)
# ═══════════════════════════════════════════════════════════════════════════
for as_ in [0.3, 0.5, 0.8]:
    sub = cora[cora.anomaly_str == as_]
    ax_b.scatter(sub["M2_A_T"], sub["AUC_T"],
                 color=AS_COLORS[as_], s=20, alpha=0.55, marker="o",
                 edgecolors="white", linewidth=0.3)

for ds, marker in DS_MARKER.items():
    sub = real[real.dataset == ds]
    if len(sub) == 0: continue
    r_ds, p_ds = ss.spearmanr(sub["M2_A_T"], sub["AUC_T"])
    ax_b.scatter(sub["M2_A_T"], sub["AUC_T"],
                 color=DS_COLOR[ds], s=32, alpha=0.85,
                 marker=marker, edgecolors="white", linewidth=0.4,
                 label=f"{ds}  ($r_S$={r_ds:.2f})")

all_M = pd.concat([cora["M2_A_T"], real["M2_A_T"]]).values
all_U = pd.concat([cora["AUC_T"], real["AUC_T"]]).values
reg2 = LinearRegression().fit(all_M.reshape(-1,1), all_U)
x2   = np.linspace(all_M.min(), all_M.max(), 100)
ax_b.plot(x2, reg2.predict(x2.reshape(-1,1)), "k--", lw=1.3, alpha=0.7)
ax_b.axvline(d, color="#888888", lw=0.9, ls=":", alpha=0.7)

r_all, _ = ss.spearmanr(all_M, all_U)
ax_b.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
ax_b.xaxis.get_offset_text().set_fontsize(7)
ax_b.xaxis.set_label_coords(0.5, -0.15)
ax_b.set_xlabel("$M^2_{A,t}$ (Anomaly Separability)", labelpad=10)
ax_b.set_ylabel("AUC")
ax_b.set_title("(b) $M^2_A \\rightarrow$ AUC (all datasets)", fontsize=10, pad=5)

cora_patch = mpatches.Patch(color=COREA_COLOR, alpha=0.5,
                             label="Cora (○, colored by anomaly str)")
ds_handles = [
    Line2D([0], [0], color=DS_COLOR[ds], marker=DS_MARKER[ds],
           lw=0, ms=5.5, label=f"{ds}")
    for ds in DS_MARKER if len(real[real.dataset == ds]) > 0
]
ax_b.legend(handles=[cora_patch] + ds_handles,
            fontsize=7, loc="lower right",
            bbox_to_anchor=(1., 0.01),
            framealpha=0.90)
set_ax_grid(ax_b)

# ═══════════════════════════════════════════════════════════════════════════
# Panel (c): M²_N self-consistency
# ═══════════════════════════════════════════════════════════════════════════
ax_c.scatter(range(len(all_m2n)), all_m2n,
             s=12, alpha=0.65, color=C["BN-adapt"],
             edgecolors="white", linewidth=0.3)
theoretical_line = ax_c.axhline(d, color="#d62728", lw=1.5, ls="--",
                                label=f"Theoretical $d = {d}$")
empirical_line = ax_c.axhline(np.mean(all_m2n), color=C["as_0.3"], lw=1.2, ls=":",
                               label=f"Empirical mean = {np.mean(all_m2n):.1f}")
ax_c.set_ylim(max(0, min(all_m2n) - 5), d + 7)
ax_c.set_xlabel("Condition index")
ax_c.set_ylabel("$\\hat{M}^2_{N,t}$")
ax_c.set_title("(c) Self-consistency: $M^2_N \\approx d$", fontsize=10, pad=5)
ax_c.legend(handles=[theoretical_line, empirical_line],
            fontsize=7, loc="upper right", framealpha=0.88)
ax_c.grid(True, color=".88", linestyle="--", linewidth=0.3, alpha=0.7)
ax_c.set_axisbelow(True)

save_fig(fig, "fig2_correlation", png=True)
