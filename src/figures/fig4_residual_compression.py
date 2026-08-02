import os, sys, numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, figure
from style import set_style, C, set_ax_grid, save_fig

set_style()

# ── Academic color scheme ──────────────────────────────────────────────────
C_METHOD = {"TENT": C["TENT"], "SHOT": C["SHOT"]}

# ── Load cross-method results ──────────────────────────────────────────────
df = pd.read_csv(results_csv("cross_method_results.csv"))
sub = df[df.method.isin(["TENT", "SHOT"])].copy()
datasets = sorted(sub["dataset"].unique())

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.8, 2.8))

# ═══════════════════════════════════════════════════════════════════════════
# Grouped bar chart — ΔAUC by dataset & method
# ═══════════════════════════════════════════════════════════════════════════
n_methods = 2
x_positions = np.arange(len(datasets))
bar_width = 0.30

for i, method in enumerate(["TENT", "SHOT"]):
    means, errs = [], []
    ds_labels = []
    for ds in datasets:
        vals = sub[(sub.dataset == ds) & (sub.method == method)]["delta_AUC"].values
        means.append(np.mean(vals))
        errs.append(np.std(vals) / np.sqrt(len(vals)))
        ds_labels.append(ds)
    offset = (i - (n_methods - 1) / 2) * bar_width
    ax.bar(x_positions + offset, means, bar_width, yerr=errs,
           label=method, color=C_METHOD[method], alpha=0.82, capsize=3,
           error_kw={"linewidth": 0.8, "alpha": 0.7})

ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.6)
ax.set_xticks(x_positions)
ax.set_xticklabels(datasets, fontsize=7.5)
ax.set_ylabel("ΔAUC (post − pre TTA)")
ax.set_title("Residual: ΔAUC by dataset\n(mean ± SEM, TENT & SHOT)",
             fontsize=9, pad=6)
ax.legend(fontsize=7, framealpha=0.88, loc="lower right")
set_ax_grid(ax, axis="y")

plt.tight_layout()
save_fig(fig, "fig4_residual_compression", png=True)