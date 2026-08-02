"""
exp_tta_with_metrics.py — TTA 轨迹图再现（属性同化）

读取 results/csv/attribute_assimilation_v3.csv，生成三张说明图：
  fig_aa_abs.png    — 双子图（距离 + AUC）
  fig_aa_dual.png   — 双轴 X-cross（论文 Introduction 关键图）
  fig_aa_norm.png   — 归一化 + 对照曲线

用法：
  python scripts/experiments/exp_tta_with_metrics.py
"""

import os, sys, pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, figure

df    = pd.read_csv(results_csv("attribute_assimilation_v3.csv"))
steps = df["step"].values
dist  = df["dist"].values
auc   = df["auc"].values
sp    = df["spread"].values

C_DIST = "#D85A30"
C_AUC  = "#533AB7"
C_CTRL = "#888780"

# ── Fig 1: abs (two subplots) ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(steps, dist, "o-", markersize=2.5, color=C_DIST,
        label="Anomaly–Normal distance")
ax.set_xlabel("TTA update steps", fontsize=12)
ax.set_ylabel("Centroid distance", fontsize=12)
ax.set_title(
    "Attribute Assimilation:\n"
    "Normal manifold expands, anomaly nodes lose relative discriminability",
    fontsize=11)
ax.legend(fontsize=10); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(steps, auc, "s-", markersize=2.5, color=C_AUC,
        label="Detection AUC")
ax.set_xlabel("TTA update steps", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title(
    "Detection Collapse:\n"
    "AUC degrades monotonically under TTA optimization",
    fontsize=11)
ax.legend(fontsize=10); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figure("fig_aa_abs.png"), dpi=300); plt.close()
print(f"Saved fig_aa_abs.png")

# ── Fig 2: dual-axis X-cross  ─────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel("TTA update steps", fontsize=12)
ax1.set_ylabel("Anomaly–Normal distance\n(normal manifold expansion)", color=C_DIST, fontsize=11)
l1, = ax1.plot(steps, dist, "o-", markersize=2.5, color=C_DIST,
               label="Anomaly–Normal distance ↑")
ax1.tick_params(axis="y", labelcolor=C_DIST)

ax2 = ax1.twinx()
ax2.set_ylabel("Detection AUC", color=C_AUC, fontsize=11)
l2, = ax2.plot(steps, auc, "s-", markersize=2.5, color=C_AUC,
               label="Detection AUC ↓")
ax2.tick_params(axis="y", labelcolor=C_AUC)

# Annotate the crossing point
cross_step = steps[np.argmin(np.abs(
    (dist - dist[0]) / (dist[-1] - dist[0]) -
    (auc[0] - auc)  / (auc[0]  - auc[-1])
))]
ax1.axvline(cross_step, color="gray", lw=0.8, ls="--", alpha=0.6)
ax1.text(cross_step + 1, dist.mean(),
         f"step ≈ {cross_step}", fontsize=9, color="gray")

ax1.set_title(
    "Attribute Assimilation under TTA:\n"
    "Normal manifold expands (↑) while detection AUC collapses (↓)",
    fontsize=11)

ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
           loc="center right", fontsize=10)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figure("fig_aa_dual.png"), dpi=300); plt.close()
print("Saved fig_aa_dual.png")

# ── Fig 3: normalized + control ───────────────────────────────────────────────
d_norm  = dist / (dist[0]  + 1e-9)
sp_norm = sp   / (sp[0]    + 1e-9)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(steps, d_norm,  "o-",  markersize=2.5, color=C_DIST,
        label="Normal manifold expansion (relative to anomaly nodes)")
ax.plot(steps, sp_norm, "^--", markersize=2.5, color=C_CTRL,
        label="Normal intra-spread (control)")
ax.axhline(1.0, color="gray", lw=0.8, ls=":")

# Shade the gap between the two curves to highlight the asymmetry
ax.fill_between(steps, sp_norm, d_norm,
                where=(d_norm >= sp_norm),
                alpha=0.12, color=C_DIST,
                label="Assimilation gap\n(anomaly coverage > normal expansion)")

ax.set_xlabel("TTA update steps", fontsize=12)
ax.set_ylabel("Relative value  (step 0 = 1.0)", fontsize=12)
ax.set_title(
    "Attribute Assimilation vs. Normal Stability\n"
    "Normal manifold engulfs anomaly nodes faster than it expands internally",
    fontsize=11)
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figure("fig_aa_norm.png"), dpi=300); plt.close()
print("Saved fig_aa_norm.png")

print("\nAll figures saved.")