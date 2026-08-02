"""
exp_multi_shift.py — 多条件 Normality Shift 实验

设计原理：
  通过控制 Normal Node Feature Drift 的强度，直接模拟论文核心假设
  "Normality Shift = P_s^N ≠ P_t^N"，制造 54 个具有不同 D_Σ 的条件。

  - Normal shift 弱 → D_Σ 小 → M²_A 大 → AUC 高
  - Normal shift 强 → D_Σ 大 → M²_A 小 → AUC 低
  → 三段链条 D_Σ → M²_A → AUC 均有足够变化范围被检验

输出：
  multi_shift_results.csv    所有 54 条记录
  multi_shift_fig.png        三面板散点图 + 回归
  multi_shift_summary.txt    关键统计量
"""

import os, sys, copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

# 添加项目根目录到 sys.path 以便导入 config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, results_csv, results_txt, figure

# ── Config ────────────────────────────────────────────────────────────────────
DATASET         = sys.argv[1] if len(sys.argv) > 1 else "cora"
OUT_SUFFIX      = sys.argv[2] if len(sys.argv) > 2 else ""
SEED            = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS      = 3
HIDDEN_DIM      = 64
PRETRAIN_EPOCHS = 300
PRETRAIN_LR     = 3e-3
MARGIN          = 2.0
TTA_STEPS       = 80
TTA_LR          = 5e-5
ANOMALY_RATIO   = 0.05

# 实验矩阵
NORMAL_SHIFT_STRENGTHS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]
ANOMALY_STRENGTHS      = [0.3, 0.5, 0.8]
SEEDS                  = [100, 200, 300]

torch.manual_seed(SEED); np.random.seed(SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
PLANETOID = {"cora":"Cora","citeseer":"Citeseer","pubmed":"Pubmed"}
print(f"Loading {DATASET}...")
data       = Planetoid(root=DATA_DIR, name=PLANETOID[DATASET])[0]
x_raw      = data.x
edge_index = data.edge_index.to(DEVICE)
print(f"  Nodes={x_raw.shape[0]}  Features={x_raw.shape[1]}")

# ── Domain creation ───────────────────────────────────────────────────────────

def inject_anomaly(x, ratio=0.05, strength=0.5, seed=None):
    """Inject Gaussian anomalies into a random subset of nodes."""
    if seed is not None: torch.manual_seed(seed)
    x_new = x.clone(); n = x.shape[0]
    idx = torch.randperm(n)[:int(n * ratio)]
    labels = torch.zeros(n, dtype=torch.long)
    labels[idx] = 1
    x_new[idx] += torch.randn_like(x_new[idx]) * strength
    return x_new, labels

def create_target_domain(x_raw, normal_shift_str, anomaly_str, seed):
    """
    Create target domain with TWO independent perturbations:
      1. Normal Node Feature Drift (normal_shift_str):
         Simulates Normality Shift P_s^N → P_t^N.
         All nodes receive a global Gaussian drift — normal nodes included.
         Larger normal_shift_str → larger D_Σ.
      2. Anomaly Injection (anomaly_str):
         Specific 5% of nodes receive additional large noise.
    """
    torch.manual_seed(seed)
    # Step 1: Normal shift — applied to ALL nodes
    x_shifted = x_raw.clone()
    if normal_shift_str > 0:
        normal_drift = torch.randn_like(x_shifted) * normal_shift_str
        x_shifted = x_shifted + normal_drift

    # Step 2: Anomaly injection on top of shifted features
    x_final, labels = inject_anomaly(
        x_shifted, ratio=ANOMALY_RATIO, strength=anomaly_str, seed=seed+1000
    )
    return x_final, labels

# ── Architecture ──────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden)] +
            [GCNConv(hidden, hidden) for _ in range(layers-1)]
        )
    def forward(self, x, ei):
        h = x
        for c in self.convs: h = F.relu(c(h, ei))
        return h

class Scorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, h): return self.fc(h).squeeze(-1)

# ── Metric helpers ────────────────────────────────────────────────────────────

def normal_stats_detached(h, labels):
    h_N = h[labels == 0].detach()
    return h_N.mean(0), h_N.var(0).clamp(min=1e-6)

def normal_stats_grad(h, labels):
    """No detach — for anchoring loss gradient flow."""
    h_N = h[labels == 0]
    return h_N.mean(0), h_N.var(0).clamp(min=1e-6)

def compute_all(h, labels, mu_s, var_s):
    h_det = h.detach()
    mu_t, var_t = normal_stats_detached(h_det, labels)
    # D_Sigma
    ratio   = var_t / var_s
    D_Sigma = max(0.5*(ratio - 1.0 - ratio.log()).sum().item(), 0.0)
    # M²_A (Mahalanobis²)
    h_A   = h_det[labels == 1]
    M2_At = ((h_A - mu_t)**2 / var_t).sum(1).mean().item()
    # M²_N (validation check ≈ HIDDEN_DIM)
    h_N   = h_det[labels == 0]
    M2_Nt = ((h_N - mu_t)**2 / var_t).sum(1).mean().item()
    # AUC (Mahalanobis scoring — consistent with M²_A)
    score = ((h_det - mu_t)**2 / var_t).sum(1).sqrt()
    auc   = roc_auc_score(labels.cpu().numpy(), score.cpu().numpy())
    return D_Sigma, M2_At, M2_Nt, auc

# ── Pre-train (once, on source domain) ───────────────────────────────────────
print("\n[1] Pre-training source model (once)...")
x_src, y_src = inject_anomaly(x_raw, strength=0.5, seed=0)
x_src = x_src.to(DEVICE); y_src = y_src.to(DEVICE)

enc_base = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
opt = torch.optim.Adam(enc_base.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
for ep in range(PRETRAIN_EPOCHS):
    enc_base.train(); opt.zero_grad()
    h  = enc_base(x_src, edge_index)
    cn = h[y_src==0].mean(0)
    d  = torch.norm(h - cn, dim=1)
    loss = (F.relu(MARGIN - d[y_src==1])**2).mean() + (d[y_src==0]**2).mean()
    loss.backward(); opt.step()
    if (ep+1) % 100 == 0:
        enc_base.eval()
        with torch.no_grad():
            h_e = enc_base(x_src, edge_index)
            mu_tmp, var_tmp = normal_stats_detached(h_e, y_src)
            sc   = ((h_e - mu_tmp)**2 / var_tmp).sum(1).sqrt()
            auc_s = roc_auc_score(y_src.cpu().numpy(), sc.cpu().numpy())
        print(f"  ep {ep+1:3d}  src-AUC={auc_s:.4f}")

# Source stats (fixed reference)
enc_base.eval()
with torch.no_grad():
    h_src = enc_base(x_src, edge_index)
mu_s, var_s = normal_stats_detached(h_src, y_src)
print(f"  Source stats fixed: mu_s norm={mu_s.norm():.2f}  var_s mean={var_s.mean():.4f}")

# ── Multi-condition experiment ────────────────────────────────────────────────
total_conditions = (len(NORMAL_SHIFT_STRENGTHS) *
                    len(ANOMALY_STRENGTHS) * len(SEEDS))
print(f"\n[2] Running {total_conditions} conditions "
      f"({len(NORMAL_SHIFT_STRENGTHS)} shifts × "
      f"{len(ANOMALY_STRENGTHS)} anomaly strengths × "
      f"{len(SEEDS)} seeds)...")

records = []
done = 0

for ns in NORMAL_SHIFT_STRENGTHS:
    for as_ in ANOMALY_STRENGTHS:
        for seed in SEEDS:

            # Create target domain
            x_tgt, y_tgt = create_target_domain(x_raw, ns, as_, seed)
            x_tgt = x_tgt.to(DEVICE); y_tgt = y_tgt.to(DEVICE)

            # Baseline (step=0)
            enc_base.eval()
            with torch.no_grad():
                h0 = enc_base(x_tgt, edge_index)
                D0, M2A0, M2N0, auc0 = compute_all(h0, y_tgt, mu_s, var_s)

            # TTA (λ=0, no anchoring — let D_Σ evolve freely)
            enc_tta = copy.deepcopy(enc_base).train()
            scorer  = Scorer(HIDDEN_DIM).to(DEVICE).train()
            opt_tta = torch.optim.Adam(
                list(enc_tta.parameters()) + list(scorer.parameters()),
                lr=TTA_LR)

            for step in range(TTA_STEPS):
                enc_tta.train(); scorer.train(); opt_tta.zero_grad()
                h     = enc_tta(x_tgt, edge_index)
                logit = scorer(h)
                prob  = torch.sigmoid(logit)
                eps   = 1e-7
                H     = -(prob*(prob+eps).log() + (1-prob)*(1-prob+eps).log())
                H.mean().backward()
                opt_tta.step()

            enc_tta.eval(); scorer.eval()
            with torch.no_grad():
                h_f = enc_tta(x_tgt, edge_index)
                D_f, M2A_f, M2N_f, auc_f = compute_all(h_f, y_tgt, mu_s, var_s)

            records.append({
                "dataset":      DATASET,
                "normal_shift": ns,
                "anomaly_str":  as_,
                "seed":         seed,
                "D_Sigma_0":    round(D0,  2),
                "D_Sigma_T":    D_f,
                "M2_A_T":       M2A_f,
                "M2_N_T":       M2N_f,
                "AUC_0":        auc0,
                "AUC_T":        auc_f,
            })

            done += 1
            if done % 9 == 0:
                print(f"  [{done:>2}/{total_conditions}] "
                      f"ns={ns:.2f} as={as_:.1f} "
                      f"D_Σ={D_f:.0f}  M²_A={M2A_f:.0f}  AUC={auc_f:.4f}")

df = pd.DataFrame(records)
df.to_csv(results_csv(f"multi_shift_results_{DATASET}{OUT_SUFFIX}.csv"), index=False)
print(f"\nSaved multi_shift_results.csv  ({len(df)} rows)")

# ── Statistics ────────────────────────────────────────────────────────────────
D  = df["D_Sigma_T"].values
M  = df["M2_A_T"].values
U  = df["AUC_T"].values
sq = np.sqrt(np.clip(D, 0, None))

r_DM, p_DM = ss.spearmanr(D, M)
r_MA, p_MA = ss.spearmanr(M, U)
r_DA, p_DA = ss.spearmanr(D, U)
r_ns_D,_   = ss.spearmanr(df["normal_shift"], D)   # manipulation check

# Regressions
reg_DM = LinearRegression().fit(sq.reshape(-1,1), M)
R2_DM  = reg_DM.score(sq.reshape(-1,1), M)
reg_MA = LinearRegression().fit(M.reshape(-1,1),  U)
R2_MA  = reg_MA.score(M.reshape(-1,1),  U)

lines = [
    "=== MULTI-SHIFT EXPERIMENT RESULTS ===",
    f"  N conditions: {len(df)}",
    f"  D_Σ range:  [{D.min():.0f}, {D.max():.0f}]",
    f"  M²_A range: [{M.min():.0f}, {M.max():.0f}]",
    f"  AUC range:  [{U.min():.4f}, {U.max():.4f}]",
    "",
    "  Manipulation check:",
    f"    Spearman(normal_shift, D_Σ) = {r_ns_D:+.4f}  (expect: positive)",
    "",
    "  Theory chain:",
    f"    Spearman(D_Σ,  M²_A): r={r_DM:+.4f}  p={p_DM:.5f}  (expect: negative)",
    f"    Spearman(M²_A, AUC):  r={r_MA:+.4f}  p={p_MA:.5f}  (expect: positive)",
    f"    Spearman(D_Σ,  AUC):  r={r_DA:+.4f}  p={p_DA:.5f}  (expect: negative)",
    "",
    "  Regressions:",
    f"    M²_A = {reg_DM.intercept_:.1f} + ({reg_DM.coef_[0]:.4f})·√D_Σ   R²={R2_DM:.4f}",
    f"    AUC  = {reg_MA.intercept_:.4f} + ({reg_MA.coef_[0]:.6f})·M²_A   R²={R2_MA:.4f}",
    "",
    f"  M²_N range: [{df.M2_N_T.min():.1f}, {df.M2_N_T.max():.1f}]  (ref: {HIDDEN_DIM})",
]

summary = "\n".join(lines)
print("\n" + summary)
with open(results_txt(f"multi_shift_summary_{DATASET}{OUT_SUFFIX}.txt"), "w") as f: f.write(summary)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = plt.cm.plasma(
    (df["normal_shift"] - df["normal_shift"].min()) /
    (df["normal_shift"].max() - df["normal_shift"].min() + 1e-8)
)

ax = axes[0]
ax.scatter(D, M, c=colors, s=30, alpha=0.7, zorder=3)
x_fit = np.linspace(sq.min(), sq.max(), 100)
ax.plot(x_fit**2,
        reg_DM.intercept_ + reg_DM.coef_[0]*x_fit,
        "k--", lw=1.5, label=f"R²={R2_DM:.3f}")
ax.set_xlabel("$D_\\Sigma$ (after TTA)", fontsize=12)
ax.set_ylabel("$M^2_{A,t}$", fontsize=12)
ax.set_title(f"D_Σ → M²_A\n"
             f"Spearman r={r_DM:.3f}  p={p_DM:.4f}", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

sm = plt.cm.ScalarMappable(cmap="plasma",
     norm=plt.Normalize(df["normal_shift"].min(), df["normal_shift"].max()))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Normal Shift Strength", shrink=0.8)

ax = axes[1]
ax.scatter(M, U, c=colors, s=30, alpha=0.7, zorder=3)
x_fit2 = np.linspace(M.min(), M.max(), 100)
ax.plot(x_fit2,
        reg_MA.intercept_ + reg_MA.coef_[0]*x_fit2,
        "k--", lw=1.5, label=f"R²={R2_MA:.3f}")
ax.set_xlabel("$M^2_{A,t}$", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title(f"M²_A → AUC\n"
             f"Spearman r={r_MA:.3f}  p={p_MA:.4f}", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.colorbar(sm, ax=ax, label="Normal Shift Strength", shrink=0.8)

ax = axes[2]
ax.scatter(D, U, c=colors, s=30, alpha=0.7, zorder=3)
ax.set_xlabel("$D_\\Sigma$ (after TTA)", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title(f"D_Σ → AUC (direct)\n"
             f"Spearman r={r_DA:.3f}  p={p_DA:.4f}", fontsize=11)
ax.grid(alpha=0.3)
plt.colorbar(sm, ax=ax, label="Normal Shift Strength", shrink=0.8)

# plt.suptitle(
#     f"Multi-Shift Experiment: Theory Chain D_Σ → M²_A → AUC\n"
#     f"(N={len(df)}, color = Normality Shift strength)",
#     fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(figure(f"multi_shift_fig_{DATASET}{OUT_SUFFIX}.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved multi_shift_fig_{DATASET}{OUT_SUFFIX}.png")
print(f"Saved multi_shift_summary_{DATASET}{OUT_SUFFIX}.txt")