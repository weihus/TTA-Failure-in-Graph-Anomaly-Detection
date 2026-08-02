"""
exp_native_multishift.py — native-label 数据集的 multi-shift 实验

对 pygod 已注入异常的数据集（native labels），跑 6 shift × 3 seed = 18 条件，
不再 inject_anomaly。用于让 inj_cora 替换 Cora(synthetic) 后在 Table 1 / fig2
有同口径的 D_Σ→M²_A→AUC 链条数据（与 Questions/Reddit 的 N=18 一致）。

复用 exp_multi_shift 的 GCNEncoder / metrics / make_target 逻辑，区别：
  - labels 来自 data.y.bool()（native），不 inject
  - create_target 只做 normal shift，不注入新异常
  - 单 anomaly 层（无 anomaly_str 维度）
输出 multi_shift_native_<DATASET><SUFFIX>.csv + summary + fig。
"""

import os, sys, copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as ss
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, results_csv, results_txt, figure

DATASET  = sys.argv[1] if len(sys.argv) > 1 else "inj_cora"
SUFFIX   = sys.argv[2] if len(sys.argv) > 2 else ""
SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS, HIDDEN_DIM = 3, 64
PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN = 300, 3e-3, 2.0
TTA_STEPS, TTA_LR = 80, 5e-5
NORMAL_SHIFT_STRENGTHS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]
SEEDS = [100, 200, 300]
torch.manual_seed(SEED); np.random.seed(SEED)

def load_dataset(name):
    from pygod.utils import load_data
    data = load_data(name)
    return data.x.float(), data.edge_index, data.y.bool().long()

print(f"Loading {DATASET} (native labels)...")
x_raw, edge_index, labels = load_dataset(DATASET)
x_raw = x_raw.to(DEVICE); edge_index = edge_index.to(DEVICE); labels = labels.to(DEVICE)
print(f"  Nodes={x_raw.shape[0]}  Features={x_raw.shape[1]}  anomaly={labels.sum().item()} ({100*labels.float().mean():.1f}%)")

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)] + [GCNConv(hidden, hidden) for _ in range(layers-1)])
    def forward(self, x, ei):
        h = x
        for c in self.convs: h = F.relu(c(h, ei))
        return h

class Scorer(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.fc = nn.Linear(dim, 1)
    def forward(self, h): return self.fc(h).squeeze(-1)

def normal_stats(h, labels):
    hN = h[labels==0].detach(); return hN.mean(0), hN.var(0).clamp(min=1e-6)

def compute_all(h, labels, mu_s, var_s):
    hd = h.detach(); mu_t, var_t = normal_stats(hd, labels)
    ratio = var_t / var_s
    D = max(0.5*(ratio - 1 - ratio.log()).sum().item(), 0.0)
    hA = hd[labels==1]; M2A = ((hA - mu_t)**2 / var_t).sum(1).mean().item()
    hN = hd[labels==0]; M2N = ((hN - mu_t)**2 / var_t).sum(1).mean().item()
    score = ((hd - mu_t)**2 / var_t).sum(1).sqrt()
    auc = roc_auc_score(labels.cpu().numpy(), score.cpu().numpy())
    return D, M2A, M2N, auc

def make_target(x_cpu, shift, seed):
    torch.manual_seed(seed)
    x = x_cpu.clone()
    if shift > 0: x = x + torch.randn_like(x) * shift
    return x.to(DEVICE)

# Pretrain (native labels)
print("[1] Pre-training source model...")
enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
for ep in range(PRETRAIN_EPOCHS):
    enc.train(); opt.zero_grad()
    h = enc(x_raw, edge_index); cn = h[labels==0].mean(0); d = torch.norm(h - cn, dim=1)
    loss = (F.relu(MARGIN - d[labels==1])**2).mean() + (d[labels==0]**2).mean()
    loss.backward(); opt.step()
enc.eval()
with torch.no_grad(): h_src = enc(x_raw, edge_index)
mu_s, var_s = normal_stats(h_src, labels)
_, _, _, src_auc = compute_all(h_src, labels, mu_s, var_s)
print(f"  Source AUC = {src_auc:.4f}")

# Multi-shift (native)
total = len(NORMAL_SHIFT_STRENGTHS) * len(SEEDS)
print(f"\n[2] Running {total} conditions ({len(NORMAL_SHIFT_STRENGTHS)} shifts × {len(SEEDS)} seeds)...")
records = []
for ns in NORMAL_SHIFT_STRENGTHS:
    for seed in SEEDS:
        x_tgt = make_target(x_raw.cpu(), ns, seed)
        enc.eval()
        with torch.no_grad(): h0 = enc(x_tgt, edge_index)
        D0, M2A0, M2N0, auc0 = compute_all(h0, labels, mu_s, var_s)
        enc_tta = copy.deepcopy(enc).train()
        scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
        opt_tta = torch.optim.Adam(list(enc_tta.parameters()) + list(scorer.parameters()), lr=TTA_LR)
        for _ in range(TTA_STEPS):
            enc_tta.train(); scorer.train(); opt_tta.zero_grad()
            h = enc_tta(x_tgt, edge_index)
            prob = torch.sigmoid(scorer(h)); eps = 1e-7
            H = -(prob*(prob+eps).log() + (1-prob)*(1-prob+eps).log())
            H.mean().backward(); opt_tta.step()
        enc_tta.eval(); scorer.eval()
        with torch.no_grad(): h_f = enc_tta(x_tgt, edge_index)
        D_f, M2A_f, M2N_f, auc_f = compute_all(h_f, labels, mu_s, var_s)
        records.append({
            "dataset": DATASET, "normal_shift": ns, "seed": seed,
            "D_Sigma_0": round(D0, 2), "D_Sigma_T": round(D_f, 2),
            "M2_A_T": round(M2A_f, 2), "M2_N_T": round(M2N_f, 2),
            "AUC_0": round(auc0, 4), "AUC_T": round(auc_f, 4),
        })

df = pd.DataFrame(records)
df.to_csv(results_csv(f"multi_shift_native_{DATASET}{SUFFIX}.csv"), index=False)
print(f"\nSaved multi_shift_native_{DATASET}{SUFFIX}.csv  ({len(df)} rows)")

D = df["D_Sigma_T"].values.astype(float)
M = df["M2_A_T"].values.astype(float)
U = df["AUC_T"].values.astype(float)
rDM, pDM = ss.spearmanr(D, M)
rMA, pMA = ss.spearmanr(M, U)
rDA, pDA = ss.spearmanr(D, U)
r_ns_D, _ = ss.spearmanr(df["normal_shift"], D)

lines = [
    "=== NATIVE MULTI-SHIFT RESULTS ===",
    f"  Dataset: {DATASET}  (native labels, no injection)",
    f"  N conditions: {len(df)}  Source AUC: {src_auc:.4f}",
    f"  D_Σ range:  [{D.min():.0f}, {D.max():.0f}]",
    f"  M²_A range: [{M.min():.0f}, {M.max():.0f}]",
    f"  AUC range:  [{U.min():.4f}, {U.max():.4f}]",
    "",
    "  Manipulation check:",
    f"    Spearman(normal_shift, D_Σ) = {r_ns_D:+.4f}  (expect: positive)",
    "",
    "  Theory chain:",
    f"    Spearman(D_Σ, M²_A):  r={rDM:+.4f}  p={pDM:.5f}  (expect: negative)",
    f"    Spearman(M²_A, AUC): r={rMA:+.4f}  p={pMA:.5f}  (expect: positive)",
    f"    Spearman(D_Σ, AUC):  r={rDA:+.4f}  p={pDA:.5f}  (expect: negative)",
    "",
    f"  M²_N range: [{df.M2_N_T.min():.1f}, {df.M2_N_T.max():.1f}]  (ref: {HIDDEN_DIM})",
    "",
    f"  σ=0 AUC: {df[df.normal_shift==0.0].AUC_0.mean():.3f} → {df[df.normal_shift==0.0].AUC_T.mean():.3f}",
]
summary = "\n".join(lines); print("\n" + summary)
open(results_txt(f"multi_shift_native_{DATASET}{SUFFIX}.txt"), "w").write(summary)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].scatter(D, M, s=30, alpha=0.7); axes[0].set(xlabel="$D_\\Sigma$", ylabel="$M^2_A$", title=f"D_Σ→M²_A r={rDM:.3f}")
axes[1].scatter(M, U, s=30, alpha=0.7); axes[1].set(xlabel="$M^2_A$", ylabel="AUC", title=f"M²_A→AUC r={rMA:.3f}")
axes[2].scatter(D, U, s=30, alpha=0.7); axes[2].set(xlabel="$D_\\Sigma$", ylabel="AUC", title=f"D_Σ→AUC r={rDA:.3f}")
plt.tight_layout()
plt.savefig(figure(f"multi_shift_native_{DATASET}{SUFFIX}.png"), dpi=300, bbox_inches="tight"); plt.close()
print(f"Saved multi_shift_native_{DATASET}{SUFFIX}.png")
