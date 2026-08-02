"""
exp_real_datasets_v3.py — 修复版本

修复项：
  1. Source AUC 显示 Bug（_, src_auc, _, _ → 改为正确位置）
  2. 加入密度自动检测：平均度 > 100 时自动降为 1 层 GCN
  3. 排除 Amazon（完全过平滑，平均度 741）
  4. 加入图统计预检，不符合条件的数据集提前跳过

推荐数据集（按密度从小到大）：
  T-Finance：预期可用（请先运行预检查）
  T-Social： 节点数 570 万，太大，跳过
  YelpChi：  平均度 168，边缘可用（降为 1 层后尝试）
  Amazon：   平均度 741，自动跳过
"""

import os, sys, dgl, copy, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, pandas as pd, scipy.stats as ss
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv

# 添加项目根目录到 sys.path 以便导入 config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, results_csv, results_txt, figure

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATASETS = [
    "questions",   # avg_deg=4.1,  N=48K, anomaly=3.0%  ← 最稀疏，首选
    "reddit",      # avg_deg=15.3, N=11K, anomaly=3.3%
    # "tolokers",    # avg_deg=45.1, N=12K, anomaly=21.8%
    # "weibo",       # avg_deg=49.5, N=8K,  anomaly=10.3%
]

# 图密度门限：平均度超过此值自动跳过（Amazon = 741，超限）
MAX_AVG_DEGREE     = 300
# 使用 1 层 GCN 的密度门限（YelpChi = 168，超限后降层）
DENSE_LAYER_THRESH = 50

# NORMAL_SHIFT_STRENGTHS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]
NORMAL_SHIFT_STRENGTHS = [0.0]
SEEDS                  = [100, 200, 300]
PRETRAIN_EPOCHS        = 300      # 增加训练轮次以应对更复杂数据集
PRETRAIN_LR            = 3e-3
MARGIN                 = 2.0
TTA_STEPS              = 80
TTA_LR                 = 5e-5
HIDDEN_DIM             = 64
DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"

# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_and_check(name):
    """
    加载数据集并做适用性预检：
      - 返回 (x, edge_index, labels, num_layers) 如果可用
      - 返回 None 如果不适合（过于稠密或节点数过多）
    """
    path = f"{DATA_DIR}/{name}"
    try:
        graphs, _ = dgl.data.utils.load_graphs(path)
    except Exception as e:
        print(f"  [{name}] 加载失败: {e}")
        return None

    graph  = graphs[0]
    x      = graph.ndata["feature"].float()
    labels = graph.ndata["label"].long()
    s, e   = graph.edges()
    ei     = torch.stack([s.long(), e.long()], dim=0)

    N        = x.shape[0]
    E        = ei.shape[1]
    avg_deg  = E / N
    n_anom   = labels.sum().item()
    anom_pct = 100 * n_anom / N

    print(f"  [{name}] N={N:,}  E={E:,}  F={x.shape[1]}  "
          f"avg_deg={avg_deg:.0f}  anomaly={anom_pct:.1f}%")

    # 过大节点数跳过
    if N > 500_000:
        print(f"  [{name}] SKIP: 节点数 {N:,} 超过 500K，计算不可行")
        return None

    # 极高密度跳过
    if avg_deg > MAX_AVG_DEGREE:
        print(f"  [{name}] SKIP: 平均度 {avg_deg:.0f} > {MAX_AVG_DEGREE}，"
              f"全图 GCN 将完全过平滑（参考 Amazon: 741）")
        return None

    # 密度自适应层数
    if avg_deg > DENSE_LAYER_THRESH:
        num_layers = 1
        print(f"  [{name}] 平均度 {avg_deg:.0f} > {DENSE_LAYER_THRESH}，"
              f"自动降为 {num_layers} 层 GCN")
    else:
        num_layers = 3
        print(f"  [{name}] 平均度 {avg_deg:.0f}，使用 {num_layers} 层 GCN")

    return x, ei, labels, num_layers

# ── 模型 ──────────────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden)] +
            [GCNConv(hidden, hidden) for _ in range(layers - 1)]
        )
    def forward(self, x, ei):
        h = x
        for c in self.convs:
            h = F.relu(c(h, ei))
        return h

class Scorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, h): return self.fc(h).squeeze(-1)

# ── 度量计算 ──────────────────────────────────────────────────────────────────

def normal_stats(h, labels, grad=False):
    h_N = h[labels == 0] if grad else h[labels == 0].detach()
    return h_N.mean(0), h_N.var(0).clamp(min=1e-6)

def compute_metrics(h, labels, mu_s, var_s):
    h_det = h.detach()
    mu_t, var_t = normal_stats(h_det, labels)
    ratio   = var_t / var_s
    D_Sigma = max(0.5 * (ratio - 1.0 - ratio.log()).sum().item(), 0.0)
    h_A   = h_det[labels == 1]
    M2_At = ((h_A - mu_t) ** 2 / var_t).sum(1).mean().item()
    h_N   = h_det[labels == 0]
    M2_Nt = ((h_N - mu_t) ** 2 / var_t).sum(1).mean().item()
    score = ((h_det - mu_t) ** 2 / var_t).sum(1).sqrt()
    auc   = roc_auc_score(labels.cpu().numpy(), score.cpu().numpy())
    return D_Sigma, M2_At, M2_Nt, auc

# ── Normality Shift 模拟 ──────────────────────────────────────────────────────

def make_target(x_cpu, shift, seed):
    torch.manual_seed(seed)
    x = x_cpu.clone()
    if shift > 0:
        x = x + torch.randn_like(x) * shift
    return x.to(DEVICE)

# ── 单数据集实验 ──────────────────────────────────────────────────────────────

def run_one(name, x_raw, edge_index, labels, num_layers):
    x_raw      = x_raw.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    labels     = labels.to(DEVICE)
    x_cpu      = x_raw.cpu()

    # 预训练
    print(f"\n  [{name}] Pre-training ({num_layers}-layer GCN, {PRETRAIN_EPOCHS} epochs)...")
    torch.manual_seed(42)
    enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, num_layers).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)

    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h  = enc(x_raw, edge_index)
        cn = h[labels == 0].mean(0)
        d  = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN - d[labels == 1]) ** 2).mean() \
             + (d[labels == 0] ** 2).mean()
        loss.backward(); opt.step()

    # ── Fix: 正确提取 AUC（第4个返回值）────────────────────────────────────
    enc.eval()
    with torch.no_grad():
        h_src = enc(x_raw, edge_index)
        mu_s, var_s = normal_stats(h_src, labels)
        _, _, _, src_auc = compute_metrics(h_src, labels, mu_s, var_s)
    print(f"  [{name}] Source AUC = {src_auc:.4f}  "
          f"({'可用' if src_auc > 0.65 else '检测能力弱，结果仅供参考'})")

    if src_auc < 0.55:
        print(f"  [{name}] WARNING: Source AUC < 0.55，接近随机，"
              f"理论链条无法在此数据集上有效验证")

    # Multi-shift 循环
    records = []
    for ns in NORMAL_SHIFT_STRENGTHS:
        for seed in SEEDS:
            x_tgt = make_target(x_cpu, ns, seed)

            enc.eval()
            with torch.no_grad():
                h0 = enc(x_tgt, edge_index)
                D0, M2A0, M2N0, auc0 = compute_metrics(h0, labels, mu_s, var_s)

            enc_tta = copy.deepcopy(enc).train()
            scorer  = Scorer(HIDDEN_DIM).to(DEVICE).train()
            opt_tta = torch.optim.Adam(
                list(enc_tta.parameters()) + list(scorer.parameters()), lr=TTA_LR)

            for _ in range(TTA_STEPS):
                enc_tta.train(); scorer.train(); opt_tta.zero_grad()
                h    = enc_tta(x_tgt, edge_index)
                prob = torch.sigmoid(scorer(h))
                eps  = 1e-7
                H    = -(prob*(prob+eps).log() + (1-prob)*(1-prob+eps).log())
                H.mean().backward(); opt_tta.step()

            enc_tta.eval(); scorer.eval()
            with torch.no_grad():
                h_f = enc_tta(x_tgt, edge_index)
                D_f, M2A_f, M2N_f, auc_f = compute_metrics(h_f, labels, mu_s, var_s)

            records.append({
                "dataset": name, "normal_shift": ns, "seed": seed,
                "D_Sigma_0": round(D0, 2), "D_Sigma_T": round(D_f, 2),
                "M2_A_T": round(M2A_f, 2), "M2_N_T": round(M2N_f, 2),
                "AUC_0": round(auc0, 4), "AUC_T": round(auc_f, 4),
            })

    df = pd.DataFrame(records)
    r_ns_D, _  = ss.spearmanr(df["normal_shift"], df["D_Sigma_T"])
    r_DM, p_DM = ss.spearmanr(df["D_Sigma_T"],   df["M2_A_T"])
    r_MA, p_MA = ss.spearmanr(df["M2_A_T"],       df["AUC_T"])
    print(f"  [{name}] manip_check r={r_ns_D:+.3f}  "
          f"Spearman(D_Σ,M²_A)={r_DM:+.3f}(p={p_DM:.3f})  "
          f"Spearman(M²_A,AUC)={r_MA:+.3f}(p={p_MA:.3f})")
    print(f"  [{name}] D_Σ range [{df.D_Sigma_T.min():.0f}, {df.D_Sigma_T.max():.0f}]  "
          f"M²_A range [{df.M2_A_T.min():.0f}, {df.M2_A_T.max():.0f}]  "
          f"AUC range [{df.AUC_T.min():.4f}, {df.AUC_T.max():.4f}]")
    return df

# ── 主流程 ────────────────────────────────────────────────────────────────────

print("=" * 62)
print(f"Real GAD Datasets v3  (DEVICE={DEVICE})")
print("=" * 62)
print("\n[Step 1] Graph density pre-check:")

viable, skipped = [], []
for name in DATASETS:
    result = load_and_check(name)
    if result is None:
        skipped.append(name)
    else:
        viable.append((name, *result))

print(f"\nViable: {[v[0] for v in viable]}")
print(f"Skipped: {skipped}")

if not viable:
    print("\nNo viable datasets. Try T-Finance or check dataset paths.")
    import sys; sys.exit(0)

print("\n[Step 2] Running multi-shift experiment:")
all_dfs = []
for name, x_raw, ei, labels, nl in viable:
    df_ds = run_one(name, x_raw, ei, labels, nl)
    all_dfs.append(df_ds)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv(results_csv("real_dataset_results.csv"), index=False)
print(f"\nSaved real_dataset_results.csv  ({len(df_all)} rows)")

# ── 最终汇总 ──────────────────────────────────────────────────────────────────
print("\n=== FINAL RESULTS ===")
D  = df_all["D_Sigma_T"].values.astype(float)
M  = df_all["M2_A_T"].values.astype(float)
U  = df_all["AUC_T"].values.astype(float)

r_DM, p_DM = ss.spearmanr(D, M)
r_MA, p_MA = ss.spearmanr(M, U)
r_DA, p_DA = ss.spearmanr(D, U)
print(f"  Spearman(D_Σ,  M²_A) = {r_DM:+.4f}  p={p_DM:.5f}")
print(f"  Spearman(M²_A, AUC)  = {r_MA:+.4f}  p={p_MA:.5f}")
print(f"  Spearman(D_Σ,  AUC)  = {r_DA:+.4f}  p={p_DA:.5f}")
print(f"\n  Cora 参考值:  r(D_Σ,M²_A)=-0.63  r(M²_A,AUC)=+0.90")
print(f"  【判断标准】  r(M²_A,AUC) > +0.6 且方向正确 → 理论链条在真实数据上成立")