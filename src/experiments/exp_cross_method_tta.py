"""
exp_cross_method_tta.py — 跨方法 TTA 对比实验

比较三种 TTA 方法（TENT / SHOT / BN-adapt）在多个数据集上的效果：
  - TENT: 纯熵最小化
  - SHOT: 熵最小化 + 多样性正则化
  - BN-adapt: 批统计量替换（零梯度对照）

输出：
  cross_method_results.csv   实验结果
  cross_method_summary.txt   统计摘要
  cross_method_fig.png       小提琴图
"""

import os, sys, dgl, copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, scipy.stats as ss
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 添加项目根目录到 sys.path 以便导入 config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, results_csv, results_txt, figure

# ── Config ──────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    DATASETS = sys.argv[1].split(",")
else:
    DATASETS        = ["cora", "citeseer", "pubmed", "questions", "reddit"]
# DATASETS        = ['questions', 'reddit']   # 仅测试单数据集
OUT_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
SHIFT_LEVELS    = [0.0, 0.1]
SEEDS           = [42, 100, 200, 300, 400]
PILOT_SEED      = 999          # 仅用于 LR 预选，不参与正式统计

PRETRAIN_EPOCHS = 300
PRETRAIN_LR     = 3e-3
MARGIN          = 2.0
TTA_STEPS       = 80
LR_CANDIDATES   = [3e-5, 1e-4, 3e-4]   # TENT / SHOT 的学习率候选
SHOT_LAMBDA_DIV = 1.0                   # SHOT 多样性正则化权重

HIDDEN_DIM = 64
NUM_LAYERS = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42); np.random.seed(42)

# ── 数据加载 ──────────────────────────────────────────────────────────────────

def inject_anomaly(x, ratio=0.05, strength=0.5, seed=None):
    if seed is not None: torch.manual_seed(seed)
    x_new = x.clone(); n = x.shape[0]
    idx = torch.randperm(n)[:int(n*ratio)]
    labels = torch.zeros(n, dtype=torch.long)
    labels[idx] = 1
    x_new[idx] += torch.randn_like(x_new[idx]) * strength
    return x_new, labels

PYGOD_DATASETS = {"inj_cora", "inj_amazon", "inj_flickr", "weibo", "disney", "books", "enron"}

def load_dataset(name):
    planetoid_names = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "Pubmed"}
    if name in planetoid_names:
        data = Planetoid(root=DATA_DIR, name=planetoid_names[name])[0]
        x, labels = inject_anomaly(data.x, ratio=0.05, strength=0.5, seed=0)
        return x, data.edge_index, labels
    elif name in PYGOD_DATASETS:
        from pygod.utils import load_data
        data = load_data(name)
        x = data.x.float()
        ei = data.edge_index
        y = data.y.bool().long()
        return x, ei, y
    graphs, _ = dgl.data.utils.load_graphs(f"{DATA_DIR}/{name}")
    g = graphs[0]
    x = g.ndata["feature"].float()
    y = g.ndata["label"].long()
    s, e = g.edges()
    return x, torch.stack([s.long(), e.long()], dim=0), y

def make_target(x_cpu, shift, seed):
    torch.manual_seed(seed)
    x = x_cpu.clone()
    if shift > 0: x = x + torch.randn_like(x) * shift
    return x.to(DEVICE)

# ── 架构（基础版，用于 TENT / SHOT）──────────────────────────────────────────

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden)] +
            [GCNConv(hidden, hidden) for _ in range(layers-1)])
    def forward(self, x, ei):
        h = x
        for c in self.convs: h = F.relu(c(h, ei))
        return h

# ── 架构（带 BatchNorm，专用于 BN-adapt）─────────────────────────────────────

class GCNEncoderBN(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden)] +
            [GCNConv(hidden, hidden) for _ in range(layers-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
    def forward(self, x, ei):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, ei)))
        return h

class Scorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, h): return self.fc(h).squeeze(-1)

# ── 度量计算（与此前所有实验完全一致）──────────────────────────────────────────

def normal_stats(h, labels, grad=False):
    hN = h[labels==0] if grad else h[labels==0].detach()
    return hN.mean(0), hN.var(0).clamp(min=1e-6)

def compute_metrics(h, labels, mu_s, var_s):
    hd = h.detach()
    mu_t, var_t = normal_stats(hd, labels)
    ratio   = var_t / var_s
    D_Sigma = max(0.5*(ratio - 1 - ratio.log()).sum().item(), 0.0)
    hA      = hd[labels==1]
    M2_A    = ((hA - mu_t)**2 / var_t).sum(1).mean().item()
    score   = ((hd - mu_t)**2 / var_t).sum(1).sqrt()
    auc     = roc_auc_score(labels.cpu().numpy(), score.cpu().numpy())
    return D_Sigma, M2_A, auc

# ── 三种 TTA 方法 ────────────────────────────────────────────────────────────

def tta_tent(enc_init, x_tgt, ei, labels, lr, steps=TTA_STEPS):
    """方法1: 纯熵最小化"""
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt = torch.optim.Adam(list(enc.parameters())+list(scorer.parameters()), lr=lr)
    final_loss = None
    for step in range(steps):
        enc.train(); scorer.train(); opt.zero_grad()
        h = enc(x_tgt, ei)
        prob = torch.sigmoid(scorer(h))
        eps = 1e-7
        loss = -(prob*(prob+eps).log() + (1-prob)*(1-prob+eps).log()).mean()
        loss.backward(); opt.step()
        final_loss = loss.item()
    enc.eval()
    with torch.no_grad():
        h_final = enc(x_tgt, ei)
    return h_final, final_loss

def tta_shot(enc_init, x_tgt, ei, labels, lr, steps=TTA_STEPS, lambda_div=SHOT_LAMBDA_DIV):
    """方法2: 熵最小化 + 多样性正则化（防止预测整体坍缩）"""
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt = torch.optim.Adam(list(enc.parameters())+list(scorer.parameters()), lr=lr)
    final_loss = None
    for step in range(steps):
        enc.train(); scorer.train(); opt.zero_grad()
        h = enc(x_tgt, ei)
        prob = torch.sigmoid(scorer(h))
        eps = 1e-7
        L_ent = -(prob*(prob+eps).log() + (1-prob)*(1-prob+eps).log()).mean()
        mean_p = prob.mean().clamp(eps, 1-eps)
        H_mean = -(mean_p*mean_p.log() + (1-mean_p)*(1-mean_p).log())  # 想要最大化
        loss = L_ent - lambda_div * H_mean
        loss.backward(); opt.step()
        final_loss = loss.item()
    enc.eval()
    with torch.no_grad():
        h_final = enc(x_tgt, ei)
    return h_final, final_loss

def tta_bn_adapt(enc_bn_init, x_tgt, ei, labels):
    """
    方法3: BN-adapt — 零梯度，零熵计算。
    逐层用目标域批统计量直接替换 BatchNorm 的 running_mean/running_var。
    这是排除 entropy collapse 解释的最有力对照：完全不涉及熵这个量。
    """
    enc = copy.deepcopy(enc_bn_init)
    enc.eval()
    with torch.no_grad():
        h = x_tgt
        for conv, bn in zip(enc.convs, enc.bns):
            h_pre = conv(h, ei)
            bn.running_mean.data = h_pre.mean(0)
            bn.running_var.data  = h_pre.var(0, unbiased=False).clamp(min=1e-6)
            h = F.relu(bn(h_pre))
    return h, None  # 无损失值（无梯度训练）

# ── LR 预选（基于训练损失收敛性，不看 AUC）────────────────────────────────────

def select_lr(method_fn, enc_init, x_pilot, ei, labels, candidates):
    """在 PILOT_SEED 上试跑短轮次，选择训练损失最低的学习率。不参与正式统计。"""
    best_lr, best_loss = candidates[0], float("inf")
    for lr in candidates:
        torch.manual_seed(PILOT_SEED)
        _, loss = method_fn(enc_init, x_pilot, ei, labels, lr, steps=20)
        if loss is not None and loss < best_loss:
            best_lr, best_loss = lr, loss
    return best_lr

# ── 单数据集主流程 ────────────────────────────────────────────────────────────

def run_dataset(name):
    print(f"\n{'='*62}")
    print(f"Dataset: {name}")
    print('='*62)

    x_raw, ei, labels = load_dataset(name)
    Fdim = x_raw.shape[1]
    n_anom = labels.sum().item()
    print(f"  N={x_raw.shape[0]:,}  F={Fdim}  anomaly={n_anom}({100*n_anom/len(labels):.1f}%)")

    x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
    x_cpu = x_raw.cpu()

    # ── 预训练：基础版（TENT/SHOT用）──────────────────────────────────────────
    torch.manual_seed(42)
    enc = GCNEncoder(Fdim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h = enc(x_raw, ei)
        cn = h[labels==0].mean(0)
        d = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN-d[labels==1])**2).mean() + (d[labels==0]**2).mean()
        loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        h_src = enc(x_raw, ei)
    mu_s, var_s = normal_stats(h_src, labels)
    _, _, src_auc = compute_metrics(h_src, labels, mu_s, var_s)
    print(f"  Source AUC (base encoder) = {src_auc:.4f}")

    # ── 预训练：BN版（BN-adapt用）─────────────────────────────────────────────
    torch.manual_seed(42)
    enc_bn = GCNEncoderBN(Fdim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    opt_bn = torch.optim.Adam(enc_bn.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc_bn.train(); opt_bn.zero_grad()
        h = enc_bn(x_raw, ei)
        cn = h[labels==0].mean(0)
        d = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN-d[labels==1])**2).mean() + (d[labels==0]**2).mean()
        loss.backward(); opt_bn.step()
    enc_bn.eval()
    with torch.no_grad():
        h_src_bn = enc_bn(x_raw, ei)
    mu_s_bn, var_s_bn = normal_stats(h_src_bn, labels)
    _, _, src_auc_bn = compute_metrics(h_src_bn, labels, mu_s_bn, var_s_bn)
    print(f"  Source AUC (BN encoder)   = {src_auc_bn:.4f}")

    # ── LR 预选（pilot seed, shift=0）────────────────────────────────────────
    x_pilot = make_target(x_cpu, 0.0, PILOT_SEED)
    lr_tent = select_lr(tta_tent, enc, x_pilot, ei, labels, LR_CANDIDATES)
    lr_shot = select_lr(tta_shot, enc, x_pilot, ei, labels, LR_CANDIDATES)
    print(f"  Selected LR — TENT: {lr_tent:.0e}   SHOT: {lr_shot:.0e}")

    # ── 正式实验：3方法 × 2shift × 5seeds ───────────────────────────────────
    records = []
    for shift in SHIFT_LEVELS:
        for seed in SEEDS:
            x_tgt = make_target(x_cpu, shift, seed)

            # No-TTA baseline
            with torch.no_grad():
                h0 = enc(x_tgt, ei)
            D0, M2A0, auc0 = compute_metrics(h0, labels, mu_s, var_s)

            # No-TTA baseline (BN encoder — for BN-adapt)
            with torch.no_grad():
                h0_bn = enc_bn(x_tgt, ei)
            D0_bn, M2A0_bn, auc0_bn = compute_metrics(h0_bn, labels, mu_s_bn, var_s_bn)

            # TENT
            torch.manual_seed(seed)
            h_tent, _ = tta_tent(enc, x_tgt, ei, labels, lr_tent)
            D_t, M2A_t, auc_t = compute_metrics(h_tent, labels, mu_s, var_s)

            # SHOT
            torch.manual_seed(seed)
            h_shot, _ = tta_shot(enc, x_tgt, ei, labels, lr_shot)
            D_s, M2A_s, auc_s = compute_metrics(h_shot, labels, mu_s, var_s)

            # BN-adapt（用 BN 专属源域统计量评估，因为架构不同）
            h_bn, _ = tta_bn_adapt(enc_bn, x_tgt, ei, labels)
            D_b, M2A_b, auc_b = compute_metrics(h_bn, labels, mu_s_bn, var_s_bn)

            for method, (D_, M2A_, auc_, auc0_) in [
                ("TENT",      (D_t, M2A_t, auc_t, auc0)),
                ("SHOT",      (D_s, M2A_s, auc_s, auc0)),
                ("BN-adapt",  (D_b, M2A_b, auc_b, auc0_bn)),
            ]:
                records.append({
                    "dataset": name, "shift": shift, "seed": seed, "method": method,
                    "AUC_0": auc0_, "D_Sigma": D_, "M2_A": M2A_, "AUC_T": auc_,
                    "delta_AUC": auc_ - auc0_,
                })

        # 简报
        sub = pd.DataFrame(records)
        sub = sub.loc[sub["shift"] == shift]
        print(f"  [shift={shift}]")
        for method in ["TENT", "SHOT", "BN-adapt"]:
            m = sub.loc[sub["method"] == method]
            print(f"    {method:<10} ΔAUC = {m.delta_AUC.mean():+.4f} ± {m.delta_AUC.std():.4f}")

    return pd.DataFrame(records)

# ── 主流程 ────────────────────────────────────────────────────────────────────

all_dfs = []
for name in DATASETS:
    # df_ds = run_dataset(name)
    try:
        df_ds = run_dataset(name)
        all_dfs.append(df_ds)
    except Exception as e:
        print(f"  SKIP {name}: {e}")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv(results_csv(f"cross_method_results{OUT_SUFFIX}.csv"), index=False)
print(f"\nSaved cross_method_results.csv  ({len(df_all)} rows)")

# ── 统计分析 ──────────────────────────────────────────────────────────────────

print("\n" + "="*62)
print("STATISTICAL ANALYSIS")
print("="*62)

summary = []

for shift in SHIFT_LEVELS:
    print(f"\n[shift={shift}]")
    summary.append(f"\n[shift={shift}]")
    sub = df_all.loc[df_all["shift"] == shift]

    # 跨方法一致性：多少比例的 (dataset×method×seed) 组合显示 AUC 下降
    n_total = len(sub)
    n_decline = (sub["delta_AUC"] < 0).sum()
    pct_decline = 100*n_decline/n_total
    line = f"  跨条件 AUC 下降比例: {n_decline}/{n_total} ({pct_decline:.1f}%)"
    print(line); summary.append(line)

    # Kruskal-Wallis: 方法间 delta_AUC 是否有显著差异
    groups = [sub.loc[sub["method"] == m, "delta_AUC"].values for m in ["TENT","SHOT","BN-adapt"]]
    kw_stat, kw_p = ss.kruskal(*groups)
    line = f"  Kruskal-Wallis (方法间 ΔAUC 差异): H={kw_stat:.3f}  p={kw_p:.4f}"
    print(line); summary.append(line)

    # 每个方法的 D_Σ vs M²_A 相关性（验证级联是否在每种方法下都成立）
    for method in ["TENT", "SHOT", "BN-adapt"]:
        m = sub.loc[sub["method"] == method]
        if m["D_Sigma"].std() > 1e-6:
            r, p = ss.spearmanr(m["D_Sigma"], m["M2_A"])
            line = f"  [{method}] Spearman(D_Σ, M²_A) = {r:+.3f}  p={p:.4f}  (N={len(m)})"
        else:
            line = f"  [{method}] D_Σ 无变化，无法计算相关性"
        print(line); summary.append(line)

with open(results_txt(f"cross_method_summary{OUT_SUFFIX}.txt"), "w") as f:
    f.write("\n".join(summary))
print("\nSaved cross_method_summary.txt")

# ── 图表 ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(DATASETS), len(SHIFT_LEVELS),
                         figsize=(5.5*len(SHIFT_LEVELS), 4*len(DATASETS)))
if len(DATASETS)==1: axes=[axes]
if len(SHIFT_LEVELS)==1: axes=[[a] for a in axes]

C = {"TENT": "#D85A30", "SHOT": "#533AB7", "BN-adapt": "#1D9E75"}

for i, name in enumerate(DATASETS):
    for j, shift in enumerate(SHIFT_LEVELS):
        ax = axes[i][j]
        sub = df_all.loc[(df_all["dataset"] == name) & (df_all["shift"] == shift)]
        if len(sub)==0: continue

        data_by_method = {m: sub.loc[sub["method"] == m, "delta_AUC"].values
                          for m in ["TENT","SHOT","BN-adapt"]}
        positions = [1,2,3]
        parts = ax.violinplot([data_by_method[m] for m in ["TENT","SHOT","BN-adapt"]],
                              positions=positions, showmedians=True)
        for pc, m in zip(parts["bodies"], ["TENT","SHOT","BN-adapt"]):
            pc.set_facecolor(C[m]); pc.set_alpha(0.7)

        for pos, m in zip(positions, ["TENT","SHOT","BN-adapt"]):
            vals = data_by_method[m]
            ax.scatter([pos]*len(vals), vals, s=14, color=C[m], zorder=3, alpha=0.8)

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(positions); ax.set_xticklabels(["TENT","SHOT","BN-adapt"], fontsize=8)
        ax.set_ylabel("ΔAUC", fontsize=9)
        ax.set_title(f"{name}  shift={shift}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

# plt.suptitle("Cross-Method TTA: ΔAUC by Method\n"
#              "(below zero = AUC degraded; consistency across methods rules out entropy-specific artifacts)",
#              fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(figure(f"cross_method_fig{OUT_SUFFIX}.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved cross_method_fig.png")