"""
exp_causal_intervention.py — Experiment 1: Causal Intervention via Controlled Covariance Injection

核心思路：
  不再通过 TTA 优化过程间接控制 D_Σ（cross-λ v2/v3 已两次失败：detach bug + 锚定梯度不足）。
  而是直接对冻结的 encoder 输出做代数变换，人为注入精确已知的 D_Σ(δ)。

关键设计（必须遵守，否则实验数学上无意义）：
  只缩放正常节点 embedding，异常节点位置完全冻结不变：
    h'_N = μ_t + s(δ)·(h_N - μ_t)
    h'_A = h_A   (UNCHANGED)
  若同时缩放所有节点，M²_A 分子分母同步缩放，结果对 δ 不变——这是错误设计，需要避免。

两种注入模式：
  1. Isotropic（各向同性）：s(δ) = sqrt(1+δ)，所有维度统一缩放
     D_Σ(δ) = (d/2)·[δ - log(1+δ)]   闭式解，shift=0时无估计噪声
  2. Anisotropic（各向异性，经验方向）：用一次真实 TTA 运行观测到的逐维度膨胀比例 r_k
     var_t'_k(δ) = var_s_k·(1 + δ·(r_k-1))
     δ=1 时精确复现真实 TTA 终态的协方差模式

最关键的比较：
  把"纯代数注入"产生的 (D_Σ, AUC) 曲线，与"真实 TTA 优化轨迹"产生的 (D_Σ(t), AUC(t)) 点
  画在同一坐标系（x=D_Σ, y=AUC）。如果真实 TTA 的点落在合成曲线附近，
  说明 D_Σ 本身就足以解释 AUC 下降，不需要诉诸 entropy collapse、scorer mismatch 等替代解释。

输出：
  causal_intervention_results.csv
  causal_intervention_fig.png
  causal_intervention_summary.txt
"""

import os, sys, dgl, copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, scipy.stats as ss
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 添加项目根目录到 sys.path 以便导入 config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, results_csv, results_txt, figure

# ── Config ──────────────────────────────────────────────────────────────────
# 默认五数据集；命令行可覆盖：python exp_causal_intervention.py "cora,citeseer" _verify
if len(sys.argv) > 1:
    DATASETS = sys.argv[1].split(",")
else:
    DATASETS        = ["cora", "citeseer", "pubmed", "questions", "reddit"]   # 已验证理论链条的五个数据集
# DATASETS        = ['tolokers', 'weibo']   # 仅测试单数据集
OUT_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""   # 输出文件后缀，避免覆盖现有结果
PRETRAIN_EPOCHS = 300
PRETRAIN_LR     = 3e-3
MARGIN          = 2.0
TTA_STEPS       = 80
TTA_LR          = 1e-4
HIDDEN_DIM      = 64
NUM_LAYERS      = 3
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# δ 网格（覆盖与真实 TTA 观测到的 D_Σ 量级相当的区间）
DELTA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

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
    """返回 (x, edge_index, labels)，CPU tensor。"""
    planetoid_names = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "Pubmed"}
    if name in planetoid_names:
        data = Planetoid(root=DATA_DIR, name=planetoid_names[name])[0]
        x_raw = data.x
        ei    = data.edge_index
        x, labels = inject_anomaly(x_raw, ratio=0.05, strength=0.5, seed=0)
        return x, ei, labels
    elif name in PYGOD_DATASETS:
        from pygod.utils import load_data
        data = load_data(name)
        x = data.x.float()
        ei = data.edge_index
        y = data.y.bool().long()            # 0=inlier, >0=anomaly → binary
        return x, ei, y
    else:
        graphs, _ = dgl.data.utils.load_graphs(f"{DATA_DIR}/{name}")
        g = graphs[0]
        x = g.ndata["feature"].float()
        y = g.ndata["label"].long()
        s, e = g.edges()
        ei = torch.stack([s.long(), e.long()], dim=0)
        return x, ei, y

# ── 模型 ──────────────────────────────────────────────────────────────────────

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

class Scorer(nn.Module):
    """仅用于生成真实 TTA 对照轨迹，不参与合成注入实验本身。"""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, h): return self.fc(h).squeeze(-1)

# ── 度量计算 ──────────────────────────────────────────────────────────────────

def normal_stats(h, labels, grad=False):
    hN = h[labels==0] if grad else h[labels==0].detach()
    return hN.mean(0), hN.var(0).clamp(min=1e-6)

def compute_metrics(h, labels, mu_s, var_s):
    """h 可以是真实 embedding 或合成注入后的 embedding，逻辑完全一致。"""
    hd = h.detach() if h.requires_grad else h
    mu_t, var_t = normal_stats(hd, labels)
    ratio   = var_t / var_s
    D_Sigma = max(0.5*(ratio - 1 - ratio.log()).sum().item(), 0.0)
    hA      = hd[labels==1]
    M2_A    = ((hA - mu_t)**2 / var_t).sum(1).mean().item()
    hN      = hd[labels==0]
    M2_N    = ((hN - mu_t)**2 / var_t).sum(1).mean().item()
    score   = ((hd - mu_t)**2 / var_t).sum(1).sqrt()
    auc     = roc_auc_score(labels.cpu().numpy(), score.cpu().numpy())
    return D_Sigma, M2_A, M2_N, auc

# ── 核心：合成协方差注入（只缩放正常节点，异常节点冻结）────────────────────────

def apply_isotropic_injection(h, labels, mu_t, delta):
    """
    各向同性注入：所有维度统一缩放 sqrt(1+delta)。
    只变换正常节点；异常节点保持原始位置不变。
    """
    h_new = h.clone()
    s = np.sqrt(1.0 + delta)
    normal_mask = (labels == 0)
    h_new[normal_mask] = mu_t + s * (h[normal_mask] - mu_t)
    # h_new[labels==1] 保持不变（继承自 h.clone()）
    return h_new

def apply_anisotropic_injection(h, labels, mu_t, r_k, delta):
    """
    各向异性注入：用真实 TTA 观测到的逐维度膨胀比例 r_k 做缩放。
    var_t'_k(delta) = var_s_k * (1 + delta*(r_k-1))
    s_k(delta) = sqrt(1 + delta*(r_k-1))，每个维度独立缩放。
    delta=1 时精确复现真实 TTA 终态协方差模式。
    """
    h_new = h.clone()
    factor = 1.0 + delta * (r_k - 1.0)
    factor = torch.clamp(factor, min=1e-4)   # 防止某些维度 r_k<1 导致负数开方
    s_k = torch.sqrt(factor)                  # [d]
    normal_mask = (labels == 0)
    h_new[normal_mask] = mu_t + s_k.unsqueeze(0) * (h[normal_mask] - mu_t)
    return h_new

# ── 单数据集完整流程 ────────────────────────────────────────────────────────────

def run_one_dataset(name):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print('='*60)

    x_raw, ei, labels = load_dataset(name)
    N, Fdim = x_raw.shape
    n_anom = labels.sum().item()
    print(f"  N={N:,}  F={Fdim}  anomaly={n_anom}({100*n_anom/N:.1f}%)")

    x_raw  = x_raw.to(DEVICE); ei = ei.to(DEVICE); labels = labels.to(DEVICE)

    # ── 预训练（一次，固定，作为冻结 encoder）────────────────────────────────
    torch.manual_seed(42)
    enc = GCNEncoder(Fdim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h  = enc(x_raw, ei)
        cn = h[labels==0].mean(0)
        d  = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN - d[labels==1])**2).mean() + (d[labels==0]**2).mean()
        loss.backward(); opt.step()

    enc.eval()
    with torch.no_grad():
        h_src = enc(x_raw, ei)
        mu_s, var_s = normal_stats(h_src, labels)
        _, _, _, src_auc = compute_metrics(h_src, labels, mu_s, var_s)
    print(f"  Source AUC = {src_auc:.4f}")

    # frozen target embedding（这里用 shift=0，即 h_frozen = h_src，
    # 这保证了 var_t_natural = var_s，D_Σ(δ=0)=0，闭式解精确成立）
    h_frozen = h_src.detach().clone()
    mu_t, var_t_natural = normal_stats(h_frozen, labels)

    # ── 真实 TTA 轨迹（用于：①提取 r_k 经验方向 ②作为对照点）────────────────
    print(f"  Running real TTA trajectory ({TTA_STEPS} steps)...")
    enc_tta = copy.deepcopy(enc).train()
    scorer  = Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt_tta = torch.optim.Adam(
        list(enc_tta.parameters()) + list(scorer.parameters()), lr=TTA_LR)

    real_traj = []
    for step in range(TTA_STEPS):
        enc_tta.train(); scorer.train(); opt_tta.zero_grad()
        h     = enc_tta(x_raw, ei)
        prob  = torch.sigmoid(scorer(h))
        eps   = 1e-7
        L_ent = -(prob*(prob+eps).log() + (1-prob)*(1-prob+eps).log()).mean()
        L_ent.backward(); opt_tta.step()

        enc_tta.eval()
        with torch.no_grad():
            h_eval = enc_tta(x_raw, ei)
            D_t, M2A_t, M2N_t, auc_t = compute_metrics(h_eval, labels, mu_s, var_s)
        real_traj.append({"step": step+1, "D_Sigma": D_t, "M2_A": M2A_t, "AUC": auc_t})
        enc_tta.train()

    df_real = pd.DataFrame(real_traj)
    print(f"  Real TTA: D_Σ {df_real.D_Sigma.iloc[0]:.1f}→{df_real.D_Sigma.iloc[-1]:.1f}  "
          f"AUC {src_auc:.4f}→{df_real.AUC.iloc[-1]:.4f}")

    # 提取最终态的逐维度方差膨胀比例 r_k（用于各向异性注入）
    enc_tta.eval()
    with torch.no_grad():
        h_tta_final = enc_tta(x_raw, ei)
        _, var_t_tta_final = normal_stats(h_tta_final, labels)
    r_k = (var_t_tta_final / var_s).detach()   # [d]，逐维度膨胀比例

    # ── 各向同性注入 ────────────────────────────────────────────────────────
    print(f"  Running isotropic injection (δ grid: {DELTA_GRID})...")
    iso_records = []
    for delta in DELTA_GRID:
        h_inj = apply_isotropic_injection(h_frozen, labels, mu_t, delta)
        D_d, M2A_d, M2N_d, auc_d = compute_metrics(h_inj, labels, mu_s, var_s)
        iso_records.append({"delta": delta, "D_Sigma": D_d, "M2_A": M2A_d,
                            "M2_N": M2N_d, "AUC": auc_d, "condition": "isotropic"})

    # ── 各向异性注入（经验方向）──────────────────────────────────────────────
    print(f"  Running anisotropic injection (empirical direction)...")
    aniso_records = []
    for delta in DELTA_GRID:
        h_inj = apply_anisotropic_injection(h_frozen, labels, mu_t, r_k, delta)
        D_d, M2A_d, M2N_d, auc_d = compute_metrics(h_inj, labels, mu_s, var_s)
        aniso_records.append({"delta": delta, "D_Sigma": D_d, "M2_A": M2A_d,
                              "M2_N": M2N_d, "AUC": auc_d, "condition": "anisotropic"})

    df_iso   = pd.DataFrame(iso_records)
    df_aniso = pd.DataFrame(aniso_records)

    print(f"  Isotropic:   D_Σ [{df_iso.D_Sigma.min():.1f}, {df_iso.D_Sigma.max():.1f}]  "
          f"AUC [{df_iso.AUC.min():.4f}, {df_iso.AUC.max():.4f}]")
    print(f"  Anisotropic: D_Σ [{df_aniso.D_Sigma.min():.1f}, {df_aniso.D_Sigma.max():.1f}]  "
          f"AUC [{df_aniso.AUC.min():.4f}, {df_aniso.AUC.max():.4f}]")
    print(f"  At δ=1.0 (matches real TTA endpoint exactly): "
          f"synthetic D_Σ={df_aniso[df_aniso.delta==1.0].D_Sigma.values[0]:.1f}  "
          f"synthetic AUC={df_aniso[df_aniso.delta==1.0].AUC.values[0]:.4f}  "
          f"vs real TTA final D_Σ={df_real.D_Sigma.iloc[-1]:.1f}  "
          f"real AUC={df_real.AUC.iloc[-1]:.4f}")

    df_iso["dataset"]   = name
    df_aniso["dataset"] = name
    df_real["dataset"]  = name
    df_real["condition"] = "real_tta"

    return df_iso, df_aniso, df_real, src_auc

# ── 主流程 ────────────────────────────────────────────────────────────────────

all_iso, all_aniso, all_real = [], [], []
src_aucs = {}

for name in DATASETS:
    try:
        df_iso, df_aniso, df_real, src_auc = run_one_dataset(name)
        all_iso.append(df_iso); all_aniso.append(df_aniso); all_real.append(df_real)
        src_aucs[name] = src_auc
    except Exception as e:
        print(f"  SKIP {name}: {e}")

df_iso_all   = pd.concat(all_iso, ignore_index=True)
df_aniso_all = pd.concat(all_aniso, ignore_index=True)
df_real_all  = pd.concat(all_real, ignore_index=True)

df_iso_all.to_csv(results_csv(f"causal_isotropic{OUT_SUFFIX}.csv"), index=False)
df_aniso_all.to_csv(results_csv(f"causal_anisotropic{OUT_SUFFIX}.csv"), index=False)
df_real_all.to_csv(results_csv(f"causal_real_tta_trajectory{OUT_SUFFIX}.csv"), index=False)
print(f"\nSaved causal_isotropic.csv, causal_anisotropic.csv, causal_real_tta_trajectory.csv")

# ── 统计分析 ──────────────────────────────────────────────────────────────────

print("\n" + "="*62)
print("STATISTICAL ANALYSIS")
print("="*62)

summary_lines = []

for name in DATASETS:
    if name not in src_aucs: continue
    sub_iso   = df_iso_all[df_iso_all.dataset==name]
    sub_aniso = df_aniso_all[df_aniso_all.dataset==name]
    sub_real  = df_real_all[df_real_all.dataset==name]

    # 回归：M²_A ~ δ，AUC ~ δ（各向同性）
    reg_M2A = LinearRegression().fit(sub_iso[["delta"]], sub_iso["M2_A"])
    R2_M2A  = reg_M2A.score(sub_iso[["delta"]], sub_iso["M2_A"])
    reg_AUC = LinearRegression().fit(sub_iso[["delta"]], sub_iso["AUC"])
    R2_AUC  = reg_AUC.score(sub_iso[["delta"]], sub_iso["AUC"])

    # Bootstrap CI on AUC~delta slope
    n_boot = 1000
    slopes = []
    rng = np.random.default_rng(0)
    idx_all = np.arange(len(sub_iso))
    for _ in range(n_boot):
        idx_b = rng.choice(idx_all, size=len(idx_all), replace=True)
        sub_b = sub_iso.iloc[idx_b]
        if sub_b["delta"].nunique() < 2: continue
        rb = LinearRegression().fit(sub_b[["delta"]], sub_b["AUC"])
        slopes.append(rb.coef_[0])
    ci_lo, ci_hi = np.percentile(slopes, [2.5, 97.5])

    # 关键比较：anisotropic δ=1 (= 真实TTA终态) vs 真实TTA最后一步
    aniso_at_1 = sub_aniso[sub_aniso.delta==1.0]
    real_final = sub_real.iloc[-1]
    auc_synth_at_1 = aniso_at_1["AUC"].values[0] if len(aniso_at_1)>0 else np.nan
    auc_real_final = real_final["AUC"]
    auc_gap = auc_real_final - auc_synth_at_1

    # Wilcoxon: 用线性插值的 anisotropic 曲线，在每个真实TTA步的 D_Σ 处预测AUC，
    # 与真实TTA观测AUC做配对检验
    aniso_sorted = sub_aniso.sort_values("D_Sigma")
    pred_auc = np.interp(sub_real["D_Sigma"], aniso_sorted["D_Sigma"], aniso_sorted["AUC"])
    residuals = sub_real["AUC"].values - pred_auc
    if len(residuals) > 5:
        try:
            wstat, wp = ss.wilcoxon(residuals)
        except ValueError:
            wstat, wp = np.nan, np.nan   # residuals all zero (no movement)
    else:
        wstat, wp = np.nan, np.nan

    line = (f"\n[{name}]  Source AUC={src_aucs[name]:.4f}\n"
            f"  Isotropic regression:   M²_A~δ R²={R2_M2A:.4f}   AUC~δ R²={R2_AUC:.4f}\n"
            f"  Bootstrap CI (AUC~δ slope, 95%): [{ci_lo:.5f}, {ci_hi:.5f}]\n"
            f"  Matched comparison (δ=1.0 anisotropic vs real TTA final step):\n"
            f"    Synthetic AUC = {auc_synth_at_1:.4f}   Real TTA AUC = {auc_real_final:.4f}   "
            f"Gap = {auc_gap:+.4f}\n"
            f"  Wilcoxon (real TTA AUC vs D_Σ-matched synthetic prediction, "
            f"N={len(residuals)}): stat={wstat:.2f}  p={wp:.4f}\n"
            f"    Mean residual = {residuals.mean():+.4f}   "
            f"({'no significant gap → D_Σ alone explains AUC' if (not np.isnan(wp) and wp>0.05) else 'significant gap → additional mechanism beyond D_Σ exists'})")
    print(line)
    summary_lines.append(line)

with open(results_txt(f"causal_intervention_summary{OUT_SUFFIX}.txt"), "w") as f:
    f.write("\n".join(summary_lines))
print("\nSaved causal_intervention_summary.txt")

# ── 图表：核心比较图 ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.5*len(DATASETS), 4.5))
if len(DATASETS) == 1: axes = [axes]

for ax, name in zip(axes, DATASETS):
    if name not in src_aucs: continue
    sub_iso   = df_iso_all[df_iso_all.dataset==name].sort_values("D_Sigma")
    sub_aniso = df_aniso_all[df_aniso_all.dataset==name].sort_values("D_Sigma")
    sub_real  = df_real_all[df_real_all.dataset==name]

    ax.plot(sub_iso["D_Sigma"], sub_iso["AUC"], "--", color="#888780",
            lw=1.5, label="Synthetic (isotropic)", zorder=2)
    ax.plot(sub_aniso["D_Sigma"], sub_aniso["AUC"], "-", color="#533AB7",
            lw=2, label="Synthetic (empirical direction)", zorder=3)
    ax.scatter(sub_real["D_Sigma"], sub_real["AUC"], color="#D85A30",
              s=14, alpha=0.7, zorder=4, label="Real TTA trajectory (80 steps)")

    # 标注 δ=1 匹配点
    aniso_at_1 = sub_aniso[sub_aniso.delta==1.0]
    if len(aniso_at_1) > 0:
        ax.scatter(aniso_at_1["D_Sigma"], aniso_at_1["AUC"],
                  marker="*", s=200, color="gold", edgecolor="black",
                  zorder=5, label="δ=1.0 (matches real TTA endpoint)")

    ax.set_xlabel("$D_\\Sigma$", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title(f"{name}\nSource AUC = {src_aucs[name]:.3f}", fontsize=11)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(alpha=0.3)

# plt.suptitle("Causal Intervention: Synthetic Covariance Injection vs Real TTA Trajectory\n"
#              "(If real TTA points fall on the synthetic curve, D_Σ alone explains the AUC drop)",
#              fontsize=11, y=1.04)
plt.tight_layout()
plt.savefig(figure(f"causal_intervention_fig{OUT_SUFFIX}.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {figure('causal_intervention_fig.png')}")