"""
analysis_planetoid_extension.py — 分析 Citeseer/Pubmed 是否符合论文结论

从已有 CSV 提取 Cora / Citeseer / Pubmed 的论文口径指标，逐条对照论文核心结论：
  结论1 TTA 损害 GAD       —— multi_shift σ=0 的 AUC_0→AUC_T
  结论2 D_Σ→M²_A→AUC 链   —— multi_shift Spearman (Table 1)
  结论3 isotropic 因果通路  —— causal_isotropic AUC~D_Σ 斜率 + bootstrap CI (Table 2)
  结论4 方向性悖论范围      —— causal anisotropic δ=1 gap (Table 3)；合成数据集应像 Cora 无大负 gap
  结论5 TENT≈SHOT          —— cross_method Friedman + post-hoc Wilcoxon (Table 4)
  结论6 压缩效应           —— AUC_T~AUC_0 控制 dataset 斜率 (Table 5)

只读 CSV + 算统计，不跑模型。Cora 同时算作口径校验（对照论文已报值）。
"""

import os, sys
import numpy as np, pandas as pd, scipy.stats as ss
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

PLANETOIDS = ["cora", "citeseer", "pubmed"]
RNG = np.random.default_rng(0)
out = []

def log(*a):
    line = " ".join(str(x) for x in a)
    print(line); out.append(line)

# ── 1. multi_shift（Table 1 + 现象）───────────────────────────────────────────
log("="*70); log("1. MULTI-SHIFT: D_Σ → M²_A → AUC 链 (Table 1) + 现象"); log("="*70)

df_cora = pd.read_csv(results_csv("multi_shift_results.csv"))
df_cora["dataset"] = "cora"
df_cit = pd.read_csv(results_csv("multi_shift_results_citeseer.csv"))
df_pub = pd.read_csv(results_csv("multi_shift_results_pubmed.csv"))
ms = pd.concat([df_cora, df_cit, df_pub], ignore_index=True)

log("\nDataset      r(D_Σ,M²_A)   p       r(M²_A,AUC)  p       N    M²_N范围")
log("-"*72)
for ds in PLANETOIDS:
    s = ms[ms.dataset==ds]
    rDM, pDM = ss.spearmanr(s["D_Sigma_T"], s["M2_A_T"])
    rMA, pMA = ss.spearmanr(s["M2_A_T"], s["AUC_T"])
    log(f"{ds:10s}   {rDM:+.3f}      {pDM:.1e}  {rMA:+.3f}      {pMA:.1e}  {len(s):3d}  [{s.M2_N_T.min():.1f}, {s.M2_N_T.max():.1f}]")

# Cora within-stratum（按 anomaly_str 分层，取平均/合并）
s = ms[ms.dataset=="cora"]
within = []
for a, g in s.groupby("anomaly_str"):
    r,_ = ss.spearmanr(g["D_Sigma_T"], g["M2_A_T"]); within.append(r)
log(f"\n  Cora within-stratum r(D_Σ,M²_A) per anomaly_str {sorted(s.anomaly_str.unique())}:")
log(f"    {[round(x,2) for x in within]}  mean={np.mean(within):.2f}  (论文≈-0.91)")

log("\n现象: σ_shift=0 时 AUC_0 → AUC_T (TTA 是否损害)：")
for ds in PLANETOIDS:
    s = ms[(ms.dataset==ds) & (ms.normal_shift==0.0)]
    # 与论文一致：取 anomaly_str=0.5 的 3 seeds 均值（Cora 口径）
    s05 = s[s.anomaly_str==0.5]
    a0, aT = s05["AUC_0"].mean(), s05["AUC_T"].mean()
    log(f"  {ds:10s}  anomaly_str=0.5: AUC {a0:.3f} → {aT:.3f}  (Δ={aT-a0:+.3f})   全σ0: AUC {s.AUC_0.mean():.3f}→{s.AUC_T.mean():.3f}")

# ── 2. causal isotropic（Table 2）─────────────────────────────────────────────
log("\n" + "="*70); log("2. CAUSAL ISOTROPIC: AUC~D_Σ 斜率 + bootstrap CI (Table 2)"); log("="*70)

iso = pd.read_csv(results_csv("causal_isotropic.csv"))
log("\nDataset      β(AUC~D_Σ)      95% bootstrap CI              β(AUC~δ)       CI排除0?")
log("-"*74)
for ds in PLANETOIDS:
    s = iso[iso.dataset==ds].sort_values("delta")
    # AUC ~ D_Σ
    reg = LinearRegression().fit(s[["D_Sigma"]], s["AUC"])
    beta_D = reg.coef_[0]
    slopes=[]
    idx=np.arange(len(s))
    for _ in range(2000):
        b = RNG.choice(idx, len(idx), replace=True)
        sb=s.iloc[b]
        if sb.D_Sigma.nunique()<2: continue
        slopes.append(LinearRegression().fit(sb[["D_Sigma"]], sb["AUC"]).coef_[0])
    ci_D = np.percentile(slopes,[2.5,97.5])
    # AUC ~ δ（口径2，对照 summary）
    reg2 = LinearRegression().fit(s[["delta"]], s["AUC"])
    beta_d = reg2.coef_[0]
    slopes2=[]
    for _ in range(2000):
        b = RNG.choice(idx, len(idx), replace=True)
        sb=s.iloc[b]
        if sb.delta.nunique()<2: continue
        slopes2.append(LinearRegression().fit(sb[["delta"]], sb["AUC"]).coef_[0])
    ci_d = np.percentile(slopes2,[2.5,97.5])
    excl = "是" if (ci_D[0]<0<ci_D[1]==False) else "是" if ci_D[1]<0 or ci_D[0]>0 else "否"
    log(f"{ds:10s}   {beta_D:+.2e}     [{ci_D[0]:+.2e}, {ci_D[1]:+.2e}]   {beta_d:+.2e}     {excl}")

# ── 3. causal anisotropic gap（Table 3）───────────────────────────────────────
log("\n" + "="*70); log("3. CAUSAL ANISOTROPIC: δ=1 gap (Table 3) — 合成数据集应无大负gap"); log("="*70)

aniso = pd.read_csv(results_csv("causal_anisotropic.csv"))
real = pd.read_csv(results_csv("causal_real_tta_trajectory.csv"))
# 论文对照值
log("\n  论文已报: Cora gap=+0.003 | Questions -0.31 | Reddit -0.20（真实数据集出现悖论）")
log("\nDataset      synth_AUC(δ=1)  real_AUC(终态)   gap=real-synth   悖论?")
log("-"*66)
for ds in PLANETOIDS:
    syn = aniso[(aniso.dataset==ds)&(aniso.delta==1.0)]["AUC"].values[0]
    rfin = real[real.dataset==ds]["AUC"].iloc[-1]
    gap = rfin - syn
    paradox = "否(像Cora)" if abs(gap)<0.05 else "是(像真实数据集)"
    log(f"  {ds:10s}  {syn:.4f}          {rfin:.4f}          {gap:+.4f}         {paradox}")

# ── 4. cross_method Table 4 + Friedman（结论5 TENT≈SHOT）──────────────────────
log("\n" + "="*70); log("4. CROSS-METHOD: ΔAUC 表 + Friedman (Table 4, 结论5 TENT≈SHOT)"); log("="*70)

cm = pd.read_csv(results_csv("cross_method_results.csv"))
log("\nTable 4 口径: mean ΔAUC (5 seeds)")
pivot = cm[cm.dataset.isin(PLANETOIDS)].pivot_table(
    index=["dataset","shift"], columns="method", values="delta_AUC", aggfunc="mean")
log(pivot.round(4).to_string())

# Friedman: block=(dataset,seed)，每block 3方法排名。分别算 shift=0 的 五数据集 vs 三数据集(论文口径)
def friedman_posthoc(sub, label):
    sub = sub.copy()
    # wide: rows=block(dataset,seed), cols=method
    w = sub.pivot_table(index=["dataset","seed"], columns="method", values="delta_AUC")
    w = w.dropna()
    if len(w) < 3: log(f"  [{label}] block 不足"); return
    chi2, p = ss.friedmanchisquare(w["TENT"], w["SHOT"], w["BN-adapt"])
    log(f"  [{label}] Friedman χ²={chi2:.2f}  p={p:.4f}  (N_blocks={len(w)})")
    # post-hoc Wilcoxon paired by block
    for a,b in [("TENT","SHOT"),("TENT","BN-adapt"),("SHOT","BN-adapt")]:
        st,pp = ss.wilcoxon(w[a], w[b])
        log(f"     {a} vs {b}: Wilcoxon p={pp:.4f}")

log("\n  shift=0.0:")
friedman_posthoc(cm[(cm['shift']==0.0)&(cm.dataset.isin(PLANETOIDS))], "三Planetoid")
friedman_posthoc(cm[(cm['shift']==0.0)&(cm.dataset.isin(["cora","questions","reddit"]))], "论文三数据集(复现)")
log("\n  shift=0.1:")
friedman_posthoc(cm[(cm['shift']==0.1)&(cm.dataset.isin(PLANETOIDS))], "三Planetoid")

# ── 5. 压缩效应（Table 5）─────────────────────────────────────────────────────
log("\n" + "="*70); log("5. 压缩效应: AUC_T~AUC_0 控制 dataset (Table 5)"); log("="*70)

def compression(sub, method):
    # 控制 dataset 固定效应：dataset one-hot + AUC_0
    ds_dummies = pd.get_dummies(sub["dataset"], prefix="ds", drop_first=True)
    X = pd.concat([sub[["AUC_0"]].reset_index(drop=True), ds_dummies.reset_index(drop=True)], axis=1)
    y = sub["AUC_T"].values
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    R2 = reg.score(X, y)
    # cluster bootstrap at dataset level
    datasets = sub["dataset"].unique()
    slopes=[]
    for _ in range(2000):
        boot_ds = RNG.choice(datasets, len(datasets), replace=True)
        parts=[]
        for d in boot_ds:
            parts.append(sub[sub.dataset==d])
        sb = pd.concat(parts)
        if sb["AUC_0"].nunique()<2: continue
        sXd = pd.get_dummies(sb["dataset"], prefix="ds", drop_first=True)
        sX = pd.concat([sb[["AUC_0"]].reset_index(drop=True), sXd.reset_index(drop=True)], axis=1)
        sy = sb["AUC_T"].values
        try:
            slopes.append(LinearRegression().fit(sX, sy).coef_[0])
        except: continue
    ci = np.percentile(slopes,[2.5,97.5])
    return beta, ci, R2

for method in ["TENT","SHOT"]:
    sub = cm[(cm.method==method)&(cm.dataset.isin(PLANETOIDS))]
    beta, ci, R2 = compression(sub, method)
    log(f"  [{method}] 三Planetoid  β={beta:.3f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]  R²={R2:.3f}  N={len(sub)}")
for method in ["TENT","SHOT"]:
    sub = cm[(cm.method==method)&(cm.dataset.isin(["cora","questions","reddit"]))]
    beta, ci, R2 = compression(sub, method)
    log(f"  [{method}] 论文三数据集  β={beta:.3f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]  R²={R2:.3f}  N={len(sub)}  (论文0.164/0.182)")

# ── 保存 ──────────────────────────────────────────────────────────────────────
with open(results_txt("planetoid_extension_analysis.txt"), "w") as f:
    f.write("\n".join(out))
log(f"\nSaved → {results_txt('planetoid_extension_analysis.txt')}")
