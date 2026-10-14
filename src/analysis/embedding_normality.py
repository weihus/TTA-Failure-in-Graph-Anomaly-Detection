"""
embedding_normality.py — P0: how non-Gaussian are the embeddings, and does
non-Gaussianity track degradation?

Reviewer #1 (R2) worries that TTA degradation is driven by the Mahalanobis
score's implicit Gaussian assumption failing on non-Gaussian embeddings.
This script measures, on the SAME adapted embeddings as the scoring
experiment:

    1. Mardia multivariate skewness / kurtosis  (overall non-Gaussianity)
    2. Henze-Zirkler multivariate normality statistic (aggregate test)
    3. Per-dimension Shapiro-Wilk rejection rate
    4. Per-dimension excess kurtosis mean

We then correlate these normality statistics against the degradation
(delta AUC under diagonal Mahalanobis) across (dataset, shift, seed, method)
to test whether *more non-Gaussian => more degradation*. If the correlation
is weak, non-Gaussianity is not the driver.

Outputs:
    embedding_normality.csv        per-run normality stats + delta_AUC
    embedding_normality_summary.txt
"""

import os, sys, copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, scipy.stats as ss
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

# reuse the TTA + scoring protocol
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from exp_nongaussian_scoring import (
    load_dataset, make_target, GCNEncoder, Scorer,
    tta_tent, tta_shot, select_lr, score_maha_diag,
    PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN, TTA_STEPS, LR_CANDIDATES, SHOT_LAMBDA_DIV,
    HIDDEN_DIM, NUM_LAYERS, DEVICE,
)

if len(sys.argv) > 1:
    DATASETS = sys.argv[1].split(",")
else:
    DATASETS = ["cora", "citeseer", "pubmed"]   # skip citeseer to save time by default
OUT_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
SHIFT_LEVELS = [0.0, 0.1]
SEEDS = [42, 100, 200, 300, 400]
N_SHAPIRO_DIMS = 40   # cap Shapiro passes for speed; subsample dims uniformly
HZ_SUBSAMPLE = 2000   # HZ is O(n^2); cap non-anom nodes per condition


def normal_stats(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.detach().cpu().numpy()
    return h_np, h_np[y == 0], y


def mardia(hN):
    """Mardia multivariate skewness b1 and kurtosis b2 (non-anom set)."""
    n, p = hN.shape
    X = hN - hN.mean(0)
    S = (X @ X.T) / n
    b1 = (np.trace(S @ S @ S)) * n / 6.0  # skewness stat (chi-square-like under H0)
    # kurtosis
    b2 = np.trace(S @ S)
    return float(b1), float(b2)


def henze_zirkler(hN):
    """Henze-Zirkler test statistic for multivariate normality. Larger => more non-Gaussian."""
    n, p = hN.shape
    X = hN - hN.mean(0)
    # bandwidth
    Cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(p)
    try:
        invCov = np.linalg.inv(Cov)
    except np.linalg.LinAlgError:
        return float("nan")
    beta = 1.0 / (np.sqrt(2.0)) * (2 * p + 1.0) / 4.0 ** (1.0 / p) * np.sqrt(n) ** (1.0 / p)
    # pairwise Mahalanobis
    D2 = np.empty((n, n))
    chunk = 512
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        diff = X[a:b, None, :] - X[None, :, :]
        D2[a:b] = np.einsum("ijk,kl,ijl->ij", diff, invCov, diff)
    HZ = (1.0 / n) * np.sum(np.exp(-beta ** 2 / 2 * D2)) \
        + (1 + 2 * (1 + beta ** 2) ** (-p / 2.0) - 2 * (1 + beta ** 2 / 2.0) ** (-p / 2.0))
    return float(HZ)


def per_dim_normality(hN):
    """Shapiro rejection rate and mean excess kurtosis across dims (subsampled)."""
    n, p = hN.shape
    dims = np.linspace(0, p - 1, min(N_SHAPIRO_DIMS, p)).astype(int)
    rejections = 0
    pvals = []
    excess_kurt = []
    for k in dims:
        col = hN[:, k]
        excess_kurt.append(ss.kurtosis(col))  # Fisher => excess kurtosis (0 for Gaussian)
        try:
            st, p = ss.shapiro(col)
            pvals.append(p)
            if p < 0.05:
                rejections += 1
        except Exception:
            pass
    return rejections / len(dims), float(np.mean(pvals)) if pvals else float("nan"), \
        float(np.mean(excess_kurt)) if excess_kurt else float("nan")


def diag_stats(h, labels, mu_s, var_s):
    """compute delta_AUC + D_Sigma + M2_A on adapted embeddings (paper's rule)."""
    h_np, hN, y = normal_stats(h, labels)
    mu_t = hN.mean(0); var_t = hN.var(0) + 1e-6
    ratio = var_t / var_s
    D_Sigma = max(0.5 * (ratio - 1.0 - np.log(ratio)).sum(), 0.0)
    hA = h_np[y == 1]
    M2_A = ((hA - mu_t) ** 2 / var_t).sum(1).mean()
    s = np.sqrt(((h_np - mu_t) ** 2 / var_t).sum(1))
    auc = roc_auc_score(y, s)
    return D_Sigma, M2_A, auc


def run_dataset(name):
    print(f"\n{'=' * 60}\nDataset: {name}\n{'=' * 60}")
    x_raw, ei, labels = load_dataset(name)
    x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
    x_cpu = x_raw.cpu()

    torch.manual_seed(42)
    enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h = enc(x_raw, ei); cn = h[labels == 0].mean(0); d = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN - d[labels == 1]) ** 2).mean() + (d[labels == 0] ** 2).mean()
        loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        h_src = enc(x_raw, ei)
    _, hN_src, _ = normal_stats(h_src, labels)
    mu_s = hN_src.mean(0); var_s = hN_src.var(0) + 1e-6

    x_pilot = make_target(x_cpu, 0.0, 999)
    lr_tent = select_lr(tta_tent, enc, x_pilot, ei, labels, LR_CANDIDATES)
    lr_shot = select_lr(tta_shot, enc, x_pilot, ei, labels, LR_CANDIDATES)
    print(f"  LR — TENT {lr_tent:.0e}  SHOT {lr_shot:.0e}")

    records = []
    for shift in SHIFT_LEVELS:
        for seed in SEEDS:
            x_tgt = make_target(x_cpu, shift, seed)
            with torch.no_grad():
                h0 = enc(x_tgt, ei)
            torch.manual_seed(seed)
            h_tent, _ = tta_tent(enc, x_tgt, ei, labels, lr_tent)
            torch.manual_seed(seed)
            h_shot, _ = tta_shot(enc, x_tgt, ei, labels, lr_shot)

            for method, h in [("No-TTA", h0), ("TENT", h_tent), ("SHOT", h_shot)]:
                h_np, hN, _ = normal_stats(h, labels)
                if hN.shape[0] > HZ_SUBSAMPLE:
                    rng = np.random.default_rng(seed * 100003 + hash(method) % 100003)
                    hN_hz = hN[rng.choice(hN.shape[0], HZ_SUBSAMPLE, replace=False)]
                else:
                    hN_hz = hN
                mardia_b1, mardia_b2 = mardia(hN_hz)
                hz = henze_zirkler(hN_hz)
                shp_rej, shp_pmean, excess_kurt = per_dim_normality(hN)
                D_S, M2_A, auc = diag_stats(h, labels, mu_s, var_s)
                records.append({
                    "dataset": name, "shift": shift, "seed": seed, "method": method,
                    "mardia_skew": mardia_b1, "mardia_kurt": mardia_b2,
                    "henze_zirkler": hz, "shapiro_rej_rate": shp_rej,
                    "shapiro_pmean": shp_pmean, "mean_excess_kurtosis": excess_kurt,
                    "D_Sigma": D_S, "M2_A": M2_A, "AUC_maha": auc,
                })
        print(f"  [shift={shift}] done")
    return pd.DataFrame(records)


all_dfs = []
for name in DATASETS:
    try:
        all_dfs.append(run_dataset(name))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  SKIP {name}: {e}")

df = pd.concat(all_dfs, ignore_index=True)
df.to_csv(results_csv(f"embedding_normality{OUT_SUFFIX}.csv"), index=False)
print(f"\nSaved embedding_normality.csv ({len(df)} rows)")

# ── Correlation: does non-Gaussianity track degradation? ──────────────────────
summary = ["=" * 64, "EMBEDDING NORMALITY vs DEGRADATION", "=" * 64]
adapted = df[df["method"].isin(["TENT", "SHOT"])].copy()

# compute delta_AUC per adapted run relative to its No-TTA match
base = df[df["method"] == "No-TTA"].set_index(["dataset", "shift", "seed"])["AUC_maha"]
adapted["delta_AUC_maha"] = adapted.apply(
    lambda r: r["AUC_maha"] - base.loc[(r["dataset"], r["shift"], r["seed"])], axis=1)

summary.append("\nMean normality stats by method (across all datasets/seeds):")
agg = df.groupby("method")[["mardia_skew", "mardia_kurt", "henze_zirkler",
                            "shapiro_rej_rate", "mean_excess_kurtosis"]].mean()
summary.append(agg.round(3).to_string())

summary.append("\nSpearman(non-Gaussianity stat, delta_AUC_maha) on adapted runs:")
for stat in ["mardia_skew", "henze_zirkler", "shapiro_rej_rate", "mean_excess_kurtosis"]:
    r, p = ss.spearmanr(adapted[stat], adapted["delta_AUC_maha"])
    summary.append(f"  {stat:<22} r={r:+.3f}  p={p:.4f}")

summary.append("\nIf |r| is small / p large => non-Gaussianity does NOT drive degradation.")

with open(results_txt(f"embedding_normality_summary{OUT_SUFFIX}.txt"), "w") as f:
    f.write("\n".join(summary))
print("Saved embedding_normality_summary.txt")
