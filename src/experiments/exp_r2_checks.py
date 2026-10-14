"""
exp_r2_checks.py — R2 quick checks: E6 (M_N^2 explanation) + E7 (circularity)

E6: The paper claims M_N^2 ∈ [46,64] (vs d=64) is explained by negative bias of
sample variance. R2 suspects variance clipping or held-out estimation instead.
We decompose empirically on source + adapted embeddings:
  (a) in-sample, MLE variance (n)        — current practice
  (b) in-sample, unbiased variance (n-1)
  (c) held-out: split normals in half; estimate mu/var on one half, score other
  (d) eps floor: count dims where var < eps margin matters (eps=1e-6)
Theory: in-sample MLE gives E[M_N^2] ≈ d*(n-1)/n (slightly < d). Values as low
as 46/64 = 0.72*d cannot come from (n-1)/n at n~10^4 — so if (a) reproduces 46,
the driver is something else (e.g. eps floor inflating denominators → scores
depressed; or heavy-tailed dims). This script adjudicates.

E7: Is the ρ(M_A^2, AUC) ≈ 0.90–0.97 correlation circular (both built from the
same Mahalanobis scores)? We join P0 CSVs: same run (dataset, shift, seed,
method), AUC under 5 scorers vs M_A^2 computed once from Maha components.
If ρ(M_A^2, AUC_knn) stays high, M_A^2 captures geometry, not scorer identity.

Outputs: results/csv/r2_e6_mn2.csv, results/txt/r2_e6_summary.txt
         results/csv/r2_e7_circularity.csv, results/txt/r2_e7_summary.txt
"""

import os, sys, copy, torch
import torch.nn.functional as F
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt  # functions: results_csv(name)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exp_nongaussian_scoring import (
    load_dataset, GCNEncoder, DEVICE,
    PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN, HIDDEN_DIM, NUM_LAYERS,
)


def pretrain(enc, x_raw, ei, labels):
    opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h = enc(x_raw, ei)
        cn = h[labels == 0].mean(0)
        d = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN - d[labels == 1]) ** 2).mean() + (d[labels == 0] ** 2).mean()
        loss.backward(); opt.step()
    enc.eval()

OUT_DIR_CSV, OUT_DIR_TXT = results_csv, results_txt

DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
EPS = 1e-6


def m_n2_variants(h, labels, seed=0):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.detach().cpu().numpy()
    hN = h_np[y == 0]
    n = hN.shape[0]
    out = {"n": n, "d": hN.shape[1]}

    # (a) in-sample MLE
    mu = hN.mean(0)
    var_mle = hN.var(0) + EPS
    out["mn2_insample_mle"] = (((hN - mu) ** 2 / var_mle).sum(1)).mean()

    # (b) in-sample unbiased
    var_unb = hN.var(0, ddof=1) + EPS
    out["mn2_insample_unbiased"] = (((hN - mu) ** 2 / var_unb).sum(1)).mean()

    # (c) held-out split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    half = n // 2
    A, B = hN[perm[:half]], hN[perm[half:]]
    muA, varA = A.mean(0), A.var(0) + EPS
    out["mn2_heldout"] = (((B - muA) ** 2 / varA).sum(1)).mean()

    # (d) eps floor diagnostics: how many dims have var <= 10*eps (clipped)
    raw_var = hN.var(0)
    out["frac_dims_eps_floor"] = float((raw_var < 10 * EPS).mean())
    # contribution of eps to denominator: sum var/(var+eps) ~ 1 means eps irrelevant
    out["eps_inflation_mean"] = float((raw_var / (raw_var + EPS)).mean())

    # theoretical in-sample MLE expectation
    out["mn2_theory_mle"] = out["d"] * (n - 1) / n
    return out


def main_e6():
    rows = []
    for ds in DATASETS:
        x_raw, ei, labels = load_dataset(ds)
        x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
        for seed in [42, 100, 200]:
            torch.manual_seed(seed)
            enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
            pretrain(enc, x_raw, ei, labels)
            with torch.no_grad():
                h = enc(x_raw, ei)
            r = {"dataset": ds, "seed": seed, "stage": "source"}
            r.update(m_n2_variants(h.cpu(), labels.cpu(), seed))
            rows.append(r)
            print(r)
    df = pd.DataFrame(rows)
    df.to_csv(results_csv("r2_e6_mn2.csv"), index=False)
    with open(results_txt("r2_e6_summary.txt"), "w") as f:
        f.write("=" * 70 + "\n")
        f.write("E6: M_N^2 DECOMPOSITION — in-sample MLE vs unbiased vs held-out\n")
        f.write("=" * 70 + "\n\n")
        cols = ["mn2_insample_mle", "mn2_insample_unbiased", "mn2_heldout",
                "mn2_theory_mle", "frac_dims_eps_floor", "eps_inflation_mean"]
        f.write(df.groupby("dataset")[cols].mean().round(3).to_string() + "\n\n")
        f.write("Reading: if mn2_insample_mle ≈ mn2_theory_mle ≈ d, the paper's\n")
        f.write("negative-bias explanation is CORRECT and 46 was an artifact of\n")
        f.write("specific adapted states, not source embeddings. If held-out ≈ d\n")
        f.write("but in-sample ≪ d, estimation bias is the driver (R2 right).\n")
        f.write("If frac_dims_eps_floor > 0 substantially, variance clipping matters.\n")


def main_e7():
    """Join P0 scoring CSVs with M_A^2 (from cross-method CSVs where available)."""
    p0 = pd.read_csv(results_csv("nongauss_scoring_results.csv"))
    cm = pd.read_csv(results_csv("cross_method_results.csv"))
    cm_key = cm.rename(columns={"shifts": "shift"})
    m = p0.merge(cm_key[["dataset", "shift", "seed", "method", "M2_A", "D_Sigma"]],
                 on=["dataset", "shift", "seed", "method"], how="inner")
    m.to_csv(results_csv("r2_e7_circularity.csv"), index=False)

    from scipy.stats import spearmanr
    lines = ["=" * 70,
             "E7: CIRCULARITY CHECK — ρ(M_A^2 [Maha-based], AUC [per scorer])",
             "=" * 70, ""]
    for scorer in sorted(m.scorer.unique()):
        sub = m[m.scorer == scorer]
        r_all, p_all = spearmanr(sub["M2_A"], sub["delta_AUC"])
        lines.append(f"[{scorer}] pooled: rho(M2_A, delta_AUC) = {r_all:+.3f}  p={p_all:.2e}  N={len(sub)}")
        for ds in sorted(sub.dataset.unique()):
            d = sub[sub.dataset == ds]
            if len(d) >= 5:
                r, p = spearmanr(d["M2_A"], d["delta_AUC"])
                lines.append(f"    {ds:10s}: rho={r:+.3f}  p={p:.3f}  N={len(d)}")
        lines.append("")
    lines.append("Reading: if rho stays high for knn/kde scorers, M_A^2 tracks")
    lines.append("GEOMETRY (scorer-independent); if it collapses, the paper's")
    lines.append("0.90-0.97 correlation is partly circular. Report both honestly.")
    with open(results_txt("r2_e7_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("e6", "all"):
        main_e6()
    if which in ("e7", "all"):
        main_e7()
