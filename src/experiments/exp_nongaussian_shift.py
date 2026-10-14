"""
exp_nongaussian_shift.py — P2: non-Gaussian synthetic Normality Shift

Reviewer #1 (R3): the existing Normality-Shift simulation adds isotropic
Gaussian noise to node features. The reviewer asks whether the conclusions
survive non-Gaussian shifts. This script adds three non-Gaussian shift
families alongside the existing Gaussian one and re-runs TENT/SHOT under
the same protocol, scored with maha_diag AND the distribution-free kNN rule.

Shift families (applied to ALL node features, independently of anomaly
injection — pure Normality Shift P_s^N -> P_t^N):
  - gaussian : additive N(0, shift^2)            (the paper's existing shift)
  - student_t: additive standardized Student-t(df=4) * shift   (heavy tails)
  - mixture  : additive bimodal: +shift or -shift per-feature (multimodal)
  - sparse   : zero out a random fraction `shift` of feature entries
               (sparse feature corruption — closest to real drift)

If degradation persists under the non-Gaussian shifts (especially under the
distribution-free kNN scorer), the conclusion is not an artifact of a
Gaussian-only shift design.

Outputs: nongauss_shift_results.csv, nongauss_shift_summary.txt
"""

import os, sys, copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from torch_geometric.datasets import Planetoid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config import results_csv, results_txt
from exp_nongaussian_scoring import (
    load_dataset, GCNEncoder, Scorer, tta_tent, tta_shot, select_lr,
    score_maha_diag, score_knn, score_kde, score_all,
    PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN, TTA_STEPS, LR_CANDIDATES,
    HIDDEN_DIM, NUM_LAYERS, DEVICE,
)

# ── Config ───────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    DATASETS = sys.argv[1].split(",")
else:
    DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
OUT_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""

SHIFT_KINDS = ["gaussian", "student_t", "mixture", "sparse"]
SHIFT_STRENGTHS = [0.0, 0.1]
SEEDS = [42, 100, 200, 300, 400]
PILOT_SEED = 999

torch.manual_seed(42); np.random.seed(42)


# ── Shift families ───────────────────────────────────────────────────────────
def make_target(x_cpu, shift, kind, seed):
    """Apply Normality Shift of family `kind` with magnitude `shift`.

    shift semantics differ by family to stay interpretable:
      gaussian / student_t / mixture : standard-deviation scale of additive noise
      sparse                         : fraction of feature entries zeroed
    """
    torch.manual_seed(seed)
    x = x_cpu.clone()
    if shift <= 0:
        return x.to(DEVICE)
    n, fdim = x.shape
    if kind == "gaussian":
        x = x + torch.randn_like(x) * shift
    elif kind == "student_t":
        # heavy-tailed additive noise: Student-t(df) standardized to unit variance,
        # then scaled by shift. Built from independent normal and chi-square draws,
        # with a floor on the denominator to guarantee finiteness.
        df = 4.0
        z = torch.randn(n, fdim)
        chi2 = (torch.randn(n, fdim) ** 2).sum(0, keepdim=False)  # chi2(1) per node-row? no -> per entry
        chi2 = torch.randn(n, fdim) ** 2  # chi2(1) per entry
        denom = torch.sqrt(chi2 / df).clamp(min=1e-3)
        t = z / denom  # Student-t(df) sample
        # standardize to unit variance (Var[Student-t df=4] = df/(df-2) = 2)
        t = t / np.sqrt(df / (df - 2.0))
        x = x + t * shift
    elif kind == "mixture":
        # bimodal additive: each feature entry gets +shift or -shift (per-entry sign)
        signs = torch.bernoulli(torch.full((n, fdim), 0.5)) * 2 - 1
        x = x + signs * shift
    elif kind == "sparse":
        # sparse feature corruption: zero out a random `shift` fraction of entries
        mask = torch.rand(n, fdim) > shift
        x = x * mask
    else:
        raise ValueError(kind)
    return x.to(DEVICE)


# ── Driver ───────────────────────────────────────────────────────────────────
def run_dataset(name):
    print(f"\n{'=' * 60}\nDataset: {name}\n{'=' * 60}")
    x_raw, ei, labels = load_dataset(name)
    x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
    x_cpu = x_raw.cpu()
    print(f"  N={x_raw.shape[0]:,} F={x_raw.shape[1]} anomaly%={100*labels.sum().item()/x_raw.shape[0]:.2f}")

    # Pretrain source encoder (margin loss, identical to main paper)
    torch.manual_seed(42)
    enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h = enc(x_raw, ei); cn = h[labels == 0].mean(0); d = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN - d[labels == 1]) ** 2).mean() + (d[labels == 0] ** 2).mean()
        loss.backward(); opt.step()
    enc.eval()

    x_pilot = make_target(x_cpu, 0.0, "gaussian", PILOT_SEED)
    lr_tent = select_lr(tta_tent, enc, x_pilot, ei, labels, LR_CANDIDATES)
    lr_shot = select_lr(tta_shot, enc, x_pilot, ei, labels, LR_CANDIDATES)
    print(f"  LR — TENT {lr_tent:.0e}  SHOT {lr_shot:.0e}")

    records = []
    def score_p2(h, labels):
        # P2 uses maha_diag/maha_full + kNN only. KDE is O(n^2) and bottlenecks
        # large datasets (Reddit 11k, Pubmed 19k); kNN is the headline
        # distribution-free scorer and suffices to answer R3.
        out = {}
        out["maha_diag"], _ = score_maha_diag(h, labels)
        out["knn_k5"], _ = score_knn(h, labels, 5)
        out["knn_k10"], _ = score_knn(h, labels, 10)
        return out
    P2_SCORERS = ["maha_diag", "knn_k5", "knn_k10"]

    for kind in SHIFT_KINDS:
        for shift in SHIFT_STRENGTHS:
            for seed in SEEDS:
                x_tgt = make_target(x_cpu, shift, kind, seed)
                with torch.no_grad():
                    h0 = enc(x_tgt, ei)
                aucs0 = score_p2(h0, labels)
                torch.manual_seed(seed)
                h_tent, _ = tta_tent(enc, x_tgt, ei, labels, lr_tent)
                aucs_tent = score_p2(h_tent, labels)
                torch.manual_seed(seed)
                h_shot, _ = tta_shot(enc, x_tgt, ei, labels, lr_shot)
                aucs_shot = score_p2(h_shot, labels)
                for method, aucs_t in [("TENT", aucs_tent), ("SHOT", aucs_shot)]:
                    for scorer in P2_SCORERS:
                        records.append({
                            "dataset": name, "shift_kind": kind, "shift": shift,
                            "seed": seed, "method": method, "scorer": scorer,
                            "AUC_0": aucs0[scorer], "AUC_T": aucs_t[scorer],
                            "delta_AUC": aucs_t[scorer] - aucs0[scorer],
                        })
        # progress per kind (knn_k10 headline)
        sub = pd.DataFrame(records).query("shift_kind == @kind and shift==0.1")
        for meth in ["TENT", "SHOT"]:
            mm = sub.query("method == @meth and scorer == 'knn_k10'")["delta_AUC"].mean()
            print(f"  [{kind:>9}] shift=0.1 mean ΔAUC(knn_k10) {meth}={mm:+.4f}")

    return pd.DataFrame(records)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_dfs = []
    for name in DATASETS:
        try:
            all_dfs.append(run_dataset(name))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  SKIP {name}: {e}")

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(results_csv(f"nongauss_shift_results{OUT_SUFFIX}.csv"), index=False)
    print(f"\nSaved nongauss_shift_results.csv ({len(df)} rows)")

    # Summary: ΔAUC by shift_kind × method × scorer (shift=0.1)
    summary = ["=" * 78,
               "P2 NON-GAUSSIAN SHIFT: mean ΔAUC at shift=0.1 by shift_kind × method",
               "=" * 78]
    s1 = df[df["shift"] == 0.1]
    for sc in ["maha_diag", "knn_k10"]:
        summary.append(f"\n[scorer = {sc}]")
        piv = s1[s1["scorer"] == sc].groupby(["dataset", "shift_kind", "method"])["delta_AUC"].mean().unstack()
        summary.append(piv.round(4).to_string())

    summary.append("\n" + "-" * 78)
    summary.append("PERSISTENCE under each shift family (knn_k10, frac of seeds with ΔAUC<0)")
    summary.append("-" * 78)
    s1_adapted = s1[s1["method"].isin(["TENT", "SHOT"])]
    pers = (s1_adapted.groupby(["dataset", "shift_kind", "method", "seed"])["delta_AUC"]
            .first().reset_index())  # placeholder
    per = s1_adapted[s1_adapted["scorer"] == "knn_k10"]
    g = per.groupby(["dataset", "shift_kind", "method"])["delta_AUC"]
    frac = (g.apply(lambda s: (s < 0).mean())).reset_index().rename(columns={"delta_AUC": "frac_neg"})
    summary.append(frac.pivot(index=["dataset", "method"], columns="shift_kind", values="frac_neg").round(2).to_string())

    summary.append("\nVERDICT: if frac_neg is high across ALL shift families (incl. student_t/mixture/sparse),")
    summary.append("degradation is NOT an artifact of Gaussian-only shift design (R3 refuted).")

    with open(results_txt(f"nongauss_shift_summary{OUT_SUFFIX}.txt"), "w") as f:
        f.write("\n".join(summary))
    print("Saved nongauss_shift_summary.txt")
