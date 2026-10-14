"""
exp_r2_ratio_sweep.py — E2 (R2-RC3): anomaly-ratio sweep

R2: the central class-imbalance mechanism (normal nodes dominate the entropy
gradient) is never tested by varying the anomaly ratio with other factors
fixed. Mechanism prediction: LOWER ratio (more imbalance) => deeper TTA
degradation; HIGHER ratio => harm shrinks and may reverse (anomalies regain
gradient share).

We inject anomalies at ratios {0.01, 0.02, 0.05, 0.10, 0.20} with strength
0.5 (paper default), pretrain the margin encoder per (dataset, seed), then
run TENT/SHOT (80 steps, entropy-selected lr per protocol) and record ΔAUC.
Additionally log the realized gradient-share diagnostic: per-step fraction of
the entropy loss mass contributed by normal nodes (sanity: ≈ 1−ε).

Datasets: cora, citeseer, pubmed, reddit (injection-capable). Questions/Weibo
have natural labels and cannot be swept — noted in the paper.

Outputs:
  results/csv/r2_e2_ratio_sweep.csv    per (dataset, ratio, seed, method)
  results/txt/r2_e2_summary.txt        mean ΔAUC table + gradient share + trend test
"""

import os, sys, copy, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, pandas as pd
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exp_nongaussian_scoring import (
    load_dataset, GCNEncoder, Scorer, DEVICE, PRETRAIN_EPOCHS, PRETRAIN_LR,
    MARGIN, HIDDEN_DIM, NUM_LAYERS, TTA_STEPS, LR_CANDIDATES, SHOT_LAMBDA_DIV,
    PILOT_SEED, make_target,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20]
SEEDS = [42, 100, 200, 300, 400]
ANOM_STRENGTH = 0.5


def inject_anomaly(x, ratio, strength, seed):
    """Fresh anomaly injection at arbitrary ratio (protocol of exp_multi_shift)."""
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed + 1000)
    idx = torch.randperm(n, generator=g)[: int(n * ratio)]
    x2 = x.clone()
    x2[idx] += strength * torch.randn(len(idx), x.shape[1], generator=g)
    return x2, idx


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


def maha_auc(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.detach().cpu().numpy()
    hN = h_np[y == 0]
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    return roc_auc_score(y, s)


def entropy_grad_share(h, labels, scorer):
    """Fraction of entropy-loss mass from normal nodes (diagnostic for mechanism)."""
    prob = torch.sigmoid(scorer(h))
    eps = 1e-7
    ent = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log())
    mass = ent.detach()
    return mass[labels == 0].sum().item() / mass.sum().item()


def tta_tent_logged(enc_init, x, ei, labels, lr, steps=TTA_STEPS):
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    shares = []
    for _ in range(steps):
        enc.train(); scorer.train(); opt.zero_grad()
        h = enc(x, ei)
        prob = torch.sigmoid(scorer(h))
        eps = 1e-7
        loss = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean()
        loss.backward(); opt.step()
        if _ % 10 == 0:
            with torch.no_grad():
                shares.append(entropy_grad_share(enc(x, ei), labels, scorer))
    enc.eval()
    with torch.no_grad():
        h_final = enc(x, ei)
    return h_final, float(np.mean(shares))


def tta_shot_logged(enc_init, x, ei, labels, lr, steps=TTA_STEPS, lambda_div=SHOT_LAMBDA_DIV):
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    shares = []
    for _ in range(steps):
        enc.train(); scorer.train(); opt.zero_grad()
        h = enc(x, ei)
        prob = torch.sigmoid(scorer(h))
        eps = 1e-7
        L_ent = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean()
        mean_p = prob.mean().clamp(eps, 1 - eps)
        H_mean = -(mean_p * mean_p.log() + (1 - mean_p) * (1 - mean_p).log())
        (L_ent - lambda_div * H_mean).backward(); opt.step()
        if _ % 10 == 0:
            with torch.no_grad():
                shares.append(entropy_grad_share(enc(x, ei), labels, scorer))
    enc.eval()
    with torch.no_grad():
        h_final = enc(x, ei)
    return h_final, float(np.mean(shares))


def select_lr(method_fn, enc_init, x_pilot, ei, labels, candidates):
    best_lr, best_loss = candidates[0], float("inf")
    for lr in candidates:
        torch.manual_seed(PILOT_SEED)
        _, loss = method_fn(enc_init, x_pilot, ei, labels, lr, steps=20)
        if loss is not None and loss < best_loss:
            best_lr, best_loss = lr, loss
    return best_lr


def run():
    rows = []
    for ds in DATASETS:
        x_raw, ei, labels = load_dataset(ds)
        x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
        x_cpu = x_raw.cpu()
        for ratio in RATIOS:
            for seed in SEEDS:
                x2, anom_idx = inject_anomaly(x_cpu, ratio, ANOM_STRENGTH, seed)
                lab2 = torch.zeros_like(labels)
                lab2[anom_idx] = 1
                x2, lab2 = x2.to(DEVICE), lab2.to(DEVICE)
                torch.manual_seed(seed)
                enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
                pretrain(enc, x2, ei, lab2)
                with torch.no_grad():
                    h0 = enc(x2, ei)
                auc0 = maha_auc(h0, lab2)

                x_pilot = make_target(x_cpu, 0.0, PILOT_SEED).to(DEVICE)
                for name, fn in [("TENT", tta_tent_logged), ("SHOT", tta_shot_logged)]:
                    lr = select_lr(fn, enc, x_pilot, ei, lab2, LR_CANDIDATES)
                    torch.manual_seed(seed)
                    hT, gshare = fn(enc, x2, ei, lab2, lr)
                    aucT = maha_auc(hT, lab2)
                    rows.append(dict(dataset=ds, ratio=ratio, seed=seed, method=name,
                                     lr=lr, AUC_0=auc0, AUC_T=aucT,
                                     delta_AUC=aucT - auc0, normal_grad_share=gshare))
                print(rows[-1]); print(rows[-2])
        pd.DataFrame(rows).to_csv(
            results_csv("r2_e2_ratio_sweep.csv"), index=False)

    df = pd.DataFrame(rows)
    with open(results_txt("r2_e2_summary.txt"), "w") as f:
        f.write("=" * 70 + "\nE2 ANOMALY-RATIO SWEEP: mean ΔAUC (maha_diag)\n" + "=" * 70 + "\n\n")
        tab = df.groupby(["dataset", "ratio", "method"])["delta_AUC"].mean().unstack(0)
        f.write(tab.round(3).to_string() + "\n\n")
        f.write("Normal-node gradient share (entropy mass), by ratio:\n")
        gs = df.groupby(["ratio"])["normal_grad_share"].mean()
        f.write(gs.round(3).to_string() + "\n\n")
        f.write("Mechanism check: degradation should DEEPEN as ratio ↓\n")
        f.write("(larger share of gradient from normal nodes).\n")
        # simple monotonic trend test per dataset: Spearman(ratio, deltaAUC)
        from scipy.stats import spearmanr
        f.write("\nSpearman(ratio, ΔAUC) per dataset×method (positive ⇒ lower ratio worse):\n")
        for ds in DATASETS:
            for m in ["TENT", "SHOT"]:
                d = df[(df.dataset == ds) & (df.method == m)]
                r, p = spearmanr(d.ratio, d.delta_AUC)
                f.write(f"  {ds:9s} {m}: rho={r:+.3f} p={p:.4f}\n")


if __name__ == "__main__":
    run()
