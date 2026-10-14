"""
exp_r9_bn_arch.py — E9 (R2 W-A): architecture-matched gradient control

R2: BN-adapt runs on a BN-augmented architecture, so the gradient-vs-
statistics contrast is confounded by architecture. Control: run TENT/SHOT
on the SAME BN-augmented encoder. If gradient adaptation still degrades
on the BN architecture (and BN-adapt stays near-neutral on it), the
family-level conclusion is architecture-robust within this pair.

Protocol: GCNEncoderBN (BN after each conv), margin pretraining in train()
mode (BN batch stats), evaluation in eval() mode (running stats), TTA in
train() mode. BN-adapt replaces running stats with target batch stats as
in the paper. Datasets: cora, citeseer, pubmed, reddit. Shift: 0 only
(the confound is about zero-shift family contrast).

Outputs: results/csv/r9_bn_arch.csv, results/txt/r9_bn_arch_summary.txt
"""

import os, sys, copy, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exp_nongaussian_scoring import (
    load_dataset, GCNEncoderBN, Scorer, DEVICE,
    PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN, HIDDEN_DIM, NUM_LAYERS,
    TTA_STEPS, LR_CANDIDATES, PILOT_SEED, SHOT_LAMBDA_DIV,
)
from exp_r2_checks import pretrain

DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
SEEDS = [42, 100, 200, 300, 400]


def pretrain_bn(enc, x_raw, ei, labels):
    """Margin pretraining with BN in train mode (batch statistics)."""
    opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h = enc(x_raw, ei)
        cn = h[labels == 0].mean(0)
        d = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN - d[labels == 1]) ** 2).mean() + (d[labels == 0] ** 2).mean()
        loss.backward(); opt.step()
    enc.eval()
    # align running stats with the data they were trained on
    with torch.no_grad():
        enc.train()
        _ = enc(x_raw, ei)
        enc.eval()


def maha_auc(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.cpu().numpy()
    hN = h_np[y == 0]
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    return roc_auc_score(y, s)


def tta_bn_run(enc_init, x_tgt, ei, labels, method, lr, seed):
    """TENT/SHOT on the BN encoder: adaptation runs in train() mode
    (batch statistics active, matching how BN-TTA updates BN layers)."""
    enc = copy.deepcopy(enc_init)
    scorer = Scorer(HIDDEN_DIM).to(DEVICE)
    torch.manual_seed(seed + 7)
    torch.nn.init.normal_(scorer.fc.weight, std=0.01)
    torch.nn.init.zeros_(scorer.fc.bias)
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    for _ in range(TTA_STEPS):
        enc.train(); scorer.train(); opt.zero_grad()
        h = enc(x_tgt, ei)
        prob = torch.sigmoid(scorer(h))
        eps = 1e-7
        L_ent = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean()
        if method == "SHOT":
            mean_p = prob.mean().clamp(eps, 1 - eps)
            H_mean = -(mean_p * mean_p.log() + (1 - mean_p) * (1 - mean_p).log())
            loss = L_ent - SHOT_LAMBDA_DIV * H_mean
        else:
            loss = L_ent
        loss.backward(); opt.step()
    # evaluation uses running statistics (eval mode) — the deployed state
    enc.eval()
    with torch.no_grad():
        return enc(x_tgt, ei)


def bn_adapt(enc_init, x_tgt, ei):
    """Zero-gradient control: replace BN running stats with target stats."""
    enc = copy.deepcopy(enc_init)
    enc.eval()
    with torch.no_grad():
        h = x_tgt
        for conv, bn in zip(enc.convs, enc.bns):
            h_pre = conv(h, ei)
            bn.running_mean.data = h_pre.mean(0)
            bn.running_var.data = h_pre.var(0, unbiased=False).clamp(min=1e-6)
            h = F.relu(bn(h_pre))
    return h


def main():
    rows = []
    for ds in DATASETS:
        x_raw, ei, labels = load_dataset(ds)
        x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
        for seed in SEEDS:
            torch.manual_seed(seed)
            enc = GCNEncoderBN(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
            pretrain_bn(enc, x_raw, ei, labels)
            with torch.no_grad():
                enc.eval()
                auc0 = maha_auc(enc(x_raw, ei), labels)

            for method in ["TENT", "SHOT"]:
                # lr selection on pilot seed (entropy criterion, paper protocol)
                best_lr, best_loss = LR_CANDIDATES[0], float("inf")
                for lr in LR_CANDIDATES:
                    torch.manual_seed(PILOT_SEED)
                    enc_p = GCNEncoderBN(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
                    pretrain_bn(enc_p, x_raw, ei, labels)
                    h = tta_bn_run(enc_p, x_raw, ei, labels, method, lr, PILOT_SEED)
                    scorer = Scorer(HIDDEN_DIM).to(DEVICE)
                    with torch.no_grad():
                        prob = torch.sigmoid(scorer(h))
                        eps = 1e-7
                        loss = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean().item()
                    if loss < best_loss:
                        best_lr, best_loss = lr, loss
                hT = tta_bn_run(enc, x_raw, ei, labels, method, best_lr, seed)
                aucT = maha_auc(hT, labels)
                rows.append(dict(dataset=ds, seed=seed, method=method, lr=best_lr,
                                 AUC_0=auc0, AUC_T=aucT, delta_AUC=aucT - auc0))

            hB = bn_adapt(enc, x_raw, ei)
            aucB = maha_auc(hB, labels)
            rows.append(dict(dataset=ds, seed=seed, method="BN-adapt", lr=0.0,
                             AUC_0=auc0, AUC_T=aucB, delta_AUC=aucB - auc0))
            print(rows[-1])
        pd.DataFrame(rows).to_csv(results_csv("r9_bn_arch.csv"), index=False)

    df = pd.DataFrame(rows)
    with open(results_txt("r9_bn_arch_summary.txt"), "w") as f:
        f.write("=" * 70 + "\nE9 BN-ARCH CONTROL: mean ΔAUC on BN-augmented GCN (shift=0)\n")
        f.write("=" * 70 + "\n\n")
        f.write(df.groupby(["dataset", "method"])["delta_AUC"].mean().round(3).to_string() + "\n\n")
        f.write(df.groupby("method")["delta_AUC"].agg(["mean", "std", lambda s: (s < 0).mean()])
                 .rename(columns={"<lambda>": "frac_neg"}).round(3).to_string() + "\n")


if __name__ == "__main__":
    main()
