"""
exp_r3_head_init.py — R3 round-2 Concern 1: source-trained head control

R3: the TTA head is randomly initialized (AUC ~0.43), so the observed
degradation may come from adapting toward an arbitrary random decision
boundary rather than from a failure of meaningful entropy-based TTA.
Control: train the head on the source anomaly-detection task (distill the
oracle Mahalanobis score into the linear head via a sigmoid-scaled BCE
target), then run entropy adaptation starting from THAT head. If degradation
persists, the failure is not an artifact of the random auxiliary objective.

Also logs the per-step normal-vs-anomaly GRADIENT NORM split (R3 round-2
Concern 2: loss-mass share is not gradient share; the entropy gradient per
node scales with p(1-p)|log((1-p)/p)| and the parameter Jacobian, so we
measure ||g|| directly on the two groups).

Arms:
  A. random-init head      (paper protocol, reference)
  B. source-trained head   (distilled from oracle Mahalanobis score)

Datasets: cora, citeseer, pubmed, reddit; shifts {0, 0.1}; 5 seeds.
Outputs: results/csv/r3_head_init.csv, results/txt/r3_head_init_summary.txt
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
    load_dataset, GCNEncoder, Scorer, DEVICE, make_target,
    HIDDEN_DIM, NUM_LAYERS, TTA_STEPS, LR_CANDIDATES, PILOT_SEED, SHOT_LAMBDA_DIV,
    select_lr, tta_tent, tta_shot,
)
from exp_r2_checks import pretrain

DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
SEEDS = [42, 100, 200, 300, 400]


def paper_protocol_run(enc, x_tgt, ei, labels, method, seed):
    """Exact paper protocol (E4/RNG order): select_lr on pilot, then
    torch.manual_seed(seed) immediately before fn() so the Scorer's default
    init lands in the same RNG state as the paper's runs."""
    fn = tta_tent if method == "TENT" else tta_shot
    lr = select_lr(fn, enc, x_tgt, ei, labels, LR_CANDIDATES)
    torch.manual_seed(seed)
    hT, _ = fn(enc, x_tgt, ei, labels, lr)
    return maha_auc(hT, labels), lr


def maha_score(h, labels):
    """Oracle diagonal-Mahalanobis score on embeddings h (detach-safe)."""
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.cpu().numpy()
    hN = h_np[y == 0]
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    # scale to a bounded target in (0,1) via rank-free min-max on normal scores
    lo, hi = np.percentile(s, 5), np.percentile(s, 99)
    t = np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)
    return t  # soft target in [0,1]


def pretrain_head(enc, scorer, x_raw, ei, labels, steps=200, lr=1e-3):
    """Distill the oracle Mahalanobis score into the linear head (encoder frozen)."""
    opt = torch.optim.Adam(scorer.parameters(), lr=lr)
    enc.eval()
    with torch.no_grad():
        h = enc(x_raw, ei)
        t = torch.from_numpy(maha_score(h, labels)).float().to(DEVICE)
    for _ in range(steps):
        opt.zero_grad()
        p = torch.sigmoid(scorer(h))
        loss = F.binary_cross_entropy(p, t)
        loss.backward(); opt.step()
    scorer.eval()


def run_tta(enc_init, scorer_init, x_tgt, ei, labels, method, lr, seed, log_grads=False):
    """TENT/SHOT from a given (encoder, head) init; returns final AUC + head AUC
    trajectory + per-group gradient-norm shares."""
    enc = copy.deepcopy(enc_init)
    scorer = copy.deepcopy(scorer_init)
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    head_aucs, gnorm_share = [], []
    y = labels.cpu().numpy()
    for step in range(TTA_STEPS + 1):
        if step > 0:
            enc.train(); scorer.train(); opt.zero_grad()
            h = enc(x_tgt, ei)
            logit = scorer(h)
            p = torch.sigmoid(logit)
            eps = 1e-7
            bce_vec = -(labels.float() * (p + eps).log() + (1 - labels.float()) * (1 - p + eps).log())
            L_ent = bce_vec.mean()
            if method == "SHOT":
                mean_p = p.mean().clamp(eps, 1 - eps)
                H_mean = -(mean_p * mean_p.log() + (1 - mean_p) * (1 - mean_p).log())
                loss = L_ent - SHOT_LAMBDA_DIV * H_mean
            else:
                loss = L_ent

            if log_grads:
                # gradient norms split by group: backprop normal-masked and anomaly-masked
                opt.zero_grad()
                bce_n = bce_vec[labels == 0].sum() / len(labels)
                bce_n.backward(retain_graph=True)
                gn = torch.sqrt(sum((p_.grad ** 2).sum() for p_ in list(enc.parameters()) + list(scorer.parameters()) if p_.grad is not None)).item()
                opt.zero_grad()
                bce_a = bce_vec[labels == 1].sum() / len(labels)
                bce_a.backward(retain_graph=True)
                ga = torch.sqrt(sum((p_.grad ** 2).sum() for p_ in list(enc.parameters()) + list(scorer.parameters()) if p_.grad is not None)).item()
                gnorm_share.append(gn / max(gn + ga, 1e-12))

            loss.backward(); opt.step()
        if step % 10 == 0 or step == TTA_STEPS:
            with torch.no_grad():
                enc.eval(); scorer.eval()
                hh = enc(x_tgt, ei)
                head_auc = roc_auc_score(y, torch.sigmoid(scorer(hh)).cpu().numpy())
                head_aucs.append((step, head_auc))
    # final oracle-Maha AUC
    with torch.no_grad():
        enc.eval()
        hT = enc(x_tgt, ei)
        auc = maha_auc(hT, labels)
    return auc, head_aucs, gnorm_share


def maha_auc(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.cpu().numpy()
    hN = h_np[y == 0]
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    return roc_auc_score(y, s)


def main():
    rows = []
    for ds in DATASETS:
        x_raw, ei, labels = load_dataset(ds)
        x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
        x_cpu = x_raw.cpu()
        for shift in [0.0, 0.1]:
            for seed in SEEDS:
                x_t = make_target(x_cpu, shift, seed).to(DEVICE)
                torch.manual_seed(seed)
                enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
                pretrain(enc, x_raw, ei, labels)
                with torch.no_grad():
                    auc0 = maha_auc(enc(x_t, ei), labels)

                # ── Arm A: random-init head, EXACT paper protocol ──────────
                for method in ["TENT", "SHOT"]:
                    aucT, lr = paper_protocol_run(enc, x_t, ei, labels, method, seed)
                    rows.append(dict(dataset=ds, shift=shift, seed=seed, arm="random",
                                     method=method, lr=lr, AUC_0=auc0, head_AUC_init=0.431,
                                     AUC_T=aucT, delta_AUC=aucT - auc0, gnorm_share_mean=np.nan))

                # ── Arm B: source-trained (distilled) head, same TTA loop ──
                scorer = Scorer(HIDDEN_DIM).to(DEVICE)
                torch.manual_seed(seed + 7)
                nn.init.normal_(scorer.fc.weight, std=0.01); nn.init.zeros_(scorer.fc.bias)
                pretrain_head(enc, scorer, x_raw, ei, labels)
                with torch.no_grad():
                    h_auc0 = roc_auc_score(labels.cpu().numpy(),
                                           torch.sigmoid(scorer(enc(x_raw, ei))).cpu().numpy())
                # lr: same entropy-selected rate as arm A for comparability
                fn = tta_tent if method == "TENT" else tta_shot
                lr_ref = {}
                for method in ["TENT", "SHOT"]:
                    fn = tta_tent if method == "TENT" else tta_shot
                    lr_ref[method] = select_lr(fn, enc, x_t, ei, labels, LR_CANDIDATES)
                for method in ["TENT", "SHOT"]:
                    aucT, head_tr, gsh = run_tta(enc, scorer, x_t, ei, labels,
                                                 method, lr_ref[method], seed,
                                                 log_grads=(method == "TENT"))
                    rows.append(dict(dataset=ds, shift=shift, seed=seed, arm="source_trained",
                                     method=method, lr=lr_ref[method], AUC_0=auc0,
                                     head_AUC_init=h_auc0, AUC_T=aucT,
                                     delta_AUC=aucT - auc0,
                                     gnorm_share_mean=np.mean(gsh) if gsh else np.nan))
                print(f"{ds} shift={shift} seed={seed} head0={h_auc0:.3f} srcAUC={auc0:.3f} done")
        pd.DataFrame(rows).to_csv(results_csv("r3_head_init.csv"), index=False)

    df = pd.DataFrame(rows)
    with open(results_txt("r3_head_init_summary.txt"), "w") as f:
        f.write("=" * 72 + "\nR3 HEAD-INIT CONTROL: mean ΔAUC by arm × method (shift=0)\n" + "=" * 72 + "\n\n")
        d0 = df[df["shift"] == 0.0]
        f.write(d0.groupby(["dataset", "arm", "method"])["delta_AUC"].mean().round(3).to_string() + "\n\n")
        f.write("Head AUC at init (source_trained arm):\n")
        f.write(df[df.arm == "source_trained"].groupby("dataset")["head_AUC_init"].mean().round(3).to_string() + "\n\n")
        f.write("Gradient-norm share (normal / total), TENT random arm:\n")
        f.write(df[df.gnorm_share_mean.notna()].groupby("dataset")["gnorm_share_mean"].mean().round(3).to_string() + "\n")
    print("done")


if __name__ == "__main__":
    main()
