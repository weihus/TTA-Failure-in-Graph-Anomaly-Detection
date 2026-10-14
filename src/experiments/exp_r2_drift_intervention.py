"""
exp_r2_drift_intervention.py — E3 (R2-RC4): drift-only interventions

R2: given that anisotropic covariance injection IMPROVES separability on real
datasets, can centroid drift (or anomaly drift) alone reproduce the real TTA
gap? This is the empirical pillar of the "drift decomposition" reframing.

Protocol (mirrors exp_causal_intervention.py, embedding-level, no gradients):
  1. Pretrain source encoder; run real TTA (TENT, 80 steps); record final
     embeddings h_T and the realized drifts:
        Δμ = μ_T^N - μ_S^N            (normal centroid drift)
        ΔA = mean anomaly displacement (anomaly-node drift vector, mean)
  2. From the FROZEN SOURCE embeddings h_S, apply single-channel synthetic
     interventions at matched magnitude δ·Δμ / δ·ΔA for δ ∈ {0,…,3}:
        - centroid-drift: translate ALL normal nodes by δ·Δμ (covariance of
          normals unchanged, anomalies fixed)
        - anomaly-drift:  translate ALL anomaly nodes by δ·ΔA (normals fixed)
        - both:            combined
     Score each with compute_metrics (same oracle Maha as paper).
  3. Compare synthetic AUC at δ=1 (matched magnitude) vs real TTA endpoint,
     and report the δ* that best matches per channel.

Datasets: cora, citeseer, pubmed, reddit (Planetoid + reddit), 3 seeds.

Outputs:
  results/csv/r2_e3_drift_intervention.csv
  results/txt/r2_e3_summary.txt
"""

import os, sys, copy, torch
import torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exp_nongaussian_scoring import (
    load_dataset, GCNEncoder, Scorer, DEVICE,
    PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN, HIDDEN_DIM, NUM_LAYERS,
    TTA_STEPS, LR_CANDIDATES, PILOT_SEED, make_target, select_lr, tta_tent,
)
from exp_r2_checks import pretrain

DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
SEEDS = [42, 100, 200]
DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def normal_stats(h, labels):
    hN = h[labels == 0].detach()
    return hN.mean(0), hN.var(0).clamp(min=1e-6)


def compute_metrics(h, labels, mu_s, var_s):
    hd = h.detach()
    mu_t, var_t = normal_stats(hd, labels)
    ratio = var_t / var_s
    D_Sigma = max(0.5 * (ratio - 1 - ratio.log()).sum().item(), 0.0)
    hA = hd[labels == 1]
    M2_A = ((hA - mu_t) ** 2 / var_t).sum(1).mean().item()
    score = ((hd - mu_t) ** 2 / var_t).sum(1).sqrt()
    auc = roc_auc_score(labels.cpu().numpy(), score.cpu().numpy())
    return D_Sigma, M2_A, auc


def apply_per_node_drift(h_S, h_T, mask_kind, delta):
    """Per-node drift: node i moves by delta * (h_T_i - h_S_i), restricted to
    nodes of mask_kind; the others stay at source positions.

    This matches the real assimilation mechanism: TTA displaces each node
    along its OWN trajectory (toward the population centroid), not by a
    common rigid translation. Rigid translation cannot change within-class
    relative geometry and therefore cannot reduce Mahalanobis separability
    (we verified: it saturates AUC to 1.0).
    """
    h_new = h_S.clone()
    delta_h = (h_T - h_S).detach()
    mask = (labels_global == 0) if mask_kind == "normal" else (labels_global == 1)
    h_new[mask] = h_S[mask] + delta * delta_h[mask]
    return h_new


def run():
    global labels_global
    rows = []
    for ds in DATASETS:
        x_raw, ei, labels = load_dataset(ds)
        x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
        labels_global = labels
        x_cpu = x_raw.cpu()
        for seed in SEEDS:
            torch.manual_seed(seed)
            enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
            pretrain(enc, x_raw, ei, labels)
            with torch.no_grad():
                h_S = enc(x_raw, ei)          # shift=0: target == source
            mu_s, var_s = normal_stats(h_S, labels)

            # real TTA on target (shift 0)
            lr = select_lr(tta_tent, enc, make_target(x_cpu, 0.0, PILOT_SEED).to(DEVICE),
                           ei, labels, LR_CANDIDATES)
            enc_t = copy.deepcopy(enc).train()
            scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
            opt = torch.optim.Adam(list(enc_t.parameters()) + list(scorer.parameters()), lr=lr)
            for _ in range(TTA_STEPS):
                enc_t.train(); opt.zero_grad()
                h = enc_t(x_raw, ei)
                prob = torch.sigmoid(scorer(h))
                eps = 1e-7
                loss = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean()
                loss.backward(); opt.step()
            enc_t.eval()
            with torch.no_grad():
                h_T = enc_t(x_raw, ei)

            # realized drift channels (magnitude-matched at delta=1):
            # per-node displacement vectors from real TTA
            #   normal nodes:   Δh_i = h_T_i - h_S_i  (includes centroid drift
            #                    AND variance change of the normal population)
            #   anomaly nodes:  Δh_i likewise
            _, _, auc_real = compute_metrics(h_T, labels, mu_s, var_s)
            rows.append(dict(dataset=ds, seed=seed, channel="real_tta",
                             delta=np.nan, D_Sigma=np.nan, M2_A=np.nan, AUC=auc_real))

            for channel, kind in [("normal_drift", "normal"),
                                  ("anomaly_drift", "anomaly")]:
                for delta in DELTAS:
                    h_syn = apply_per_node_drift(h_S, h_T, kind, delta)
                    Dsig, m2a, auc = compute_metrics(h_syn, labels, mu_s, var_s)
                    rows.append(dict(dataset=ds, seed=seed, channel=channel,
                                     delta=delta, D_Sigma=Dsig, M2_A=m2a, AUC=auc))
            print(f"{ds} seed={seed}: real={auc_real:.3f} done")
        pd.DataFrame(rows).to_csv(results_csv("r2_e3_drift_intervention.csv"), index=False)

    df = pd.DataFrame(rows)
    with open(results_txt("r2_e3_summary.txt"), "w") as f:
        f.write("=" * 72 + "\nE3 DRIFT-ONLY INTERVENTION: synthetic AUC by channel × delta\n")
        f.write("(δ=1 = magnitude-matched to real TTA drift)\n" + "=" * 72 + "\n\n")
        real = df[df.channel == "real_tta"].groupby("dataset")["AUC"].mean()
        f.write("Real TTA endpoint AUC:\n" + real.round(3).to_string() + "\n\n")
        syn = df[df.channel != "real_tta"].groupby(["dataset", "channel", "delta"])["AUC"].mean().unstack(0)
        f.write("Synthetic AUC by channel × delta:\n" + syn.round(3).to_string() + "\n\n")
        # gap at delta=1
        f.write("Gap (synthetic δ=1 − real TTA) per channel:\n")
        for ch in ["normal_drift", "anomaly_drift"]:
            gaps = []
            for ds in DATASETS:
                s = df[(df.dataset == ds) & (df.channel == ch) & (df.delta == 1.0)]["AUC"].mean()
                gaps.append(f"{ds}: {s - real[ds]:+.3f}")
            f.write(f"  {ch:15s} " + "  ".join(gaps) + "\n")
        f.write("\nReading: normal_drift applies each normal node's OWN real-TTA\n")
        f.write("displacement (assimilation toward the population), anomaly_drift\n")
        f.write("likewise for anomalies. At δ=1 a channel reproduces the real endpoint\n")
        f.write("iff that channel alone suffices. Note normal_drift subsumes centroid\n")
        f.write("drift AND normal-covariance change (they are entangled in per-node\n")
        f.write("displacement); anomaly_drift isolates the anomaly-side channel.\n")
        f.write("A rigid common translation was rejected as a protocol: it cannot\n")
        f.write("change within-class geometry and trivially saturates AUC→1.\n")


if __name__ == "__main__":
    run()
