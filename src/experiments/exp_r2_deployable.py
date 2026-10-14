"""
exp_r2_deployable.py — E1 (R2-RC1/RC2): head AUC trajectory + deployable scoring

R2's sharpest question: TTA optimizes the entropy of a randomly-initialized
prediction head, while the paper's AUC uses an oracle Mahalanobis probe on
embeddings. Does DEPLOYABLE detection degrade, or is the degradation specific
to the oracle probe?

Evaluation layers (same runs):
  L1  oracle maha_diag            — the paper's probe (oracle target labels)
  L2  oracle-ref dist-free kNN    — still oracle reference set, scorer-free
  L3a source-stat maha            — FROZEN source μ/Σ applied to target
                                    embeddings: deployable (no target labels;
                                    exactly what a deployed source model uses)
  L3b trained-head sigmoid score  — the head TTA actually optimizes; AUC of
                                    sigmoid(scorer(h)) vs oracle labels

Also records head initialization (random Linear) and its step-0 AUC to answer
"how is the head initialized / what does its own AUC do across 80 steps".

Datasets: cora, citeseer, pubmed, reddit. Methods: TENT, SHOT (No-TTA base).

Outputs:
  results/csv/r2_e1_deployable.csv   per (dataset, shift, seed, method, layer)
  results/csv/r2_e1_head_traj.csv    head AUC every 10 steps
  results/txt/r2_e1_summary.txt
"""

import os, sys, copy, torch
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exp_nongaussian_scoring import (
    load_dataset, make_target, GCNEncoder, Scorer, DEVICE,
    PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN, HIDDEN_DIM, NUM_LAYERS,
    TTA_STEPS, LR_CANDIDATES, SHOT_LAMBDA_DIV, PILOT_SEED, select_lr,
)
from exp_r2_checks import pretrain

DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
SEEDS = [42, 100, 200, 300, 400]


def split_np(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.detach().cpu().numpy()
    return h_np, h_np[y == 0], y


def auc_maha_diag(h, labels):
    h_np, hN, y = split_np(h, labels)
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    return roc_auc_score(y, s)


def auc_knn10(h, labels):
    h_np, hN, y = split_np(h, labels)
    nn = NearestNeighbors(n_neighbors=11).fit(hN)
    dist, _ = nn.kneighbors(h_np)
    return roc_auc_score(y, dist[:, 10])


def auc_src_maha(h, labels, mu_s, var_s):
    h_np, _, y = split_np(h, labels)
    s = np.sqrt(((h_np - mu_s) ** 2 / var_s).sum(1))
    return roc_auc_score(y, s)


def m2_a(h, labels):
    """M_A^2: Mahalanobis-separability of anomalies (same formula as paper)."""
    h_np, hN, y = split_np(h, labels)
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    hA = h_np[y == 1]
    return float((((hA - mu) ** 2 / var).sum(1)).mean())


def run_method(enc_init, x, ei, labels, method, lr, mu_s, var_s, seed):
    """Run TTA, return per-layer AUCs at final step + head-AUC trajectory."""
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    torch.manual_seed(seed + 7)  # head init reproducible
    torch.nn.init.normal_(scorer.fc.weight, std=0.01)
    torch.nn.init.zeros_(scorer.fc.bias)
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    head_traj = []
    for step in range(TTA_STEPS + 1):
        if step > 0:
            enc.train(); scorer.train(); opt.zero_grad()
            h = enc(x, ei)
            prob = torch.sigmoid(scorer(h))
            eps = 1e-7
            L_ent = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean()
            loss = L_ent
            if method == "SHOT":
                mean_p = prob.mean().clamp(eps, 1 - eps)
                H_mean = -(mean_p * mean_p.log() + (1 - mean_p) * (1 - mean_p).log())
                loss = L_ent - SHOT_LAMBDA_DIV * H_mean
            loss.backward(); opt.step()
        enc.eval(); scorer.eval()
        with torch.no_grad():
            h = enc(x, ei)
            head_score = torch.sigmoid(scorer(h)).cpu().numpy()
        y = labels.cpu().numpy()
        head_auc = roc_auc_score(y, head_score)
        if step % 10 == 0:
            head_traj.append(dict(step=step, head_auc=head_auc))
        if step == TTA_STEPS:
            layers = {
                "L1_oracle_maha": auc_maha_diag(h, labels),
                "L2_oracle_knn10": auc_knn10(h, labels),
                "L3a_src_maha": auc_src_maha(h, labels, mu_s, var_s),
                "L3b_head": head_auc,
            }
            return layers, head_traj, m2_a(h, labels)


def run():
    rows, trajs = [], []
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
                    h_src = enc(x_raw, ei)
                hN = h_src[labels == 0]
                mu_s = hN.mean(0).cpu().numpy()
                var_s = (hN.var(0) + 1e-6).cpu().numpy()

                # No-TTA baseline at this shift
                with torch.no_grad():
                    h0 = enc(x_t, ei)
                base = {
                    "L1_oracle_maha": auc_maha_diag(h0, labels),
                    "L2_oracle_knn10": auc_knn10(h0, labels),
                    "L3a_src_maha": auc_src_maha(h0, labels, mu_s, var_s),
                }
                for layer, v in base.items():
                    rows.append(dict(dataset=ds, shift=shift, seed=seed,
                                     method="No-TTA", layer=layer, AUC=v, M2_A=m2_a(h0, labels)))

                x_pilot = make_target(x_cpu, shift, PILOT_SEED).to(DEVICE)
                for method in ["TENT", "SHOT"]:
                    from exp_nongaussian_scoring import tta_tent, tta_shot
                    tta_fn = tta_tent if method == "TENT" else tta_shot
                    lr = select_lr(tta_fn, enc, x_pilot, ei, labels, LR_CANDIDATES)
                    torch.manual_seed(seed)
                    layers, head_traj, m2a = run_method(enc, x_t, ei, labels, method, lr, mu_s, var_s, seed)
                    for layer, v in layers.items():
                        rows.append(dict(dataset=ds, shift=shift, seed=seed,
                                         method=method, layer=layer, AUC=v, M2_A=m2a))
                    for t in head_traj:
                        trajs.append(dict(dataset=ds, shift=shift, seed=seed,
                                          method=method, **t))
                print(f"{ds} shift={shift} seed={seed} done")
        pd.DataFrame(rows).to_csv(results_csv("r2_e1_deployable.csv"), index=False)
        pd.DataFrame(trajs).to_csv(results_csv("r2_e1_head_traj.csv"), index=False)

    df = pd.DataFrame(rows)
    with open(results_txt("r2_e1_summary.txt"), "w") as f:
        f.write("=" * 72 + "\nE1 DEPLOYABLE EVALUATION: ΔAUC by layer (vs No-TTA same layer)\n" + "=" * 72 + "\n\n")
        piv = df.pivot_table(index=["dataset", "shift", "method", "layer"], values="AUC", aggfunc="mean")
        f.write(piv.round(3).to_string() + "\n\n")
        # delta table
        base = df[df.method == "No-TTA"].set_index(["dataset", "shift", "seed", "layer"])["AUC"]
        d = df[df.method != "No-TTA"].copy()
        d["delta"] = d.apply(lambda r: r.AUC - base.get((r.dataset, r.shift, r.seed, r.layer), np.nan), axis=1)
        tab = d.groupby(["layer", "method"])["delta"].mean().unstack()
        f.write("MEAN ΔAUC by layer × method:\n" + tab.round(3).to_string() + "\n\n")
        f.write("Reading: L3a (source-stat maha, deployable) is the key row.\n")
        f.write("If its ΔAUC matches L1, degradation is representation-level and\n")
        f.write("deployable — R2-RC2 answered affirmatively. L3b head AUC trajectory\n")
        f.write("answers RC1 (what the optimized head itself does).\n")
        ht = pd.DataFrame(trajs)
        f.write("\nHead AUC trajectory (mean over runs, every 10 steps):\n")
        f.write(ht.groupby(["method", "step"])["head_auc"].mean().unstack(0).round(3).to_string() + "\n")
        # E7 circularity: rho(M2_A, delta per layer) within gradient methods
        from scipy.stats import spearmanr
        f.write("\nE7 CIRCULARITY: ρ(M_A^2, ΔAUC per layer), gradient methods:\n")
        dd = d.dropna(subset=["delta"])
        for layer in sorted(dd.layer.unique()):
            sub = dd[dd.layer == layer]
            if len(sub) >= 10:
                r, p = spearmanr(sub["M2_A"], sub["delta"])
                f.write(f"  {layer:16s}: rho={r:+.3f}  p={p:.3g}  N={len(sub)}\n")


if __name__ == "__main__":
    run()
