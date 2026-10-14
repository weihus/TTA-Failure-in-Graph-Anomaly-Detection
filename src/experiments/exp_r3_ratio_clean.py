"""
exp_r3_ratio_clean.py — R3 round-2 Concern 2: clean ratio sweep

R3's protocol: one pretrained source model held fixed, anomaly prevalence
varied via a nested anomaly construction (the ε=20% anomaly pool contains
the ε=1% pool as a prefix), and normal-vs-anomaly gradient contributions
measured directly. Only the anomaly set changes with ε; the encoder, the
injection realization, and everything else are identical across ε.

For each dataset: pretrain ONE source encoder (seed 42); build a shuffled
anomaly pool once (largest ε); for each ε take the first int(n*ε) pool
nodes as the anomaly set (nested by construction); label = pool membership;
run TENT/SHOT 80 steps from the same head init; log per-step normal and
anomaly gradient norms; report ΔAUC (oracle Maha) + gradient-norm share.

Sweep seeds vary ONLY the pool order (i.e., which nodes are anomalies),
giving 5 nested-pool replicates per dataset — the encoder stays fixed.

Outputs: results/csv/r3_ratio_clean.csv, results/txt/r3_ratio_clean_summary.txt
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
    load_dataset, GCNEncoder, Scorer, DEVICE, HIDDEN_DIM, NUM_LAYERS,
    TTA_STEPS, SHOT_LAMBDA_DIV, LR_CANDIDATES, PILOT_SEED,
)
from exp_r2_checks import pretrain

DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
SEEDS = [42, 100, 200, 300, 400]   # pool-order replicates ONLY
RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20]
ANCHOR_SEED = 42                    # the single fixed source checkpoint
LR = 3e-4                           # entropy-selected scale (paper protocol)


def maha_auc(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.cpu().numpy()
    hN = h_np[y == 0]
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    return roc_auc_score(y, s)


def grad_norms(params):
    return torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))


def tta_with_gnorms(enc_init, x, ei, labels, method, seed):
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE)
    torch.manual_seed(seed)  # same head-init RNG state across ε (nested comparison)
    enc_t = enc
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=LR)
    shares = []
    for step in range(TTA_STEPS):
        enc.train(); opt.zero_grad()
        h = enc(x, ei)
        p = torch.sigmoid(scorer(h))
        eps = 1e-7
        bce = -(labels.float() * (p + eps).log() + (1 - labels.float()) * (1 - p + eps).log())
        loss = bce.mean()
        if method == "SHOT":
            mp = p.mean().clamp(eps, 1 - eps)
            loss = loss - SHOT_LAMBDA_DIV * (-(mp * mp.log() + (1 - mp) * (1 - mp).log()))
        # split gradient norms: masked backprops (same loss, group weights)
        opt.zero_grad()
        (bce[labels == 0].sum() / len(labels)).backward(retain_graph=True)
        gn = grad_norms(list(enc.parameters()) + list(scorer.parameters())).item()
        opt.zero_grad()
        (bce[labels == 1].sum() / len(labels)).backward(retain_graph=True)
        ga = grad_norms(list(enc.parameters()) + list(scorer.parameters())).item()
        shares.append(gn / max(gn + ga, 1e-12))
        loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        hT = enc(x, ei)
    return maha_auc(hT, labels), float(np.mean(shares)), float(np.mean(shares[-20:]))


def main():
    rows = []
    for ds in DATASETS:
        x_raw, ei, _ = load_dataset(ds)          # default labels unused; we inject our own
        x_raw = x_raw.to(DEVICE); ei = ei.to(DEVICE)
        n = x_raw.shape[0]
        F = x_raw.shape[1]

        # ── ONE single injection + ONE fixed source checkpoint per dataset ──
        # Inject the LARGEST anomaly pool once (ε=20%), pretrain the encoder on it,
        # then for each smaller ε take the first k injected nodes (nested: the 1%
        # pool is a prefix of the 20% pool) and DROP the unselected injected nodes
        # from the graph so the normal set is always "truly un-injected" nodes.
        g = torch.Generator().manual_seed(ANCHOR_SEED + 1000)
        pool_order = torch.randperm(n, generator=g)
        kmax = int(n * RATIOS[-1])
        injected_idx = pool_order[:kmax]                          # the 20% pool (fixed)
        noise = (torch.randn(kmax, F, generator=g) * 0.5).to(DEVICE)  # fixed noise realization
        x_inj = x_raw.clone()
        x_inj[injected_idx.to(DEVICE)] = x_inj[injected_idx.to(DEVICE)] + noise  # single injection event

        lab_full = torch.zeros(n, dtype=torch.long)
        lab_full[injected_idx] = 1
        lab_full = lab_full.to(DEVICE)
        torch.manual_seed(ANCHOR_SEED)
        enc = GCNEncoder(F, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
        pretrain(enc, x_inj, ei, lab_full)                       # sees the full 20% pool
        enc.eval(); enc_src = copy.deepcopy(enc)
        print(f"[{ds}] fixed source checkpoint (pretrained on 20% injected pool, {kmax} anomalies)")

        # also vary pool order for 5 replicates — but the checkpoint is FIXED to the
        # ANCHOR_SEED pool above. Replicate = different nested subsample of the SAME pool.
        # (i.e., permute which injected nodes count as the "first k"; encoder never changes.)
        for rep_seed in SEEDS:
            g2 = torch.Generator().manual_seed(rep_seed + 1000)
            perm = torch.randperm(kmax, generator=g2)           # reorder the injected pool only
            for eps_r in RATIOS:
                k = int(n * eps_r)
                if k > kmax: k = kmax
                sel = injected_idx[perm[:k]]                     # nested: perm[:k1] ⊂ perm[:k2]
                # build the sub-graph: keep ALL non-injected nodes (normal) + selected injected (anomaly)
                keep_mask = torch.ones(n, dtype=torch.bool)
                keep_mask[injected_idx] = False                  # drop all injected first
                keep_mask[sel] = True                            # then re-add the selected k
                idx_map = torch.where(keep_mask)[0]
                remap = torch.full((n,), -1, dtype=torch.long); remap[idx_map] = torch.arange(len(idx_map))
                ei_cpu = ei.cpu()                                # build sub-graph on CPU then move
                ei_sub = remap[ei_cpu]                           # remap edge_index to sub-graph
                ei_sub = ei_sub[:, (ei_sub >= 0).all(0)]        # drop edges into removed nodes
                x_sub = x_inj[idx_map].to(DEVICE)
                ei_sub = ei_sub.to(DEVICE)
                lab_sub = torch.zeros(len(idx_map), dtype=torch.long)
                # which kept positions are the selected anomalies?
                anom_pos = (remap[sel][remap[sel] >= 0]).to(DEVICE)
                lab_sub[anom_pos] = 1
                lab_sub = lab_sub.to(DEVICE)

                with torch.no_grad():
                    h0 = enc_src(x_sub, ei_sub)
                auc0 = maha_auc(h0, lab_sub)
                for method in ["TENT", "SHOT"]:
                    aucT, gshare, gshare_late = tta_with_gnorms(enc_src, x_sub, ei_sub, lab_sub, method, rep_seed)
                    rows.append(dict(dataset=ds, rep=rep_seed, ratio=eps_r, method=method,
                                     k_anom=k, n_sub=int(len(idx_map)),
                                     AUC_0=auc0, AUC_T=aucT, delta_AUC=aucT - auc0,
                                     gnorm_share=gshare, gnorm_share_late=gshare_late))
                print(f"[{ds}] rep={rep_seed} eps={eps_r}: n={len(idx_map)} AUC0={auc0:.3f} done")
        pd.DataFrame(rows).to_csv(results_csv("r3_ratio_clean.csv"), index=False)

    df = pd.DataFrame(rows)
    from scipy.stats import spearmanr
    with open(results_txt("r3_ratio_clean_summary.txt"), "w") as f:
        f.write("=" * 74 + "\nR3 CLEAN RATIO SWEEP: ONE fixed checkpoint, nested sub-graphs, gradient norms\n" + "=" * 74 + "\n\n")
        f.write("Source AUC by ε (5 nested-pool replicates):\n")
        t = df.groupby(["dataset", "method", "ratio"])["delta_AUC"].mean().unstack("ratio")
        f.write(t.round(3).to_string() + "\n\n")
        f.write("Gradient-norm share (normal/total), TENT, mean over run:\n")
        gt = df[df.method == "TENT"].groupby(["dataset", "ratio"])["gnorm_share"].mean().unstack("ratio")
        f.write(gt.round(3).to_string() + "\n\n")
        f.write("Spearman(ε, ΔAUC) per dataset×method (positive = lower ratio worse, mechanism direction):\n")
        for ds in DATASETS:
            for m in ["TENT", "SHOT"]:
                sub = df[(df.dataset == ds) & (df.method == m)]
                rho, p = spearmanr(sub["ratio"], sub["delta_AUC"])
                f.write(f"  {ds:9s} {m:5s} rho={rho:+.2f} p={p:.3f}\n")


if __name__ == "__main__":
    main()
