"""
exp_r2_untuned_lr.py — E4 (R2-RC6): untuned / loss-agnostic learning rate

R2: the learning rate is selected by minimizing the entropy loss on a pilot
seed — the same objective hypothesized to cause harm — so the tuned result may
partly reflect selection bias. Check: run the main TTA comparison with
(a) entropy-selected lr (paper protocol), (b) fixed lr=1e-4 (middle candidate,
no selection), (c) fixed lr=3e-5 (smallest candidate, most conservative).

If degradation persists under (b)/(c), the selection procedure is not driving
the headline result.

Datasets: cora, citeseer, pubmed, reddit; TENT + SHOT; shift {0, 0.1}.
Outputs: results/csv/r2_e4_untuned_lr.csv, results/txt/r2_e4_summary.txt
"""

import os, sys, copy, torch
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exp_nongaussian_scoring import (
    load_dataset, make_target, GCNEncoder, DEVICE,
    PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN, HIDDEN_DIM, NUM_LAYERS,
    TTA_STEPS, LR_CANDIDATES, PILOT_SEED, select_lr, tta_tent, tta_shot,
)
from exp_r2_checks import pretrain

DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
SEEDS = [42, 100, 200, 300, 400]


def maha_auc(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.cpu().numpy()
    hN = h_np[y == 0]
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    return roc_auc_score(y, s)


def run():
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
                x_pilot = make_target(x_cpu, shift, PILOT_SEED).to(DEVICE)
                for method, fn in [("TENT", tta_tent), ("SHOT", tta_shot)]:
                    lr_sel = select_lr(fn, enc, x_pilot, ei, labels, LR_CANDIDATES)
                    for lr_mode, lr in [("entropy_selected", lr_sel),
                                        ("fixed_1e-4", 1e-4),
                                        ("fixed_3e-5", 3e-5)]:
                        torch.manual_seed(seed)
                        hT, _ = fn(enc, x_t, ei, labels, lr)
                        aucT = maha_auc(hT, labels)
                        rows.append(dict(dataset=ds, shift=shift, seed=seed,
                                         method=method, lr_mode=lr_mode, lr=lr,
                                         AUC_0=auc0, AUC_T=aucT, delta_AUC=aucT - auc0))
                print(rows[-1])
        pd.DataFrame(rows).to_csv(results_csv("r2_e4_untuned_lr.csv"), index=False)

    df = pd.DataFrame(rows)
    with open(results_txt("r2_e4_summary.txt"), "w") as f:
        f.write("=" * 70 + "\nE4 UNTUNED LR: mean ΔAUC by lr_mode × method (shift=0)\n")
        f.write("=" * 70 + "\n\n")
        d0 = df[df["shift"] == 0.0]
        f.write(d0.groupby(["method", "lr_mode"])["delta_AUC"].agg(["mean", "std", lambda s: (s < 0).mean()]).rename(columns={"<lambda>": "frac_neg"}).round(3).to_string() + "\n\n")
        f.write("Selected lr distribution:\n")
        f.write(df.groupby(["dataset", "method"])["lr"].agg(lambda s: s.mode()[0]).to_string() + "\n\n")
        f.write("Reading: if fixed-lr degradation ≈ entropy-selected degradation,\n")
        f.write("selection bias is ruled out as the driver of the headline result.\n")


if __name__ == "__main__":
    run()
