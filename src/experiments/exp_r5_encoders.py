"""
exp_r5_encoders.py — E5-lite (R2-RC5): architecture generality

R2: the phenomenon is evaluated only with a 3-layer GCN; does it survive
under other message-passing encoders? We swap the encoder for GAT and
GraphSAGE (same depth/width, same margin pretraining, same TTA protocol)
and reproduce the headline comparison: TENT/SHOT/BN-adapt ΔAUC at shift 0.

Gate: source AUC > 0.65 required for a (dataset, encoder) cell to count
(same threshold as the paper's dataset scan); cells below the gate are
reported as excluded, mirroring the Mean-Submergence exclusion protocol.

Datasets: cora, citeseer, pubmed, reddit. Encoders: gcn (reference), gat,
sage. Methods: TENT, SHOT (+ BN-adapt only on BN variants, not here).
Outputs: results/csv/r5_encoders.csv, results/txt/r5_encoders_summary.txt
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
    load_dataset, GCNEncoder, Scorer, DEVICE,
    PRETRAIN_EPOCHS, PRETRAIN_LR, MARGIN, HIDDEN_DIM, NUM_LAYERS,
    TTA_STEPS, LR_CANDIDATES, PILOT_SEED, SHOT_LAMBDA_DIV,
)
from torch_geometric.nn import GATConv, SAGEConv
from exp_r2_checks import pretrain

DATASETS = ["cora", "citeseer", "pubmed", "reddit"]
SEEDS = [42, 100, 200, 300, 400]
SOURCE_GATE = 0.65


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3, heads=2):
        super().__init__()
        # hidden is the OUTPUT dim; per-head dim = hidden // heads
        assert hidden % heads == 0
        self.convs = nn.ModuleList(
            [GATConv(in_dim, hidden // heads, heads=heads)] +
            [GATConv(hidden, hidden // heads, heads=heads) for _ in range(layers - 1)])

    def forward(self, x, ei):
        h = x
        for c in self.convs:
            h = F.elu(c(h, ei))
        return h


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [SAGEConv(in_dim, hidden)] + [SAGEConv(hidden, hidden) for _ in range(layers - 1)])

    def forward(self, x, ei):
        h = x
        for c in self.convs:
            h = F.relu(c(h, ei))
        return h


ENCODERS = {
    "gcn": GCNEncoder,
    "gat": GATEncoder,
    "sage": SAGEEncoder,
}


def maha_auc(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.cpu().numpy()
    hN = h_np[y == 0]
    mu, var = hN.mean(0), hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    return roc_auc_score(y, s)


def tta_run(enc_init, x_tgt, ei, labels, method, lr, seed):
    """TENT/SHOT with the paper protocol; returns final embeddings."""
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    torch.manual_seed(seed + 7)
    torch.nn.init.normal_(scorer.fc.weight, std=0.01)
    torch.nn.init.zeros_(scorer.fc.bias)
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    for _ in range(TTA_STEPS):
        enc.train(); opt.zero_grad()
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
    enc.eval()
    with torch.no_grad():
        return enc(x_tgt, ei)


def select_lr(method, enc, x_pilot, ei, labels):
    best_lr, best_loss = LR_CANDIDATES[0], float("inf")
    for lr in LR_CANDIDATES:
        torch.manual_seed(PILOT_SEED)
        h = tta_run(enc, x_pilot, ei, labels, method, lr, PILOT_SEED)
        # loss proxy: recompute entropy loss at pilot end state
        enc2 = copy.deepcopy(enc).train()
        scorer = Scorer(HIDDEN_DIM).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(scorer(h))
            eps = 1e-7
            loss = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean().item()
        if loss < best_loss:
            best_lr, best_loss = lr, loss
    return best_lr


def main():
    rows = []
    for ds in DATASETS:
        x_raw, ei, labels = load_dataset(ds)
        x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
        x_cpu = x_raw.cpu()
        for enc_name, Enc in ENCODERS.items():
            for seed in SEEDS:
                torch.manual_seed(seed)
                enc = Enc(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
                pretrain(enc, x_raw, ei, labels)
                with torch.no_grad():
                    auc0 = maha_auc(enc(x_raw, ei), labels)
                passed = auc0 > SOURCE_GATE
                x_pilot = make_target0(x_cpu)
                for method in ["TENT", "SHOT"]:
                    lr = select_lr(method, enc, x_pilot, ei, labels) if passed else LR_CANDIDATES[-1]
                    hT = tta_run(enc, x_raw, ei, labels, method, lr, seed)
                    aucT = maha_auc(hT, labels)
                    rows.append(dict(dataset=ds, encoder=enc_name, seed=seed, method=method,
                                     lr=lr, AUC_0=auc0, AUC_T=aucT, delta_AUC=aucT - auc0,
                                     source_gate=passed))
                print(f"{ds} {enc_name} seed={seed}: AUC0={auc0:.3f} gate={'PASS' if passed else 'FAIL'}")
        pd.DataFrame(rows).to_csv(results_csv("r5_encoders.csv"), index=False)

    df = pd.DataFrame(rows)
    with open(results_txt("r5_encoders_summary.txt"), "w") as f:
        f.write("=" * 70 + "\nE5-lite ENCODER GENERALITY: mean ΔAUC (shift=0)\n")
        f.write("=" * 70 + "\n\n")
        g = df[df.source_gate].groupby(["dataset", "encoder", "method"])["delta_AUC"].mean().round(3)
        f.write(g.to_string() + "\n\n")
        gate = df.groupby(["dataset", "encoder"])["AUC_0"].mean().round(3)
        f.write("Source AUC by (dataset, encoder):\n" + gate.to_string() + "\n\n")
        f.write("Gate (AUC_0 > 0.65) pass/fail counts:\n")
        f.write(df.groupby(["dataset", "encoder"])["source_gate"].mean().round(2).to_string() + "\n")


def make_target0(x_cpu):
    return x_cpu.clone().to(DEVICE)


if __name__ == "__main__":
    main()
