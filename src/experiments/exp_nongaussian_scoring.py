"""
exp_nongaussian_scoring.py — P0: non-Gaussian / distribution-free scoring

Motivation (Reviewer #1, R2): the current anomaly score is the diagonal
Mahalanobis distance, which embeds an implicit multivariate-Gaussian
assumption. The reviewer worries that TTA-induced degradation may be an
artifact of the scoring rule failing on non-Gaussian embeddings, rather
than a genuine representation change.

This script tests that hypothesis directly. We run No-TTA / TENT / SHOT
(same protocol as exp_cross_method_tta) and re-score the SAME adapted
embeddings with four scoring rules:

    1. maha_diag   — diagonal Mahalanobis (the paper's rule; Gaussian-ish)
    2. maha_full   — full-covariance Mahalanobis (still Gaussian, but richer)
    3. knn         — k-th NN distance in the non-anomalous reference set
                     (distribution-free; no Gaussian assumption)
    4. kde         — Gaussian-kernel KDE on PCA-reduced embeddings
                     (non-parametric; no Gaussian assumption on the support)

If degradation persists under the distribution-free scorers (knn, kde),
the Gaussian-assumption artifact is ruled out as the root cause.

Outputs:
    nongauss_scoring_results.csv   per (dataset, shift, seed, method, scorer)
    nongauss_scoring_summary.txt   ΔAUC by method × scorer + comparison
"""

import os, sys, copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, scipy.stats as ss
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

# ── Config ───────────────────────────────────────────────────────────────────
# Data lives at the repo root (not under tta-gad-release/data). This file is at
# <repo>/tta-gad-release/src/experiments/, so the repo root is three levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

PLANETOID = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "Pubmed"}
if len(sys.argv) > 1:
    DATASETS = sys.argv[1].split(",")
else:
    DATASETS = ["cora", "citeseer", "pubmed"]
OUT_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""

SHIFT_LEVELS = [0.0, 0.1]
SEEDS = [42, 100, 200, 300, 400]
PILOT_SEED = 999

PRETRAIN_EPOCHS = 300
PRETRAIN_LR = 3e-3
MARGIN = 2.0
ANOMALY_RATIO = 0.05
TTA_STEPS = 80
LR_CANDIDATES = [3e-5, 1e-4, 3e-4]
SHOT_LAMBDA_DIV = 1.0

HIDDEN_DIM = 64
NUM_LAYERS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Non-parametric scorer hyperparameters
KNN_KS = [5, 10]           # kNN scorer: distance to k-th NN
KDE_PCA_DIM = 8            # KDE scorer: PCA dim to keep KDE tractable in d=64

torch.manual_seed(42); np.random.seed(42)


# ── Data ─────────────────────────────────────────────────────────────────────
def inject_anomaly(x, ratio=0.05, strength=0.5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x_new = x.clone(); n = x.shape[0]
    idx = torch.randperm(n)[:int(n * ratio)]
    labels = torch.zeros(n, dtype=torch.long)
    labels[idx] = 1
    x_new[idx] += torch.randn_like(x_new[idx]) * strength
    return x_new, labels


PYGOD_DATASETS = {"reddit", "inj_cora", "weibo", "inj_amazon", "enron", "inj_flickr"}


def load_dataset(name):
    if name in PLANETOID:
        data = Planetoid(root=DATA_DIR, name=PLANETOID[name])[0]
        x, labels = inject_anomaly(data.x, ratio=ANOMALY_RATIO, strength=0.5, seed=0)
        return x, data.edge_index, labels
    if name in PYGOD_DATASETS:
        from pygod.utils import load_data as pygod_load
        data = pygod_load(name)
        x = data.x.float()
        ei = data.edge_index
        y = data.y
        # flatten pygod's multi-hot / score labels to a binary anomaly indicator
        if y.dim() > 1:
            y = (y.float().sum(-1) > 0).long()
        else:
            y = (y > 0).long()
        return x, ei, y
    raise ValueError(f"dataset {name} not supported here (needs dgl/pygod)")


def make_target(x_cpu, shift, seed):
    torch.manual_seed(seed)
    x = x_cpu.clone()
    if shift > 0:
        x = x + torch.randn_like(x) * shift
    return x.to(DEVICE)


# ── Architecture ─────────────────────────────────────────────────────────────
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden)] + [GCNConv(hidden, hidden) for _ in range(layers - 1)])

    def forward(self, x, ei):
        h = x
        for c in self.convs:
            h = F.relu(c(h, ei))
        return h


class GCNEncoderBN(nn.Module):
    """GCN with batch-norm after each conv — used by the normalization-based
    adaptation baseline (BN-adapt), which updates only running statistics."""
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden)] + [GCNConv(hidden, hidden) for _ in range(layers - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])

    def forward(self, x, ei):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, ei)))
        return h


class Scorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, h):
        return self.fc(h).squeeze(-1)


# ── TTA methods (verbatim protocol from exp_cross_method_tta) ────────────────
def tta_tent(enc_init, x_tgt, ei, labels, lr, steps=TTA_STEPS):
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    final_loss = None
    for step in range(steps):
        enc.train(); scorer.train(); opt.zero_grad()
        h = enc(x_tgt, ei)
        prob = torch.sigmoid(scorer(h))
        eps = 1e-7
        loss = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean()
        loss.backward(); opt.step()
        final_loss = loss.item()
    enc.eval()
    with torch.no_grad():
        h_final = enc(x_tgt, ei)
    return h_final, final_loss


def tta_shot(enc_init, x_tgt, ei, labels, lr, steps=TTA_STEPS, lambda_div=SHOT_LAMBDA_DIV):
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    final_loss = None
    for step in range(steps):
        enc.train(); scorer.train(); opt.zero_grad()
        h = enc(x_tgt, ei)
        prob = torch.sigmoid(scorer(h))
        eps = 1e-7
        L_ent = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log()).mean()
        mean_p = prob.mean().clamp(eps, 1 - eps)
        H_mean = -(mean_p * mean_p.log() + (1 - mean_p) * (1 - mean_p).log())
        loss = L_ent - lambda_div * H_mean
        loss.backward(); opt.step()
        final_loss = loss.item()
    enc.eval()
    with torch.no_grad():
        h_final = enc(x_tgt, ei)
    return h_final, final_loss


def tta_pseudolabel(enc_init, src_scorer, x_tgt, ei, labels, lr, steps=TTA_STEPS,
                    pos_thr=0.5, neg_thr=0.5):
    """Pseudo-label / self-training TTA (Reviewer #1 R1).

    Mechanism differs structurally from entropy methods:
      - A FROZEN source scorer produces Mahalanobis-based anomaly scores on
        the target embeddings.
      - High-confidence nodes are assigned pseudo-labels; the encoder + a fresh
        scoring head are then fine-tuned with binary cross-entropy on those
        pseudo-labels.
    Unlike entropy minimization, the gradient here flows from BOTH pseudo
    anomaly and pseudo normal nodes (those that clear the confidence bar), so
    it is NOT dominated by the majority class the way TENT/SHOT are. This is
    the cleanest test of whether the degradation is specific to the
    normal-dominated entropy objective.

    Returns (h_final, final_loss) like the other methods so select_lr works.
    """
    enc = copy.deepcopy(enc_init).train()
    scorer = Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt = torch.optim.Adam(list(enc.parameters()) + list(scorer.parameters()), lr=lr)
    final_loss = None
    with torch.no_grad():
        h0 = enc_init.eval()(x_tgt, ei)
        src_score = src_scorer(h0)
    pl = torch.full((x_tgt.shape[0],), -1.0, device=DEVICE)  # -1 = ignored
    pl[src_score >= pos_thr] = 1.0   # confident anomaly
    pl[src_score <= neg_thr] = 0.0   # confident normal
    mask = pl >= 0
    if mask.sum() < 2:
        return enc.eval()(x_tgt, ei).detach(), float("nan")
    for step in range(steps):
        enc.train(); scorer.train(); opt.zero_grad()
        h = enc(x_tgt, ei)
        logits = scorer(h)
        prob = torch.sigmoid(logits)
        eps = 1e-7
        # BCE on confident pseudo-labels only
        loss = -(pl[mask] * (prob[mask] + eps).log()
                 + (1 - pl[mask]) * (1 - prob[mask] + eps).log()).mean()
        loss.backward(); opt.step()
        final_loss = loss.item()
    enc.eval()
    with torch.no_grad():
        h_final = enc(x_tgt, ei)
    return h_final, final_loss


def tta_bn_adapt(enc_bn_init, x_tgt, ei, labels):
    """Normalization-based statistic adaptation (zero gradient).

    Replaces BN running statistics with target-domain batch statistics in a
    single forward pass. No entropy objective, no encoder-weight update.
    This is the normalization-family baseline the reviewer asked about; it is
    deterministic given the source encoder.
    """
    enc = copy.deepcopy(enc_bn_init)
    enc.eval()
    with torch.no_grad():
        h = x_tgt
        for conv, bn in zip(enc.convs, enc.bns):
            h_pre = conv(h, ei)
            bn.running_mean.data = h_pre.mean(0)
            bn.running_var.data = h_pre.var(0, unbiased=False).clamp(min=1e-6)
            h = F.relu(bn(h_pre))
    return h, None  # no loss (no gradient training)


def select_lr(method_fn, enc_init, x_pilot, ei, labels, candidates):
    best_lr, best_loss = candidates[0], float("inf")
    for lr in candidates:
        torch.manual_seed(PILOT_SEED)
        _, loss = method_fn(enc_init, x_pilot, ei, labels, lr, steps=20)
        if loss is not None and loss < best_loss:
            best_lr, best_loss = lr, loss
    return best_lr


def make_src_scorer(mu_s, var_s):
    """Frozen source-domain Mahalanobis scorer for pseudo-label generation.

    Returns a callable scoring an embedding tensor h -> source-Mahalanobis
    scores (per node), computed under the FIXED source normal statistics.
    Using source (not target) stats is the correct 'source model' reference
    for generating pseudo-labels at deployment."""
    mu = mu_s.detach()
    var = (var_s + 1e-6).detach()

    def score(h):
        return (((h.detach() - mu) ** 2) / var).sum(1).sqrt()

    return score


def select_lr_pseudo(enc_init, src_scorer, x_pilot, ei, labels, candidates):
    best_lr, best_loss = candidates[0], float("inf")
    for lr in candidates:
        torch.manual_seed(PILOT_SEED)
        _, loss = tta_pseudolabel(enc_init, src_scorer, x_pilot, ei, labels, lr, steps=20)
        if loss is not None and not np.isnan(loss) and loss < best_loss:
            best_lr, best_loss = lr, loss
    return best_lr


# ── Scorers ──────────────────────────────────────────────────────────────────
def _split(h, labels):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    y = labels.detach().cpu().numpy()
    return h_np, h_np[y == 0], h_np[y == 1], y


def score_maha_diag(h, labels):
    h_np, hN, _, y = _split(h, labels)
    mu = hN.mean(0)
    var = hN.var(0) + 1e-6
    s = np.sqrt(((h_np - mu) ** 2 / var).sum(1))
    return roc_auc_score(y, s), s


def score_maha_full(h, labels):
    h_np, hN, _, y = _split(h, labels)
    mu = hN.mean(0)
    cov = EmpiricalCovariance().fit(hN).covariance_
    # ridge for numerical stability
    cov = cov + 1e-4 * np.eye(cov.shape[0])
    diff = h_np - mu
    s = np.sqrt(np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff))
    return roc_auc_score(y, s), s


def score_knn(h, labels, k):
    """Distance to the k-th nearest non-anomalous neighbor. Distribution-free.

    To avoid a node being its own neighbor leaking its label, we compute
    NN distances among the non-anomalous reference set only, then query all
    nodes against it (k+1 to skip self for known-normals, reported as k-th).
    """
    h_np, hN, _, y = _split(h, labels)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(hN)
    dist, _ = nn.kneighbors(h_np)
    s = dist[:, k]  # k-th NN distance (index 0 is the nearest)
    return roc_auc_score(y, s), s


def score_kde(h, labels):
    """Gaussian-kernel KDE log-density on PCA-reduced embeddings.

    Non-parametric: no Gaussian assumption on the embedding support. Lower
    density => higher anomaly score. PCA reduces dimensionality so KDE stays
    tractable and well-conditioned (d=64 full KDE is degenerate).
    """
    h_np, hN, _, y = _split(h, labels)
    pca = PCA(n_components=min(KDE_PCA_DIM, hN.shape[1]), random_state=0)
    z_all = pca.fit_transform(h_np)
    zN = z_all[y == 0]
    scaler = StandardScaler().fit(zN)
    zN_s = scaler.transform(zN)
    z_all_s = scaler.transform(z_all)
    # bandwidth via Silverman on the non-anomalous set
    sigma = max(zN_s.std(), 1e-3)
    bw = sigma * (4.0 / (3.0 * zN.shape[1] + 1.0)) ** (1.0 / (zN.shape[1] + 4.0)) \
        * zN.shape[0] ** (-1.0 / (zN.shape[1] + 4.0))
    bw = max(bw, 1e-3)
    # log-density = logsumexp of Gaussian kernels (chunked for memory)
    log_density = np.empty(z_all_s.shape[0])
    chunk = 512
    for a in range(0, z_all_s.shape[0], chunk):
        b = min(a + chunk, z_all_s.shape[0])
        d2 = ((z_all_s[a:b, None, :] - zN[None, :, :]) ** 2).sum(2)
        log_density[a:b] = logsumexp(-0.5 * d2 / bw ** 2) - np.log(zN.shape[0]) \
            - zN.shape[1] * np.log(bw) - 0.5 * zN.shape[1] * np.log(2 * np.pi)
    s = -log_density  # low density -> high score
    return roc_auc_score(y, s), s


SCORERS = [("maha_diag", None), ("maha_full", None),
           ("knn_k5", 5), ("knn_k10", 10), ("kde", None)]


def score_all(h, labels):
    """Return {scorer_name: auc} for one set of embeddings."""
    out = {}
    out["maha_diag"], _ = score_maha_diag(h, labels)
    out["maha_full"], _ = score_maha_full(h, labels)
    out["knn_k5"], _ = score_knn(h, labels, 5)
    out["knn_k10"], _ = score_knn(h, labels, 10)
    out["kde"], _ = score_kde(h, labels)
    return out


# ── Per-dataset driver ───────────────────────────────────────────────────────
def run_dataset(name):
    print(f"\n{'=' * 62}\nDataset: {name}\n{'=' * 62}")
    x_raw, ei, labels = load_dataset(name)
    n_anom = labels.sum().item()
    print(f"  N={x_raw.shape[0]:,}  F={x_raw.shape[1]}  anomaly={n_anom}({100*n_anom/len(labels):.1f}%)")
    x_raw, ei, labels = x_raw.to(DEVICE), ei.to(DEVICE), labels.to(DEVICE)
    x_cpu = x_raw.cpu()

    # Pretrain source encoder (margin loss, identical to main paper)
    torch.manual_seed(42)
    enc = GCNEncoder(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h = enc(x_raw, ei)
        cn = h[labels == 0].mean(0)
        d = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN - d[labels == 1]) ** 2).mean() + (d[labels == 0] ** 2).mean()
        loss.backward(); opt.step()
    enc.eval()

    # Source-domain normal statistics (frozen) for pseudo-label generation
    with torch.no_grad():
        h_src = enc(x_raw, ei)
    hN_src = h_src[labels == 0]
    mu_s = hN_src.mean(0)
    var_s = hN_src.var(0).clamp(min=1e-6)
    src_scorer = make_src_scorer(mu_s, var_s)
    # pseudo-label confidence thresholds from source-score quantiles (pilot)
    with torch.no_grad():
        s_pilot = src_scorer(enc(make_target(x_cpu, 0.0, PILOT_SEED).to(DEVICE), ei))
    neg_thr = torch.quantile(s_pilot, 0.90).item()   # bottom 90% => "confident normal"
    pos_thr = torch.quantile(s_pilot, 0.97).item()   # top 3% => "confident anomaly"

    # Pretrain BN encoder (for normalization-based BN-adapt baseline)
    torch.manual_seed(42)
    enc_bn = GCNEncoderBN(x_raw.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    opt_bn = torch.optim.Adam(enc_bn.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc_bn.train(); opt_bn.zero_grad()
        h = enc_bn(x_raw, ei)
        cn = h[labels == 0].mean(0)
        d = torch.norm(h - cn, dim=1)
        loss = (F.relu(MARGIN - d[labels == 1]) ** 2).mean() + (d[labels == 0] ** 2).mean()
        loss.backward(); opt_bn.step()
    enc_bn.eval()

    # LR selection (pilot seed, minimize training loss)
    x_pilot = make_target(x_cpu, 0.0, PILOT_SEED)
    lr_tent = select_lr(tta_tent, enc, x_pilot, ei, labels, LR_CANDIDATES)
    lr_shot = select_lr(tta_shot, enc, x_pilot, ei, labels, LR_CANDIDATES)
    lr_pseudo = select_lr_pseudo(enc, src_scorer, x_pilot, ei, labels, LR_CANDIDATES)
    print(f"  Selected LR — TENT: {lr_tent:.0e}  SHOT: {lr_shot:.0e}  "
          f"Pseudo: {lr_pseudo:.0e}  (BN-adapt: n/a)")

    records = []
    for shift in SHIFT_LEVELS:
        for seed in SEEDS:
            x_tgt = make_target(x_cpu, shift, seed)

            with torch.no_grad():
                h0 = enc(x_tgt, ei)
            aucs0 = score_all(h0, labels)

            torch.manual_seed(seed)
            h_tent, _ = tta_tent(enc, x_tgt, ei, labels, lr_tent)
            aucs_tent = score_all(h_tent, labels)

            torch.manual_seed(seed)
            h_shot, _ = tta_shot(enc, x_tgt, ei, labels, lr_shot)
            aucs_shot = score_all(h_shot, labels)

            torch.manual_seed(seed)
            h_pseudo, _ = tta_pseudolabel(enc, src_scorer, x_tgt, ei, labels, lr_pseudo,
                                         pos_thr=pos_thr, neg_thr=neg_thr)
            aucs_pseudo = score_all(h_pseudo, labels)

            # BN-adapt: deterministic, uses BN encoder; score with its own BN-domain stats
            with torch.no_grad():
                h0_bn = enc_bn(x_tgt, ei)
            aucs0_bn = score_all(h0_bn, labels)
            h_bn, _ = tta_bn_adapt(enc_bn, x_tgt, ei, labels)
            aucs_bn = score_all(h_bn, labels)

            runs = [("No-TTA", aucs0, aucs0), ("TENT", aucs0, aucs_tent),
                    ("SHOT", aucs0, aucs_shot), ("Pseudo-label", aucs0, aucs_pseudo),
                    ("BN-adapt", aucs0_bn, aucs_bn)]
            for method, base, aucs_t in runs:
                for scorer in ["maha_diag", "maha_full", "knn_k5", "knn_k10", "kde"]:
                    records.append({
                        "dataset": name, "shift": shift, "seed": seed, "method": method,
                        "scorer": scorer,
                        "AUC_0": base[scorer],
                        "AUC_T": aucs_t[scorer],
                        "delta_AUC": aucs_t[scorer] - base[scorer],
                    })
        # progress
        sub = pd.DataFrame(records).query("shift == @shift and method in ['TENT','SHOT','Pseudo-label','BN-adapt']")
        for meth in ["TENT", "SHOT", "Pseudo-label", "BN-adapt"]:
            mm = sub.query("method == @meth and scorer == 'maha_diag'")["delta_AUC"].mean()
            print(f"  [shift={shift}] mean ΔAUC (maha_diag) {meth:<12}={mm:+.4f}")

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
    df.to_csv(results_csv(f"nongauss_scoring_results{OUT_SUFFIX}.csv"), index=False)
    print(f"\nSaved nongauss_scoring_results.csv ({len(df)} rows)")

    # ── Summary: ΔAUC by method × scorer ─────────────────────────────────────
    summary = []
    summary.append("=" * 70)
    summary.append("NON-GAUSSIAN SCORING SUMMARY  (mean ΔAUC over seeds)")
    summary.append("=" * 70)

    for shift in SHIFT_LEVELS:
        summary.append(f"\n[shift={shift}]")
        for name in DATASETS:
            summary.append(f"  -- {name} --")
            sub = df.query("dataset == @name and shift == @shift")
            if sub.empty:
                continue
            tbl = sub.groupby(["method", "scorer"])["delta_AUC"].mean().unstack()
            cols = [c for c in ["maha_diag", "maha_full", "knn_k5", "knn_k10", "kde"] if c in tbl.columns]
            tbl = tbl[cols]
            for method in ["No-TTA", "TENT", "SHOT", "Pseudo-label", "BN-adapt"]:
                if method in tbl.index:
                    row = tbl.loc[method]
                    summary.append(f"    {method:<12} " + "  ".join(
                        f"{s}={row[s]:+.3f}" for s in tbl.columns))

    # Persistence-of-degradation check: does degradation survive distribution-free scorers?
    summary.append("\n" + "=" * 70)
    summary.append("PERSISTENCE CHECK: degradation under distribution-free scorers")
    summary.append("=" * 70)
    for shift in SHIFT_LEVELS:
        summary.append(f"\n[shift={shift}] fraction of (dataset,seed) with ΔAUC<0")
        for method in ["TENT", "SHOT", "Pseudo-label", "BN-adapt"]:
            for name in DATASETS:
                sub = df.query("dataset==@name and shift==@shift and method==@method")
                if sub.empty:
                    continue
                per_seed = sub.groupby(["seed", "scorer"])["delta_AUC"].mean().unstack()
                for scorer in ["maha_diag", "knn_k5", "knn_k10", "kde"]:
                    frac = (per_seed[scorer] < 0).mean() if scorer in per_seed else float("nan")
                    mean_d = per_seed[scorer].mean() if scorer in per_seed else float("nan")
                    summary.append(f"    {name:<9} {method:<5} {scorer:<9}: frac<0={frac*100:4.0f}%  meanΔ={mean_d:+.4f}")

    with open(results_txt(f"nongauss_scoring_summary{OUT_SUFFIX}.txt"), "w") as f:
        f.write("\n".join(summary))
    print("Saved nongauss_scoring_summary.txt")
