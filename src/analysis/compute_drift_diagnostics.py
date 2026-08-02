"""
compute_drift_diagnostics.py — 补算 Table 3 的 anomaly drift + centroid drift

为 enron/inj_amazon 补 drift（gap 已有），并跑 cora 验证口径（对照论文 1.33/1.96）。
复用 causal_intervention 的配置与 load_dataset 逻辑（TTA_LR=1e-4）。

drift 定义（论文 §4.3）:
  anomaly drift = mean ‖h_{A,T} - h_{A,s}‖_2  (anomaly embedding L2 位移)
  normal drift  = mean ‖h_{N,T} - h_{N,s}‖_2
  Anomaly drift / Normal drift  (Table 3 列2)
  centroid drift = ‖(μ_T - μ_s)‖_{Σ_s^{-1}}  (diagonal Mahalanobis)
  Centroid drift / √M²_{A,T}    (Table 3 列3)
"""
import os, sys, copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, results_txt

PRETRAIN_EPOCHS=300; PRETRAIN_LR=3e-3; MARGIN=2.0; TTA_STEPS=80; TTA_LR=1e-4
HIDDEN_DIM=64; NUM_LAYERS=3
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)

def inject_anomaly(x, ratio=0.05, strength=0.5, seed=None):
    if seed is not None: torch.manual_seed(seed)
    x_new=x.clone(); n=x.shape[0]; idx=torch.randperm(n)[:int(n*ratio)]
    labels=torch.zeros(n, dtype=torch.long); labels[idx]=1
    x_new[idx]+=torch.randn_like(x_new[idx])*strength
    return x_new, labels

def load_dataset(name):
    planetoid={"cora":"Cora","citeseer":"Citeseer","pubmed":"Pubmed"}
    if name in planetoid:
        data=Planetoid(root=DATA_DIR, name=planetoid[name])[0]
        x, labels=inject_anomaly(data.x, 0.05, 0.5, 0)
        return x, data.edge_index, labels
    else:
        from pygod.utils import load_data
        data=load_data(name)
        return data.x.float(), data.edge_index, data.y.bool().long()

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs=nn.ModuleList([GCNConv(in_dim, hidden)]+[GCNConv(hidden, hidden) for _ in range(layers-1)])
    def forward(self, x, ei):
        h=x
        for c in self.convs: h=F.relu(c(h, ei))
        return h
class Scorer(nn.Module):
    def __init__(self, dim): super().__init__(); self.fc=nn.Linear(dim, 1)
    def forward(self, h): return self.fc(h).squeeze(-1)

def normal_stats(h, labels):
    hN=h[labels==0].detach(); return hN.mean(0), hN.var(0).clamp(min=1e-6)

def run(name):
    print(f"\n=== {name} ===")
    x, ei, labels=load_dataset(name)
    x=x.to(DEVICE); ei=ei.to(DEVICE); labels=labels.to(DEVICE)
    torch.manual_seed(42)
    enc=GCNEncoder(x.shape[1], HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    opt=torch.optim.Adam(enc.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    for ep in range(PRETRAIN_EPOCHS):
        enc.train(); opt.zero_grad()
        h=enc(x, ei); cn=h[labels==0].mean(0); d=torch.norm(h-cn, dim=1)
        loss=(F.relu(MARGIN-d[labels==1])**2).mean()+(d[labels==0]**2).mean()
        loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad(): h_src=enc(x, ei)
    mu_s, var_s=normal_stats(h_src, labels)
    enc_tta=copy.deepcopy(enc).train(); scorer=Scorer(HIDDEN_DIM).to(DEVICE).train()
    opt_tta=torch.optim.Adam(list(enc_tta.parameters())+list(scorer.parameters()), lr=TTA_LR)
    for _ in range(TTA_STEPS):
        enc_tta.train(); scorer.train(); opt_tta.zero_grad()
        h=enc_tta(x, ei); prob=torch.sigmoid(scorer(h)); eps=1e-7
        L=-(prob*(prob+eps).log()+(1-prob)*(1-prob+eps).log()).mean()
        L.backward(); opt_tta.step()
    enc_tta.eval()
    with torch.no_grad(): h_T=enc_tta(x, ei)
    mu_T, var_T=normal_stats(h_T, labels)
    hA_s=h_src[labels==1]; hA_T=h_T[labels==1]
    hN_s=h_src[labels==0]; hN_T=h_T[labels==0]
    anom_drift=(hA_T-hA_s).norm(dim=1).mean().item()
    norm_drift=(hN_T-hN_s).norm(dim=1).mean().item()
    ratio=anom_drift/norm_drift
    delta_mu=mu_T-mu_s
    centroid_drift=(delta_mu**2/var_s).sum().sqrt().item()
    M2_A_T=((hA_T-mu_T)**2/var_T).sum(1).mean().item()
    centroid_over_sqrtM2A=centroid_drift/(M2_A_T**0.5)
    print(f"  anom/norm={ratio:.2f}  centroid/√M²_A={centroid_over_sqrtM2A:.2f}  (M²_A_T={M2_A_T:.1f})")
    return name, ratio, centroid_over_sqrtM2A

results=[]
for name in ["cora","enron","inj_amazon"]:
    try: results.append(run(name))
    except Exception as e: print(f"  SKIP {name}: {e}")

print("\n=== 汇总 ===")
print(f"{'dataset':10s} {'anom/norm':>10s} {'centroid/√M²_A':>15s}")
for name, ratio, cdr in results:
    print(f"{name:10s} {ratio:10.2f} {cdr:15.2f}")
print("\n论文对照: Cora 1.33/1.96  Questions 1.38/10.60  Reddit 1.03/1.99")
