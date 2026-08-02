"""
dataset_scan.py — 图数据集扫描与可用性预检

遍历 data/ 下的所有图数据集，输出每张图的统计信息：
  节点数、边数、平均度、特征维数、异常比例
并按照密度给出使用建议：
  - avg_deg ≤ 50  → 推荐直接使用（3 层 GCN）
  - 50 < avg_deg ≤ 150 → 可尝试（降为 1 层 GCN）
  - avg_deg > 150 → 跳过（过平滑风险）

用法：
  python scripts/utils/dataset_scan.py
"""

import os, sys, dgl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR
DATASETS = [
    "yelp", "tsocial", "weibo", "tfinance",
    "tolokers", "questions", "reddit", "amazon"
]

results = []
print(f"{'Dataset':<12} {'N':>8} {'E':>12} {'avg_deg':>9} "
      f"{'F':>5} {'anomaly%':>9} {'usable?':>10}")
print("-" * 70)

for name in DATASETS:
    path = f"{DATA_DIR}/{name}"
    try:
        graphs, _ = dgl.data.utils.load_graphs(path)
        g       = graphs[0]
        feature = g.ndata.get("feature", g.ndata.get("feat", None))
        labels  = g.ndata.get("label",   g.ndata.get("labels", None))
        s, e    = g.edges()

        if feature is None or labels is None:
            raise ValueError("Missing feature or label")

        N       = feature.shape[0]
        F       = feature.shape[1]
        E       = s.shape[0]
        avg_deg = E / N
        n_anom  = int(labels.sum().item())
        anom_pct = 100.0 * n_anom / N

        if N > 500_000:
            tag = "SKIP(too large)"
            rec = 0
        elif avg_deg > 150:
            tag = "SKIP(too dense)"
            rec = 0
        elif avg_deg > 50:
            tag = "TRY(1-layer)"
            rec = 1
        else:
            tag = "✓ USE"
            rec = 2

        results.append({
            "name": name, "N": N, "E": E, "avg_deg": avg_deg,
            "F": F, "anom_pct": anom_pct, "tag": tag, "rec": rec
        })
        print(f"{name:<12} {N:>8,} {E:>12,} {avg_deg:>9.1f} "
              f"{F:>5} {anom_pct:>8.1f}% {tag:>12}")

    except Exception as ex:
        print(f"{name:<12}  ERROR: {ex}")

# 推荐列表
recommended = [r for r in results if r["rec"] == 2]
borderline  = [r for r in results if r["rec"] == 1]

print()
print("=== 推荐直接使用（avg_deg ≤ 50）===")
for r in sorted(recommended, key=lambda x: x["avg_deg"]):
    print(f"  {r['name']:<12}  avg_deg={r['avg_deg']:.1f}  "
          f"N={r['N']:,}  anomaly={r['anom_pct']:.1f}%")

print()
print("=== 可尝试使用（50 < avg_deg ≤ 150，降为 1 层）===")
for r in sorted(borderline, key=lambda x: x["avg_deg"]):
    print(f"  {r['name']:<12}  avg_deg={r['avg_deg']:.1f}  "
          f"N={r['N']:,}  anomaly={r['anom_pct']:.1f}%")

print()
if recommended:
    best = sorted(recommended, key=lambda x: x["avg_deg"])
    print(f"建议在 exp_real_datasets_v3.py 中设置：")
    print(f"DATASETS = {[r['name'] for r in best]}")