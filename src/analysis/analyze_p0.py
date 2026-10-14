"""
analyze_p0.py — consolidate P0 (non-Gaussian scoring) + P0 (normality) results
into the branch-decision summary that drives the revision narrative.

Decision rule (from the revision plan):
  - degradation persists (frac<0 high, mean<0) under distribution-free scorers
    => Gaussian-assumption artifact ruled out; core mechanism is real.
  - magnitude differs => Mahalanobis amplifies; report kNN alongside.

Outputs: prints the decision table; writes p0_consolidated.csv.
"""
import os, sys
import pandas as pd, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

SCORERS = ["maha_diag", "maha_full", "knn_k5", "knn_k10", "kde"]
Gaussian = {"maha_diag", "maha_full"}
DistFree = {"knn_k5", "knn_k10", "kde"}


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    df = pd.read_csv(results_csv(f"nongauss_scoring_results{suffix}.csv"))
    adapted = df[df["method"].isin(["TENT", "SHOT"])].copy()

    out = []
    out.append("=" * 78)
    out.append("P0 DECISION TABLE: mean ΔAUC by dataset × method × scorer")
    out.append("=" * 78)
    piv = adapted.groupby(["dataset", "method", "scorer"])["delta_AUC"].mean().unstack()
    piv = piv[[s for s in SCORERS if s in piv.columns]]
    out.append(piv.round(4).to_string())

    out.append("\n" + "-" * 78)
    out.append("PERSISTENCE: fraction of seeds with ΔAUC<0, and mean ΔAUC")
    out.append("-" * 78)
    pers = []
    for (ds, m), g in adapted.groupby(["dataset", "method"]):
        per_seed = g.groupby(["seed", "scorer"])["delta_AUC"].mean().unstack()
        for s in [c for c in SCORERS if c in per_seed.columns]:
            pers.append({
                "dataset": ds, "method": m, "scorer": s,
                "frac_neg": float((per_seed[s] < 0).mean()),
                "mean_delta": float(per_seed[s].mean()),
                "kind": "Gaussian" if s in Gaussian else "dist-free",
            })
    pers_df = pd.DataFrame(pers)
    out.append(pers_df.pivot(index=["dataset", "method"], columns="scorer", values="frac_neg")
               .round(2).to_string() if not pers_df.empty else "(no data)")

    out.append("\n" + "-" * 78)
    out.append("AMPLIFICATION: maha_diag mean ΔAUC vs best distribution-free (knn_k10)")
    out.append("-" * 78)
    for (ds, m), g in adapted.groupby(["dataset", "method"]):
        per_seed = g.groupby(["seed", "scorer"])["delta_AUC"].mean().unstack()
        md = per_seed.get("maha_diag")
        kn = per_seed.get("knn_k10")
        if md is not None and kn is not None:
            amp = md.mean() / kn.mean() if kn.mean() < 0 else float("nan")
            out.append(f"  {ds:9} {m:5}: maha_diag={md.mean():+.4f}  knn_k10={kn.mean():+.4f}"
                       f"  amplification={amp:+.2f}x")

    # Overall verdict
    out.append("\n" + "=" * 78)
    out.append("VERDICT")
    out.append("=" * 78)
    df_pers = pers_df[pers_df["kind"] == "dist-free"]
    if not df_pers.empty:
        overall_frac = df_pers["frac_neg"].mean()
        overall_mean = df_pers["mean_delta"].mean()
        out.append(f"Distribution-free scorers: mean frac<0 = {overall_frac*100:.0f}%, "
                   f"mean ΔAUC = {overall_mean:+.4f}")
        if overall_frac >= 0.8 and overall_mean < -0.005:
            out.append("=> DEGRADATION PERSISTS under distribution-free scoring.")
            out.append("=> Gaussian-assumption artifact RULED OUT as root cause (R2 refuted).")
        else:
            out.append("=> degradation weak/absent under distribution-free scoring (CHECK).")
    else:
        out.append("(no distribution-free data yet)")

    text = "\n".join(out)
    print(text)
    with open(results_txt("p0_consolidated.txt"), "w") as f:
        f.write(text)
    pers_df.to_csv(results_csv("p0_consolidated.csv"), index=False)
    print("\nSaved p0_consolidated.csv / p0_consolidated.txt")


if __name__ == "__main__":
    main()
