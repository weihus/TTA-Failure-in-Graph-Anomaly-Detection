"""
analyze_p2.py — consolidate P2 (non-Gaussian shift) results.

Answers Reviewer #1 R3: is degradation an artifact of Gaussian-only shift?
We check, at shift=0.1, whether ΔAUC stays negative across all four shift
families (gaussian, student_t, mixture, sparse) under the distribution-free
kNN scorer. If yes, the conclusion is not a Gaussian-shift artifact.

Reads nongauss_shift_results.csv. Writes p2_consolidated.txt.
"""
import os, sys
import pandas as pd, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

SHIFT_KINDS = ["gaussian", "student_t", "mixture", "sparse"]


def main():
    df = pd.read_csv(results_csv("nongauss_shift_results.csv"))
    s1 = df[df["shift"] == 0.1].copy()

    out = ["=" * 80,
           "P2 NON-GAUSSIAN SHIFT: mean ΔAUC at shift=0.1 by shift_kind × method",
           "=" * 80]
    for sc in ["maha_diag", "knn_k10"]:
        out.append(f"\n[scorer = {sc}]")
        piv = (s1[s1["scorer"] == sc]
               .groupby(["dataset", "shift_kind", "method"])["delta_AUC"].mean().unstack())
        out.append(piv.round(4).to_string())

    # Persistence per shift family (knn_k10)
    out.append("\n" + "-" * 80)
    out.append("PERSISTENCE under each shift family (knn_k10, frac seeds ΔAUC<0) + mean ΔAUC")
    out.append("-" * 80)
    per = s1[(s1["scorer"] == "knn_k10") & (s1["method"].isin(["TENT", "SHOT"]))]
    rows = []
    for (ds, kind, meth), g in per.groupby(["dataset", "shift_kind", "method"]):
        ps = g.groupby("seed")["delta_AUC"].first()
        rows.append({"dataset": ds, "shift_kind": kind, "method": meth,
                     "frac_neg": float((ps < 0).mean()), "mean_delta": float(ps.mean())})
    P = pd.DataFrame(rows)
    out.append("\nfrac<0:")
    out.append(P.pivot(index=["dataset", "method"], columns="shift_kind", values="frac_neg")
               .round(2).to_string())
    out.append("\nmean ΔAUC:")
    out.append(P.pivot(index=["dataset", "method"], columns="shift_kind", values="mean_delta")
               .round(4).to_string())

    # Verdict
    out.append("\n" + "=" * 80)
    out.append("VERDICT (R3: Gaussian-shift artifact?)")
    out.append("=" * 80)
    nongauss = P[~P["shift_kind"].isin(["gaussian"])]
    overall_frac = nongauss["frac_neg"].mean()
    overall_mean = nongauss["mean_delta"].mean()
    out.append(f"Non-Gaussian families (student_t/mixture/sparse): mean frac<0={overall_frac*100:.0f}%, "
               f"mean ΔAUC={overall_mean:+.4f}")
    if overall_frac >= 0.8 and overall_mean < -0.005:
        out.append("=> DEGRADATION PERSISTS under non-Gaussian shifts. Gaussian-shift artifact RULED OUT (R3 refuted).")
    else:
        out.append("=> degradation weak/absent under non-Gaussian shifts (CHECK).")

    text = "\n".join(out)
    print(text)
    with open(results_txt("p2_consolidated.txt"), "w") as f:
        f.write(text)
    print("\nSaved p2_consolidated.txt")


if __name__ == "__main__":
    main()
