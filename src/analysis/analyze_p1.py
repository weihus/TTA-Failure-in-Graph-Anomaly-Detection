"""
analyze_p1.py — consolidate P1 four-method comparison (TENT / SHOT /
Pseudo-label / BN-adapt) and run the statistical tests that quantify
"mechanism heterogeneity" (Reviewer #1 R1).

Reads nongauss_scoring_results_p1.csv (produced by running
exp_nongaussian_scoring.py with OUT_SUFFIX='_p1').

We use two scorers:
  - maha_diag  (the paper's rule; reported for continuity)
  - knn_k10    (distribution-free; the honest headline from P0)

Outputs: prints + writes p1_consolidated.txt and p1_pairwise.csv.
"""
import os, sys, itertools
import pandas as pd, numpy as np, scipy.stats as ss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import results_csv, results_txt

METHODS = ["TENT", "SHOT", "Pseudo-label", "BN-adapt"]
HEAD_SCORERS = ["maha_diag", "knn_k10"]


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "_p1"
    results_tag = f"nongauss_scoring_results{suffix}.csv"
    out_tag = "p1_consolidated" if suffix == "_p1" else f"p1_consolidated{suffix}"
    df = pd.read_csv(results_csv(results_tag))

    out = []
    out.append("=" * 80)
    out.append("P1 FOUR-METHOD COMPARISON: mean ΔAUC by dataset × method")
    out.append("=" * 80)

    for sc in HEAD_SCORERS:
        out.append(f"\n[scorer = {sc}]")
        sub = df[df["scorer"] == sc]
        piv = sub.groupby(["dataset", "shift", "method"])["delta_AUC"].mean().unstack()
        piv = piv[[m for m in METHODS if m in piv.columns]]
        out.append(piv.round(4).to_string())

    # Headline: average across datasets (per shift)
    out.append("\n" + "-" * 80)
    out.append("Average ΔAUC across datasets (per shift × method × scorer)")
    out.append("-" * 80)
    avg = (df[df["scorer"].isin(HEAD_SCORERS)]
           .groupby(["shift", "method", "scorer"])["delta_AUC"].mean().unstack())
    out.append(avg.round(4).to_string())

    # ── Statistical tests (Friedman + pairwise Wilcoxon, TMLR appendix protocol) ─
    out.append("\n" + "=" * 80)
    out.append("STATISTICAL TESTS (Friedman + pairwise Wilcoxon, blocks=dataset×seed)")
    out.append("=" * 80)
    pairwise_rows = []
    for sc in HEAD_SCORERS:
        out.append(f"\n[scorer = {sc}]")
        for shift in sorted(df["shift"].unique()):
            sub = df[(df["scorer"] == sc) & (df["shift"] == shift)]
            # build block table: rows=dataset×seed, cols=method
            tbl = sub.pivot_table(index=["dataset", "seed"], columns="method", values="delta_AUC")
            tbl = tbl[[m for m in METHODS if m in tbl.columns]].dropna()
            present = list(tbl.columns)
            if len(present) < 2:
                continue
            # Friedman on per-block ΔAUC
            try:
                chi2, pfr = ss.friedmanchisquare(*[tbl[m].values for m in present])
                out.append(f"  shift={shift}: Friedman χ²={chi2:.2f}, p={pfr:.4g}  "
                           f"(methods={present})")
            except Exception as e:
                out.append(f"  shift={shift}: Friedman failed ({e})")
                pfr = np.nan
            # pairwise Wilcoxon (paired by block)
            for a, b in itertools.combinations(present, 2):
                try:
                    stat, pw = ss.wilcoxon(tbl[a].values, tbl[b].values, zero_method="wilcox")
                except Exception:
                    stat, pw = np.nan, np.nan
                out.append(f"      {a:<12} vs {b:<12}: W={stat:.1f}, p={pw:.4g}")
                pairwise_rows.append({"scorer": sc, "shift": shift,
                                      "A": a, "B": b, "wilcoxon_W": stat, "p": pw,
                                      "friedman_p": pfr})

    # ── Mechanism heterogeneity verdict ──
    out.append("\n" + "=" * 80)
    out.append("MECHANISM HETEROGENEITY (Reviewer R1)")
    out.append("=" * 80)
    out.append("Interpretation guide:")
    out.append("  - Entropy family (TENT, SHOT): dominated by majority non-anomalous nodes.")
    out.append("  - Pseudo-label: gradient flows from BOTH confident-anomaly and confident-normal")
    out.append("    pseudo-labels, so it is NOT majority-dominated in the same way.")
    out.append("  - BN-adapt: zero-gradient statistic adaptation.")
    out.append("If Pseudo-label / BN-adapt behave DIFFERENTLY from TENT/SHOT, the degradation is")
    out.append("NOT a generic property of 'all adaptation' but specific to normal-dominated")
    out.append("entropy objectives — which is the narrowed, honest claim of the revision.")

    text = "\n".join(out)
    print(text)
    with open(results_txt(f"{out_tag}.txt"), "w") as f:
        f.write(text)
    if pairwise_rows:
        pd.DataFrame(pairwise_rows).to_csv(results_csv(f"{out_tag}_pairwise.csv"), index=False)
        print(f"\nSaved {out_tag}.txt / {out_tag}_pairwise.csv")


if __name__ == "__main__":
    main()
