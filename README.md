# When Does Test-Time Adaptation Harm Graph Anomaly Detection? The Role of Prediction-Objective Alignment in Entropy Adaptation

Code source for the paper *When Does Test-Time Adaptation Harm Graph Anomaly Detection? The Role of Prediction-Objective Alignment in Entropy Adaptation*.

## Overview

Entropy-based test-time adaptation (TTA), despite its success in image classification, can *harm* graph anomaly detection (GAD): with a misaligned prediction objective, AUC degrades across six datasets and five seeds, and the degradation persists under distribution-free scoring, non-Gaussian shifts, and fixed conservative learning rates.

The paper's central findings, all reproducible from this repository:

1. **Prediction-objective alignment is the strongest determinant of the failure.** A head-initialization control (Table 4) re-runs adaptation with the entropy head trained on the source anomaly-detection task: on the citation datasets the large random-head degradation nearly disappears, and degradation is largest when the objective is arbitrary.
2. **The failure tracks majority-dominated gradient objectives.** A four-method comparison (Table 3) ranks harm as TENT > SHOT > Pseudo-label >> BN-adapt; gradient-free statistics adaptation is near-neutral at zero shift.
3. **Single-channel mechanisms are ruled out by magnitude-matched intervention.** Covariance expansion of the normal manifold is sufficient in isolation but too weak at real-TTA magnitudes (Table 5); each drift channel alone fails to reproduce the real gap (Table 6); Gaussian-scoring artifacts are excluded (Table 10). What remains is an undecomposed interaction among the channels.
4. **The protocol's learning-rate selection amplifies the magnitude 4–20x.** Selecting the rate by the entropy objective — the same objective hypothesized to cause harm — amplifies degradation relative to fixed conservative rates, though degradation persists in essentially every run untuned (Table 11).
5. **Class imbalance remains a plausible, mechanism-consistent explanation, not an established cause.** A clean fixed-checkpoint anomaly-ratio sweep (Table 8) confirms the normal majority dominates the aggregate gradient (monotone in the ratio), but the ratio–degradation trend is not separable from a compression effect (Table 13: adaptation contracts initial AUC differences, β ≈ 0.17).
6. **A source-calibration-free anomaly separability measure** $M^2_A$, grounded in the identity $\mathbb{E}[M^2_\text{normal}] = d$, for post-hoc analysis of adaptation behavior with oracle labels (Figure 2; a Mahalanobis-geometry monitor rather than a scorer-independent index).

## Repository Structure

```
tta-gad-release/
├── src/
│   ├── config.py                 # Path configuration (root-relative)
│   ├── style.py                  # Unified academic plotting style
│   ├── experiments/              # Core experiments (produce result CSVs)
│   │   ├── exp_multi_shift.py            # D_Σ→M²_A→AUC chain (Cora, 54 conditions)
│   │   ├── exp_real_datasets_v3.py       # Real datasets (Questions, Reddit)
│   │   ├── exp_causal_intervention.py    # Isotropic/anisotropic covariance injection
│   │   ├── exp_cross_method_tta.py       # TENT vs SHOT vs BN-adapt
│   │   ├── exp_tta_with_metrics.py       # Figure 1(a): AUC trajectory
│   │   ├── exp_native_multishift.py      # Appendix: native multi-shift for inj_cora
│   │   ├── exp_r3_head_init.py           # Table 4: head-initialization control (strongest finding)
│   │   ├── exp_r3_ratio_clean.py         # Table 8: clean fixed-checkpoint ratio sweep
│   │   ├── exp_r2_ratio_sweep.py         # Table 7: earlier per-ratio re-pretraining sweep
│   │   ├── exp_r2_deployable.py          # Table 9: deployable frozen-source scoring + head AUC trajectory
│   │   ├── exp_r2_untuned_lr.py          # Tables 11 & 21: fixed/untuned learning rates
│   │   ├── exp_r2_drift_intervention.py  # Table 6: drift-only magnitude-matched interventions
│   │   ├── exp_r2_checks.py              # Appendix: M_N² decomposition + circularity checks
│   │   ├── exp_r5_encoders.py            # Table 14: encoder generality (GCN/GAT/SAGE)
│   │   ├── exp_r9_bn_arch.py             # Table 12: architecture-matched gradient-vs-statistics control
│   │   ├── exp_nongaussian_scoring.py    # Table 10 & 20: kNN/KDE distribution-free scoring
│   │   └── exp_nongaussian_shift.py      # Appendix: non-Gaussian Normality-Shift variants
│   ├── figures/                  # Figure generation scripts (Figures 1–4)
│   ├── analysis/                 # Statistical analysis (paper table values)
│   │   ├── analyze_p0.py                 # Distribution-free scoring consolidation
│   │   ├── analyze_p1.py                 # Four-method comparison tests (Friedman/Wilcoxon)
│   │   ├── analyze_p2.py                 # Non-Gaussian shift consolidation
│   │   ├── analysis_pygod_appendix.py    # Table 18: PyGOD robustness summary
│   │   ├── analysis_planetoid_extension.py  # Citeseer/Pubmed extension + compression (Table 13)
│   │   ├── compute_drift_diagnostics.py  # Table 1: residual drift diagnostics
│   │   └── embedding_normality.py        # M_N² embedding normality checks
│   └── utils/
│       └── dataset_scan.py              # Table 15: dataset density pre-check
├── results/
│   ├── csv/                      # Raw result tables (42 CSVs, reproducible from src/)
│   └── txt/                      # Statistical summaries and run logs
└── figures/                      # Generated figure PDFs + PNGs (Figures 1–4)
```

## Datasets

| Dataset | Source | Nodes | Anomaly | Loading |
|---|---|---|---|---|
| Cora / Citeseer / Pubmed | Planetoid (PyG) | 2,708–19,717 | 5% injected (σ=0.5 Gaussian) | `Planetoid(name=...)` |
| Questions | Platonov et al. | 48,921 | 3.0% native | DGL graph |
| Reddit | Kumar et al. | 10,984 | 3.3% native | DGL graph |
| inj_cora, weibo, enron, inj_amazon, inj_flickr | PyGOD | 2,708–89,250 | injected/organic | `pygod.utils.load_data` |

Cora/Citeseer/Pubmed are loaded via `torch_geometric.datasets.Planetoid`; Questions/Reddit via `dgl.data.utils.load_graphs`; PyGOD benchmarks via `pygod.utils.load_data`. Dataset files are expected under `data/` (see `src/config.py`); download Planetoid/PyGOD through their standard loaders.

## Reproducing the Results

### Requirements
- Python 3.8+
- PyTorch, PyTorch Geometric, DGL
- scikit-learn, scipy, numpy, pandas, matplotlib, seaborn
- pygod (for the PyGOD robustness benchmarks, Table 18)

### Running the pipeline

All scripts are path-configured via `src/config.py` and should be run from the repository root. Outputs are written to `results/` and `figures/`.

```bash
# 1. Core experiments → results/csv/
python src/experiments/exp_multi_shift.py        # D_Σ→M²_A→AUC chain (Cora, 54 conditions)
python src/experiments/exp_real_datasets_v3.py   # Questions, Reddit
python src/experiments/exp_causal_intervention.py "cora,questions,reddit"   # covariance injection
python src/experiments/exp_cross_method_tta.py "cora,questions,reddit"      # TENT/SHOT/BN-adapt
python src/experiments/exp_tta_with_metrics.py   # Figure 1(a) trajectory

# 2. Head-initialization control (Table 4, the paper's strongest finding)
python src/experiments/exp_r3_head_init.py

# 3. Anomaly-ratio sweeps
python src/experiments/exp_r3_ratio_clean.py     # Table 8: clean fixed-checkpoint protocol
python src/experiments/exp_r2_ratio_sweep.py     # Table 7: earlier per-ratio protocol

# 4. Learning rate, deployable scoring, drift interventions, architecture
python src/experiments/exp_r2_untuned_lr.py       # Tables 11 & 21
python src/experiments/exp_r2_deployable.py       # Table 9
python src/experiments/exp_r2_drift_intervention.py  # Table 6
python src/experiments/exp_r5_encoders.py         # Table 14
python src/experiments/exp_r9_bn_arch.py          # Table 12

# 5. Distribution-free scoring & non-Gaussian shifts (Tables 10 & 20)
python src/experiments/exp_nongaussian_scoring.py "cora,questions,reddit,citeseer,pubmed,weibo" _p1
python src/experiments/exp_nongaussian_shift.py "cora,questions,reddit"

# 6. PyGOD robustness benchmarks (Table 18)
python src/experiments/exp_causal_intervention.py "inj_cora,weibo,enron,inj_amazon,inj_flickr" _pygod
python src/experiments/exp_cross_method_tta.py "inj_cora,weibo,enron,inj_amazon" _pygod
python src/experiments/exp_native_multishift.py inj_cora _inj  # native multi-shift for inj_cora

# 7. Analysis (produces paper table values)
python src/analysis/analyze_p0.py                # scoring consolidation
python src/analysis/analyze_p1.py                # four-method Friedman/Wilcoxon tests
python src/analysis/analyze_p2.py                # non-Gaussian shift consolidation
python src/analysis/analysis_pygod_appendix.py   # Table 18 summary
python src/analysis/analysis_planetoid_extension.py  # Citeseer/Pubmed + compression (Table 13)
python src/analysis/compute_drift_diagnostics.py # Table 1 drift measurements
python src/analysis/embedding_normality.py       # M_N² checks

# 8. Figures → figures/*.pdf + *.png
python src/figures/fig1_phenomenon.py
python src/figures/fig2_correlation.py
python src/figures/fig3_paradox.py
python src/figures/fig4_residual_compression.py
```

Scripts named `exp_r*` use fixed dataset lists matched to the paper's protocols and take no arguments. The `causal_intervention`, `cross_method`, `native_multishift`, and `nongaussian_*` scripts accept command-line arguments `<datasets> <suffix>` to run arbitrary dataset subsets without overwriting existing results (suffix is appended to output filenames).

## Citation

If you use this code, please cite the paper (citation to be added upon publication).
