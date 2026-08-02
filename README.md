# Anonymous

Code source for the paper studying when and why test-time adaptation (TTA) fails for graph anomaly detection (GAD).

## Overview

Standard entropy-minimization TTA, despite its success in image classification, can *harm* graph anomaly detection: AUC declines in 76–89% of tested conditions, with drops as large as 0.29. This repository provides the full experimental pipeline, analysis scripts, result data, and figure/PDF sources to reproduce every table and figure in the paper.

The central analysis disentangles the **covariance-expansion** hypothesis ($D_\Sigma \to M^2_A \to \text{AUC}$) from other adaptation-induced representation dynamics through a controlled algebraic intervention on frozen encoder embeddings, and introduces a calibration-free anomaly separability measure $M^2_A$ grounded in the distribution-free identity $\mathbb{E}[M^2_\text{normal}] = d$.

## Repository Structure

```
tta-gad-release/
├── src/                          # All Python code
│   ├── config.py                 # Path configuration (root-relative)
│   ├── style.py                  # Unified academic plotting style
│   ├── experiments/              # Core experiments (produce result CSVs)
│   │   ├── exp_multi_shift.py            # Table 1: D_Σ→M²_A→AUC chain (Cora, 54 conditions)
│   │   ├── exp_real_datasets_v3.py       # Table 1: real datasets (Questions, Reddit)
│   │   ├── exp_causal_intervention.py   # Tables 2 & 3: isotropic/anisotropic injection + gap
│   │   ├── exp_cross_method_tta.py       # Table 4: TENT vs SHOT vs BN-adapt
│   │   ├── exp_native_multishift.py       # Appendix: native multi-shift for inj_cora
│   │   └── exp_tta_with_metrics.py       # Figure 1(a): AUC trajectory
│   ├── figures/                  # Figure generation scripts
│   │   ├── fig1_phenomenon.py            # Figure 1: TTA degradation phenomenon
│   │   ├── fig2_correlation.py           # Figure 2: D_Σ→M²_A→AUC correlations
│   │   ├── fig3_paradox.py               # Figure 3: directional paradox
│   │   └── fig4_residual_compression.py # Figure 4: residual + compression
│   ├── analysis/                 # Statistical analysis (paper table values)
│   │   ├── analysis_pygod_appendix.py    # Appendix C: PyGOD robustness summary
│   │   ├── analysis_planetoid_extension.py  # Citeseer/Pubmed extension analysis
│   │   └── compute_drift_diagnostics.py # Table 3: anomaly/centroid drift
│   └── utils/
│       └── dataset_scan.py              # Dataset density pre-check
├── results/                      # Pre-computed experimental outputs
│   ├── csv/                      # Raw result tables (reproducible from src/)
│   └── txt/                      # Statistical summaries
└── figures/                      # Generated figure PDFs + PNGs (Figures 1–4)
```

## Datasets

| Dataset | Source | Nodes | Features | Anomaly | Loading |
|---|---|---|---|---|---|
| Cora | Planetoid (PyG) | 2,708 | 1,433 | 5% injected (σ=0.5 Gaussian) | `Planetoid(name="Cora")` |
| Questions | Platonov et al. | 48,921 | 301 | 3.0% native | DGL graph |
| Reddit | Kumar et al. | 10,984 | 64 | 3.3% native | DGL graph |
| inj_cora, weibo, enron, inj_amazon, inj_flickr | PyGOD | 2,708–89,250 | varies | injected/organic | `pygod.utils.load_data` |

Cora/Citeseer/Pubmed are loaded via `torch_geometric.datasets.Planetoid`; Questions/Reddit via `dgl.data.utils.load_graphs`; PyGOD benchmarks via `pygod.utils.load_data`. Dataset files are expected under `data/` (see `src/config.py`); download Planetoid/PyGOD through their standard loaders.

## Reproducing the Results

### Requirements
- Python 3.8+
- PyTorch, PyTorch Geometric, DGL
- scikit-learn, scipy, numpy, pandas, matplotlib, seaborn
- pygod (for the Appendix C robustness benchmarks)

### Running the pipeline

All scripts are path-configured via `src/config.py` and should be run from the repository root. Outputs are written to `results/` and `figures/`.

```bash
# 1. Core experiments → results/csv/
python src/experiments/exp_multi_shift.py        # Table 1 (Cora, 54 conditions)
python src/experiments/exp_real_datasets_v3.py   # Table 1 (Questions, Reddit)
python src/experiments/exp_causal_intervention.py "cora,questions,reddit"   # Tables 2 & 3
python src/experiments/exp_cross_method_tta.py "cora,questions,reddit"      # Table 4
python src/experiments/exp_tta_with_metrics.py   # Figure 1(a) trajectory

# 2. Appendix C: PyGOD robustness benchmarks
python src/experiments/exp_causal_intervention.py "inj_cora,weibo,enron,inj_amazon,inj_flickr" _pygod
python src/experiments/exp_cross_method_tta.py "inj_cora,weibo,enron,inj_amazon" _pygod
python src/experiments/exp_native_multishift.py inj_cora _inj   # native multi-shift for inj_cora

# 3. Analysis (produces paper table values)
python src/analysis/analysis_pygod_appendix.py    # Appendix C summary table
python src/analysis/compute_drift_diagnostics.py  # Table 3 drift measurements
python src/analysis/analysis_planetoid_extension.py  # Citeseer/Pubmed extension

# 4. Figures → figures/*.pdf + *.png
python src/figures/fig1_phenomenon.py
python src/figures/fig2_correlation.py
python src/figures/fig3_paradox.py
python src/figures/fig4_residual_compression.py
```

The `causal_intervention` and `cross_method` scripts accept command-line arguments `<datasets> <suffix>` to run arbitrary dataset subsets without overwriting existing results (suffix is appended to output filenames).


## Key Findings

1. **TTA harms GAD.** Entropy-minimization TTA reduces AUC in 76–89% of conditions across Cora/Questions/Reddit, with declines up to 0.29, even without distribution shift.
2. **The covariance-expansion pathway is causally real but not the operative mechanism.** Isotropic covariance injection reduces AUC with bootstrap CIs excluding zero on all datasets (Tables 2); however, the anisotropic injection using real-TTA variance ratios reverses the prediction on real-world graphs (Figure 3), establishing a directional paradox.
3. **A calibration-free diagnostic.** $M^2_A$ satisfies $\mathbb{E}[M^2_\text{normal}] = d$ without source-domain calibration, confirmed empirically within 6% of $d=64$.
4. **Robustness to injection mechanism.** PyGOD benchmarks with independent injection schemes (Appendix C) replicate all main findings: 5/5 causal CIs exclude zero; the synthetic-vs-real paradox split mirrors the main paper; TENT≈SHOT ($p=0.15$) holds.

## Citation

If you use this code, please cite the paper (citation to be added upon acceptance).
