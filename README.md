# Sacubitril/Valsartan Co-crystal Tablet — QbD Computational Framework

Python and TensorFlow 2.16 implementation of the multi-criteria decision-making (MCDM) ranking algorithm and deep-learning sensitivity analysis from:

> Tonmoy, P.R., Sarkar, M.R. *In-Silico Predictive Modeling of Process Parameter Interactions in Sacubitril/Valsartan Co-crystal Tablet Manufacturing: A Quality by Design Strategy for Real-Time Quality Assurance.*

## Overview

This repository contains the complete open-source computational pipeline for the manuscript:

1. **Multi-Criteria Decision-Making (MCDM):** Ranks the 15 Box-Behnken design (BBD) experimental runs by a weighted **Final Efficiency Score (FES)** combining three critical quality attributes (CQAs):
   - Dissolution Q15 (%) — Weight 0.50
   - Tablet Hardness, TH (N) — Weight 0.30
   - Disintegration Time, DT (s) — Weight 0.20 (lower is better)
   *(Assay % was excluded from scoring as it remained non-differentiating across all runs).*

2. **TensorFlow 2.16 MLP Sensitivity Engine:** Generates over 1,000 virtual batch simulations via Monte Carlo noise sampling to evaluate process sensitivity and identify critical compression thresholds (>22 kN).

---

## Repository Structure

- `canonical_bbd_dataset.csv` — The single canonical 15-run Box-Behnken design experimental dataset (TH, Dissolution, DT, Assay).
- `mcdm_ranking.py` — Min-max normalization (1–10 scale) + weighted FES scoring + ranking script.
- `mlp_sensitivity.py` — TensorFlow 2.16 surrogate neural network & Monte Carlo simulation engine.
- `requirements.txt` — Python package dependencies (`numpy`, `pandas`, `tensorflow`, `scikit-learn`).

---

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
