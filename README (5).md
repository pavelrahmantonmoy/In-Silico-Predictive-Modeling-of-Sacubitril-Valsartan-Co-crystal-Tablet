# In-Silico Predictive Modeling of Sacubitril/Valsartan Co-crystal Tablet Manufacturing

Complete, reproducible computational pipeline and datasets for the in-silico
Quality-by-Design (QbD) and machine-learning study described in the manuscript
*"In-Silico Predictive Modeling of Process Parameter Interactions in
Sacubitril/Valsartan Co-crystal Tablet Manufacturing: A Quality by Design
Strategy for Real-Time Quality Assurance."*

All results are **simulation-derived hypotheses** that require experimental
confirmation. A single fixed random seed (42) is used throughout so that every
reported number and figure can be regenerated exactly.

## Repository contents

| File | Purpose |
|------|---------|
| `canonical_bbd_dataset.csv` | Simulated 15-run Box-Behnken design dataset (process parameters + the four responses TH, Q15, DT, Assay). Reference data only; the scripts regenerate their own virtual batches internally. |
| `IJAP.dxpx` | Stat-Ease Design-Expert project file for the four RSM models (TH, Q15, DT, Assay). Contains the full ANOVA, fit statistics, **VIF and PRESS outputs**, and the residual diagnostics (normal-probability, residual-vs-predicted, predicted-vs-actual) underlying Table 2 and Figures 1-10. Open with Design-Expert / Stat-Ease 360. |
| `mcdm_ranking.py` | Multi-Criteria Decision-Making (MCDM) / Final Efficiency Score (FES) weighted scoring algorithm (Appendix A). Reproduces the run ranking (Figures 11 and 18). |
| `mlp_sensitivity.py` | TensorFlow/Keras MLP surrogate for Q15: Monte-Carlo virtual-batch generation, process-parameter sensitivity, and the 22 kN compression failure-rate analysis (Figures 14-16). |
| `ann_surrogate_parity.py` | TensorFlow/Keras MLP surrogates trained independently for the three continuous responses (TH, Q15, DT); reproduces the parity plots in **Figure 17**. |
| `requirements.txt` | Python dependencies. |

## Reproducibility and model settings

- **Random seed:** 42 (fixed across all Monte-Carlo sampling and TensorFlow runs).
- **Virtual batches:** 1,000 samples drawn uniformly over the studied ranges
  (impeller speed 150-350 rpm, binder rate 60-160 g/min, compression 12-28 kN)
  and evaluated with the fitted coded RSM polynomials, with residual-magnitude
  Gaussian noise added (noise CVs: TH 2.0%, Q15 1.8%, DT 3.0%).
- **ANN architecture:** Input(3) -> Dense(64, ReLU, L2=0.01) -> Dropout(0.2)
  -> Dense(32, ReLU, L2=0.01) -> Dense(16, ReLU) -> Dense(1, linear);
  Adam optimiser, MSE loss, EarlyStopping (patience 25), up to 300 epochs,
  batch size 32, 80/20 train-test split (`random_state=42`).
- **Scaling:** features and targets standardised with `StandardScaler`.
- **Scope note:** the ANN surrogate models the **three continuous responses that
  have a fitted RSM equation** (Tablet Hardness, Dissolution Q15, Disintegration
  Time). **Assay% is intentionally excluded** — no response surface was fitted
  for it and it was omitted from the MCDM scoring — so it does not appear in the
  Figure 17 parity plots.
- **Note on determinism:** TensorFlow training is not bit-for-bit deterministic
  across machines, so held-out R² values may vary by roughly ±0.01-0.02 between
  environments. The values reported in the manuscript are from the authors' run.

## How to run

```bash
pip install -r requirements.txt

python mcdm_ranking.py          # FES ranking (Figures 11, 18)
python mlp_sensitivity.py       # ANN sensitivity + 22 kN threshold (Figures 14-16)
python ann_surrogate_parity.py  # 3-panel parity plot -> Figure17_ANN_parity.png (Figure 17)
```

## Reference R² (held-out validation set, authors' run)

| Response | R² |
|----------|-----|
| Tablet Hardness (R1) | 0.968 |
| Dissolution Q15 (R2) | 0.973 |
| Disintegration Time (R3) | 0.991 |
