# Sacubitril/Valsartan Co-crystal Tablet — MCDM Ranking

Python implementation of the multi-criteria decision-making (MCDM) ranking
algorithm from Appendix A of:

> Tonmoy, P.R., Sarkar, M.R. *In-Silico Predictive Modeling of Process
> Parameter Interactions in Sacubitril/Valsartan Co-crystal Tablet
> Manufacturing: A Quality by Design Strategy for Real-Time Quality
> Assurance.*

Ranks the 15 Box-Behnken design (BBD) experimental runs by a weighted
**Final Efficiency Score (FES)** combining three critical quality
attributes:

- Dissolution Q15 (%) — weight 0.50
- Tablet Hardness, TH (N) — weight 0.30
- Disintegration Time, DT (s) — weight 0.20 (lower is better)

Assay (%) was excluded from scoring — it stayed within 98–101% across
all runs and did not meaningfully differentiate batches.

## What this repo contains

- `mcdm_ranking.py` — normalization + FES scoring + ranking
- `data/bbd_experimental_data.csv` — the 15-run raw data (TH, Dissolution, DT)

## What this repo does **not** contain

The manuscript states "all source code is provided in Appendix A," but
only the MCDM script above was actually included in the paper text.
The following, referenced in the Methods section, are **not** included
here because their code was not published in the appendix:

- The Box-Behnken RSM quadratic model fitting (done in Stat-Ease
  Design-Expert v.13, not Python)
- The TensorFlow 2.16 multilayer perceptron sensitivity analysis
  (>1,000 virtual batch simulations)

If you have that code, add it here (e.g. `mlp_sensitivity.py`) and
update this README accordingly.

## Usage

```bash
pip install -r requirements.txt
python mcdm_ranking.py
```

Expected output (top 5 of 15 runs):

```
Top 5 Runs:
 Run  S_Dissolution     S_TH      S_DT      FES
   7      10.000000 8.978667 10.000000 9.693600
   4       7.325667 8.320000  9.470903 8.053014
  13       8.766329 6.402667  8.426325 7.989230
   1       8.120515 6.714667  7.807748 7.636207
   6       8.041858 7.061333  6.799157 7.499161
```

Run 7 ranks first with FES = 9.69, matching the value reported in the
manuscript.

## License

Add a license of your choice (MIT/Apache-2.0 are common for research code)
before making the repo public.
