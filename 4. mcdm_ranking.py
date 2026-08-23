"""
Multi-Criteria Decision-Making (MCDM) ranking algorithm for
Sacubitril/Valsartan co-crystal tablet Box-Behnken design runs.

Reference:
    Tonmoy, P.R., Sarkar, M.R. "In-Silico Predictive Modeling of Process
    Parameter Interactions in Sacubitril/Valsartan Co-crystal Tablet
    Manufacturing: A Quality by Design Strategy for Real-Time Quality
    Assurance." Appendix A.

Method:
    Weighted min-max normalization (1-10 scale) of three critical
    quality attributes, followed by a weighted-sum Final Efficiency
    Score (FES):

        FES = 0.50 * S_Dissolution + 0.30 * S_TH + 0.20 * S_DT

    Assay was excluded from scoring (98-101% across all runs, judged
    non-differentiating).

Environment: Python 3.10+ | NumPy, Pandas
"""

import numpy as np
import pandas as pd

# Experimental data: TH = Tablet Hardness (N), Dissolution = Q15 (%), DT = Disintegration Time (s)
data = {
    "Run": np.arange(1, 16),
    "TH": [
        114.78, 104.47, 121.00, 120.80, 104.22, 116.08, 123.27,
        111.57, 116.65, 93.35, 127.10, 109.39, 113.61, 103.32, 102.34,
    ],
    "Dissolution": [
        89.66, 91.82, 86.51, 87.74, 88.77, 89.47, 94.20, 86.88,
        72.46, 91.48, 78.86, 88.05, 91.22, 80.72, 90.40,
    ],
    "DT": [
        295.08, 304.27, 374.60, 260.88, 255.79, 315.82, 250.00, 261.40,
        375.37, 271.00, 435.07, 314.25, 282.36, 346.45, 273.33,
    ],
}


def normalize(series: pd.Series, weight_type: str = "beneficial") -> pd.Series:
    """Min-max normalize a response to a 1-10 dimensionless scale.

    Args:
        series: raw response values.
        weight_type: 'beneficial' if higher is better (e.g. Dissolution, TH),
            'non-beneficial' if lower is better (e.g. DT).
    """
    if weight_type == "beneficial":
        return ((series - series.min()) / (series.max() - series.min())) * 9 + 1
    return ((series.max() - series) / (series.max() - series.min())) * 9 + 1


def rank_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Score and rank all BBD runs by Final Efficiency Score (FES)."""
    df = df.copy()
    df["S_TH"] = normalize(df["TH"], "beneficial")
    df["S_Dissolution"] = normalize(df["Dissolution"], "beneficial")
    df["S_DT"] = normalize(df["DT"], "non-beneficial")

    w_dissolution, w_th, w_dt = 0.50, 0.30, 0.20  # Dissolution | TH | DT
    df["FES"] = (
        df["S_Dissolution"] * w_dissolution
        + df["S_TH"] * w_th
        + df["S_DT"] * w_dt
    )

    return df.sort_values(by="FES", ascending=False).reset_index(drop=True)


def main() -> None:
    df = pd.DataFrame(data)
    df_ranked = rank_runs(df)

    print("Top 5 Runs:")
    print(
        df_ranked[["Run", "S_Dissolution", "S_TH", "S_DT", "FES"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
