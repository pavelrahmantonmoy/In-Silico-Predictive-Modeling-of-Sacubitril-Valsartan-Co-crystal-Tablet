"""
Multi-Criteria Decision-Making (MCDM) ranking algorithm for
Sacubitril/Valsartan co-crystal tablet Box-Behnken design runs.

Reference:
    Tonmoy, P.R., Sarkar, M.R. "In-Silico Predictive Modeling of Process
    Parameter Interactions in Sacubitril/Valsartan Co-crystal Tablet
    Manufacturing: A Quality by Design Framework for Real-Time Quality
    Assurance." Appendix A.

Method:
    Weighted min-max normalization (1-10 scale) of three critical
    quality attributes, followed by a weighted-sum Final Efficiency Score (FES):

        FES = 0.50 * S_Dissolution + 0.30 * S_TH + 0.20 * S_DT

    Assay was excluded from scoring (98-101% across all runs, judged
    non-differentiating).

Environment: Python 3.10+ | NumPy, Pandas
"""

import numpy as np
import pandas as pd

# Canonical BBD Experimental Data
data = {
    "Run": np.arange(1, 16),
    "TH": [
        82.00, 114.00, 125.00, 78.00, 68.00, 134.00, 124.71,
        118.00, 114.73, 95.00, 125.00, 92.00, 121.00, 108.00, 110.00,
    ],
    "Dissolution": [
        89.00, 78.00, 70.00, 93.00, 96.00, 65.00, 91.88,
        78.00, 89.91, 82.00, 72.00, 88.00, 90.94, 92.00, 85.00,
    ],
    "DT": [
        230.0, 360.0, 450.0, 190.0, 150.0, 520.0, 248.6,
        400.0, 239.6, 280.0, 460.0, 260.0, 226.6, 210.0, 320.0,
    ],
}


def normalize(series: pd.Series, weight_type: str = "beneficial") -> pd.Series:
    """Min-max normalize a response to a 1-10 dimensionless scale."""
    if weight_type == "beneficial":
        return ((series - series.min()) / (series.max() - series.min())) * 9 + 1
    return ((series.max() - series) / (series.max() - series.min())) * 9 + 1


def rank_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Score and rank all BBD runs by Final Efficiency Score (FES)."""
    df = df.copy()
    df["S_TH"] = normalize(df["TH"], "beneficial")
    df["S_Dissolution"] = normalize(df["Dissolution"], "beneficial")
    df["S_DT"] = normalize(df["DT"], "non-beneficial")

    w_dissolution, w_th, w_dt = 0.50, 0.30, 0.20
    df["FES"] = (
        df["S_Dissolution"] * w_dissolution
        + df["S_TH"] * w_th
        + df["S_DT"] * w_dt
    )

    return df.sort_values(by="FES", ascending=False).reset_index(drop=True)


def main() -> None:
    df = pd.DataFrame(data)
    df_ranked = rank_runs(df)

    print("Top 5 Runs (Canonical BBD Dataset):")
    print(
        df_ranked[["Run", "S_Dissolution", "S_TH", "S_DT", "FES"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
