"""
ann_surrogate_parity.py
-----------------------------------------------------------------------------
Exploratory ANN surrogate for the sacubitril/valsartan co-crystal tablet QbD
study. Trains an independent multilayer-perceptron (MLP) surrogate for each of
the THREE continuous responses that have a fitted response-surface (RSM) model
- tablet hardness (TH, R1), dissolution Q15 (R2) and disintegration time
(DT, R3) - on Monte-Carlo virtual batches sampled from the RSM surfaces, and
produces the parity (actual-vs-predicted) plot used as Figure 17.

Assay (R4) is deliberately NOT modelled here: no response-surface equation was
fitted for assay and it was excluded from the MCDM, so it has no surrogate and
is not shown in the parity figure.

All results are simulation-derived hypotheses and require experimental
confirmation. Random seed is fixed (42) for full reproducibility.
-----------------------------------------------------------------------------
Reproduces manuscript Figure 17 R^2 (held-out validation set):
    Tablet Hardness (R1): 0.966
    Dissolution Q15 (R2): 0.972
    Disintegration Time (R3): 0.991
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------- virtual batches
def generate_monte_carlo_batches(n_samples=1000):
    """Sample process parameters uniformly over the studied ranges and evaluate
    the fitted RSM (coded) polynomials, adding residual-magnitude Gaussian noise
    (noise CVs: TH 2.0%, Q15 1.8%, DT 3.0%)."""
    A = np.random.uniform(150, 350, n_samples)   # impeller speed (rpm)
    B = np.random.uniform(60, 160, n_samples)    # binder addition rate (g/min)
    C = np.random.uniform(12, 28, n_samples)     # compression force (kN)

    A_c = (A - 250) / 100
    B_c = (B - 110) / 50
    C_c = (C - 18) / 6

    th = (118.0 + 11.25*A_c + 0.75*B_c + 20.50*C_c + 6.75*A_c*B_c
          - 3.62*A_c**2 - 4.63*B_c**2 - 12.62*C_c**2)
    q15 = (91.00 - 8.25*A_c - 2.75*B_c - 7.50*C_c - 1.00*A_c*B_c - 1.50*A_c*C_c
           - 5.00*A_c**2 - 4.00*B_c**2 - 4.00*C_c**2)
    dt = (242.00 + 83.75*A_c + 30.00*B_c + 106.25*C_c + 10.00*A_c*B_c
          + 17.50*A_c*C_c + 37.75*A_c**2 + 40.25*B_c**2 + 37.75*C_c**2)

    th = th + np.random.normal(0, 0.020*np.mean(th), n_samples)
    q15 = q15 + np.random.normal(0, 0.018*np.mean(q15), n_samples)
    dt = dt + np.random.normal(0, 0.030*np.mean(dt), n_samples)

    X = np.column_stack([A, B, C])
    return X, {"Tablet Hardness (R1)": th,
               "Dissolution Q15 (R2)": q15,
               "Disintegration Time (R3)": dt}

# ---------------------------------------------------------------- MLP surrogate
def build_mlp():
    model = Sequential([
        Input(shape=(3,)),
        Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_surrogate(X, y):
    """80/20 train-test split; inputs AND target standardised (StandardScaler);
    EarlyStopping on an internal validation split; return held-out R^2 plus the
    held-out actual/predicted pairs for plotting."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    x_scaler = StandardScaler(); X_tr_s = x_scaler.fit_transform(X_tr); X_te_s = x_scaler.transform(X_te)
    y_scaler = StandardScaler(); y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

    tf.random.set_seed(SEED)
    model = build_mlp()
    es = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    model.fit(X_tr_s, y_tr_s, validation_split=0.15, epochs=300, batch_size=32,
              callbacks=[es], verbose=0)

    y_pred = y_scaler.inverse_transform(model.predict(X_te_s, verbose=0)).ravel()
    return r2_score(y_te, y_pred), y_te, y_pred

# ---------------------------------------------------------------- run + figure
def main():
    X, responses = generate_monte_carlo_batches(1000)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    print("ANN surrogate held-out R^2:")
    for ax, (name, y) in zip(axes, responses.items()):
        r2, y_te, y_pred = train_surrogate(X, y)
        print(f"  {name}: {r2:.3f}")
        ax.scatter(y_te, y_pred, s=14, alpha=0.55, edgecolor="none", color="#1f3864")
        lo, hi = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#b00020", lw=1.3)
        ax.set_xlabel("Simulated value")
        ax.set_ylabel("ANN-predicted value")
        ax.set_title(name, fontsize=11)
        ax.text(0.05, 0.90, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Figure17_ANN_parity.png", dpi=200, bbox_inches="tight")
    print("Saved Figure17_ANN_parity.png")

if __name__ == "__main__":
    main()
