"""
TensorFlow 2.16 Multilayer Perceptron (MLP) Surrogate Engine & Monte Carlo Simulation
for Sacubitril/Valsartan Co-crystal Tablet Process Sensitivity Analysis.

Reference:
    Tonmoy, P.R., Sarkar, M.R. "In-Silico Predictive Modeling of Process Parameter
    Interactions in Sacubitril/Valsartan Co-crystal Tablet Manufacturing: A Quality
    by Design Framework for Real-Time Quality Assurance."

Methodology:
    1. Generates 1,000+ virtual batch configurations across the BBD parameter space
       using RSM model equations augmented with zero-mean Gaussian Monte Carlo noise
       scaled to published Coefficients of Variation (CV: TH 2.0%, Q15 1.8%, DT 3.0%).
    2. Trains a TensorFlow 2.16 MLP with L2 regularization and Early Stopping as a fast
       surrogate model to evaluate process robustness and compression force thresholds (>22 kN).

Environment: Python 3.10+ | TensorFlow 2.16+, NumPy, Pandas, Scikit-learn
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_monte_carlo_batches(n_samples: int = 1000) -> pd.DataFrame:
    """Generate virtual batches using RSM equations with Monte Carlo noise sampling."""
    
    # Process Parameters (CPPs)
    # A: Impeller Speed (150 - 350 rpm), B: Binder Rate (60 - 160 g/min), C: Compression Force (12 - 28 kN)
    A = np.random.uniform(150, 350, n_samples)
    B = np.random.uniform(60, 160, n_samples)
    C = np.random.uniform(12, 28, n_samples)
    
    # Coded values (-1 to +1) for RSM equations
    A_coded = (A - 250) / 100
    B_coded = (B - 110) / 50
    C_coded = (C - 18) / 6

    # Base RSM Quadratic Model Equations
    th_pred = 118.0 + 11.25*A_coded + 0.75*B_coded + 20.50*C_coded + 6.75*A_coded*B_coded - 3.62*(A_coded**2) - 4.63*(B_coded**2) - 12.62*(C_coded**2)
    q15_pred = 91.00 - 8.25*A_coded - 2.75*B_coded - 7.50*C_coded - 1.00*A_coded*B_coded - 1.50*A_coded*C_coded - 5.00*(A_coded**2) - 4.00*(B_coded**2) - 4.00*(C_coded**2)
    dt_pred = 242.00 + 83.75*A_coded + 30.00*B_coded + 106.25*C_coded + 10.00*A_coded*B_coded + 17.50*A_coded*C_coded + 37.75*(A_coded**2) + 40.25*(B_coded**2) + 37.75*(C_coded**2)

    # Adding Uncorrelated Zero-Mean Gaussian Noise based on published CVs
    th_noise = np.random.normal(0, 0.020 * np.mean(th_pred), n_samples)      # CV = 2.0%
    q15_noise = np.random.normal(0, 0.018 * np.mean(q15_pred), n_samples)    # CV = 1.8%
    dt_noise = np.random.normal(0, 0.030 * np.mean(dt_pred), n_samples)      # CV = 3.0%

    df_virtual = pd.DataFrame({
        'Impeller_Speed': A,
        'Binder_Rate': B,
        'Compression_Force': C,
        'TH': th_pred + th_noise,
        'Q15': q15_pred + q15_noise,
        'DT': dt_pred + dt_noise
    })
    
    return df_virtual

def build_and_train_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Build and train TensorFlow 2.16 MLP Surrogate Model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # Single CQA target prediction (Q15 Dissolution)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, history

def main():
    print("--- Step 1: Generating 1,000 Virtual Batches via Monte Carlo Noise Sampling ---")
    df_sim = generate_monte_carlo_batches(n_samples=1000)
    print(f"Generated Dataset Shape: {df_sim.shape}")
    print(df_sim.head())

    # Features and Target
    X = df_sim[['Impeller_Speed', 'Binder_Rate', 'Compression_Force']].values
    y = df_sim['Q15'].values  # Predicting Q15 Dissolution

    # Train/Validation Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\n--- Step 2: Training TensorFlow MLP Surrogate Model ---")
    model, history = build_and_train_mlp(X_train_scaled, y_train, X_val_scaled, y_val)

    val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f"Validation MSE Loss: {val_loss:.6f}")
    print(f"Validation MAE: {val_mae:.4f}")

    # Process Failure Threshold Evaluation (>22 kN)
    high_cf_batches = df_sim[df_sim['Compression_Force'] > 22]
    failure_rate = (high_cf_batches['Q15'] < 85).mean() * 100
    print(f"\n--- Step 3: Threshold Sensitivity Analysis ---")
    print(f"Predicted Q15 Failure Rate at Compression > 22 kN: {failure_rate:.2f}%")

if __name__ == "__main__":
    main()
