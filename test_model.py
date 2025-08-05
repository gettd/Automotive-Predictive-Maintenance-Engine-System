import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import accuracy_score
from path_params import *

os.makedirs(PLOT_TEST_DIR, exist_ok=True)

# Load model and scaler
model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.pkl")

SEQUENCE_LENGTH = 30
TARGET_COL = "Engine Condition"
DATA_DIR = Path(SYNTHETIC_OUTPUT_TEST_DIR)

selected_files = sorted(DATA_DIR.glob("synthetic_timeseries_*.csv"))

print(f"Found {len(selected_files)} files for testing:")
for i, file in enumerate(selected_files, 1):
    print(f"[{i}] {file.name}")


def create_sequences(df, sequence_length, feature_cols, target_col):
    X, y = [], []
    data = df[feature_cols].values
    labels = df[target_col].values

    for i in range(len(df) - sequence_length):
        seq_x = data[i:i + sequence_length]
        seq_y = labels[i + sequence_length - 1]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

#test each files
accuracies = []
for i, file in enumerate(selected_files, 1):
    df = pd.read_csv(file)

    feature_cols = [col for col in df.columns if col != TARGET_COL]
    df[feature_cols] = scaler.transform(df[feature_cols])

    X_test, y_test = create_sequences(df, SEQUENCE_LENGTH, feature_cols, TARGET_COL)

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print("\n=================================")
    print(f"\n Test File {i} — {file.name}")
    print(f"Accuracy: {acc:.4f}")
    print("\n=================================")

    plt.figure(figsize=(12, 4))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred_probs, label='Predicted Probability', alpha=0.7)
    plt.title(f"Prediction  {file.name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Engine Condition")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = os.path.join(PLOT_TEST_DIR, file.name.replace(".csv", "_test.png"))
    plt.savefig(output_file)
    plt.close()

print("\nTesting complete. All plots saved.")

if accuracies:
    avg_acc = sum(accuracies) / len(accuracies)
    print(f"\n===================================================")
    print(f"Average Accuracy Across All Test Files: {avg_acc:.4f}")
    print(f"Tested {len(accuracies)} file(s).")
    print(f"\n===================================================")
else:
    print("No valid test files were processed.")