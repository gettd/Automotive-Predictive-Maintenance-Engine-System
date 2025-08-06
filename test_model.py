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

os.makedirs(PLOT_RESULT_DIR, exist_ok=True)

# Load model and scaler
model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.pkl")

SEQUENCE_LENGTH = 30
TARGET_COL = "Engine Condition"
DATA_DIR = Path(SYNTHETIC_OUTPUT_TEST_DIR)
PREDICT_HORIZON = 800 #10 min threshold that model should be able to predict before failure

selected_files = sorted(DATA_DIR.glob("synthetic_timeseries_*.csv"))

print(f"Found {len(selected_files)} files for testing:")
for i, file in enumerate(selected_files, 1):
    print(f"[{i}] {file.name}")



def create_sequences(df, sequence_length, feature_cols, target_col, pdh):
    X, y, y_actual = [], [], []
    data = df[feature_cols].values
    labels = df[target_col].values

    for i in range(len(df) - sequence_length - pdh + 1):
        if i + sequence_length + pdh - 1 >= len(df): break
        seq_x = data[i:i + sequence_length]
        seq_y = labels[i + sequence_length + pdh - 1]
        actual_y = labels[i + sequence_length - 1]
        X.append(seq_x)
        y.append(seq_y)
        y_actual.append(actual_y)

    return np.array(X), np.array(y), np.array(y_actual)

#for analysis
def first_crossing(signal, timestamps, threshold=0.8):
    for i, v in enumerate(signal):
        if v > threshold:
            return timestamps[i]
    return None  # no crossing found

#locking the predicted value since we only care about the initial classification
def apply_locking(preds, threshold=0.5):
    locked = []
    lock = False
    for p in preds:
        if lock or p > threshold:
            lock = True
            locked.append(1.0)
        else:
            locked.append(p)
    return np.array(locked)

#test each files
accuracies = []
for i, file in enumerate(selected_files, 1):
    df = pd.read_csv(file)
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    df[feature_cols] = scaler.transform(df[feature_cols])

    X_test, y_test, y_actual = create_sequences(df, SEQUENCE_LENGTH, feature_cols, TARGET_COL, PREDICT_HORIZON)

    y_pred_probs = model.predict(X_test).flatten()
    y_pred_locked = apply_locking(y_pred_probs)

    # Binary classification for evaluation
    y_pred_binary = (y_pred_locked > 0.5).astype(int)
    y_test_binary = (y_test > 0.5).astype(int)
    acc = accuracy_score(y_test_binary, y_pred_binary)
    accuracies.append(acc)

    timestamps = np.arange(len(y_test))
    print("\n=================================")
    print(f"Test File {i} — {file.name}")
    print(f"Accuracy: {acc:.4f}")

    true_cross = first_crossing(y_actual, timestamps)
    expected_cross = first_crossing(y_test, timestamps)
    predicted_cross = first_crossing(y_pred_locked, timestamps)

    print(f"Expected Prediction at: {expected_cross} sec")
    print(f"Actual Condition at: {true_cross} sec")
    print(f"Predicted at: {predicted_cross} sec")
    if predicted_cross is not None and true_cross is not None:
        print(f"Predicted - Actual Condition: {predicted_cross - true_cross} sec")
    print("=================================\n")

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(y_test, label='Expected Prediction', alpha=0.7, linestyle='--', color = "red")
    plt.plot(y_actual, label='Actual Condition', alpha=0.7, color = "green")
    plt.plot(y_pred_probs, label='Raw Prediction', alpha=0.7, linestyle=':')
    plt.plot(y_pred_locked, label='Locked Prediction', alpha=0.9, color = "blue")
    plt.title(f"Prediction Result — {file.name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Engine Condition")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = os.path.join(PLOT_RESULT_DIR, file.name.replace(".csv", "_test.png"))
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