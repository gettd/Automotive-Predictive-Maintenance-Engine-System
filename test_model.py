import os
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
model = load_model("model/lstm_model.keras", compile=False)  # no need to recompile custom loss
scaler = joblib.load("model/scaler.pkl")

SEQUENCE_LENGTH = 30
TARGET_COL = "Engine Condition"
DATA_DIR = Path(SYNTHETIC_OUTPUT_TEST_DIR)
THRESHOLD = 0.1

#selected_files = sorted(DATA_DIR.glob("synthetic_timeseries_*.csv"))
#selected_files = sorted(DATA_DIR.glob("obd_21-10-2025_21-50-55.csv"))
selected_files = sorted(DATA_DIR.glob("*.csv"))

print(f"Found {len(selected_files)} files for testing:")
for i, file in enumerate(selected_files, 1):
    print(f"[{i}] {file.name}")


def create_sequences(df, sequence_length, feature_cols, target_col):
    """Create overlapping sequences"""
    X, y = [], []
    data = df[feature_cols].values
    labels = df[target_col].values

    for i in range(len(df) - sequence_length):
        seq_x = data[i:i + sequence_length]
        seq_y = labels[i + sequence_length - 1]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def first_crossing(signal, timestamps):
    """Find first index where signal crosses above threshold"""
    for i, v in enumerate(signal):
        if v >= 1:
            return timestamps[i]
    return None


def apply_locking(preds, threshold=THRESHOLD ):
    """Lock prediction to 'failure' once threshold is crossed"""
    locked = []
    lock = False
    j=0
    for i, p in enumerate(preds):
        if lock:
            locked.append(1.0)
        elif i >= 50 and p > threshold:
            j+=1
            if j>=60:          #to ignore jumps
                lock = True
                locked.append(1.0)
            else:
                locked.append(0)
        else:
            #j=0
            locked.append(0)
    return np.array(locked)


# Test each file
accuracies = []
earliness_list = []

for i, file in enumerate(selected_files, 1):
    df = pd.read_csv(file)
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    df[feature_cols] = scaler.transform(df[feature_cols])

    X_test, y_test = create_sequences(df, SEQUENCE_LENGTH, feature_cols, TARGET_COL)

    # Predict
    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred_locked = apply_locking(y_pred_probs)

    # Binary classification
    y_pred_binary = (y_pred_locked > 0.2).astype(int)
    y_test_binary = (y_test > 0.2).astype(int)
    acc = accuracy_score(y_test_binary, y_pred_binary)
    accuracies.append(acc)

    # Analysis of first crossing
    timestamps = np.arange(len(y_test))
    true_cross = first_crossing(y_test, timestamps)
    predicted_cross = first_crossing(y_pred_locked, timestamps)

    # Earliness (how much earlier prediction happens before actual failure)
    earliness = None
    if predicted_cross is not None and true_cross is not None:
        #earliness = max(0, true_cross - predicted_cross)
        earliness = true_cross - predicted_cross
        earliness_list.append(earliness)

    print("\n=================================")
    print(f"Test File {i} — {file.name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"True Failure at: {true_cross} sec")
    print(f"Predicted at: {predicted_cross} sec")
    if earliness is not None:
        print(f"Earliness (how much earlier prediction occurs): {earliness} sec")
    print("=================================\n")

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(y_test, label='Actual Condition', alpha=0.7, color="green")
    plt.plot(y_pred_probs, label='Prediction Probability', alpha=0.7, linestyle="--", color="blue")
    plt.plot(y_pred_locked, label='Locked Prediction', alpha=0.9, color="red")
    plt.title(f"Prediction Result — {file.name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Engine Condition")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = os.path.join(PLOT_RESULT_DIR, file.name.replace(".csv", "_test.png"))
    plt.savefig(output_file)
    plt.close()

print("\nTesting complete. All plots saved.")

# Summary stats
if accuracies:
    avg_acc = sum(accuracies) / len(accuracies)
    avg_earliness = np.mean(earliness_list) if earliness_list else 0
    print(f"\n===================================================")
    print(f"Average Accuracy Across All Test Files: {avg_acc:.4f}")
    print(f"Average Earliness (sec): {avg_earliness:.2f}")
    print(f"Tested {len(accuracies)} file(s).")
    print(f"===================================================\n")
else:
    print("No valid test files were processed.")
