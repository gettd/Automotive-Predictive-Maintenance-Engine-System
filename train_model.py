import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import joblib
from path_params import *

SYNTHETIC_OUTPUT_DIR = SYNTHETIC_OUTPUT_DIR
DATA_DIR = Path(SYNTHETIC_OUTPUT_DIR)
FILE_PREFIX = "synthetic_timeseries_"
FILE_SUFFIX = ".csv"
TOTAL_FILES = 40

SEQUENCE_LENGTH = 30
TARGET_COL = "Engine Condition"
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

all_files = [DATA_DIR / f"{FILE_PREFIX}{i+1}{FILE_SUFFIX}" for i in range(TOTAL_FILES)]
train_files = all_files[:-1]
test_file = all_files[-1]

train_dfs = [pd.read_csv(f) for f in train_files]
test_df = pd.read_csv(test_file)
train_df = pd.concat(train_dfs, ignore_index=True)

feature_cols = [col for col in train_df.columns if col != TARGET_COL]

# normalize data and create sequences
scaler = MinMaxScaler()
scaler.fit(train_df[feature_cols])

for df in train_dfs:
    df[feature_cols] = scaler.transform(df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

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

#prepare train and test sequences
X_train_list, y_train_list = [], []
for df in train_dfs:
    x_seq, y_seq = create_sequences(df, SEQUENCE_LENGTH, feature_cols, TARGET_COL)
    X_train_list.append(x_seq)
    y_train_list.append(y_seq)

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)

X_test, y_test = create_sequences(test_df, SEQUENCE_LENGTH, feature_cols, TARGET_COL)

#build lstm model using keras
input_dim = X_train.shape[2]

model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(SEQUENCE_LENGTH, input_dim), dropout=0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

#train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) #if val_loss dont improve for 5 consec epochs, then stop and restore the model with lowest val_loss

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

#save model
model.save("lstm_model.keras")
joblib.dump(scaler, "scaler.pkl")
print("Model saved.")

'''
#evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")

#binary prediction
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
binary_acc = accuracy_score(y_test, y_pred)
print(f"Binary Accuracy (threshold 0.5): {binary_acc:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Actual Engine Condition', alpha=0.7)
plt.plot(y_pred_probs, label='Predicted Probability', alpha=0.7)
plt.title('Engine Condition: Actual vs Predicted Probability')
plt.xlabel('Sample Index')
plt.ylabel('Engine Condition')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Actual Engine Condition', alpha=0.7)
plt.plot(y_pred, label='Predicted (Binary)', alpha=0.7)
plt.title('Engine Condition: Actual vs Predicted (Binary)')
plt.xlabel('Sample Index')
plt.ylabel('Engine Condition')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''