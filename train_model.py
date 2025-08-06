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

DATA_DIR = Path(SYNTHETIC_OUTPUT_TRAIN_DIR)
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
    tf.random.set_seed(seed)

set_seed()

all_files = sorted(DATA_DIR.glob(f"{FILE_PREFIX}*{FILE_SUFFIX}"))
train_dfs = [pd.read_csv(f) for f in all_files]
train_df = pd.concat(train_dfs, ignore_index=True)

feature_cols = [col for col in train_df.columns if col != TARGET_COL]

# normalize data and create sequences
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

PREDICT_HORIZON = 1200 #predict next 20 min(max len of possible early sign)
def create_sequences(df, sequence_length, feature_cols, target_col, pdh):
    X, y = [], []
    data = df[feature_cols].values
    labels = df[target_col].values

    for i in range(len(df) - sequence_length - pdh + 1):
        if i + sequence_length + pdh - 1 >= len(df): break
        seq_x = data[i:i + sequence_length]
        seq_y = labels[i + sequence_length + pdh - 1]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


#prepare train and test sequences
X_train, y_train = create_sequences(train_df, SEQUENCE_LENGTH, feature_cols, TARGET_COL, PREDICT_HORIZON)


#build lstm model
input_dim = X_train.shape[2]
model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(SEQUENCE_LENGTH, input_dim), dropout=0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

#save model and scaler
model.save("model/lstm_model.keras")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler saved.")