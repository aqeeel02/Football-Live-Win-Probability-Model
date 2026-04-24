"""Model builders and sequence utilities."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Masking
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor


def build_lstm(timesteps, num_features):

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(timesteps, num_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dense(2, activation="softplus")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Poisson(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    return model, early_stopping

def build_random_forest():

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    return model

def build_gru(timesteps, num_features):

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(timesteps, num_features)),
        GRU(64, return_sequences=True),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dense(2, activation="softplus")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Poisson(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    return model, early_stopping

def make_early_stopping(patience: int = 5) -> EarlyStopping:
    """Default early stopping callback."""
    return EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)



def split_and_scale(X_padded, y_padded, mask_padded, match_ids, random_state: int = 42):
    """Create match-level train/validation/test splits and scale features."""
    train_ids, temp_ids = train_test_split(match_ids, test_size=0.30, random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=random_state)

    def filter_by_ids(ids):
        keep = [i for i, mid in enumerate(match_ids) if mid in set(ids)]
        return X_padded[keep], y_padded[keep], mask_padded[keep]

    X_train, y_train, mask_train = filter_by_ids(train_ids)
    X_val, y_val, mask_val = filter_by_ids(val_ids)
    X_test, y_test, mask_test = filter_by_ids(test_ids)

    n_train, timesteps, n_features = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    train_real = mask_train.reshape(-1) == 1
    scaler.fit(X_train.reshape(-1, n_features)[train_real])

    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(n_val, timesteps, n_features)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(n_test, timesteps, n_features)

    X_train_scaled[mask_train == 0] = 0
    X_val_scaled[mask_val == 0] = 0
    X_test_scaled[mask_test == 0] = 0

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "mask_train": mask_train,
        "mask_val": mask_val,
        "mask_test": mask_test,
        "scaler": scaler,
    }
