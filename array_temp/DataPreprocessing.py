import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from RNN_Dataset import RNN_Dataset


def make_sequence_datasets(
    df_xy,
    state_cols,
    control_cols,
    seq_len,
    stride=100,
    train_frac=0.8,
    batch_size=64,
):


    cols_to_scale = state_cols + control_cols

    # Train/test split (time-series safe)
    n_total = len(df_xy)
    train_len = int(train_frac * n_total)

    df_train_raw = df_xy.iloc[:train_len].reset_index(drop=True)
    df_test_raw  = df_xy.iloc[train_len:].reset_index(drop=True)

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    scaler.fit(df_train_raw[cols_to_scale])

    # Apply scaling
    df_train = df_train_raw.copy()
    df_test  = df_test_raw.copy()

    df_train[cols_to_scale] = scaler.transform(df_train_raw[cols_to_scale])
    df_test[cols_to_scale]  = scaler.transform(df_test_raw[cols_to_scale])

    # Create datasets
    train_dataset = RNN_Dataset(
        df_train,
        state_cols,
        control_cols,
        seq_len,
        stride
    )

    test_dataset = RNN_Dataset(
        df_test,
        state_cols,
        control_cols,
        seq_len,
        stride
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, test_dataset, train_loader, test_loader, scaler
