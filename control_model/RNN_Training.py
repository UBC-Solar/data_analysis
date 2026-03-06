#necessary imports
import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from RNN_Dataset import RNN_Dataset
from RNN import RNN
from DataPreprocessing import *
from data_tools import query
from data_tools.collections import TimeSeries
import matplotlib.pyplot as plt
import pandas as pd
import dill
import os
import pytz
from datetime import datetime, time, date


#create training loop
def train_model(model, train_loader, test_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss() #mean squared error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train_losses, test_losses = [], []
    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
    #check inputs before forward pass
            if torch.isnan(x_batch).any():
                print("nan in inputs, skip batch")
                continue
            optimizer.zero_grad() #reset gradients of all parameters to zero
            outputs = model(x_batch)
            loss    = criterion(outputs, y_batch)

            if torch.isnan(loss):
                print(f"  [warn] NaN loss at epoch {epoch+1}, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # evaluate
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)
                test_loss  += criterion(predictions, y_batch).item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        # Free memory each epoch
        gc.collect()

    return train_losses, test_losses


#training script.
if __name__ == "__main__":

    # requirements
    STATE_COLS   = ["position", "speed"] #states
    CONTROL_COLS = ["brake_pressed", "accel_position"]  #controls
    SEQ_LEN      = 600 #one minute
    STRIDE       = 100 #sliding window change between consecutive sequences, reduces overlapping
    BATCH_SIZE   = 128
    EPOCHS       = 30
    HIDDEN_SIZE  = 64
    NUM_LAYERS   = 2

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #load required data

    df = make_single_df()
    gc.collect()

    print("Building datasets...")
    train_dataset, test_dataset, train_loader, test_loader, scaler = make_sequence_datasets(
        df, STATE_COLS, CONTROL_COLS,
        seq_len=SEQ_LEN, stride=STRIDE, batch_size=BATCH_SIZE
    )

    # Free original dataframe — no longer needed
    del df
    gc.collect()

    print(f"Train sequences : {len(train_dataset)}")
    print(f"Test  sequences : {len(test_dataset)}")
    print(f"Batches / epoch : {len(train_loader)}")

    model = RNN(
        input_size=len(STATE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, seq_length=SEQ_LEN,output_size=len(CONTROL_COLS)
    ).to(device)

    train_losses, test_losses = train_model(
        model, train_loader, test_loader, epochs=EPOCHS
    )

    # Save model
    torch.save(model.state_dict(), "rnn_model.pth")
    print("Model saved to rnn_model.pth")
