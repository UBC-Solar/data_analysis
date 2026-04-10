import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from RNN import RNN
from RNN_Dataset import RNN_Dataset
from DataPreprocessing import make_single_df, make_sequence_datasets
from DataPreprocessing import *

#constants
STATE_COLS   = ["speed", "ROC"] #states
CONTROL_COLS = ["brake_pressed", "accel_position"]  #controls
SEQ_LEN      = 150 #30 seconds
STRIDE       = 50 #sliding window change between consecutive sequences, reduces overlapping
BATCH_SIZE   = 64
EPOCHS       = 320
HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
MODEL_PATH = "rnn_model.pt"


"""
- the purpose of this class is to load the RNN model from its path onto the device. 
- Also includes 2 visualization functions for single and multisequence comparisons. 
"""




def load_model(model_path: str = MODEL_PATH, device: torch.device = None):
    """
    Load the RNN checkpoint from ``self.model_path`` onto the target device.
    :param device: cpu or cuda, depending on user device. Defaults to CUDA automatically when available.
    :return: model: RNN model loaded for evaluation mode

    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RNN(
        input_size=len(STATE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        seq_length=SEQ_LEN,
        output_size=len(CONTROL_COLS),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[load_model] Loaded '{model_path}' on {device}")
    return model

def plot_control_trajectory_extended(model, df_test_scaled, scaler, state_cols, control_cols, seq_len, start_idx):
    model.eval() #eval model and disable dropout layer
    device = next(model.parameters()).device #send tensors to relevatn device (cpu/gpu)

    n_states   = len(state_cols)
    n_controls = len(control_cols)
    cols       = state_cols + control_cols

    h, c = None, None #initialise hidden states
    all_states, all_y_true, all_y_pred = [], [], []

    for i in range(start_idx, start_idx + seq_len): #loop over consecutive timestamps
        row = df_test_scaled[cols].iloc[i].values.astype(np.float32)

        state_vals   = row[:n_states]
        control_vals = row[n_states:]
        #split row into state and control
        #convert to tensors
        #LSTM expects shape [1,1,states] adding batch and sequence dimension
        x_tensor = torch.tensor(state_vals).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad(): #disable gradient calculations at inference.
            hidden = (h, c) if h is not None else None #carry over hidden state from previous
            y_pred_tensor, (h, c) = model(x_tensor, hidden=hidden) #run the model and get back predicted controls (tensor +hideen states)
            h, c = h.detach(), c.detach() #break gradient flow
            y_pred = y_pred_tensor.squeeze().cpu().numpy() #remove batch/seq dimension to convert back to numpy

        all_states.append(state_vals)
        all_y_true.append(control_vals)
        all_y_pred.append(y_pred)

    # unscale
    #note that scaler was fit on all columns together, so subsets cannot be inverted in isolatoin
    #build full width dummy array
    #insert subset into the column adn then inverse transform

    def unscale(arr, col_slice):
        dummy = np.zeros((arr.shape[0], n_states + n_controls))
        dummy[:, col_slice] = arr
        return scaler.inverse_transform(dummy)[:, col_slice]

    all_states = unscale(np.array(all_states), slice(None, n_states))
    all_y_true = unscale(np.array(all_y_true), slice(n_states, None))
    all_y_pred = unscale(np.array(all_y_pred).reshape(-1, n_controls), slice(n_states, None)) #reshape after squeeze()


    time_axis  = np.arange(seq_len) * 0.1

    n_rows = n_controls + n_states
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True)

    for i, col in enumerate(control_cols):
        axes[i].plot(time_axis, all_y_true[:, i], 'g-',  label='Actual',    linewidth=1)
        axes[i].plot(time_axis, all_y_pred[:, i], 'r--', label='Predicted', linewidth=1)
        axes[i].set_ylabel(col)
        axes[i].legend()
        axes[i].grid(True)

    for i, col in enumerate(state_cols):
        axes[n_controls + i].plot(time_axis, all_states[:, i], 'b-', label=col, linewidth=1)
        axes[n_controls + i].set_ylabel(col)
        axes[n_controls + i].legend()
        axes[n_controls + i].grid(True)

    axes[-1].set_xlabel("Time (seconds)")
    plt.suptitle(f"Control Trajectory  {seq_len/10} consecutive seconds")
    plt.tight_layout()
    plt.show()





def plot_control_trajectory(model, test_dataset, scaler, state_cols, control_cols, sample_idx):
    # normalize sample_idx to always be a list
    if isinstance(sample_idx, int):
        sample_idx = [sample_idx]

    model.eval()
    device = next(model.parameters()).device
    n_states   = len(state_cols)
    n_controls = len(control_cols)
    seq_len    = 300

    def unscale_states(arr):
        state_mean = scaler.mean_[:n_states]
        state_std  = scaler.scale_[:n_states]
        return arr * state_std + state_mean

    def unscale_controls(arr):
        dummy = np.zeros((seq_len, n_states + n_controls))
        dummy[:, n_states:] = arr
        return scaler.inverse_transform(dummy)[:, n_states:]

    time_controls = np.arange(seq_len) * 0.1


    results = []
    for idx in sample_idx:
        x_input, y_target = test_dataset[idx]

        with torch.no_grad():
            x_tensor = x_input.to(device).unsqueeze(0)
            y_pred   = model(x_tensor).squeeze(0).cpu().numpy()

        x_np     = x_input.numpy()
        y_target = y_target.numpy()

        results.append({
            "idx":       idx,
            "x_np":      x_np,
            "y_target":  unscale_controls(y_target),
            "y_pred":    unscale_controls(y_pred),
            "x_unscaled": unscale_states(x_np),
        })

    n_samples      = len(results)
    time_states    = np.arange(results[0]["x_np"].shape[0]) * 0.1

#plot controls
    fig, axes = plt.subplots(
        n_controls, n_samples,
        figsize=(8 * n_samples, 4 * n_controls),
        sharex=True, sharey="row",
        squeeze=False,
    )

    for col_j, r in enumerate(results):
        for row_i, col_name in enumerate(control_cols):
            ax = axes[row_i, col_j]
            ax.plot(time_controls, r["y_target"][:, row_i], "g-",  label="Actual (Driver)")
            ax.plot(time_controls, r["y_pred"][:,   row_i], "r--", label="Predicted (RNN)")
            ax.set_ylabel(col_name)
            ax.grid(True)
            if row_i == 0:
                ax.set_title(f"Sample {r['idx']}")
            if row_i == n_controls - 1:
                ax.set_xlabel("Time (seconds)")
            if col_j == 0:
                ax.legend()

    fig.suptitle("Control Trajectories", y=1.01)
    plt.tight_layout()
    plt.show()


    fig2, axes2 = plt.subplots(
        n_states, n_samples,
        figsize=(8 * n_samples, 4 * n_states),
        sharex=True, sharey="row",
        squeeze=False,
    )

    for col_j, r in enumerate(results):
        for row_i, col_name in enumerate(state_cols):
            ax = axes2[row_i, col_j]
            ax.plot(time_states, r["x_unscaled"][:, row_i], "b-", label=col_name)
            ax.set_ylabel(col_name)
            ax.grid(True)
            if row_i == 0:
                ax.set_title(f"Sample {r['idx']}")
            if row_i == n_states - 1:
                ax.set_xlabel("Time (seconds)")
            if col_j == 0:
                ax.legend()

    fig2.suptitle("Input State Sequences", y=1.01)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df = make_single_df()
        train_dataset, test_dataset, train_loader, test_loader, scaler = make_sequence_datasets(
            df, STATE_COLS, CONTROL_COLS,
            seq_len=SEQ_LEN, stride=STRIDE, batch_size=BATCH_SIZE
        )
        model = load_model("rnn_model (1).pth", device)
        #evaluate_and_plot(model_path=args.model, save_fig=args.save_fig)
        plot_control_trajectory(model, test_dataset, scaler, STATE_COLS, CONTROL_COLS, [3820, 3821, 3822])