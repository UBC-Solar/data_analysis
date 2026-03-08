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

#constants
STATE_COLS   = ["speed"]
CONTROL_COLS = ["brake_pressed", "accel_position"]
SEQ_LEN      = 300
STRIDE       = 100
BATCH_SIZE   = 128
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
MODEL_PATH   = "rnn_model.pth"


# the purpose of this class is majorly to evaluate and visualize


def load_model(model_path: str = MODEL_PATH, device: torch.device = None):
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


import matplotlib.pyplot as plt
import numpy as np
def plot_control_trajectory(model, test_dataset, scaler, state_cols, control_cols, sample_idx):
    # Normalize sample_idx to always be a list
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

    # ── Collect predictions for all samples ──────────────────────────────────
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

    # ── Plot controls ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_controls, n_samples,
        figsize=(8 * n_samples, 4 * n_controls),
        sharex=True, sharey="row",
        squeeze=False,          # always 2-D array of axes
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
#
# def plot_control_trajectory(model, test_dataset, scaler, state_cols, control_cols, sample_idx):
#     model.eval()
#     x_input, y_target = test_dataset[sample_idx]
#
# # x_input should be [seq_len, n_states]
#     print(x_input.shape)
#
#     x_np = x_input.numpy()  # [seq_len, n_states]
#
#     # derive time axes from actual array shapes, not seq_len variable
#     time_controls = np.arange(y_target.shape[0]) * 0.1
#     time_states   = np.arange(x_np.shape[0]) * 0.1
#     x_input, y_target = test_dataset[sample_idx]
#
#     with torch.no_grad():
#         device = next(model.parameters()).device
#         x_tensor = x_input.to(device).unsqueeze(0)
#         y_pred = model(x_tensor).squeeze(0).cpu().numpy()
#
#     y_target = y_target.numpy()
#     x_np = x_input.numpy()  # [seq_len, n_states]
#
#     # inverse transform controls
#     # scaler was fit on state_cols + control_cols so controls start at index n_states
#     n_states = len(state_cols)
#     n_controls = len(control_cols)
#     seq_len = 300
#     def unscale_states(arr):
#         state_mean = scaler.mean_[:n_states]
#         state_std  = scaler.scale_[:n_states]
#         return arr * state_std + state_mean
#     def unscale_controls(arr):
#         dummy = np.zeros((seq_len, n_states + n_controls))
#         dummy[:, n_states:] = arr
#         return scaler.inverse_transform(dummy)[:, n_states:]
#
#     y_target_unscaled = unscale_controls(y_target)
#     y_pred_unscaled   = unscale_controls(y_pred)
#     x_unscaled        = unscale_states(x_np)
#
#     time_controls = np.arange(seq_len) * 0.1
#     time_states   = np.arange(x_np.shape[0]) * 0.1
#
#     # plot controls
#     fig, axes = plt.subplots(n_controls, 1, figsize=(10, 4 * n_controls), sharex=True)
#     if n_controls == 1:
#         axes = [axes]
#
#     for i, col in enumerate(control_cols):
#         axes[i].plot(time_controls, y_target_unscaled[:, i], 'g-', label='Actual (Driver)')
#         axes[i].plot(time_controls, y_pred_unscaled[:, i], 'r--', label='Predicted (RNN)')
#         axes[i].set_ylabel(col)
#         axes[i].legend()
#         axes[i].grid(True)
#
#     axes[-1].set_xlabel("Time (seconds)")
#     plt.suptitle(f"Control Trajectory — Sample {sample_idx}")
#     plt.tight_layout()
#
#     # plot states
#     fig2, axes2 = plt.subplots(n_states, 1, figsize=(10, 4 * n_states), sharex=True)
#     if n_states == 1:
#         axes2 = [axes2]
#     print(x_np.shape)        # should be [seq_len, n_states]
#     print(x_unscaled.shape)  # should match
#     print(time_states.shape)
#     for i, col in enumerate(state_cols):
#         axes2[i].plot(time_states, x_unscaled[:, i], 'b-', label=col)
#         axes2[i].set_ylabel(col)
#         axes2[i].legend()
#         axes2[i].grid(True)
#
#     axes2[-1].set_xlabel("Time (seconds)")
#     plt.suptitle(f"Input State Sequence — Sample {sample_idx}")
#     plt.tight_layout()
#     plt.show()



if __name__ == "__main__":

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df = make_single_df()
        train_dataset, test_dataset, train_loader, test_loader, scaler = make_sequence_datasets(
            df, STATE_COLS, CONTROL_COLS,
            seq_len=SEQ_LEN, stride=STRIDE, batch_size=BATCH_SIZE
        )
        model = load_model("rnn_model.pth", device)
        #evaluate_and_plot(model_path=args.model, save_fig=args.save_fig)
        plot_control_trajectory(model, test_dataset, scaler, STATE_COLS, CONTROL_COLS, [3820, 3821, 3822])