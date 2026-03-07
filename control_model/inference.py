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
STATE_COLS   = ["position", "speed"]
CONTROL_COLS = ["brake_pressed", "accel_position"]
SEQ_LEN      = 300
STRIDE       = 100
BATCH_SIZE   = 128
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
MODEL_PATH   = "rnn_model.pth"


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



def predict_loader(
    model: RNN,
    loader: torch.utils.data.DataLoader,
    scaler: StandardScaler,
    device: torch.device = None,
):
    #returns rescaled real y values and predicted, given the input RNN and scaler.
    if device is None:
        device = next(model.parameters()).device

    all_real, all_pred = [], []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds   = model(x_batch).cpu().numpy()
            real    = y_batch.numpy()
            all_pred.append(preds)
            all_real.append(real)

    y_pred_scaled = np.concatenate(all_pred, axis=0)
    y_real_scaled = np.concatenate(all_real, axis=0)

    # perform inverse transform
    n_state   = len(STATE_COLS)
    n_control = len(CONTROL_COLS)
    n_total   = n_state + n_control

    def _unscale(arr: np.ndarray) -> np.ndarray:
        original_shape = arr.shape
        if arr.ndim == 3:
            N, T, C = arr.shape
            arr_2d = arr.reshape(-1, C)
        else:
            arr_2d = arr

        dummy = np.zeros((len(arr_2d), n_total), dtype=np.float32)
        dummy[:, n_state:] = arr_2d
        unscaled = scaler.inverse_transform(dummy)[:, n_state:]

        if original_shape.ndim if hasattr(original_shape, 'ndim') else len(original_shape) == 3:
            unscaled = unscaled.reshape(N, T, C)

        return unscaled

    # Handle both 2D and 3D outputs cleanly
    def _safe_unscale(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            N, T, C = arr.shape
            flat = arr.reshape(-1, C)
            dummy = np.zeros((len(flat), n_total), dtype=np.float32)
            dummy[:, n_state:] = flat
            unscaled_flat = scaler.inverse_transform(dummy)[:, n_state:]
            return unscaled_flat.reshape(N, T, C)
        else:
            dummy = np.zeros((len(arr), n_total), dtype=np.float32)
            dummy[:, n_state:] = arr
            return scaler.inverse_transform(dummy)[:, n_state:]

    y_real = _safe_unscale(y_real_scaled)
    y_pred = _safe_unscale(y_pred_scaled)

    # If 3D (N, seq_len, n_controls), take the last timestep for plotting/metrics
    # This gives one prediction per sequence — change to [:, 0, :] for first step
    # or reshape to (N*T, C) if you want every timestep unrolled.
    if y_pred.ndim == 3:
        print(f"[predict_loader] Model output is 3D {y_pred.shape} — "
              f"using last timestep per sequence for y_pred/y_real.")
        y_real = y_real[:, -1, :]   # (N, n_controls)
        y_pred = y_pred[:, -1, :]   # (N, n_controls)

    return y_real, y_pred


def predict_dataframe(
    model: RNN,
    df: pd.DataFrame,
    scaler: StandardScaler,
    device: torch.device = None,
    stride: int = STRIDE,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
):
    """
    Run inference directly from a raw (unscaled) DataFrame.

    The function scales the data using the provided scaler, builds sequences,
    and returns unscaled real + predicted arrays.

    Parameters
    ----------
    df       : DataFrame with at least STATE_COLS + CONTROL_COLS columns.
    scaler   : The fitted StandardScaler from training.

    Returns
    -------
    y_real, y_pred : unscaled numpy arrays of shape (N, n_controls)
    """
    if device is None:
        device = next(model.parameters()).device

    features = STATE_COLS + CONTROL_COLS
    scaled   = scaler.transform(df[features].values.astype(np.float32))
    scaled_df = pd.DataFrame(scaled, columns=features)

    dataset = RNN_Dataset(scaled_df, STATE_COLS, CONTROL_COLS,
                          seq_len=seq_len, stride=stride)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=False)

    return predict_loader(model, loader, scaler, device)


def predict_single_sequence(
    model: RNN,
    sequence: np.ndarray,          # shape (seq_len, n_states)  – UNSCALED
    scaler: StandardScaler,
    device: torch.device = None,
) -> np.ndarray:
    """
    Predict control outputs for ONE sequence (e.g. a live window).

    Parameters
    ----------
    sequence : raw (unscaled) state values, shape (seq_len, len(STATE_COLS))

    Returns
    -------
    prediction : unscaled control outputs, shape (len(CONTROL_COLS),)
    """
    if device is None:
        device = next(model.parameters()).device

    n_state   = len(STATE_COLS)
    n_control = len(CONTROL_COLS)
    n_total   = n_state + n_control

    # Scale the state part only (pad controls with 0 for the scaler)
    dummy = np.zeros((len(sequence), n_total), dtype=np.float32)
    dummy[:, :n_state] = sequence
    scaled_states = scaler.transform(dummy)[:, :n_state]

    x = torch.tensor(scaled_states, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, n_state)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy()[0]  # (n_control,)

    # Unscale prediction
    dummy_out = np.zeros((1, n_total), dtype=np.float32)
    dummy_out[0, n_state:] = pred_scaled
    unscaled = scaler.inverse_transform(dummy_out)[0, n_state:]

    return unscaled


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    control_cols: list[str] = CONTROL_COLS,
    n_samples: int = 5000,
    title_prefix: str = "RNN",
    save_path: str = None,
) -> None:
    """
    Plot unscaled real vs predicted control outputs side-by-side.

    Parameters
    ----------
    y_real, y_pred : outputs from predict_loader / predict_dataframe
    n_samples      : how many time-steps to display (keeps plots readable)
    save_path      : if given, save figure to this file instead of showing
    """
    n_controls = len(control_cols)
    xs = np.arange(min(n_samples, len(y_real)))

    fig, axes = plt.subplots(n_controls, 1,
                             figsize=(14, 4 * n_controls),
                             sharex=True)

    if n_controls == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, control_cols)):
        ax.plot(xs, y_real[:len(xs), i], label="Real",      linewidth=1.2,
                color="steelblue")
        ax.plot(xs, y_pred[:len(xs), i], label="Predicted", linewidth=1.2,
                color="tomato", linestyle="--")
        ax.set_ylabel(col, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # residual shading
        ax.fill_between(xs,
                        y_real[:len(xs), i],
                        y_pred[:len(xs), i],
                        alpha=0.15, color="red", label="Error")

    axes[-1].set_xlabel("Sequence index", fontsize=11)
    fig.suptitle(f"{title_prefix} – Real vs Predicted (unscaled)", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Figure saved to '{save_path}'")
    else:
        plt.show()


def plot_error_distribution(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    control_cols: list[str] = CONTROL_COLS,
    save_path: str = None,
) -> None:
    """Histogram of residuals (real – predicted) for each control output."""
    n_controls = len(control_cols)
    fig, axes = plt.subplots(1, n_controls,
                             figsize=(6 * n_controls, 4))
    if n_controls == 1:
        axes = [axes]

    for ax, col, i in zip(axes, control_cols, range(n_controls)):
        residuals = y_real[:, i] - y_pred[:, i]
        ax.hist(residuals, bins=60, color="steelblue", edgecolor="white",
                alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{col} – residual distribution", fontsize=11)
        ax.set_xlabel("Error (real – predicted)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Figure saved to '{save_path}'")
    else:
        plt.show()


def print_metrics(y_real: np.ndarray, y_pred: np.ndarray,
                  control_cols: list[str] = CONTROL_COLS) -> None:
    """Print MAE, RMSE and max-error per control output."""
    print("\n── Prediction Metrics (unscaled) ────────────────────────")
    for i, col in enumerate(control_cols):
        err  = y_real[:, i] - y_pred[:, i]
        mae  = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        maxe = np.max(np.abs(err))
        print(f"  {col:25s}  MAE={mae:.4f}  RMSE={rmse:.4f}  MaxErr={maxe:.4f}")
    print("─" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Convenience "run everything" function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(
    model_path: str = MODEL_PATH,
    save_fig: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end helper:
      1. Loads data & rebuilds the scaler (same pipeline as training).
      2. Loads the saved model.
      3. Runs inference on the test split.
      4. Prints metrics and shows the comparison plot.

    Returns
    -------
    y_real, y_pred  – unscaled arrays you can use for further analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[evaluate_and_plot] Loading data …")
    df = make_single_df()
    gc.collect()

    _, test_dataset, _, test_loader, scaler = make_sequence_datasets(
        df, STATE_COLS, CONTROL_COLS,
        seq_len=SEQ_LEN, stride=STRIDE, batch_size=BATCH_SIZE
    )
    del df
    gc.collect()

    model  = load_model(model_path, device)
    y_real, y_pred = predict_loader(model, test_loader, scaler, device)

    print_metrics(y_real, y_pred)
    plot_predictions(y_real, y_pred, save_path=save_fig)
    plot_error_distribution(y_real, y_pred)

    return y_real, y_pred


def get_sequence_timestamps(test_dataset, sample_idx, df_raw,
                            seq_len=SEQ_LEN, stride=STRIDE,
                            timestamp_col="timestamp"):
    """
    Get the timestamps corresponding to a sequence from test_dataset.

    Parameters
    ----------
    test_dataset : RNN_Dataset (test split)
    sample_idx   : which sequence you're inspecting
    df_raw       : the original unprocessed dataframe (with timestamp column)
    timestamp_col: name of the datetime column in df_raw

    Returns
    -------
    timestamps : pd.Series of timestamps for that sequence window
    start_row  : integer row index in df_raw where the sequence starts
    end_row    : integer row index in df_raw where the sequence ends
    """
    # ── how many sequences are in the train split? ─────────────────────────
    # test_dataset starts after the train split in df_raw.
    # make_sequence_datasets does an 80/20 split on df_raw rows first,
    # so the test portion starts at this row:
    n_total = len(df_raw)
    train_rows = int(n_total * 0.8)  # must match your split ratio

    # within the test portion, sequence i starts at row: i * stride
    seq_start_in_test = sample_idx * stride
    start_row = train_rows + seq_start_in_test
    end_row = start_row + seq_len

    if end_row > n_total:
        raise IndexError(
            f"sample_idx {sample_idx} goes out of bounds "
            f"(start_row={start_row}, df length={n_total})"
        )

    timestamps = df_raw[timestamp_col].iloc[start_row:end_row].reset_index(drop=True)

    print(f"Sequence {sample_idx}")
    print(f"  df rows  : {start_row} → {end_row}")
    print(f"  start    : {timestamps.iloc[0]}")
    print(f"  end      : {timestamps.iloc[-1]}")
    print(f"  duration : {timestamps.iloc[-1] - timestamps.iloc[0]}")

    return timestamps, start_row, end_row



import matplotlib.pyplot as plt
import numpy as np
def plot_control_trajectory(model, test_dataset, scaler, state_cols, control_cols, sample_idx):
    """
    sample_idx: int or list of ints
    """
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

    # ── Plot states ───────────────────────────────────────────────────────────
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