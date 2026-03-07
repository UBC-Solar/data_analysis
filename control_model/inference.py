"""
rnn_inference.py
================
Inference and visualisation utilities for the trained RNN model.

Usage examples
--------------
# 1. Quick plot – compare real vs predicted on the test split
python rnn_inference.py --mode plot

# 2. Run inference on a raw dataframe and get unscaled predictions back
python rnn_inference.py --mode predict

# 3. Step-by-step single-sequence prediction (e.g. live / streaming use)
python rnn_inference.py --mode single
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ── project imports (adjust paths if needed) ──────────────────────────────────
from RNN import RNN
from RNN_Dataset import RNN_Dataset
from DataPreprocessing import make_single_df, make_sequence_datasets

# ── constants – must match training ──────────────────────────────────────────
STATE_COLS   = ["position", "speed"]
CONTROL_COLS = ["brake_pressed", "accel_position"]
SEQ_LEN      = 600
STRIDE       = 100
BATCH_SIZE   = 128
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
MODEL_PATH   = "rnn_model.pth"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH, device: torch.device = None) -> RNN:
    """Load a saved RNN from its state-dict file."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Core prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_loader(
    model: RNN,
    loader: torch.utils.data.DataLoader,
    scaler: StandardScaler,
    device: torch.device = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over an entire DataLoader.

    Returns
    -------
    y_real : np.ndarray  shape (N, n_controls)  – unscaled ground-truth
    y_pred : np.ndarray  shape (N, n_controls)  – unscaled predictions
    """
    if device is None:
        device = next(model.parameters()).device

    all_real, all_pred = [], []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds   = model(x_batch).cpu().numpy()   # (B, n_controls) scaled
            real    = y_batch.numpy()                # (B, n_controls) scaled
            all_pred.append(preds)
            all_real.append(real)

    y_pred_scaled = np.concatenate(all_pred, axis=0)
    y_real_scaled = np.concatenate(all_real, axis=0)

    # ── inverse-transform ────────────────────────────────────────────────────
    # The scaler was fitted on [STATE_COLS + CONTROL_COLS].
    # Controls occupy the last len(CONTROL_COLS) columns.
    n_state   = len(STATE_COLS)
    n_control = len(CONTROL_COLS)
    n_total   = n_state + n_control

    def _unscale(arr: np.ndarray) -> np.ndarray:
        """
        Inverse-transform control outputs regardless of shape:
          - 2D (N, n_controls)          → return (N, n_controls)
          - 3D (N, seq_len, n_controls) → flatten, unscale, reshape back
        """
        original_shape = arr.shape
        if arr.ndim == 3:
            # (N, T, C) → (N*T, C)
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
) -> tuple[np.ndarray, np.ndarray]:
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
    n_samples: int = 500,
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
                        alpha=0.15, color="orange", label="Error")

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


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNN inference & visualisation")
    parser.add_argument("--mode", choices=["plot", "predict", "single"],
                        default="plot",
                        help="plot  – evaluate on test split and show plots\n"
                             "predict – return predictions from a raw df\n"
                             "single  – demo of single-sequence inference")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to saved model state-dict (.pth)")
    parser.add_argument("--save_fig", default=None,
                        help="Optional path to save the comparison figure")
    args = parser.parse_args()

    if args.mode == "plot":
        evaluate_and_plot(model_path=args.model, save_fig=args.save_fig)

    elif args.mode == "predict":
        # Example: load your own df here and call predict_dataframe
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df     = make_single_df()
        _, _, _, _, scaler = make_sequence_datasets(
            df, STATE_COLS, CONTROL_COLS,
            seq_len=SEQ_LEN, stride=STRIDE, batch_size=BATCH_SIZE
        )
        model  = load_model(args.model, device)
        y_real, y_pred = predict_dataframe(model, df, scaler, device)
        print_metrics(y_real, y_pred)
        plot_predictions(y_real, y_pred, save_path=args.save_fig)

    elif args.mode == "single":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df     = make_single_df()
        _, _, _, _, scaler = make_sequence_datasets(
            df, STATE_COLS, CONTROL_COLS,
            seq_len=SEQ_LEN, stride=STRIDE, batch_size=BATCH_SIZE
        )
        model = load_model(args.model, device)

        # Grab one raw window as a demo
        raw_window = df[STATE_COLS].values[:SEQ_LEN].astype(np.float32)
        pred = predict_single_sequence(model, raw_window, scaler, device)
        print(f"\n[single] Predicted controls for first {SEQ_LEN} steps:")
        for col, val in zip(CONTROL_COLS, pred):
            print(f"  {col}: {val:.4f}")