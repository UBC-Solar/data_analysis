# necessary imports:
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
from visualization import *
from inference import *

# Constants as they must match with RNN training requirements:
state_cols = ["speed", "ROC"]  # input state features
control_cols = ["brake_pressed", "accel_position"]  # output control targets
SEQ_LEN = 150  # sequence length (15 seconds at 0.1 granularity)
STRIDE = 50  # sliding-window step between consecutive sequences
BATCH_SIZE = 64  # training batch size
EPOCHS = 320  # number of training epochs
HIDDEN_SIZE = 256  # LSTM hidden layer dimensionality
NUM_LAYERS = 2  # number of stacked LSTM layers
MODEL_PATH = "rnn_model.pt"  # default model checkpoint path


class Control_Model():
    """
    Class for loading, running inference and evaluation of the state-control RNN.
    The model works on a state to control array mapping based on a sequence-to-sequence LSTM.
    State array consists of curvature and speed as inputs and predicts corresponding driver control outputs (brake pressed, accelerator position) as a 0-1 continuous value.

    This class handles all internal preprocessing steps required for using the RNN, including:
    - scaling raw inputs with the training scaler
    - propagating LSTM hidden states across timesteps
    - unscale predicted outputs to reflect original physical units.

    :param model_path: str filepath of the loaded RNN (.pt / .pth)
    :param input_speed: np.ndarray Unscaled speed array input for model use.
    :param input_curvature: unscaled curvature array input for model use.
    :param hidden_size: Fixed hidden size for LSTM cells.
    :param num_layers: Number of stacked LSTM layers.
    :param seq_length: Constant sequence length of 15 seconds at 0.1 granularity
    :param device: Device to load RNN, CUDA by default, else resorts to CPU.



    :return output_brake_pressed: np.ndarray Unscaled brake pressed array output predicted by model.
    :return output_accel_position: np.ndarray Unscaled acceleration array output predicted by model.

    """

    state_cols = ["speed", "ROC"]  # input state features
    control_cols = ["brake_pressed", "accel_position"]  # output control features

    def __init__(
            self,
            model_path: str,
            scaler,
            input_speed: np.ndarray,
            input_curvature: np.ndarray,
            hidden_size: int = 256,
            num_layers: int = 2,
            seq_length: int = 150,
            device: torch.device = None,
    ):
        self.model_path = model_path
        self.scaler = scaler
        self.input_speed = np.asarray(input_speed, dtype=np.float32)
        self.input_curvature = np.asarray(input_curvature, dtype=np.float32)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.n_states = len(self.state_cols)  # 2 inputs
        self.n_controls = len(self.control_cols)  # 2 outputs
        self.n_total = self.n_states + self.n_controls  # 4

        self.model = self.load_model()

    def load_model(self, device: torch.device = None):
        """
        Load the RNN checkpoint from ``self.model_path`` onto the target device.
        :param device: cpu or cuda, depending on user device. Defaults to CUDA automatically when available.
        :return model: RNN model loaded for evaluation mode

        """

        model_path = self.model_path
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = RNN(
            input_size=len(state_cols),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            seq_length=SEQ_LEN,
            output_size=len(control_cols),
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"[load_model] Loaded '{model_path}' on {device}")
        return model

    def unscale_outputs(self, arr) -> np.ndarray:
        """

        Method to reverse scaled transformation applied during preprocessing.
        As the scaler was fit on both state and control arrays, individual features cannot be inverted in isolation.
        This method will reconstruct a full-width array to be inverse transformed and then extracts only the relevant subsets.

        :param arr: np.ndarray: Scaled array to be inverted of shape (n_controls)
        :return unscaled_states: np.ndarray: Unscaled state values (speed and curvature)

        """
        T = arr.shape[0]
        dummy = np.zeros((T, self.n_total), dtype=np.float32)
        dummy[:, self.n_states:] = arr  # insert at control columns

        unscaled = self.scaler.inverse_transform(dummy)
        return unscaled[:, self.n_states:]  # return only control columns

    def scale_inputs(self) -> np.ndarray:
        """
        Method to stack speed and curvature into a nparray of shape (T,2) and then standardise using the training scaler.
        Stack speed and curvature into a (T, 2) array and standardise using
        the training scaler (Sci-kit Learn's StandardScaler()).
        :return  scaled_states : np.ndarray, of shape (T, 2) that has been standardised and is ready to feed into LSTM

        """
        T = len(self.input_speed)

        # full width dummy array
        # note that scaler was fit on all columns together, so subsets cannot be inverted in isolatoin
        # build full width dummy array
        # insert subset into the column adn then transform
        dummy = np.zeros((T, self._n_total), dtype=np.float32)
        dummy[:, 0] = self.input_speed  # speed is column 0
        dummy[:, 1] = self.input_curvature  # curvature is col 1

        scaled = self.scaler.transform(dummy)
        return scaled[:, :self._n_states]  # return only the state columns

    def predict(self):
        """
        Runs the entire inference process on input state arrays and returns the unscaled predicted control values (physical units)
        Internal steps include:
            - scaling input speed and input curvature on the training scalar.
            - Inference is performed row-by-row i.e. stride is set to 1 in order to avoid scaling inconsistencies due to boundary spikes in data.
            - Predicted outputs are unscaled back to physical units.


        :return accel_position : np.ndarray,
            Predicted accelerator position in original (unscaled) units.
        :return brake_pressed  : np.ndarray,
                   Predicted brake signal in original (unscaled) units.
        """

        scaled_states = self.scale_inputs()
        n_samples = scaled_states.shape[0]  # length of array for prediction

        h, c = None, None  # # cold initialise hidden states
        scaled_pred = np.zeros((n_samples, self.n_controls), dtype=np.float32)

        with torch.no_grad():
            for i in range(n_samples):
                # LSTM requires tensor of shape (1,1,n_states) i.e. (batch=1, seq=1, features=2)
                x = (
                    torch.tensor(scaled_states[i])
                    .unsqueeze(0)  # adds batch dimension
                    .unsqueeze(0)  # adds sequence dimension
                    .to(self.device)
                )

                hidden = (h, c) if h is not None else None  # carry over hidden states, do not reinitialise every time the loop runs.
                y_pred_tensor, (h, c) = self.model(x, hidden=hidden)
                h, c = h.detach(), c.detach()  # break gradient flow
                # remove added batch and sequence dimensios.
                scaled_pred[i] = y_pred_tensor.squeeze().detach().cpu().numpy()


        # convert from standardised space back to physical units
        unscaled = self.unscale_controls(scaled_pred)

        # col 0 = brake_pressed, col 1 = accel_position (to match training order)
        brake_pressed = unscaled[:, 0]
        accel_position = unscaled[:, 1]

        return accel_position, brake_pressed




#example usage:

if __name__ == "__main__":
    #for the sake of an example usage, i'm directly loading data from my downloaded dataframe.


    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    STATE_COLS = ["speed", "ROC"]
    CONTROL_COLS = ["brake_pressed", "accel_position"]

    final_df = pd.read_csv("fsgp_2024_training_data.csv")
    final_df = final_df.set_index("Unnamed: 0")

    train_len = (0.8 * len(final_df))
    df_train = final_df.iloc[:train_len]

    scaler = StandardScaler()
    scaler.fit(df_train[STATE_COLS + CONTROL_COLS])

    speed = np.ndarray(final_df['speed'])
    curvature = np.ndarray(final_df['ROC'])
    #
    # speed     = np.random.uniform(10, 15, size=500).astype(np.float32)
    # curvature = np.random.uniform(-0.05, 0.05, size=500).astype(np.float32)

    cm = Control_Model(
        model_path      = "rnn_model (1).pth",
        scaler          = scaler,
        input_speed     = speed,
        input_curvature = curvature,
    )

    accel_position, brake_pressed = cm.predict()
    import matplotlib.pyplot as plt
    plot_control_trajectory_extended(model, speed, curvature, accel_position, brake_pressed, state_cols=STATE_COLS, control_cols=CONTROL_COLS, seq_len=SEQ_LEN, start_idx = 0)
    plt.show()

    print(f"accel_position : min={accel_position.min():.3f}  max={accel_position.max():.3f}")
    print(f"brake_pressed  : min={brake_pressed.min():.3f}   max={brake_pressed.max():.3f}")