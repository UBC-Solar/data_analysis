from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from RNN_Dataset import RNN_Dataset
#necessary imports
from sklearn.preprocessing import MinMaxScaler
from data_tools import query
import pandas as pd
import numpy as np
import os
import dill
from data_tools import *
from localization_roc import *


#this file will create a single dataframe to use for the RNN. Further scales the data, creates a testing/training split and makes individual sequences to furhter feed into the RNN.


def combine_dfs(telemetry_names, index_common, all_dfs):
    combined_df = pd.DataFrame(index=index_common)
    combined_df.dropna()

    for name, df in zip(telemetry_names, all_dfs):
        combined_df[name] = df

    return combined_df
# get data from sunbeam and influx.
# use sunbeam instead to save yourself a headache

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
#from RNN_Dataset import RNN_Dataset
#necessary imports
from sklearn.preprocessing import MinMaxScaler
from data_tools import query
import pandas as pd
import numpy as np
import os
import dill
from data_tools import *
import control_model.localization_roc
from control_model.localization_roc import *


#this file will create a single dataframe to use for the RNN. Further scales the data, creates a testing/training split and makes individual sequences to furhter feed into the RNN.


def combine_dfs(telemetry_names, index_common, all_dfs):
    combined_df = pd.DataFrame(index=index_common)
    combined_df.dropna()

    for name, df in zip(telemetry_names, all_dfs):
        combined_df[name] = df

    return combined_df
# get data from sunbeam and influx.
# use sunbeam instead to save yourself a headache
def make_df(source, event):
    dfs = []
    files = []
    client = query.SunbeamClient()
    for name in ["VehicleVelocity", "MechBrakePressed", "AcceleratorPosition"]:
        file = client.get_file(
            origin="production",
            event=event,
            source=source,
            name=name
        ).unwrap().data
        files.append(file)


    file_pos = client.get_file(
            origin="production",
            event=event,
            source="localization",
            name="TrackIndex"
        ).unwrap().data

    files = TimeSeries.align(files[0], files[1], files[2], file_pos);
    last_idx = np.where(np.isnan(file_pos))[0][0]
    file_pos = file_pos[0:last_idx]
    files.append(file_pos)
    files = TimeSeries.align(files[0], files[1], files[2], files[3]);
    for file2 in files:
            dfs.append(
                pd.DataFrame(
                    data=file2,
                    index=file2.datetime_x_axis
                )
            )




    return pd.concat(dfs).sort_index()


def make_single_df():
   for event in ["FSGP_2024_Day_1", "FSGP_2024_Day_2", "FSGP_2024_Day_3"]:
       speed_kph, mech_brake_pressed, accel_position, position = make_df(source="ingress", event=event)


   pd.merge_asof(speed_kph.sort_index(), mech_brake_pressed)
   scaler = MinMaxScaler(feature_range=(0, 1))
  #scale acceleration position before standard scaling.
   df_accel_position = scaler.fit_transform(accel_position.reshape(-1, 1))
   radius_of_curvature = calculate_circular_track_curvature(coords(), step=2)
   calculated_roc = [radius_of_curvature[int(i) - 1] for i in position]
   calculated_roc_df = pd.DataFrame(calculated_roc).sort_index()

   all_dfs = [mech_brake_pressed, df_accel_position]
   final_df = pd.merge_asof(
        mech_brake_pressed.sort_index(),
        pd.DataFrame(df_accel_position, index = accel_position.datetime_x_axis),
        left_index=True,
        right_index=True,
        direction="nearest"
    )
   dfs = pd.concat([calculated_roc_df, speed_kph], axis=1)
   final_df = pd.merge_asof(
        final_df.sort_index(),
        dfs.sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest"
    )
   final_df.columns = ["brake_pressed", "accel_position", "speed", "ROC"]
   final_df = final_df.sort_index()
   final_df = final_df.ffill().dropna()
   return final_df


#given the raw dataframe, creates a testing / training split. Only training data is scaled.
#create sequences of given length and feed to dataloaders (tensor conversions are done via class RNN_Dataset).
# returns scaled training dataset, unscaled testing dataset, train_loader and test_loader (Dataloaders for iterating over the dataset and can return batches of samples).
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
    df_xy = df_xy.dropna(subset=state_cols + control_cols).reset_index(drop=True)

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
        batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory = True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,num_workers=0,
        shuffle=False, pin_memory = True
    )

    return train_dataset, test_dataset, train_loader, test_loader, scaler


