from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from RNN_Dataset import RNN_Dataset
#necessary imports
from sklearn.preprocessing import MinMaxScaler
from data_tools import query
import pandas as pd
import os
import dill


# this file will create a single dataframe to use for the RNN. Further scales the data, creates a testing/training split and makes individual sequences to furhter feed into the RNN.

def combine_dfs(telemetry_names, index_common, all_dfs):
    combined_df = pd.DataFrame(index=index_common)
    combined_df.dropna()

    for name, df in zip(telemetry_names, all_dfs):
        combined_df[name] = df

    return combined_df
# get data from sunbeam and influx.
# use sunbeam instead to save yourself a headache
def make_df(source, name):
    dfs = []

    client = query.SunbeamClient()
    for event in ["FSGP_2024_Day_1", "FSGP_2024_Day_2", "FSGP_2024_Day_3"]:
        file = client.get_file(
            origin="production",
            event=event,
            source=source,
            name=name
        ).unwrap()

        dfs.append(
            pd.DataFrame(
                data=file.data,
                index=file.data.datetime_x_axis
            )
        )

    return pd.concat(dfs).sort_index()

def make_single_df():
    #mech_brake_pressed, accel_position, speed_kph, position = get_data()
    out_dir = os.path.join("../../array_temp", "data", "control_state_fsgp_2024")

    brake_path = os.path.join(out_dir, "brake_pressed.bin")
    accel_path = os.path.join(out_dir, "acceleration.bin")
    speed_path = os.path.join(out_dir, "speed_kph.bin")

    filepaths = [brake_path, accel_path, speed_path]

    loaded_datasets = []

    for filepath in filepaths:
        with open(filepath, "rb") as f:
            data = dill.load(f)
            loaded_datasets.append(data)

    # unnpack
    mech_brake_pressed, accel_position, speed_kph = loaded_datasets
    df_mech_brake_pressed = pd.DataFrame(mech_brake_pressed, index=mech_brake_pressed.datetime_x_axis)
   # df_accel_position = pd.DataFrame(accel_position, index=accel_position.datetime_x_axis)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_accel_position = scaler.fit_transform(accel_position.reshape(-1, 1))
   # pos_df = make_df(source="localization", name="TrackIndex")
    speed_df = make_df(source="ingress", name="VehicleVelocity")
    all_dfs = [df_mech_brake_pressed, df_accel_position]
    final_df = pd.merge_asof(
        df_mech_brake_pressed.sort_index(),
        pd.DataFrame(df_accel_position, index = accel_position.datetime_x_axis),
        left_index=True,
        right_index=True,
        direction="nearest"
    )
    # dfs = pd.concat([pos_df, speed_df], axis=1)
    final_df = pd.merge_asof(
        final_df.sort_index(),
        speed_df.sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest"
    )
    final_df.columns = ["brake_pressed", "accel_position", "speed"]
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
