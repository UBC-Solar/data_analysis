#helper class to clean data from influx and organise further




from os import rename
import pandas
from sqlalchemy.testing.util import total_size
from data_tools import query
from data_tools.collections import TimeSeries
from datetime import datetime, date, time, timezone
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill
import os
import pytz
from datetime import datetime, time, date

def data_preprocessing():

    def __init__(self, influx):
        self.influx = influx

    # #queries brake_pressed, accelerator position and speed
    # def query_data(self, db, start_time, stop_time):
    #     utc_offset_h = 7
    #     start_utc = time(start_time)
    #     stop_utc = time(stop_time)
    #
    #     date_start = date(2024, 7, 14)
    #     date_stop = date(2024, 7, 16)
    #
    #     vancouver = pytz.timezone("America/Vancouver")
    #
    #     start_local = vancouver.localize(datetime.combine(date_start, start_utc))
    #     stop_local = vancouver.localize(datetime.combine(date_stop, stop_utc))
    #
    #     start_time = start_local.astimezone(pytz.utc)
    #     stop_time = stop_local.astimezone(pytz.utc)
    #
    #     client = query.DBClient()
    #     mech_brake_pressed: TimeSeries = client.query_time_series(start_time, stop_time, field="MechBrakePressed")
    #     accel_position: TimeSeries = client.query_time_series(start_time, stop_time, field="AcceleratorPosition")
    #     speed_kph: TimeSeries = client.query_time_series(start_time, stop_time, "VehicleVelocity")


    def combine_dfs(self, telemetry_names, index_common, all_dfs):
        combined_df = pd.DataFrame(index=index_common)

        for name, df in zip(telemetry_names, all_dfs):
            #df_interp = self.resample(df, index_common)
            combined_df[name] = pd.to_numeric(df).values

        return combined_df

