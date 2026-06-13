from data_tools.collections.time_series import TimeSeries
from data_tools.query.influxdb_query import DBClient
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd


client = DBClient()

# ISO 8601-compliant times corresponding to pre-competition testing

# Day 1
start_dt = datetime(2024, 7, 16, 14, 23, 57, tzinfo=timezone.utc)
stop_dt = datetime(2024, 7, 16, 21, 34, 15, tzinfo=timezone.utc)



# We can, in one line, make a query to InfluxDB and parse 
# the data into a powerful format: the `TimeSeries` class.
pack_current_data: TimeSeries = client.query_time_series(
    start=start_dt,
    stop=stop_dt,
    field="PackCurrent",
)


battery_voltage_data: TimeSeries = client.query_time_series(
    start=start_dt,
    stop=stop_dt,
    field="TotalPackVoltage",
)

voltage_aligned, current_aligned = TimeSeries.align(battery_voltage_data, pack_current_data)


def timeseries_to_csv(timeseries_instance, output_file):
    # Extract time (as datetime) and values
    times = timeseries_instance.datetime_x_axis
    values = timeseries_instance
    
    # Create a DataFrame with the time and value columns
    df = pd.DataFrame({
        'Time': times,
        'Value': values
    })
    
    # Write the DataFrame to a CSV file
    df.to_csv(output_file, index=False)


timeseries_to_csv(voltage_aligned, 'voltage.csv')
timeseries_to_csv(current_aligned, 'current.csv')


