from data_tools.collections.time_series import TimeSeries
from data_tools.query.influxdb_query import DBClient
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

#3 pm UTC 16th July


client = DBClient()

# ISO 8601-compliant times corresponding to pre-competition testing

# Day 1 -- CORRECT
start_dt = datetime(2024, 7, 16, 15, 0, 0, tzinfo=timezone.utc)
stop_dt = datetime(2024, 7, 16, 20, 39, 40, tzinfo=timezone.utc)

# Day 2 -- CORRECT


# We can, in one line, make a query to InfluxDB and parse 
# the data into a powerful format: the `TimeSeries` class.
battery_current_data_raw: TimeSeries = client.query_time_series(
    start=start_dt,
    stop=stop_dt,
    field="BatteryCurrent",
    granularity=1

)

battery_current_direction_raw: TimeSeries = client.query_time_series(
    start=start_dt,
    stop=stop_dt,
    field="BatteryCurrentDirection",
    granularity=1
)

battery_voltage_raw: TimeSeries = client.query_time_series(
    start=start_dt,
    stop=stop_dt,
    field="BatteryVoltage",
    granularity=1
)

# this is m/s
vehicle_velocity_raw: TimeSeries = client.query_time_series(
    start=start_dt,
    stop=stop_dt,
    field="VehicleVelocity",
    granularity=1
)


motor_voltage, motor_current_direction, motor_current, vehicle_velocity = TimeSeries.align(battery_voltage_raw, battery_current_direction_raw, battery_current_data_raw, vehicle_velocity_raw)


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


def plot_time_series(timeseries_instance, title="Time Series Data", ylabel="Value", xlabel="Time"):
    """
    Plots a given TimeSeries instance.

    :param timeseries_instance: A TimeSeries object containing time and values.
    :param title: Title of the plot.
    :param ylabel: Label for the Y-axis.
    :param xlabel: Label for the X-axis (default is Time).
    """
    # Extract time and values
    times = timeseries_instance.datetime_x_axis
    values = timeseries_instance

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(times, values, label=title, linewidth=2)

    # Formatting
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Rotate timestamps for better readability
    plt.xticks(rotation=45)

    # Show plot
    plt.show()

def adjust_motor_current(current_direction, current):
    
    # Multiply motor_current_direction by -2 and shift it up by 1
    adjusted_direction = current_direction * -2 + 1
    
    # Multiply adjusted current direction with motor current
    adjusted_current = adjusted_direction * current
    
    return adjusted_current


adjusted_current = adjust_motor_current(motor_current_direction, motor_current)

timeseries_to_csv(motor_voltage, '../motor_validation/motor_voltage_day1.csv')
timeseries_to_csv(adjusted_current, '../motor_validation/motor_current_day1.csv')
timeseries_to_csv(vehicle_velocity, "../motor_validation/vehicle_velocity_day1.csv")

plot_time_series(battery_current_data_raw, title="Vehicle Velocity Over Time", ylabel="Speed (km/h)")


