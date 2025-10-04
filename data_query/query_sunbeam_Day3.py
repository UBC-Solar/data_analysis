from data_tools.collections.time_series import TimeSeries
from data_tools.query import SunbeamClient
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

client = SunbeamClient()


# We can, in one line, make a query to InfluxDB and parse 
# the data into a powerful format: the `TimeSeries` class.
battery_current_data_raw: TimeSeries = client.get_file(
    origin="soc_stage",
    event="FSGP_2024_Day_3",
    source="ingress",
    name="BatteryCurrent",
).unwrap().data


print(battery_current_data_raw[100:20000])

battery_current_direction_raw: TimeSeries = client.get_file(
    origin="soc_stage",
    event="FSGP_2024_Day_3",
    source="ingress",
    name="BatteryCurrentDirection",
).unwrap().data

battery_voltage_raw: TimeSeries = client.get_file(
    origin="soc_stage",
    event="FSGP_2024_Day_3",
    source="ingress",
    name="TotalPackVoltage",
).unwrap().data

vehicle_velocity_raw: TimeSeries = client.get_file(
    origin="soc_stage",
    event="FSGP_2024_Day_3",
    source="ingress",
    name="VehicleVelocity",
).unwrap().data

start_time = vehicle_velocity_raw.start()

# this corresponds to the first lap start time of 17:00
adjusted_time = vehicle_velocity_raw + 1061
print(f'start time {vehicle_velocity_raw.start()}')
# vehicle_velocity_raw.index_of()


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

timeseries_to_csv(motor_voltage, '../motor_validation/motor_voltage_day3.csv')
timeseries_to_csv(adjusted_current, '../motor_validation/motor_current_day3.csv')
timeseries_to_csv(vehicle_velocity, "../motor_validation/vehicle_velocity_day3.csv")

plot_time_series(battery_current_data_raw, title="Vehicle Velocity Over Time", ylabel="Speed (km/h)")


