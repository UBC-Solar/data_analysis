from data_tools import InfluxClient, TimeSeries
from fsgp_2024_laps import FSGPDayLaps
import matplotlib.pyplot as plt
import numpy as np


def get_energy_power(day, lap):
    laps = FSGPDayLaps(day)
    start_utc = laps.get_start_utc(lap)
    finish_utc = laps.get_finish_utc(lap)

    client = InfluxClient()
    motor_voltage: TimeSeries = client.query_time_series(start_utc, finish_utc, "BatteryVoltage", units="V")
    raw_motor_current: TimeSeries = client.query_time_series(start_utc, finish_utc, "BatteryCurrent", units="A")
    motor_current_dir: TimeSeries = client.query_time_series(start_utc, finish_utc, "BatteryCurrentDirection")

    # Align x-axes
    raw_motor_current, motor_voltage, motor_current_dir = TimeSeries.align(
        raw_motor_current, motor_voltage, motor_current_dir
    )
    # Make direction -1 or 1 instead of 1 or 0
    motor_current_sign = motor_current_dir * -2 + 1

    # Account for regen direction
    motor_current = raw_motor_current.promote(raw_motor_current * motor_current_sign)
    motor_power: TimeSeries = motor_current.promote(motor_current * motor_voltage)
    motor_energy: TimeSeries = motor_power.promote(np.cumsum(motor_power) * motor_power.granularity)

    motor_power.units = "W"
    motor_energy.units = "J"

    return motor_power, motor_energy


if __name__ == "__main__":
    laps_3 = FSGPDayLaps(day=3)
    num_laps = laps_3.get_lap_count()

    lap_energies = np.zeros(num_laps)
    lap_speeds = np.zeros(num_laps)
    for lap_idx in range(num_laps):
        lap_num = lap_idx + 1
        try:
            motor_power, motor_energy = get_energy_power(3, lap_num)
        except ValueError:
            print(f"Warning: data from {lap_num=} is not in Influx")
        lap_energies[lap_idx] = motor_energy[-1] - motor_energy[0]
        lap_speeds[lap_idx] = laps_3.get_lap_mph(lap_num)

    # Create a figure and axis object
    fig, ax1 = plt.subplots()

    # Plot the first set of data (lap_energies) on the primary y-axis
    ax1.plot(lap_energies, color='blue', label='Lap Energies')
    ax1.set_xlabel('Lap Number')
    ax1.set_ylabel('Energy (Joules)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis, sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(lap_speeds, color='red', label='Lap Speeds')
    ax2.set_ylabel('Speed (mi/h)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Show the plot
    plt.title('Lap Energies and Speeds')
    plt.show()
