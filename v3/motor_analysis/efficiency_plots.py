from data_tools import DBClient, TimeSeries, FSGPDayLaps
import numpy as np
import matplotlib.pyplot as plt
import datetime

data_client = DBClient()


def get_lap_speeds(start_time: datetime, end_time: datetime, client: DBClient) -> np.ndarray:

    motor_voltage: TimeSeries = client.query_time_series(start_time, end_time, "VehicleVelocity")

    motor_voltage.units = "m/s"
    motor_voltage.meta["field"] = "Speed"
    return motor_voltage

def calculate_lap_power(start_time: datetime, end_time: datetime, client: DBClient) -> TimeSeries:

    motor_voltage: TimeSeries = client.query_time_series(start_time, end_time, "BatteryVoltage")
    raw_motor_current: TimeSeries = client.query_time_series(start_time, end_time, "BatteryCurrent")
    motor_current_dir: TimeSeries = client.query_time_series(start_time, end_time, "BatteryCurrentDirection")

    # Align x-axes
    raw_motor_current, motor_voltage, motor_current_dir = TimeSeries.align(raw_motor_current, motor_voltage,
                                                                           motor_current_dir)
    # Make direction -1 or 1 instead of 1 or 0
    motor_current_sign = motor_current_dir * -2 + 1

    # Account for regen direction
    motor_current = raw_motor_current.promote(raw_motor_current * motor_current_sign)
    motor_power = motor_current.promote(motor_current * motor_voltage)
    motor_power.units = "W"
    motor_power.meta["field"] = "Motor Power (adjusted for regen)"
    return motor_power

def calculate_lap_energy(start_time: datetime, end_time: datetime, client) -> TimeSeries:
    motor_power: TimeSeries= calculate_lap_power(start_time, end_time, client)
    motor_energy = motor_power.promote(np.cumsum(motor_power) * motor_power.granularity)
    motor_energy.units = "J"
    motor_energy.meta["field"] = "Motor Energy (regen adjusted)"
    return motor_energy

if __name__ == "__main__":

    # Select which FSGP 2024 day to calculate lap efficiency for (1, 2 or 3)
    laps1 = FSGPDayLaps(1)  # Corresponds to July 16th
    laps3 = FSGPDayLaps(3)  # Corresponds to July 18th
    day_1_idx = range(laps1.get_lap_count())
    day_3_idx = range(laps3.get_lap_count())
    num_laps = len(day_1_idx) + len(day_3_idx)

    # Initialize lists of lap data
    lap_energies = []
    lap_powers = []
    lap_drivers = []
    lap_speed_variances = []
    lap_power_variances = []
    lap_speed_arrs = []
    lap_speeds = []

    # Iterate through all selected laps
    for day_laps, lap_indices in zip((laps1, laps3), (day_1_idx, day_3_idx)):
        for lap_idx in lap_indices:
            lap_num = lap_idx + 1

            lap_start = day_laps.get_start_utc(lap_num)
            lap_end = day_laps.get_finish_utc(lap_num)

            # Fill in relevant lap data
            lap_energies.append(calculate_lap_energy(lap_start, lap_end, data_client))
            lap_drivers.append(day_laps.get_lap_driver(lap_num))
            lap_powers.append(calculate_lap_power(lap_start, lap_end, data_client))
            lap_power_variances.append(np.var(lap_powers[-1].base))
            lap_speed_arrs.append(get_lap_speeds(lap_start, lap_end, data_client))
            lap_speed_variances.append(np.var(lap_speed_arrs[-1].base))
            lap_speeds.append(day_laps.get_lap_mph(lap_num))

            print(f"Processed data for day {day_laps.day} lap {lap_num}")
            print(f"{lap_start=}\n{lap_end=}")
            print(f"Total motor energy for lap {lap_num}: {lap_energies[-1][-1]}J\n"
                  f"Driver: {lap_drivers[-1]}\n"
                  f"Power Variance: {lap_power_variances[-1]}\n"
                  f"Speed Variance: {lap_speed_variances[-1]}\n"
                  f"Average Speed: {lap_speeds[-1]}mph\n")

    lap_energies = np.array([arr[-1] for arr in lap_energies])
    lap_drivers = np.array(lap_drivers)
    lap_speed_variances = np.array(lap_speed_variances)
    lap_power_variances = np.array(lap_power_variances)
    lap_speeds = np.array(lap_speeds)

    driver_colours = {
        "Alex": "red",
        "Bryan": "orange",
        "Diego": "green",

        "Phoebe": "blue"
    }

    FSGP_TRACK_LEN_M = 5_070
    lap_efficiencies = lap_energies / FSGP_TRACK_LEN_M

    # -------- PLOT SPEED VARIANCE VS EFFICIENCY --------

    for driver, colour in driver_colours.items():
        plt.scatter(lap_speed_variances[lap_drivers == driver],
                    lap_efficiencies[lap_drivers == driver],
                    c=colour,
                    label=f"Driver: {driver}")
    plt.xlabel("Lap Speed Variance")
    plt.ylabel("Lap Efficiency (J/m)")
    plt.legend()
    plt.title(f"Lap Speed Variance vs. Lap Efficiency (J/m) by Driver")
    plt.show()

    # -------- PLOT POWER VARIANCE VS EFFICIENCY --------

    for driver, colour in driver_colours.items():
        plt.scatter(lap_power_variances[lap_drivers == driver],
                    lap_efficiencies[lap_drivers == driver],
                    c=colour,
                    label=f"Driver: {driver}")
    plt.xlabel("Lap Power Variance")
    plt.ylabel("Lap Efficiency (J/m)")
    plt.legend()
    plt.title(f"Lap Power Variance vs. Lap Efficiency (J/m) by Driver")
    plt.show()

    # -------- PLOT SPEED, POWER VARIANCE AND EFFICIENCY --------

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for driver, colour in driver_colours.items():
        ax.scatter(lap_speeds[lap_drivers == driver],
                   lap_power_variances[lap_drivers == driver],
                   lap_efficiencies[lap_drivers == driver],
                   c=colour,
                   label=f"Driver: {driver}")
    ax.set_xlabel('Average Speed (mph)')
    ax.set_ylabel('Power Variance')
    ax.set_zlabel('Lap Efficiency (J/m)')
    plt.show()
    plt.title(f"Lap Efficiency vs. Lap Power Variance and Speed by Driver")
    plt.show()
