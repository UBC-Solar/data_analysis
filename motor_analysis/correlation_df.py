from data_tools import TimeSeries, DBClient, FSGPDayLaps
from typing import Callable
import numpy as np
import pandas as pd
import datetime

def lap_current_ts(start_time: datetime, end_time: datetime, client: DBClient) -> TimeSeries:
    raw_motor_current: TimeSeries = client.query_time_series(start_time, end_time, "BatteryCurrent")
    motor_current_dir: TimeSeries = client.query_time_series(start_time, end_time, "BatteryCurrentDirection")

    # Align x-axes
    raw_motor_current, motor_current_dir = TimeSeries.align(raw_motor_current, motor_current_dir)
    # Make direction -1 or 1 instead of 1 or 0
    motor_current_sign = motor_current_dir * -2 + 1

    # Account for regen direction
    motor_current = raw_motor_current.promote(raw_motor_current * motor_current_sign)
    motor_current.units = "A"
    motor_current.meta["field"] = "Motor Current (adjusted for regen)"
    return motor_current

def lap_power_ts(start_time: datetime, end_time: datetime, client: DBClient) -> TimeSeries:
    motor_voltage: TimeSeries = client.query_time_series(start_time, end_time, "BatteryVoltage")
    motor_current = lap_current_ts(start_time, end_time, client)

    # Align x-axes
    motor_voltage, motor_current = TimeSeries.align(motor_voltage, motor_current)

    motor_power = motor_current.promote(motor_current * motor_voltage)
    motor_power.units = "W"
    motor_power.meta["field"] = "Motor Power (adjusted for regen)"
    return motor_power

def lap_energy_ts(start_time: datetime, end_time: datetime, client) -> TimeSeries:
    motor_power = lap_power_ts(start_time, end_time, client)
    motor_energy = motor_power.promote(np.cumsum(motor_power) * motor_power.granularity)
    motor_energy.units = "J"
    motor_energy.meta["field"] = "Motor Energy (regen adjusted)"
    return motor_energy

def lap_speed_ts(start_time: datetime, end_time: datetime, client: DBClient) -> TimeSeries:
    lap_speed: TimeSeries = client.query_time_series(start_time, end_time, "VehicleVelocity")

    lap_speed.units = "m/s"
    lap_speed.meta["field"] = "Speed"
    return lap_speed

def lap_accel_ts(start_time: datetime, end_time: datetime, client: DBClient) -> TimeSeries:
    accelerator_position: TimeSeries = client.query_time_series(start_time, end_time, "AcceleratorPosition")

    accelerator_position.units = "%"
    accelerator_position.meta["field"] = "Accelerator Position"
    return accelerator_position

def lap_battery_temp(start_time: datetime, end_time: datetime, client: DBClient) -> float:
    battery_temp = client.query_time_series(start_time, end_time, "AverageTemp")    
    return battery_temp.mean()

def lap_regen_energy(start_time: datetime, end_time: datetime, client: DBClient) -> float:
    """
    Calculate the regen energy for a given motor power time series.
    """

    motor_voltage: TimeSeries = client.query_time_series(start_time, end_time, "BatteryVoltage")
    raw_motor_current: TimeSeries = client.query_time_series(start_time, end_time, "BatteryCurrent")
    motor_current_dir: TimeSeries = client.query_time_series(start_time, end_time, "BatteryCurrentDirection")
    
    # Align x-axes
    raw_motor_current, motor_voltage, motor_current_dir = TimeSeries.align(raw_motor_current, motor_voltage, motor_current_dir)
    # Make direction -1 or 1 instead of 1 or 0
    motor_current_sign = motor_current_dir * -2 + 1
    
    # Account for regen direction
    motor_current = raw_motor_current.promote(raw_motor_current * motor_current_sign)
    motor_power = motor_current.promote(motor_current * motor_voltage)
    
    # Calculate the regen energy by integrating the regen power
    regen_power = motor_power[motor_power < 0]
    if len(regen_power) == 0: # no regen power
        return 0.0
    regen_energy = np.cumsum(regen_power) * regen_power.granularity
 
    return abs(regen_energy[-1])

def lap_pack_current(start_time: datetime, end_time: datetime, client: DBClient) -> float:
    pack_current = client.query_time_series(start_time, end_time, "PackCurrent")    
    return pack_current.mean() 



if __name__ == "__main__":

    data_client = DBClient("can_log_prod")

    # Use data from FSGP day 1 and 3
    laps1 = FSGPDayLaps(1)  # Corresponds to July 16th
    laps3 = FSGPDayLaps(3)  # Corresponds to July 18th
    day_1_idx = range(laps1.get_lap_count())
    day_3_idx = range(laps3.get_lap_count())
    num_laps = len(day_1_idx) + len(day_3_idx)

    def collect_lap_data(query_func: Callable) -> np.ndarray:
        """
        Higher order function - computes `query_func` for all laps and returns the resulting array.

        :param query_func: must take in parameters (lap_start: datetime, lap_end:datetime, data_client:DBClient)
        :return: array of query_func results for all laps
        """
        lap_data = []
        # Iterate through all selected laps
        for day_laps, lap_indices in zip((laps1, laps3), (day_1_idx, day_3_idx)):
            for lap_idx in lap_indices:
                lap_num = lap_idx + 1
                lap_start = day_laps.get_start_utc(lap_num)
                lap_end = day_laps.get_finish_utc(lap_num)
                lap_data.append(query_func(lap_start, lap_end, data_client))
                print(f"Processed data for day {day_laps.day} lap {lap_num}")
                print(f"{lap_start=}\n{lap_end=}")
                print(f"{query_func.__name__} result for lap {lap_num}: {lap_data[-1]}\n")
        return np.array(lap_data)


    # -------- get lap average speeds and driver data from timing spreadsheet --------

    lap_speeds = []
    lap_drivers = []
    for day_laps, lap_indices in zip((laps1, laps3), (day_1_idx, day_3_idx)):
        for lap_idx in lap_indices:
            lap_num = lap_idx + 1
            lap_start = day_laps.get_start_utc(lap_num)
            lap_end = day_laps.get_finish_utc(lap_num)
            lap_drivers.append(day_laps.get_lap_driver(lap_num))
            lap_speeds.append(day_laps.get_lap_mph(lap_num))


    # -------- define various querying functions --------

    def get_lap_distance(start, end, client):
        speed_array: TimeSeries = lap_speed_ts(start, end, client)  # in meters per second
        return np.sum(speed_array) * speed_array.granularity  # in meters

    def lap_total_energy(start, end, client):
        return lap_energy_ts(start, end, client)[-1]

    def get_speed_variance(start, end, client):
        return np.var(lap_speed_ts(start, end, client).base)

    def get_power_variance(start, end, client):
        return np.var(lap_power_ts(start, end, client).base)

    def get_accelerator_variance(start, end, client):
        return np.var(lap_accel_ts(start, end, client).base)
    
    def get_current_variance(start, end, client):
        return np.var(lap_current_ts(start, end, client).base)

    def get_accel_variance(start, end, client):
        return np.var(np.diff(lap_speed_ts(start, end, client)))
    
    def get_accel_mean(start, end, client):
        return np.mean(np.diff(lap_speed_ts(start, end, client)))


    # -------- define various querying functions --------

    FSGP_TRACK_LEN_M = 5_070

    lap_drivers = np.array(lap_drivers)
    lap_speeds = np.array(lap_speeds)
    lap_distances_m = collect_lap_data(get_lap_distance)
    lap_energies = collect_lap_data(lap_total_energy)
    lap_practical_efficiencies = lap_energies / FSGP_TRACK_LEN_M
    lap_real_efficiencies = lap_energies / lap_distances_m
    lap_speed_variances = collect_lap_data(get_speed_variance)
    lap_power_variances = collect_lap_data(get_power_variance)
    lap_accelerator_variances = collect_lap_data(get_accelerator_variance)
    lap_current_variances = collect_lap_data(get_current_variance)
    lap_accel_variance = collect_lap_data(get_accel_variance)
    lap_accel_mean = collect_lap_data(get_accel_mean)
    lap_avg_battery_temp = collect_lap_data(lap_battery_temp)
    lap_regen = collect_lap_data(lap_regen_energy)
    lap_avg_pack_current = collect_lap_data(lap_pack_current)


    # Filter out laps that are not full laps (pitted or started from previous pitted lap)
    distance_filter = np.logical_and(lap_distances_m > 5000, lap_distances_m < 5200)

    # df = pd.DataFrame({
    #     'Driver': lap_drivers,
    #     'Average Speed (mph)': lap_speeds,
    #     'Lap Distance (m)': lap_distances_m,
    #     'Energy (J)': lap_energies,
    #     'Practical Efficiency (J/m)': lap_practical_efficiencies,
    #     'Real Efficiency (J/m)': lap_real_efficiencies,
    #     'Speed Variance': lap_speed_variances,
    #     'Motor Power Variance': lap_power_variances,
    #     'Accelerator Variance': lap_accelerator_variances,
    #     'Motor Current Variance': lap_current_variances,
    #     'Avg Acceleration (m/s^2)': lap_accel_mean,
    #     'Accel Variance': lap_accel_variance,
    #     'Avg Battery Temp (C)': lap_avg_battery_temp,
    #     'Regened Energy (J)': lap_regen,
    #     'Avg Pack Current (A)': lap_avg_pack_current,
    # })

    df = pd.DataFrame({
        # Basic lap metrics
        'driver': lap_drivers,
        'lap_distance (m)': lap_distances_m,
        
        # Speed and acceleration metrics
        'speed_avg (mph)': lap_speeds,
        'speed_variance': lap_speed_variances,
        'acceleration_avg (m/s^2)': lap_accel_mean,
        'acceleration_variance': lap_accel_variance,
        
        # Energy and efficiency metrics
        'energy_total (J)': lap_energies,
        'energy_regen (J)': lap_regen,
        'efficiency_practical (J/m)': lap_practical_efficiencies,
        'efficiency_real (J/m)': lap_real_efficiencies,
        
        # Power and current metrics
        'motor_power_variance': lap_power_variances,
        'motor_current_variance': lap_current_variances,
        'pack_current_avg (A)': lap_avg_pack_current,
        
        # Control metrics
        'accelerator_variance': lap_accelerator_variances,  # dimensionless
        
        # Temperature metrics
        'battery_temp_avg (C)': lap_avg_battery_temp,
    })


    df.to_csv('lap_data.csv', index=False)