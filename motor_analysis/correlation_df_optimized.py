from data_tools import TimeSeries, DBClient, FSGPDayLaps
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
import datetime

class LapData:
    """Container for all raw time series data for a single lap"""
    def __init__(self, start_time: datetime, end_time: datetime, client: DBClient):
        # Query all needed time series at once
        queries = {
            "battery_voltage": "BatteryVoltage",
            "battery_current": "BatteryCurrent",
            "current_direction": "BatteryCurrentDirection",
            "vehicle_speed": "VehicleVelocity",
            "accelerator": "AcceleratorPosition",
            "battery_temp": "AverageTemp",
            "pack_current": "PackCurrent"
        }
        
        # Collect all time series
        self.raw_data = {
            name: client.query_time_series(start_time, end_time, query)
            for name, query in queries.items()
        }
        
        # Align all time series to same time base
        self.data = TimeSeries.align(*self.raw_data.values())
        self._cache = {}  # Cache for computed values
        
    def get_motor_current(self) -> TimeSeries:
        """Get motor current adjusted for regen"""
        if "motor_current" not in self._cache:
            current_dir = self.data[2] * -2 + 1  # Convert 0/1 to -1/1
            self._cache["motor_current"] = self.data[1].promote(self.data[1] * current_dir)
            self._cache["motor_current"].units = "A"
        return self._cache["motor_current"]
    
    def get_motor_power(self) -> TimeSeries:
        """Get motor power"""
        if "motor_power" not in self._cache:
            motor_current = self.get_motor_current()
            self._cache["motor_power"] = motor_current.promote(motor_current * self.data[0])
            self._cache["motor_power"].units = "W"
        return self._cache["motor_power"]
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate all metrics for this lap"""
        # Get required time series
        speed = self.data[3]
        motor_power = self.get_motor_power()
        motor_current = self.get_motor_current()
        
        # Calculate distance
        distance = np.sum(speed) * speed.granularity
        
        # Calculate energy metrics
        energy = np.sum(motor_power) * motor_power.granularity
        regen_power = motor_power[motor_power < 0]
        regen_energy = abs(np.sum(regen_power) * motor_power.granularity) if len(regen_power) > 0 else 0.0
        
        # Calculate variances
        speed_var = np.var(speed.base)
        power_var = np.var(motor_power.base)
        current_var = np.var(motor_current.base)
        accel_var = np.var(np.diff(speed))
        accelerator_var = np.var(self.data[4].base)
        
        # Calculate means
        accel_mean = np.mean(np.diff(speed))
        battery_temp_mean = np.mean(self.data[5].base)
        pack_current_mean = np.mean(self.data[6].base)
        
        return {
            "lap_distance_(m)": distance,
            "energy_total_(J)": energy,
            "energy_regen_(J)": regen_energy,
            "speed_variance_(mph^2)": speed_var,
            "motor_power_variance_(W^2)": power_var,
            "motor_current_variance_(A^2)": current_var,
            "acceleration_variance_(m^2/s^4)": accel_var,
            "accelerator_variance": accelerator_var,
            "acceleration_avg_(m/s^2)": accel_mean,
            "battery_temp_avg_(C)": battery_temp_mean,
            "pack_current_avg_(A)": pack_current_mean
        }

def collect_lap_data(laps1: FSGPDayLaps, laps3: FSGPDayLaps, client: DBClient) -> pd.DataFrame:
    """Collect all lap data efficiently"""
    all_data = []
    FSGP_TRACK_LEN_M = 5_070
    
    # Process each day's laps
    for day_laps in (laps1, laps3):
        for lap_idx in range(day_laps.get_lap_count()):
            lap_num = lap_idx + 1
            print(f"Processing day {day_laps.day} lap {lap_num}")
            
            # Get lap timing data
            lap_start = day_laps.get_start_utc(lap_num)
            lap_end = day_laps.get_finish_utc(lap_num)
            
            # Collect all raw data for this lap
            lap = LapData(lap_start, lap_end, client)
            metrics = lap.get_metrics()
            
            # Add driver and speed from timing data
            metrics.update({
                "lap_index": lap_idx,
                "lap_number": lap_idx + 1,
                "lap_end_time": lap_end,
                "day": day_laps.day,
                "driver": day_laps.get_lap_driver(lap_num),
                "speed_avg_(mph)": day_laps.get_lap_mph(lap_num)
            })
            
            # Calculate efficiencies
            metrics["efficiency_practical_(J/m)"] = metrics["energy_total_(J)"] / FSGP_TRACK_LEN_M
            metrics["efficiency_real_(J/m)"] = metrics["energy_total_(J)"] / metrics["lap_distance_(m)"]
            
            all_data.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    return df

if __name__ == "__main__":
    data_client = DBClient("can_log_prod")
    
    # Use data from FSGP day 1 and 3
    laps1 = FSGPDayLaps(1)  # July 16th
    laps3 = FSGPDayLaps(3)  # July 18th
    
    # Collect all data efficiently
    df = collect_lap_data(laps1, laps3, data_client)
    df.to_csv('lap_data.csv', index=False)