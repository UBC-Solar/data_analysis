# Methods to calculate array power from telemetry data.
# Focusing on analyzing FSGP 2024 data
import pytz
from data_tools import DBClient, FSGPDayLaps, TimeSeries
from datetime import datetime, timedelta
from enum import Enum
from irradiance_timeseries import open_meteo_archive_timeseries, IrradianceData
import matplotlib.pyplot as plt
import numpy as np

class ArraySource(Enum):
    STRING_1 = 1
    STRING_2 = 2
    ALL = 3

def _get_string_power(start: datetime, stop: datetime, array_string: int, client: DBClient):
    assert array_string in (1, 2), "Array string must be 1 or 2"
    current = client.query_time_series(start, stop, f"CurrentSensor{array_string}")
    voltage = client.query_time_series(start, stop, f"VoltSensor{array_string}")
    current, voltage = TimeSeries.align(current, voltage)
    power = current.promote(current * voltage)
    return power

def get_array_power(start: datetime,
                    stop: datetime,
                    source: ArraySource=ArraySource.ALL,
                    client: DBClient=None) -> TimeSeries:
    if client is None:
        client = DBClient()
    if source == ArraySource.STRING_1:
        power_1 = _get_string_power(start, stop, 1, client)
        power_1.units = "W"
        power_1.meta["field"] = "String 1 Array Power"
        return power_1
    if source == ArraySource.STRING_2:
        power_2 = _get_string_power(start, stop, 2, client)
        power_2.units = "W"
        power_2.meta["field"] = "String 2 Array Power"
        return power_2
    power_1 = _get_string_power(start, stop, 1, client)
    power_2 = _get_string_power(start, stop, 2, client)
    power_1, power_2 = TimeSeries.align(power_1, power_2)
    array_power = power_1.promote(power_1 + power_2)
    array_power.units = "W"
    array_power.meta["field"] = "Total Array Power"
    return array_power


if __name__ == "__main__":

    query_client = DBClient()

    for day in (1, 3):
        race_day = FSGPDayLaps(day)
        start = race_day.get_start_utc(1)
        last_lap = 28 if day == 3 else race_day.get_lap_count()  # Only have 28 laps of data for day 3
        stop = race_day.get_finish_utc(last_lap)

        string_1_pow = get_array_power(start, stop, ArraySource.STRING_1, query_client)
        string_2_pow = get_array_power(start, stop, ArraySource.STRING_2, query_client)
        combined_pow = get_array_power(start, stop, client=query_client)

        plt.plot(string_1_pow, label="String 1")
        plt.plot(string_2_pow, label="String 2")
        plt.plot(combined_pow, label="Combined")
        plt.title(f"Day {day} Race Array Power")
        plt.xlabel("Time Since Start (s * 0.1)")
        plt.ylabel("Power (W)")
        plt.grid(True)
        plt.legend(loc="best")
        plt.show()

        # ----------- PLOT PREDICTED POWER FOR DAY 1 AND DAY 3 -----------

        start_str = race_day.get_start_utc_string(1)
        end_str = race_day.get_finish_utc_string(last_lap)

        # declare parameters for open-meteo query
        # open-meteo provides historical data in sets of days, each with 24 hourly data points
        # we need at least 2024-07-16T10:00:00 to 2024-07-18T17:00:00 America/Chicago time
        QUERY_FIRST_DAY = "2024-07-16"  # midnight UTC -> 19:00 previous day America/Chicago time
        QUERY_LAST_DAY = "2024-07-18"  # will get data until end of 18th UTC --> 19:00 America/Chicago time, i.e. 7pm
        query_start: datetime = datetime.strptime(QUERY_FIRST_DAY, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        query_stop: datetime = (datetime.strptime(QUERY_LAST_DAY, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                               + timedelta(hours=24))
        NCM_MOTORSPORTS_LAT = 37.0006332
        NCM_MOTORSPORTS_LONG = -86.3709907
        TIMEZONE = "GMT"  # same as UTC

        ghi_hourly = open_meteo_archive_timeseries(NCM_MOTORSPORTS_LAT,
                                            NCM_MOTORSPORTS_LONG,
                                            QUERY_FIRST_DAY,
                                            QUERY_LAST_DAY,
                                            IrradianceData.GHI,
                                            TIMEZONE)

        ghi = ghi_hourly.promote(np.interp(combined_pow.x_axis, ghi_hourly.x_axis, ghi_hourly))

        combined_pow, ghi = TimeSeries.align(combined_pow, ghi)

        plt.plot(combined_pow, label="Array Power (W)")
        plt.plot(ghi, label="GHI (W/m^2)")
        plt.title(f"Day {day} Array Power vs. GHI")
        plt.xlabel("Time Since Start (s * 0.1)")
        plt.ylabel("W or W/m^2 lol")
        plt.grid(True)
        plt.legend(loc="best")
        plt.show()
