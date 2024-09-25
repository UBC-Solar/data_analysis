import pandas as pd
from os import path
from datetime import datetime
import pytz


class FSGPDayLaps:

    def __init__(self, day: int):
        self.day = day

        self.filepath = path.normpath(path.join(path.dirname(__file__), f"..\\data\\fsgp_timing_day_{day}.csv"))
        self._df_full = pd.read_csv(self.filepath)

        header = self._df_full.iloc[2]
        data = self._df_full.iloc[4:]
        self.df = pd.DataFrame(data.values, columns=header)

        selected_cols = [
            'Lap #',
            'Start Time (Only if Diff than Prev Finish)',
            'Finish Time (HH:MM:SS)',
            'Lap Time (HH:MM:SS)',
            'Pit Time Before Lap (Min)',
            'Lap Time (Min)',
            'Avg Lap Speed (MPH)',
            'DRVR Name',
        ]

        dtypes = {
            'Lap #': 'int',
            'Pit Time Before Lap (Min)': 'float',
            'Lap Time (Min)': 'float',
            'Avg Lap Speed (MPH)': 'float',
        }

        self.df = self.df[selected_cols].dropna(axis=0, how='all').astype(dtypes)
        self.df.set_index('Lap #', inplace=True)

    @staticmethod
    def _pad_timestamp(timestamp: str):
        """
        Pad a timestamp to match HH:MM:SS format

        :param timestamp: timestamp with H:MM:SS or HH:MM:SS format
        :return: timestamp in HH:MM:SS format
        """
        assert timestamp.count(":") == 2, "timestamp is not in HH:MM:SS format - should have 2 colons"
        if len(timestamp) == 8:
            return timestamp
        elif len(timestamp) == 7:
            return "0" + timestamp
        else:
            raise ValueError('timestamp did not have H:MM:SS format or HH:MM:SS format, should have len 7 or 8')

    def get_lap_count(self):
        return self.df.index.max()

    def _get_utc(self, time_str):

        date_time_str = f"{'2024-07-'}{15 + self.day} {time_str}"
        naive_datetime = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")

        central_tz = pytz.timezone('America/Chicago')
        local_datetime = central_tz.localize(naive_datetime)
        utc_datetime = local_datetime.astimezone(pytz.utc)

        # Format the UTC time to ISO 8601 string format
        return utc_datetime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

    def get_start_utc(self, lap) -> str:
        """
        Get lap start time as a UTC timestamp

        :param lap: Lap number from the race day
        :return: Start time in %Y-%m-%dT%H:%M:%SZ format e.g. 2024-07-16T20:46:32Z
        """
        start_time = self.df.loc[lap, 'Start Time (Only if Diff than Prev Finish)']
        if isinstance(start_time, str) and len(start_time) > 0:
            time_str = start_time
        else:
            time_str = self.df.loc[lap - 1, 'Finish Time (HH:MM:SS)']
        return self._get_utc(time_str)

    def get_finish_utc(self, lap) -> str:
        """
        Get lap finish time as a UTC timestamp

        :param lap: Lap number from the race day
        :return: Finish time in %Y-%m-%dT%H:%M:%SZ format e.g. 2024-07-16T20:46:32Z
        """
        time_str = self.df.loc[lap, 'Finish Time (HH:MM:SS)']
        return self._get_utc(time_str)

    def get_time(self, lap) -> str:
        """
        Get lap time as an HH:MM:SS timestamp

        :param lap: Lap number from the race day
        :return: Time in HH:MM:SS format e.g. 00:06:59
        """
        return self._pad_timestamp(self.df.loc[lap, 'Lap Time (HH:MM:SS)'])

    def get_pit_time(self, lap) -> float:
        """
        Get pit time before lap in minutes as a decimal, or 0. if Brightside didn't pit

        :param lap: Lap number from the race day
        :return: Pit time before lap in minutes as a decimal, e.g. 3.583
        """
        return self.df.loc[lap, 'Pit Time Before Lap (Min)']

    def get_time_minutes(self, lap) -> float:
        """
        Get lap time in minutes as a decimal

        :param lap: Lap number from the race day
        :return: Time in minutes as a decimal, e.g. 6.324
        """
        return self.df.loc[lap, 'Lap Time (Min)']

    def get_lap_mph(self, lap) -> float:
        """
        Get the average lap speed in mph

        :param lap: Lap number from the race day
        :return: Speed in mph, e.g. 26.745
        """
        return self.df.loc[lap, 'Avg Lap Speed (MPH)']

    def get_lap_driver(self, lap) -> str:
        """
        Get the name of a driver for a lap

        :param lap: Lap number from the race day
        :return: Name of driver, e.g. 'Diego'
        """
        return self.df.loc[lap, 'DRVR Name']


if __name__ == "__main__":
    laps1 = FSGPDayLaps(1)
    laps2 = FSGPDayLaps(2)
    laps3 = FSGPDayLaps(3)
    breakpoint()
    for i, race_day in enumerate((laps1, laps2, laps3), 1):
        for lap_idx in range(1, race_day.get_lap_count()+1):
            print(f'Day {i}, lap {lap_idx}, start_utc: {race_day.get_start_utc(lap_idx)}, '
                  f'stop_utc: {race_day.get_finish_utc(lap_idx)}')
