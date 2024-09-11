import pandas as pd
from os import path
from datetime import datetime
import pytz


class LapDataHelper:

    def __init__(self, day: int):
        self.day = day

        self.filepath = path.normpath(path.join(path.dirname(__file__), f"..\\data\\fsgp_timing_day_{day}.csv"))
        self._df_full = pd.read_csv(self.filepath)

        header = self._df_full.iloc[2]
        data = self._df_full.iloc[4:]

        dtypes = {
            'Lap #': 'int',
            'Pit Time Before Lap (Min)': 'float',
            'Lap Time (Min)': 'float',
            'Avg Lap Speed (MPH)': 'float',
        }

        self.df = pd.DataFrame(data.values, columns=header).astype(dtypes)
        self.df.set_index('Lap #', inplace=True)

    def get_day_laps(self):
        return self.df.iloc[:, 0].max()

    def _get_utc(self, time_str):

        date_time_str = f"{'2024-07-'}{15 + self.day} {time_str}"
        naive_datetime = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")

        central_tz = pytz.timezone('America/Chicago')
        local_datetime = central_tz.localize(naive_datetime)
        utc_datetime = local_datetime.astimezone(pytz.utc)

        # Format the UTC time to ISO 8601 string format
        return utc_datetime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

    def get_finish(self, lap):
        time_str = self.df.loc[lap, 'Finish Time (HH:MM:SS)']
        return self._get_utc(time_str)

    def get_start(self, lap):
        start_time = self.df.loc[lap, 'Start Time (Only if Diff than Prev Finish)']
        if isinstance(start_time, str) and len(start_time) > 0:
            time_str = start_time
        else:
            time_str = self.df.loc[lap - 1, 'Finish Time (HH:MM:SS)']
        return self._get_utc(time_str)


if __name__ == "__main__":
    laps = LapDataHelper(3)
    breakpoint()
