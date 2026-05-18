from data_tools.collections import TimeSeries
from data_tools.query import DBClient
from datetime import datetime

PACK_CURRENT_THRESHOLD = 0.10682  # pack current never reads 0; it settles to this constant (it is rounded up)

if __name__ == "__main__":
    client = DBClient()

    start_times = [datetime.fromisoformat("2026-04-02T20:02:30Z")]
    stop_times = [datetime.fromisoformat("2026-04-02T20:03:10Z")]

    for start, stop in zip(start_times, stop_times):
        motor_rotating_speed: TimeSeries = client.query_time_series(
            start=start,
            stop=stop,
            field="MotorRotatingSpeed",
            units="km/h"
        )

        pack_current: TimeSeries = client.query_time_series(
            start=start,
            stop=stop,
            field="PackCurrent",
            units="A"
        )

        brake_pressed: TimeSeries = client.query_time_series(
            start=start,
            stop=stop,
            field="BreakPressed",
            units=""
        )




    