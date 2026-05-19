import numpy as np
from data_tools.collections import TimeSeries
from data_tools.query import DBClient
from datetime import datetime

PACK_CURRENT_THRESHOLD = 0.10682  # pack current never reads 0; its lowest value is this constant (constant is rounded up here)

if __name__ == "__main__":
    client = DBClient()

    start_times = [datetime.fromisoformat("2026-04-02T20:02:30Z"),
                   datetime.fromisoformat("2026-04-02T20:07:10Z"),
                   datetime.fromisoformat("2026-04-02T20:26:05Z"),
                   datetime.fromisoformat("2026-04-02T20:35:40Z"),
                   datetime.fromisoformat("2026-04-02T21:05:25Z"),
                   datetime.fromisoformat("2026-04-02T21:10:30Z"),]
    
    stop_times = [datetime.fromisoformat("2026-04-02T20:03:10Z"),
                  datetime.fromisoformat("2026-04-02T20:07:30Z"),
                  datetime.fromisoformat("2026-04-02T20:26:24Z"),
                  datetime.fromisoformat("2026-04-02T20:36:15Z"),
                  datetime.fromisoformat("2026-04-02T21:05:40Z"),
                  datetime.fromisoformat("2026-04-02T21:10:40Z"),]

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
            field="BrakePressed",
            units=""
        )

        cruise_start_index = np.where(pack_current <= PACK_CURRENT_THRESHOLD)[0][0]
        cruise_end_index = np.where(brake_pressed > 0)[0][0]

        cruise_start_time = pack_current.datetime_x_axis[cruise_start_index]
        cruise_end_time = brake_pressed.datetime_x_axis[cruise_end_index]

        cruise_speed: TimeSeries = motor_rotating_speed.slice(cruise_start_time, cruise_end_time)
        cruise_speed.meta = motor_rotating_speed.meta
        cruise_speed.plot()


        



    