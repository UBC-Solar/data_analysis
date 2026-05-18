from data_tools.collections import TimeSeries
from data_tools.query import DBClient
from datetime import datetime

if __name__ == "__main__":
    client = DBClient()

    start = datetime.fromisoformat("2026-04-02T20:00:00Z")
    stop = datetime.fromisoformat("2026-04-02T20:45:00Z")

    data: TimeSeries = client.query_time_series(
        start=start,
        stop=stop,
        field="MotorRotatingSpeed",
        units="km/h"
    )

    # Plot the data
    data.plot()