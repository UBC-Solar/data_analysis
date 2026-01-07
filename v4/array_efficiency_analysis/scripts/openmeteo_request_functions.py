from datetime import datetime, timedelta

import openmeteo_requests
import pytz

import pandas as pd
from data_tools import TimeSeries

from data_tools import DBClient, TimeSeries
from datetime import datetime
import pytz


# Below is modified from auto-generated code from open-meteo.com

def open_meteo_archive_timeseries(latitude: float,
                                      longitude: float,
                                      start_date: str,
                                      end_date: str,
                                      field: str,
                                      timezone: str = "GMT") -> TimeSeries:
    """
    Gets an archive of the forecast from a start and end date
    :param latitude:
    :param longitude: 
    :param start_date: YYYY-MM-DD
    :param end_date: YYYY-MM-DD
    :param timezone:
    :param instant: Set to true to get instantaneous values rather than average of the past hour
    :return:
    """
        
    # Set up the Open-Meteo API client with cache and retry on error
    # cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    # retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client()

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [field],
        "timezone": timezone
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    #print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_data = hourly.Variables(0).ValuesAsNumpy()

    query_start: datetime = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    query_stop: datetime = (datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                            + timedelta(hours=24))

    return TimeSeries(hourly_data, meta={
        "start": query_start + timedelta(hours = 7),
        "stop": query_stop + timedelta(hours = 7),
        "car": "Brightside",
        "measurement": "Irradiance",
        "field": "GHI",
        "period": 3600,  # seconds per hour
        "length": len(hourly_data)*3600,
        "units": "W/m^2",
    })

