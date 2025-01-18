import numpy as np
from data_tools import TimeSeries
import openmeteo_requests
from datetime import datetime, timedelta
import pytz
import enum


class IrradianceData(enum.Enum):
    """
    List of some relevant fields to be queried from open-meteo.

    See https://open-meteo.com/en/docs/historical-weather-api for other possible values to query.
    """
    GHI = "shortwave_radiation_instant"  # equal to DIR + DHI
    DIR = "direct_radiation_instant"  # in theory, equal to DNI*cos(solar_zenith), non-diffuse portion of GHI
    DNI = "direct_normal_irradiance_instant"
    DHI = "diffuse_radiation_instant"


def open_meteo_archive_timeseries(
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        field: IrradianceData,
        timezone: str = "GMT",
        granularity = 0.1) -> TimeSeries:

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
        "hourly": [field.value],
        "timezone": timezone,
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    return response.Hourly().Variables(0).ValuesAsNumpy()