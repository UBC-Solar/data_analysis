import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry


# Below is modified from auto-generated code from open-meteo.com


def get_irradiance_archive(latitude: float, longitude: float, start_date: str, end_date: str, timezone: str = "GMT"):
    """

    :param latitude:
    :param longitude: YYYY-MM-DD
    :param start_date: YYYY-MM-DD
    :param end_date:
    :param timezone:
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
        "hourly": ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
                   "terrestrial_radiation"],
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
    hourly = response.Hourly()
    hourly_shortwave_radiation = hourly.Variables(0).ValuesAsNumpy()
    hourly_direct_radiation = hourly.Variables(1).ValuesAsNumpy()
    hourly_diffuse_radiation = hourly.Variables(2).ValuesAsNumpy()
    hourly_direct_normal_irradiance = hourly.Variables(3).ValuesAsNumpy()
    hourly_global_tilted_irradiance = hourly.Variables(4).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "shortwave_radiation": hourly_shortwave_radiation, "direct_radiation": hourly_direct_radiation,
        "diffuse_radiation": hourly_diffuse_radiation, "direct_normal_irradiance": hourly_direct_normal_irradiance,
        "global_tilted_irradiance": hourly_global_tilted_irradiance}

    return pd.DataFrame(data=hourly_data)


def get_irradiance_forecast(latitude: float, longitude: float, timezone: str = "GMT"):
    # Set up the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "hourly": ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
                   "global_tilted_irradiance", "terrestrial_radiation"],
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_shortwave_radiation = hourly.Variables(0).ValuesAsNumpy()
    hourly_direct_radiation = hourly.Variables(1).ValuesAsNumpy()
    hourly_diffuse_radiation = hourly.Variables(2).ValuesAsNumpy()
    hourly_direct_normal_irradiance = hourly.Variables(3).ValuesAsNumpy()
    hourly_global_tilted_irradiance = hourly.Variables(4).ValuesAsNumpy()
    hourly_terrestrial_radiation = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "shortwave_radiation": hourly_shortwave_radiation, "direct_radiation": hourly_direct_radiation,
        "diffuse_radiation": hourly_diffuse_radiation, "direct_normal_irradiance": hourly_direct_normal_irradiance,
        "global_tilted_irradiance": hourly_global_tilted_irradiance,
        "terrestrial_radiation": hourly_terrestrial_radiation}

    return pd.DataFrame(data=hourly_data)
