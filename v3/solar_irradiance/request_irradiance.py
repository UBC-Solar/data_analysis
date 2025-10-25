from datetime import datetime, timedelta

import openmeteo_requests
import pytz

import requests_cache
import pandas as pd
from data_tools import TimeSeries
from retry_requests import retry


# Below is modified from auto-generated code from open-meteo.com


def get_irradiance_archive(latitude: float,
                           longitude: float,
                           start_date: str,
                           end_date: str,
                           timezone: str = "GMT",
                           instant: bool = False) -> pd.DataFrame:
    """
    :param latitude:
    :param longitude: YYYY-MM-DD
    :param start_date: YYYY-MM-DD
    :param end_date:
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
    if instant: fields = [
            "shortwave_radiation_instant",
            "direct_radiation_instant",
            "diffuse_radiation_instant",
            "direct_normal_irradiance_instant",
            "terrestrial_radiation_instant"
        ]
    else : fields = [
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation"
        ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": fields,
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

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "shortwave_radiation": hourly_shortwave_radiation,
        "direct_radiation": hourly_direct_radiation,
        "diffuse_radiation": hourly_diffuse_radiation,
        "direct_normal_irradiance": hourly_direct_normal_irradiance,
        "global_tilted_irradiance": hourly_global_tilted_irradiance
    }

    return pd.DataFrame(data=hourly_data)


def get_irradiance_forecast(latitude: float,
                            longitude: float,
                            timezone: str = "GMT",
                            instant: bool = False) -> pd.DataFrame:

    # Set up the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    if instant: fields = [
            "shortwave_radiation_instant",
            "direct_radiation_instant",
            "diffuse_radiation_instant",
            "direct_normal_irradiance_instant",
            "terrestrial_radiation_instant"
        ]
    else : fields = [
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation"
        ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "hourly": fields
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


def open_meteo_archive_timeseries(latitude: float,
                                      longitude: float,
                                      start_date: str,
                                      end_date: str,
                                      field: str,
                                      timezone: str = "GMT") -> TimeSeries:

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
    hourly_data = hourly.Variables(0).ValuesAsNumpy()

    query_start: datetime = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    query_stop: datetime = (datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                            + timedelta(hours=24))

    return TimeSeries(hourly_data, meta={
        "start": query_start,
        "stop": query_stop,
        "car": "Brightside",
        "measurement": "Irradiance",
        "field": "GHI",
        "granularity": 3600,  # seconds per hour
        "length": len(hourly_data),
        "units": "W/m^2",
    })
