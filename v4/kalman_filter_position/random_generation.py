import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import os

# the goal here is to create a csv dataset that consists of random data so that we can simulate necessary failure conditions as necessary.
# Overall, this includes the sensor class and the Path generator (this will interpolate lat/lon over time).
# The Voronoi Track Generator is responsible for generating the track and speed based on the curvature profile. It also exports this to a csv format.
RADIUS_EARTH = 6371000.0


# required coordinate utilites.

def latlon_to_xy(lat, lon, lat0, lon0):
    # this will take in the gps coordinates and return it in east north emtres.

    # using the haversine formula to find the bearing and distance betweeen two GPS points.

    x = RADIUS_EARTH * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon - lon0)
    y = RADIUS_EARTH * np.deg2rad(lat - lat0)
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + np.rad2deg(y / RADIUS_EARTH)
    lon = lon0 + np.rad2deg(x / (RADIUS_EARTH * np.cos(np.deg2rad(lat0))))
    return lat, lon


def haversine(lon1, lat1, lon2, lat2):
    """
    calculates the great circle distance in km between the two specified points on the eearth

    :param lon1:
    :param lat1:
    :param lon2:
    :param lat2:
    :return: uses the haversine formula to calculate the great circle distance in between two specified points on the Earth.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2)
    return RADIUS_EARTH * 2 * asin(sqrt(a))


class Path:
    """
    reads the csv (ground_truth_path)  and interpolates the input data to produce linear lat/lon time series at the required time. the CSV consists of the columns time_s, lat, lon


    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.t = df["time_s"].values.astype(float)
        self._lat = df["lat"].values.astype(float)
        self._lon = df["lon"].values.astype(float)

        self.duration = self.t[-1]
        self.lat0 = self._lat[0]
        self.lon0 = self._lon[0]

        # convert to local E-N
        self._x, self._y = latlon_to_xy(self._lat, self._lon, self.lat0, self.lon0)

        # Pre-compute derivatives
        self._vx = np.gradient(self._x, self.t)
        self._vy = np.gradient(self._y, self.t)
        self._ax = np.gradient(self._vx, self.t)
        self._ay = np.gradient(self._vy, self.t)

    def calculate_distance(self, lat, lon):
        dist = np.zeros(len(lat))

        for i in np.arange(1, len(lat)):
            lat1 = lat[i - 1]
            lon1 = (lon[i - 1])
            lat2 = (lat[i])
            lon2 = (lon[i])

            dist[i] = haversine(lon1, lat1, lon2, lat2)

        return (dist)

    def gps_interpolate(self, lat, lon, tstamp):
        dist = self.calculate_distance(lat, lon)
        dist_cum_norm = np.cumsum(dist) / np.sum(dist)
        data = [lat, lon]
        lat_interp = np.interp(tstamp, self.t, self._lat)
        lon_interp = np.interp(tstamp, self.t, self._lon)
        return lat_interp, lon_interp

    def latitude(self, t):
        # method to create a time series object of the latitude using gps interpolation
        return np.interp(t, self.t, self._lat)

    def lat(self, t):
        # interpolates raw lat array
        return np.interp(t, self.t, self._lat)

    def lon(self, t):
        return np.interp(t, self.t, self._lon)

    def x(self, t):
        return np.interp(t, self.t, self._x)

    def y(self, t):
        return np.interp(t, self.t, self._y)

    # velocity in x and y:
    def vx(self, t):
        # differentiate the x values and then interpolate to requested time t.
        return np.interp(t, self.t, self._vx)

    def vy(self, t):
        return np.interp(t, self.t, self._vy)

    def speed(self, t):
        return np.hypot(self.vx(t), self.vy(t))

    def ax(self, t):
        # double derivative of x coordinate.
        # v = np.gradient(self.vx(t), self.t)

        return np.interp(t, self.t, self._ax)

    def ay(self, t):
        return np.interp(t, self.t, self._ay)


class Sensor:
    def __init__(self, rate_hz,
                 noise_std,
                 bias, drift_rate,
                 rng_seed: int = 21):
        self.rate_hz = rate_hz
        self.noise_std = noise_std or {}
        self.bias = bias or {}
        self.drift_rate = drift_rate or {}
        self.rng = np.random.default_rng(rng_seed)

    def sample(self, path: Path):
        """
        generates measuremesnts over the full path duration.
        :param path:
        :return the sample dataframe
        """
        t = np.arange(0, path.duration, 1.0 / self.rate_hz)
        df = self.measure(path, t)
        df.insert(0, "t", t)
        self.apply_noise(df)
        return df

    def measure(self, path, t):
        """generate measurements for the certain time instance, different for the subclasses so will be overridden. """
        raise NotImplementedError("to be implemented by subclasses")

    def apply_noise(self, df):
        """applies Gaussian noise, constant bias and time variable drift to sensor data. """
        t = df["t"].values
        for col, std in self.noise_std.items():
            if col in df.columns:
                df[col] += self.rng.normal(0, std, len(df))  # generate gaussian noise.
        for col, b in self.bias.items():
            if col in df.columns:
                df[col] += b
        for col, rate in self.drift_rate.items():
            if col in df.columns:
                df[col] += rate * t


# consider noise in GPS vs IMU:
class GPSSensor(Sensor):
    """
    the GPS noise is mostly high frequency, struggles with precision over accuracy and quick velocity changes.
    the IMU tracks relative movement i.e. errors can accumulate as we integrate creating a "drift" over time.
    sensors also might have inherent biases. the imu cannot provide absolute position reference on its own.


    - the gps sensor class will output lat, lon at a fixed rate with configurable noise and bias. our current GPS Is at 4Hz and has ~5m noise. in our data, we have also observed weird spikes and a lack of consistency between cellular and telemetry (radio) transfer of data.
    - drift should consider both spatial and temporal aspects. error is +- (lat, lon) tuple.

    """

    def __init__(self, rate_hz=4.0, noise_m=5.0, bias_m=(0, 0), drift_ms=(0, 0), rng_seed=21):
        # convert all metre noise to degrees using small angle approximation
        # use: dely = dellat * pi/180 * r_earth

        noise_lat = noise_m / RADIUS_EARTH * (180 / np.pi)
        noise_lon = noise_m / RADIUS_EARTH * (180 / np.pi)

        bias_deg_lat = bias_m[1] / RADIUS_EARTH * (180 / np.pi)
        bias_deg_lon = bias_m[0] / RADIUS_EARTH * (180 / np.pi)
        drift_deg_lat = drift_ms[1] / RADIUS_EARTH * (180 / np.pi)
        drift_deg_lon = drift_ms[0] / RADIUS_EARTH * (180 / np.pi)

        noise_std = {"lat": noise_lat, "lon": noise_lon}
        bias = {"lat": bias_deg_lat, "lon": bias_deg_lon}
        drift_rate = {"lat": drift_deg_lat, "lon": drift_deg_lon}
        # could potentially add noise by direclty adding to the x,y coordinates too?
        super().__init__(rate_hz, noise_std, bias, drift_rate, rng_seed=rng_seed)

    def measure(self, path, t):
        return pd.DataFrame({"lat": path.lat(t), "lon": path.lon(t)})


class IMUSensor(Sensor):
    """
    Outputs acceleration along the three coordinate axes
     (ax, ay and az)

    """

    def __init__(self, rate_hz=30.0, noise_std_mss=0.05,  # gaussian noise in m/s^2,
                 bias_mss=0.02, drift_rate=0.0, rng_seed=23):
        noise_std = {"ax": noise_std_mss, "ay": noise_std_mss, "az": 0.02}
        bias = {"ax": bias_mss, "ay": bias_mss}
        drift = {"ax": drift_rate, "ay": drift_rate}
        super().__init__(rate_hz, noise_std, bias, drift, rng_seed)

    def measure(self, path: Path, t: np.ndarray):
        return pd.DataFrame({
            "ax": path.ax(t),
            "ay": path.ay(t),
            "az": np.full(len(t), 9.81),
        })


class SpeedSensor(Sensor):
    """
    Outputs scalar ground speed in m/s.

    """

    def __init__(self, rate_hz=30.0, noise_mps=0.5, bias_mps=0.0, rng_seed=44):
        noise_std = {"speed": noise_mps}
        bias = {"speed": bias_mps}
        super().__init__(rate_hz, noise_std, bias, {}, rng_seed)

    def measure(self, path: Path, t: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "speed": path.speed(t)
        })
