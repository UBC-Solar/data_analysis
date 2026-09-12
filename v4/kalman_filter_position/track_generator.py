import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, spatial, interpolate
from shapely.geometry import Point, LineString, Polygon

EARTH_RADIUS_M = 6_371_000.0


def closest_node(node, nodes, k):
    deltas = nodes - node
    distance = np.einsum('ij,ij->i', deltas, deltas)
    return np.argpartition(distance, k)[k]


def clockwise_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2):
    return (dx_dt ** 2 + dy_dt ** 2) ** -1.5 * (dx_dt * d2y_dt2 - dy_dt * d2x_dt2)


def xy_to_latlon(x: np.ndarray, y: np.ndarray,
                 lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + np.rad2deg(y / EARTH_RADIUS_M)
    lon = lon0 + np.rad2deg(x / (EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))))
    return lat, lon


def assign_speed(curvature_abs: np.ndarray,
                 v_max: float = 18.0,
                 v_min: float = 6.0,
                 curvature_threshold: float = 1 / 20.0, rng: np.random.Generator = None):
    """
    Assign speed at each point on the centerline based on local curvature.

    Straight sections (low curvature) get v_max
    Tight corners  (high curvature)   get v_min
    Linearly interpolated in between.


    curvature_abs         : absolute curvature at each centerline point
    v_max                 : maximum speed on straights (m/s)
    v_min                 : minimum speed in corners (m/s)
    curvature_threshold   : curvature at which v_min is used
                            1/20 = radius of 20 m
    :return numpy array of the speed values.
    """
    # Normalise curvature to [0, 1] where 0 = straight, 1 = max corner
    # rng = rng or np.random.default_rng()
    #
    # #random baseline
    # speed = rng.uniform(v_min, v_max, len(curvature_abs))
    # #sharp corners use hard minimum
    # sharp = curvature_abs >=curvature_threshold
    # straight_threshold = 1/100
    # speed[sharp] = v_min
    # speed = np.convolve(speed, np.ones(25)/25, mode="same")
    #
    # straight = curvature_abs <= straight_threshold
    # speed[straight] = v_max
    # return np.clip(speed, v_min, v_max)
    # Normalise curvature to [0, 1] where 0 = straight, 1 = max corner
    k_norm = np.clip(curvature_abs / curvature_threshold, 0, 1)
    speed = v_max - k_norm * (v_max - v_min)
    return speed


def curvature_to_time(x: np.ndarray, y: np.ndarray,
                      speed: np.ndarray):
    """
    Convert spatial centerline points to a time array using the local speed.

    time per segment is give by  Arc length between consecutive points / average speed of those points.
    :return numpy array
    """
    ds = np.hypot(np.diff(x), np.diff(y))  # segment lengths
    v_avg = 0.5 * (speed[:-1] + speed[1:])  # average speed per segment
    dt = ds / v_avg  # time per segment
    t = np.concatenate([[0], np.cumsum(dt)])
    return t


class VoronoiTrackGenerator:
    """
    Generates a random closed-loop solar car track.



    n_points     : number of Voronoi seed points
    n_regions    : how many Voronoi regions to merge into the track shape
                   (more = larger, more complex track)
    min_bound    : lower bound of the generation area (m)
    max_bound    : upper bound of the generation area (m)
    origin_lat   : reference latitude for output
    origin_lon   : reference longitude for output (WGS-84)
    v_max        : maximum vehicle speed (m/s)
    v_min        : minimum vehicle speed in corners (m/s)
    min_radius_m : minimum allowed corner radius (m) this sets curvature threshold
    n_centerline : number of points on the output centerline
    rng_seed     : random seed for reproducibility
    """

    MIN_RADIUS_M = 20.0  # tightest allowable corner radius
    STRAIGHT_THRESHOLD = 1.0 / 100.0  # curvature below this is considered straight.

    def __init__(self,
                 n_points: int = 70,
                 n_regions: int = 10,
                 min_bound: float = 10.0,
                 max_bound: float = 500.0,
                 origin_lat: float = 49.2606,
                 origin_lon: float = -123.2460,
                 v_max: float = 18.0,
                 v_min: float = 6.0,
                 min_radius_m: float = 20.0,
                 n_centerline: int = 1000,
                 rng_seed: int = None):

        self.n_points = n_points
        self.n_regions = n_regions
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.v_max = v_max
        self.v_min = v_min
        self.curvature_threshold = 1.0 / min_radius_m
        self.n_centerline = n_centerline

        if rng_seed is not None:
            np.random.seed(rng_seed)

    def _bounded_voronoi(self, input_points):
        bb = np.array([self.min_bound, self.max_bound] * 2)
        x_min, x_max, y_min, y_max = bb

        def _mirror(pts, boundary, axis):
            m = pts.copy()
            m[:, axis] = 2 * boundary - m[:, axis]
            return m

        points = np.concatenate([
            input_points,
            _mirror(input_points, x_min, 0),
            _mirror(input_points, x_max, 0),
            _mirror(input_points, y_min, 1),
            _mirror(input_points, y_max, 1),
        ])

        vor = spatial.Voronoi(points)
        vor.filtered_points = input_points
        vor.filtered_regions = np.array(
            vor.regions, dtype=object
        )[vor.point_region[:vor.npoints // 5]]
        return vor

    # Track generation

    def generate(self):
        """
        generate track and return as a dataframe (time_s, lat, lon)
        :return: dataframe
        """

        pts = np.random.uniform(self.min_bound, self.max_bound,
                                (self.n_points, 2))
        vor = self._bounded_voronoi(pts)

        for attempt in range(1, 51):
            idx = np.random.randint(0, self.n_points)
            r_idxs = [idx] + [closest_node(pts[idx], pts, k=i + 1)
                              for i in range(self.n_regions - 1)]

            regions = np.array(vor.regions, dtype=object)
            vertices = np.unique(
                vor.vertices[np.concatenate(regions[vor.point_region[r_idxs]])],
                axis=0
            )
            verts = clockwise_sort(vertices)
            verts = np.vstack([verts, verts[0]])

            # Iteratively remove vertices that cause excessive curvature
            valid = True
            for _ in range(50):
                if len(verts) < 4:
                    valid = False;
                    break
                try:
                    tck, _ = interpolate.splprep(
                        [verts[:, 0], verts[:, 1]], s=0, per=True)
                except Exception:
                    valid = False;
                    break

                t_spl = np.linspace(0, 1, self.n_centerline)
                x, y = interpolate.splev(t_spl, tck, der=0)
                dx, dy = interpolate.splev(t_spl, tck, der=1)
                d2x, d2y = interpolate.splev(t_spl, tck, der=2)
                k_abs = np.abs(curvature(dx, d2x, dy, d2y))

                peaks, _ = signal.find_peaks(k_abs)
                if len(peaks) == 0:
                    break
                if k_abs[peaks].max() > self.curvature_threshold:
                    worst = peaks[k_abs[peaks].argmax()]
                    v_idx = closest_node((x[worst], y[worst]), verts, k=0)
                    verts = np.delete(verts, v_idx, axis=0)
                    if not np.array_equal(verts[0], verts[-1]):
                        verts = np.vstack([verts, verts[0]])
                else:
                    break

            if not valid:
                continue
            if not (Polygon(zip(x, y)).is_valid):
                continue

            # Speed profile and time axis
            speed = assign_speed(k_abs, self.v_max, self.v_min, self.curvature_threshold)
            t_arr = curvature_to_time(x, y, speed)

            # Centre track and convert to lat/lon
            x_c = x - x.mean()
            y_c = y - y.mean()
            lat, lon = xy_to_latlon(x_c, y_c, self.origin_lat, self.origin_lon)

            self._x, self._y = x_c, y_c
            self._speed, self._k = speed, k_abs
            self._lat, self._lon = lat, lon

            df = pd.DataFrame({"time_s": t_arr, "lat": lat, "lon": lon})
            print(f"Track generated (attempt {attempt}): "
                  f"{t_arr[-1]:.1f}s, "
                  f"{np.sum(np.hypot(np.diff(x), np.diff(y))):.0f}m, "
                  f"min radius {1 / k_abs.max():.1f}m, "
                  f"speed {speed.min() * 3.6:.0f}–{speed.max() * 3.6:.0f} km/h")

            # if out_csv:
            #     df.to_csv(out_csv, index=False)
            #     print(f"Saved → {out_csv}")

            return df

        raise RuntimeError("Could not generate valid track in 50 attempts.")

    def generate_multilap(self, n_laps, speed_change=0.50, rng=None):
        """
        Build continuous multilap path that uses the same spatial geometry with a slight shift in the v_min / v_max values


        """

        rng = rng or np.random.default_rng()
        frames = []
        t_cursor = 0.0
        self.lap_info = []

        for lap in range(
                n_laps):  # for every lap we randomly either increment/decrement the v_min / v_max by a number in a specified range.
            change = 1.0 + rng.uniform(-speed_change, speed_change)
            v_max_lap = self.v_max * change
            v_min_lap = self.v_min * change

            speed_lap = assign_speed(self._k, v_max_lap, v_min_lap, self.curvature_threshold)
            t_arr = curvature_to_time(self._x, self._y, speed_lap)  # arc length unaffected by centering

            lap_df = pd.DataFrame({
                "time_s": t_arr + t_cursor,
                "lat": self._lat,
                "lon": self._lon,
            })
            if lap > 0:
                lap_df = lap_df.iloc[1:]  # drop duplicate (same time stamps as previous lap's last row)

            frames.append(lap_df)
            lap_duration = t_arr[-1]
            self.lap_info.append({
                "lap": lap + 1, "v_max": v_max_lap, "v_min": v_min_lap,
                "duration": lap_duration, "start_t": t_cursor,
            })

            t_cursor += lap_duration

        return pd.concat(frames, ignore_index=True)

    def plot(self, df: pd.DataFrame, title: str = "Voronoi Track",
             save_path: str = None):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(title, fontweight="bold")

        ax = axes[0]
        sc = ax.scatter(self._x, self._y,
                        c=self._speed * 3.6,

                        vmin=self.v_min * 3.6, vmax=self.v_max * 3.6)
        plt.colorbar(sc, ax=ax, label="Speed (km/h)")
        ax.plot(self._x[0], self._y[0], "ko", ms=8, label="Start", zorder=5)
        n = len(self._x)
        for i in range(0, n, max(1, n // 10)):
            if i + 1 < n:
                ax.annotate("", xy=(self._x[i + 1], self._y[i + 1]),
                            xytext=(self._x[i], self._y[i]),
                            arrowprops=dict(arrowstyle="->", color="navy", lw=0.8))
        ax.set_aspect("equal");
        ax.grid(alpha=0.3)
        ax.set_xlabel("East (m)");
        ax.set_ylabel("North (m)")
        ax.set_title("Track (coloured by speed)");
        ax.legend()

        axes[1].plot(df["time_s"], self._speed * 3.6, "tomato", lw=1.2)
        axes[1].axhline(self.v_max * 3.6, color="green", ls="--", lw=0.8,
                        label=f"v_max {self.v_max * 3.6:.0f} km/h")
        axes[1].axhline(self.v_min * 3.6, color="orange", ls="--", lw=0.8,
                        label=f"v_min {self.v_min * 3.6:.0f} km/h")
        axes[1].set_xlabel("Time (s)");
        axes[1].set_ylabel("Speed (km/h)")
        axes[1].set_title("Speed Profile");
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        axes[2].plot(df["time_s"], self._k, "royalblue", lw=1.0)
        axes[2].axhline(self.curvature_threshold, color="red", ls="--", lw=0.8,
                        label=f"min radius {1 / self.curvature_threshold:.0f}m")
        axes[2].set_xlabel("Time (s)");
        axes[2].set_ylabel("Curvature (m⁻¹)")
        axes[2].set_title("Curvature Profile");
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()

    def plot_multilap(self, df: pd.DataFrame, title: str = "Multi-Lap Track", save_path: str = None):

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontweight="bold")

        # Trajectory geometry is identical every lap
        ax = axes[0]
        ax.plot(self._x, self._y, "royalblue", lw=1.5)
        ax.plot(self._x[0], self._y[0], "ko", ms=8, label="Start/Finish", zorder=5)
        ax.set_aspect("equal");
        ax.grid(alpha=0.3)
        ax.set_xlabel("East (m)");
        ax.set_ylabel("North (m)")
        ax.set_title("Track (fixed geometry)");
        ax.legend()
        ax = axes[1]
        for info in self.lap_info:
            speed_lap = assign_speed(self._k, info["v_max"], info["v_min"], self.curvature_threshold)
            t_arr = curvature_to_time(self._x, self._y, speed_lap) + info["start_t"]
            ax.plot(t_arr, speed_lap * 3.6, lw=1.2, label=f"Lap {info['lap']}")
        ax.set_xlabel("Time (s)");
        ax.set_ylabel("Speed (km/h)")
        ax.set_title("Speed Profile per Lap");
        ax.legend(fontsize=8);
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            print(f"Saved → {save_path}")
        plt.show()
