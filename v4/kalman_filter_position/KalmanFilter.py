import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from random_generation import Path, xy_to_latlon, latlon_to_xy
from random_generation import *
from track_generator import *


class ExtendedKalmanFilter:
    """
    Fuses GPS + IMU + speed to estimate position and velocity.

    State: x = [x_m, y_m, vx, vy]
    x_m, y_m  : position in local East-North frame (metres)
    vx, vy   : East-North velocity in m/s

    the motion model is non-linear because acceleration heading is used to estimate the velocity direction, making F and H Jacobians necessary.

    Implementation:
    - predict step: uses IMU data
    - update step: GPS+speed updates
    """

    def __init__(self, x0, P0, Q, R_gps, R_speed):
        self.x = x0.astype(float)  # inital state [x,y,vx,vy]
        self.P = P0.astype(float)  # inital covariance 4x4
        self.Q = Q.astype(float)  # process noise covariance 4x4
        self.R_gps = R_gps.astype(float)  # gps measurement noise 2x2
        self.R_speed = float(R_speed)  # scalar speed measurement noise (m/s)^2
        self._log: list[dict] = []  # stores history

    def predict(self, ax: float, ay: float, dt: float):
        """
            IMU prediction step runs at 30 Hz.
            Propagate state forward by dt seconds using IMU derived acceleration.
            Process model uses constant acceleration over dt
            The Jacobian matrix linearises the motion model, F is the matrix of differentials
        """
        x, y, vx, vy = self.x
        self.x = np.array([  # using kinematic equations
            x + vx * dt + 0.5 * ax * dt ** 2,  # s= ut + 0.5 a t^2
            y + vy * dt + 0.5 * ay * dt ** 2,
            vx + ax * dt,  # v = u + at
            vy + ay * dt,
        ])
        # state transition Jacobian
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)

        # covariance prediction: P- = F. P. F^T + Q
        self.P = F @ self.P @ F.T + self.Q  # @ for matrix multiplication.

    # measurement update using core EKF equations.
    def _update(self, z, H, R, z_pred):
        """
        Core EKF update equations (Joseph form for numerical stability).

        innovation (y or v) the difference between the actual measurement (z) and the expected measurement based on the prior state prediction (h(x)), defined as y = z - h(x)
        z is the measurement vector i.e. the sensor observation at a specific timestep. 
        H is the observation matrix.

        y = z - h(x) innovation residual
        S = H . P. H^T + R innovation covariance
        K = P. H ^T . S-1  Kalman gain
        X = X + K.Y  State update
        P = (I - K.H).P covariance update in the Joseph form. The Joseph form
        The Joseph form  P = (I-KH)P(I-KH)^T + KRK^T  is numerically more
        stable than the simpler  P = (I-KH)P  despite a higher computation time.

        """

        n = len(self.x)
        I = np.eye(n)  # builds the 2d identity matrix.
        y_inn = z - z_pred  # innovation
        S = H @ self.P @ H.T + R  # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y_inn
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def update_gps(self, x_m: float, y_m: float):
        """
        GPS directly observes position in the E-N frame.
        Measurement model (linear):

         h(x) = [x_m, y_m],
         H = [[1,0,0,0], [0,1,0,0]]

         """
        Z = np.array([x_m, y_m])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        Z_pred = H @ self.x  # multiply with state vector.
        self._update(np.array([x_m, y_m]), H, self.R_gps, H @ self.x)

    def update_speed(self, speed_ms: float):
        """
        Motor speed gives the scalar magnitude of velocity. Measurement model (non-linear):
        h(x) = sqrt(vx^2 +vy^2)

        Linearised Jacobian:
        H  = [0,0, vx / |v|, vy/|v|]
        if |v| is less than 0.1 m/s this update is skipped to avoid division by zero issues.

        """
        vx, vy = self.x[2], self.x[3]
        v_norm = np.hypot(vx, vy)
        if v_norm < 0.1:
            return
        H = np.array([[0, 0, vx / v_norm, vy / v_norm]])
        self._update(np.array([speed_ms]), H,
                     np.array([[self.R_speed]]), np.array([v_norm]))

    def log(self, t: float):
        self._log.append({
            "t": t,
            "x_m": self.x[0], "y_m": self.x[1],
            "vx": self.x[2], "vy": self.x[3],
            "std_x": np.sqrt(self.P[0, 0]),
            "std_y": np.sqrt(self.P[1, 1]),
        })

    def trajectory(self) -> pd.DataFrame:
        df = pd.DataFrame(self._log)
        return df


# MAIN RUN EKF LOOP:


def build_ekf(gps_df, path):
    """Constructs a new EKF, initialize based on the first GPS values."""
    lat0, lon0 = path.lat0, path.lon0
    gps_x, gps_y = latlon_to_xy(gps_df["lat"].values, gps_df["lon"].values, lat0, lon0)
    x0 = np.array([gps_x[0], gps_y[0], 0.0, 0.0])
    # initial covariance: gps uncertainty is given as 5m and then variance is 25m^2
    # velocity variance is 10 (m/s)^2

    P0 = np.diag([25.0, 25.0, 10.0, 10.0])

    # process noise so position and velocity noise
    Q = np.diag([0.01, 0.01, 0.10, 0.10])

    # gps measurement noise ~5m
    R_gps = np.diag([25.0, 25.0])
    R_speed = 0.25

    return ExtendedKalmanFilter(x0, P0, Q, R_gps, R_speed)


def run_ekf(ekf, gps_df, imu_df, speed_df, path, t_offset=0.0):
    """
    interleave GPS and IMU/speed events in time order. Runs predict() on each IMU tick and update_gps/update_speed() when those measurements are available.

    :param gps_df
    :param imu_df
    :param speed_df
    :param path
    :return: traj_df: EKF state trajectory
    :return: gps_df: raw GPS positions in E-N (coordinates, needs to be converted to lat/lon)
    """
    lat0, lon0 = path.lat0, path.lon0

    # convert gps to xy
    gps_x, gps_y = latlon_to_xy(gps_df["lat"].values, gps_df["lon"].values, lat0, lon0)
    # x0 = np.array([gps_x[0], gps_y[0], 0.0, 0.0])

    # building the gps lookup at each IMU tick index.

    gps_ptr = 0
    speed_ptr = 0
    t_imu = imu_df["t"].values

    # fusion loop that repeats the prediction state (imu) and update state (gps and speed). the prediction state uses the system model to guess the next state and thisestimate and uncertainty are propagated forward in time. the update state compares the real measurement to the predicted measurement and then applies the kalman gain. this is used to adjust the final state estimate.
    for i in range(len(imu_df)):
        t = t_imu[i]
        dt = t - t_imu[i - 1] if i > 0 else 1 / 30.0

        ekf.predict(imu_df["ax"].iloc[i], imu_df["ay"].iloc[i], dt)
        while speed_ptr < len(speed_df) and speed_df["t"].iloc[speed_ptr] <= t:
            ekf.update_speed(speed_df["speed"].iloc[speed_ptr])
            speed_ptr += 1

        while gps_ptr < len(gps_df) and gps_df["t"].iloc[gps_ptr] <= t:
            ekf.update_gps(gps_x[gps_ptr], gps_y[gps_ptr])
            gps_ptr += 1

        ekf.log(t + t_offset)  # global time, so logs across laps dont overlap

    traj = ekf.trajectory()
    traj["lat"], traj["lon"] = xy_to_latlon(traj["x_m"].values, traj["y_m"].values, lat0, lon0)
    return traj


# evaluate and compare ekf version with ground truth.


def evaluate(traj: pd.DataFrame, path: Path,
             gps_df: pd.DataFrame, lap_duration, label: str = ""):

    """


    :param traj:
    :param path:
    :param gps_df:
    :param lap_duration:
    :param label:
    :return: RMSE error associated with the velocity and position results of the EKF.
    """
    t = traj["t"].values
    # t_local = np.mod(t, lap_duration)

    # position error
    gt_x = path.x(t)
    gt_y = path.y(t)
    err_ekf = np.hypot(traj["x_m"].values - gt_x,
                       traj["y_m"].values - gt_y)

    t_gps = gps_df["t"].values
    # t_gps_local = np.mod(t_gps,
    #                      lap_duration)  # to wrap time into [o, lap_duration] before beig interpolated for path.x, vx, vy etc so that it works correctly across multiple ocncatenated laps.

    gps_x, gps_y = latlon_to_xy(gps_df["lat"].values, gps_df["lon"].values,
                                path.lat0, path.lon0)
    err_gps = np.hypot(gps_x - path.x(t_gps),
                       gps_y - path.y(t_gps))

    # velocity error
    gt_vx = path.vx(t)
    gt_vy = path.vy(t)

    err_vx = traj["vx"].values - gt_vx  # signed
    err_vy = traj["vy"].values - gt_vy
    err_speed = np.hypot(traj["vx"].values, traj["vy"].values) \
                - np.hypot(gt_vx, gt_vy)  # scalar speed error of signed values.
    err_vel_magnitude = np.hypot(err_vx, err_vy)  # combined, like position

    rmse_vx = np.sqrt(np.mean(err_vx ** 2))
    rmse_vy = np.sqrt(np.mean(err_vy ** 2))
    rmse_speed = np.sqrt(np.mean(err_speed ** 2))
    rmse_vel = np.sqrt(np.mean(err_vel_magnitude ** 2))

    tag = f"[{label}] " if label else ""
    print(f"\n{tag} Position ")
    print(f"  GPS  RMSE : {np.sqrt(np.mean(err_gps ** 2)):.2f} m")
    print(f"  EKF  RMSE : {np.sqrt(np.mean(err_ekf ** 2)):.2f} m")
    print(f"  Improvement : {(1 - np.mean(err_ekf) / np.mean(err_gps)) * 100:.1f}%")
    print(f"\n{tag} Velocity")
    print(f"  RMSE vx   : {rmse_vx:.3f} m/s")
    print(f"  RMSE vy   : {rmse_vy:.3f} m/s")
    print(f"  RMSE speed: {rmse_speed:.3f} m/s")
    print(f"  RMSE |v|  : {rmse_vel:.3f} m/s")

    return {
        "err_ekf": err_ekf, "err_gps": err_gps,
        "t": t, "t_gps": gps_df["t"].values,
        "gps_rmse": np.sqrt(np.mean(err_gps ** 2)),
        "ekf_rmse": np.sqrt(np.mean(err_ekf ** 2)),
        "err_vx": err_vx, "err_vy": err_vy,
        "err_speed": err_speed, "err_vel": err_vel_magnitude,
        "rmse_vx": rmse_vx, "rmse_vy": rmse_vy,
        "rmse_speed": rmse_speed, "rmse_vel": rmse_vel,
        "gt_speed": np.hypot(gt_vx, gt_vy),
    }


def plot_results(traj, gps_df, path, lap_duration, results, title="EKF Results", save_path=None):
    t = traj["t"].values
    #  t_local = np.mod(t, lap_duration)
    gt_x = path.x(t)
    gt_y = path.y(t)
    gps_x, gps_y = latlon_to_xy(gps_df["lat"].values, gps_df["lon"].values,
                                path.lat0, path.lon0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontweight="bold")

    # Trajectory
    ax = axes[0, 0]
    ax.plot(gt_x, gt_y, "k-", lw=2, label="Ground truth", zorder=3)
    ax.scatter(gps_x, gps_y, s=8, c="tomato", alpha=0.5,
               label=f"GPS (RMSE={results['gps_rmse']:.1f}m)", zorder=2)
    ax.plot(traj["x_m"], traj["y_m"], "royalblue", lw=1.5,
            label=f"EKF (RMSE={results['ekf_rmse']:.1f}m)", zorder=4)
    ax.set_aspect("equal");
    ax.grid(alpha=0.3)
    ax.set_xlabel("East (m)");
    ax.set_ylabel("North (m)")
    ax.set_title("Trajectory");
    ax.legend(fontsize=8)

    # Position error over time
    ax = axes[0, 1]
    ax.plot(results["t_gps"], results["err_gps"],
            "tomato", lw=0.8, alpha=0.7, label="GPS error")
    ax.plot(results["t"], results["err_ekf"],
            "royalblue", lw=1.2, label="EKF error")
    ax.set_xlabel("Time (s)");
    ax.set_ylabel("Position error (m)")
    ax.set_title("Position Error vs Ground Truth")
    ax.legend(fontsize=8);
    ax.grid(alpha=0.3)

    # Speed over time (EKF estimate vs ground truth)
    ax = axes[1, 0]
    ekf_speed = np.hypot(traj["vx"].values, traj["vy"].values)
    ax.plot(t, results["gt_speed"] * 3.6, "k-", lw=1.5,
            label="True speed", alpha=0.8)
    ax.plot(t, ekf_speed * 3.6, "royalblue", lw=1.2,
            label=f"EKF speed (RMSE={results['rmse_speed'] * 3.6:.2f} km/h)")
    ax.set_xlabel("Time (s)");
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed Estimate");
    ax.legend(fontsize=8);
    ax.grid(alpha=0.3)

    # Velocity component errors. these are signed so you can see bias direction
    ax = axes[1, 1]
    ax.plot(t, results["err_vx"], "royalblue", lw=1.0,
            alpha=0.8, label=f"vx error (RMSE={results['rmse_vx']:.3f} m/s)")
    ax.plot(t, results["err_vy"], "darkorange", lw=1.0,
            alpha=0.8, label=f"vy error (RMSE={results['rmse_vy']:.3f} m/s)")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Time (s)");
    ax.set_ylabel("Velocity error (m/s)")
    ax.set_title("Velocity Error (signed)");
    ax.legend(fontsize=8);
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def plot_results_per_lap(traj, gps_df, path, results, lap_info, title_prefix="EKF Lap", save_dir=None):
    """
    Renders plot_results per lap to avoid crowding, slices results based on [start_t, start_t + lap_duration] as returned by the lap_info field from the Voronoi Track Generator.

    """
    t_all = traj["t"].values
    t_gps_all = gps_df["t"].values

    for info in lap_info:
        lap_num = info["lap"]
        t0, t1 = info["start_t"], info["start_t"] + info["duration"]

        # slice trajectory to this lap's time window
        mask_traj = (t_all >= t0) & (t_all < t1)
        traj_lap = traj[mask_traj].reset_index(drop=True)

        # slice gps to this lap's time window
        mask_gps = (t_gps_all >= t0) & (t_gps_all < t1)
        gps_lap = gps_df[mask_gps].reset_index(drop=True)

        if len(traj_lap) == 0:
            print(f"Lap {lap_num}: no trajectory samples in window, skipping")
            continue

        # re-derive per-lap ground truth/error to avoid misalignment issues.
        lap_results = evaluate(traj_lap, path, gps_lap, None,
                               label=f"Lap {lap_num}")

        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"ekf_lap{lap_num}.png")

        plot_results(traj_lap, gps_lap, path, None, lap_results,
                     title=f"{title_prefix} {lap_num}",
                     save_path=save_path)


# main loop:
if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    n_laps = 3
    print("Generating track")
    gen = VoronoiTrackGenerator(
        n_points=70, n_regions=10,
        min_bound=10.0, max_bound=500.0,
        origin_lat=49.2606, origin_lon=-123.2460,
        v_max=18.0, v_min=6.0,
        min_radius_m=20.0, n_centerline=1000,
        rng_seed=41,
    )

    gen.generate()
    speed_rng = np.random.default_rng(4)
    gt_df = gen.generate_multilap(n_laps=n_laps, speed_change=0.50, rng=speed_rng)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(HERE, "ground_truth_path.csv")
    gt_df.to_csv(csv_path, index=False)
    print("[Saved ground_truth_path.csv]")

    gen.plot_multilap(gt_df, title="Ground Truth Track",
                      save_path=os.path.join(HERE, "track.png"))

    path = Path(csv_path)
    lap_duration = np.float64(path.duration)

    gps_df = GPSSensor(rate_hz=4, noise_m=5.0, rng_seed=21).sample(path)
    imu_df = IMUSensor(rate_hz=30, noise_std_mss=0.05, bias_mss=0.2, rng_seed=32).sample(path)
    speed_df = SpeedSensor(rate_hz=30, noise_mps=0.43, rng_seed=42).sample(path)

    print(f"GPS    : {len(gps_df)} samples @ 4 Hz")
    print(f"IMU    : {len(imu_df)} samples @ 30 Hz")
    print(f"Speed  : {len(speed_df)} samples @ 30 Hz")

    ekf = build_ekf(gps_df, path)
    full_traj = run_ekf(ekf, gps_df, imu_df, speed_df, path)  # t_offset not needed, one continuous pass

    results = evaluate(full_traj, path, gps_df, lap_duration=None, label=f"{n_laps}-lap variable pace")
    plot_results(full_traj, gps_df, path, None, results,
                 title=f"EKF  {n_laps} Laps, Variable Pace",
                 save_path=os.path.join(HERE, "ekf_multilap_variable_pace.png"))
    # # to analyse multiple laps, i'm still using the same initial lat, lon point as a reference. concatenating laps does not necessarily require a change in reference frame.
    plot_results_per_lap(full_traj, gps_df, path, results, gen.lap_info,
                         title_prefix="EKF Lap", save_dir=HERE)
