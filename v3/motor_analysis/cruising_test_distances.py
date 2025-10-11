import matplotlib.pyplot as plt
import numpy as np

# config in SI units
max_accel = 1.1111 # 0-20km/h in 5s
cruising_decel = -0.3
brake_decel = -1

# cruising start/stop speeds
start_speeds_kmh = [80, 70, 60, 50, 40, 30]
stop_speeds_kmh = [50, 40, 30, 20, 10, 0]

# 2d list
accel_distances = np.ones([len(start_speeds_kmh), len(stop_speeds_kmh)]) * np.nan
cruise_distances = np.ones([len(start_speeds_kmh), len(stop_speeds_kmh)]) * np.nan
brake_distances = np.ones([len(start_speeds_kmh), len(stop_speeds_kmh)]) * np.nan

for i, start_speed in enumerate(start_speeds_kmh):
    for j, stop_speed in enumerate(stop_speeds_kmh):

        if stop_speed > start_speed:
            continue

        # convert to meters per second
        start_mps = start_speed / 3.6
        stop_mps = stop_speed / 3.6

        # delta_t = delta_v / a
        # delta_x = v_i * t + 1/2at^2
        accel_time = start_mps / max_accel  # v_i = 0
        accel_distance = 1/2 * max_accel * accel_time ** 2
        cruise_time = (stop_mps - start_mps) / cruising_decel
        cruise_distance = (start_mps * cruise_time) + 1 / 2 * cruising_decel * cruise_time ** 2
        brake_time = (0 - stop_mps) / brake_decel
        brake_distance = stop_mps * brake_time + 1/2 * brake_decel * brake_time ** 2

        accel_distances[i, j] = np.abs(accel_distance)
        cruise_distances[i, j] = np.abs(cruise_distance)
        brake_distances[i, j] = np.abs(brake_distance)

total_distances = accel_distances + cruise_distances + brake_distances

# make 2x2 grid of plots
fig, axes_2d = plt.subplots(2, 2)
axs = axes_2d.ravel()

data_to_plot = [total_distances, accel_distances, cruise_distances, brake_distances]
labels = ["Total Required Distance (m)", "Accelerating Distance (m)", "Cruising Distance (m)", "Braking Distance (m)",]
# total distances
for ax, data, label in zip(axs, data_to_plot, labels):
    im = ax.imshow(data)
    ax.set_yticks(range(len(start_speeds_kmh)), labels=start_speeds_kmh,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(range(len(stop_speeds_kmh)), labels=stop_speeds_kmh)
    for i in range(len(start_speeds_kmh)):
        for j in range(len(stop_speeds_kmh)):
            text = ax.text(j, i, f"{data[i, j]:.0f}",
                           ha="center", va="center", color="w")
    ax.set_title(f"{label}")
    ax.set_ylabel("Start Speed (km/h)")
    ax.set_xlabel("Stop Speed (km/h)")
fig.tight_layout()
fig.suptitle("Cruising Test Required Distances\n"
             f"{max_accel=}m/s^2\n{cruising_decel=}m/s^2\n{brake_decel=}m/s^2")
plt.subplots_adjust(top=0.8)  # make space for title
plt.show()