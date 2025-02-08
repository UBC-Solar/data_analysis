import numpy as np
import math
from physics.environment.gis import GIS
from geopy.distance import geodesic  # For distance calculation
import matplotlib.pyplot as plt
import pandas as pd


from typing import Union


bs_config = {
  "panel_efficiency": 0.2432,
  "panel_size": 4.0,

  "max_voltage": 134.4,
  "min_voltage": 86.4,
  "num_cells_per_module": 13,
  "num_modules": 32,
  "cell_charge_rating": 3300,
  "max_current_capacity": 48.9,
  "max_energy_capacity": 5400,

  "lvs_voltage": 12,
  "lvs_current": 1.50,

  "vehicle_mass": 350,
  "road_friction": 0.012,
  "tire_radius": 0.2032,
  "vehicle_frontal_area": 1.1853,
  "drag_coefficient": 0.11609,

  "max_acceleration_kmh_per_s": 6,
  "max_deceleration_kmh_per_s": 6,

  "max_speed_during_turn": 20,
  "max_cruising_speed": 80
}

AIR_DENSITY = 1.225
ACCELERATION_G = 9.81


def calculate_motor_efficiency_value(motor_output_power, revolutions_per_minute):
    return 0.7382 - (6.281e-5 * motor_output_power) + (6.708e-4 * revolutions_per_minute) \
        - (2.89e-8 * motor_output_power ** 2) + (2.416e-7 * motor_output_power * revolutions_per_minute) \
        - (8.672e-7 * revolutions_per_minute ** 2) + (5.653e-12 * motor_output_power ** 3) \
        - (1.74e-11 * motor_output_power ** 2 * revolutions_per_minute) \
        - (7.322e-11 * motor_output_power * revolutions_per_minute ** 2) \
        + (3.263e-10 * revolutions_per_minute ** 3)


def calculate_motor_controller_efficiency_value(motor_angular_speed, motor_torque_array):
    return 0.7694 + (0.007818 * motor_angular_speed) + (0.007043 * motor_torque_array) \
        - (1.658e-4 * motor_angular_speed ** 2) - (1.806e-5 * motor_torque_array * motor_angular_speed) \
        - (1.909e-4 * motor_torque_array ** 2) + (1.602e-6 * motor_angular_speed ** 3) \
        + (4.236e-7 * motor_angular_speed ** 2 * motor_torque_array) \
        - (2.306e-7 * motor_angular_speed * motor_torque_array ** 2) \
        + (2.122e-06 * motor_torque_array ** 3) - (5.701e-09 * motor_angular_speed ** 4) \
        - (2.054e-9 * motor_angular_speed ** 3 * motor_torque_array) \
        - (3.126e-10 * motor_angular_speed ** 2 * motor_torque_array ** 2) \
        + (1.708e-09 * motor_angular_speed * motor_torque_array ** 3) \
        - (8.094e-09 * motor_torque_array ** 4)

def forward_fill_nans(array: np.ndarray):
    for i in range(1, len(array)):
        if np.isnan(array[i]).any():
            array[i] = array[i - 1]
    return array

def calculate_motor_efficiency(motor_angular_speed, motor_output_energy, tick):
    """

    Calculates a NumPy array of motor efficiency from NumPy array of operating angular speeds and NumPy array
        of output power. Based on data obtained from NGM SC-M150 Datasheet and modelling done in MATLAB

    r squared value: 0.873

    :param np.ndarray motor_angular_speed: (float[N]) angular speed motor operates in rad/s
    :param np.ndarray motor_output_energy: (float[N]) energy motor outputs to the wheel in J
    :param float tick: length of 1 update cycle in seconds
    :returns e_m: (float[N]) efficiency of the motor
    :rtype: np.ndarray

    """

    # Power = Energy / Time
    motor_output_power = motor_output_energy * tick
    rads_rpm_conversion_factor = 30 / math.pi

    revolutions_per_minute = motor_angular_speed * rads_rpm_conversion_factor

    e_m = calculate_motor_efficiency_value(motor_output_power, revolutions_per_minute)

    e_m[e_m < 0.7382] = 0.7382
    e_m[e_m > 1] = 1

    return e_m

def calculate_total_motor_power(array, tick_start, tick_end):
    return np.cumsum(array[tick_start:tick_end]) * tick_end

def calculate_motor_controller_efficiency(motor_angular_speed, motor_output_energy, tick):
    """

    Calculates a NumPy array of motor controller efficiency from NumPy array of operating angular speeds and
    NumPy array of output power. Based on data obtained from the WaveSculptor Motor Controller Datasheet efficiency
    curve for a 90 V DC Bus and modelling done in MATLAB.

    r squared value: 0.7431

    :param np.ndarray motor_angular_speed: (float[N]) angular speed motor operates in rad/s
    :param np.ndarray motor_output_energy: (float[N]) energy motor outputs to the wheel in J
    :param float tick: length of 1 update cycle in seconds
    :returns e_mc (float[N]) efficiency of the motor controller
    :rtype: np.ndarray

    """

    # Power = Energy / Time
    motor_output_power = motor_output_energy / tick

    # Torque = Power / Angular Speed
    motor_torque_array = np.nan_to_num(motor_output_power / motor_angular_speed)

    np.seterr(divide='warn', invalid='warn')

    e_mc = calculate_motor_controller_efficiency_value(motor_angular_speed, motor_torque_array)

    e_mc[e_mc < 0.9] = 0.9
    e_mc[e_mc > 1] = 1

    return e_mc


def calculate_energy_in(required_speed_kmh, gradients, wind_speeds, tick):
    """

    Create a function which takes in array of elevation, array of wind speed, required
        speed, returns the consumed energy.

    :param np.ndarray required_speed_kmh: (float[N]) required speed array in km/h
    :param np.ndarray gradients: (float[N]) gradient at parts of the road
    :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
    :param float tick: length of 1 update cycle in seconds
    :returns: (float[N]) energy expended by the motor at every tick
    :rtype: np.ndarray

    """

    DRAG_COEF = 1

    # Temporary constants for testing
    vehicle_mass = bs_config["vehicle_mass"]
    drag_coefficient = bs_config["drag_coefficient"]
    vehicle_frontal_area = bs_config["vehicle_frontal_area"]
    road_friction = bs_config["road_friction"]
    tire_radius = bs_config["tire_radius"]

    required_speed_ms = required_speed_kmh / 3.6

    acceleration_ms2 = np.clip(np.gradient(required_speed_ms), a_min=0, a_max=None)
    acceleration_force = acceleration_ms2 * vehicle_mass

    required_angular_speed_rads = required_speed_ms / tire_radius

    drag_forces = DRAG_COEF * 0.5 * AIR_DENSITY * (
            (required_speed_ms + wind_speeds) ** 2) * drag_coefficient * vehicle_frontal_area

    angles = np.arctan(gradients)
    g_forces = vehicle_mass * ACCELERATION_G * np.sin(angles)

    road_friction_array = road_friction * vehicle_mass * ACCELERATION_G * np.cos(angles)

    net_force = road_friction_array + drag_forces + g_forces + acceleration_force

    motor_output_energies = required_angular_speed_rads * net_force * tire_radius * tick
    motor_output_energies = np.clip(motor_output_energies, a_min=0, a_max=None)

    e_m = calculate_motor_efficiency(required_angular_speed_rads, motor_output_energies, tick)
    e_mc = calculate_motor_controller_efficiency(required_angular_speed_rads, motor_output_energies, tick)

    motor_controller_input_energies = motor_output_energies / (e_m * e_mc)

    # Filter out and replace negative energy consumption as 0
    motor_controller_input_energies = np.where(motor_controller_input_energies > 0,
                                               motor_controller_input_energies, 0)

    return motor_controller_input_energies

def calculate_speed_kmh(gis_coords: np.ndarray) -> np.ndarray:
    """
    Calculate speed (km/h) between GIS waypoints, assuming 1 second intervals.

    :param gis_coords: A NumPy array of shape (N, 2) containing latitude and longitude pairs.
    :return: A NumPy array of speed values (km/h) for each segment.
    """
    num_points = len(gis_coords)
    if num_points < 2:
        return np.array([])

    # Create a copy to prevent modifying the original array
    gis_coords = gis_coords.copy()

    # TODO: this should really zero out speeds associated with Nan values since these occured during pitting
    # Forward fill NaN values since they mess with distance calculations
    for i in range(1, len(gis_coords)):
        if np.isnan(gis_coords[i]).any():
            gis_coords[i] = gis_coords[i - 1]


    speed_kmh = np.zeros(num_points - 1)

    for i in range(1, num_points):
        # Get previous and current coordinates (lat, lon)
        prev_point = tuple(gis_coords[i - 1])
        curr_point = tuple(gis_coords[i])

        # Compute distance in meters using geopy (Haversine formula)
        distance_m = geodesic(prev_point, curr_point).meters

        # Compute speed in km/h (since time difference = 1 second)
        speed_kmh[i - 1] = (distance_m / 1) * 3.6  # Convert m/s to km/h

    return speed_kmh

def run_motor_model() -> (np.ndarray, np.ndarray):
    """
    Adapted from Simulation's motor model
    """
    tick = 1 # seconds

    # from /Simulation/simulation/cache/route/route_data.npz
    route_data = np.load("route_data_FSGP.npz")
    gis_coords = np.load("coords_day1.npy")
    gis_indices = np.load("coord_indices_day1.npy")


    # Day 1 index 0 = index 269 of Simulation's gis index 0
    start_index = 269
    gis_indices = np.roll(gis_indices, start_index)  # np.ndarray[float]
    gis_indices = forward_fill_nans(gis_indices)     # ensure there are no Nans since they mess with indexing
    gis_indices = np.round(gis_indices).astype(int)  # Convert to integers

    # ----- Expected distance estimate -----
    speed_kmh = load_data("vehicle_velocity.csv")
    speed_kmh = np.append(speed_kmh, 0)  # Proper way to append a value to a NumPy array

    """ closest_gis_indices is a 1:1 mapping between each point which has within it a timestamp and cumulative
            distance from a starting point, to its closest point on a map.

        closest_weather_indices is a 1:1 mapping between a weather condition, and its closest point on a map.
    """


    # from /Simulation/simulation/config/settings_FSGP.json
    origin_coord, current_coord = [
        38.9281815,
        -95.677021
    ]

    gis = GIS(
        route_data,
        origin_coord,
        current_coord
    )

    closest_gis_indices = gis_indices # set to the route data provided by Miguel


    path_distances = gis.get_path_distances()
    cumulative_distances = np.cumsum(path_distances)  # [cumulative_distances] = meters

    max_route_distance = cumulative_distances[-1]

    route_length = max_route_distance / 1000.0  # store the route length in kilometers

    # Array of elevations at every route point
    gis_route_elevations = gis.get_path_elevations()

    # Get the azimuth angle of the vehicle at every location
    # gis_vehicle_bearings = simulation.vehicle_bearings[closest_gis_indices]

    # Get array of path gradients
    gradients = gis.get_gradients(closest_gis_indices)


    # ----- Timing Calculations -----

    # Get time zones at each point on the GIS path

    # absolute_wind_speeds = simulation.meteorology.wind_speed
    # wind_directions = simulation.meteorology.wind_direction
    wind_speeds = np.zeros(len(gis_indices)) # zero out wind speeds for now

    # ----- Energy Calculations -----


    motor_consumed_energy = calculate_energy_in(
        speed_kmh,
        gradients,
        wind_speeds,
        tick
    )

    return motor_consumed_energy, gradients


# load data from a csv using dataframe
def load_data(path):
    data = pd.read_csv(path)
    values = data["Value"].to_numpy()
    return values


motor_power_predicted, gradient_global = run_motor_model()

# Generate tick indices
ticks = np.arange(len(motor_power_predicted))

motor_current = load_data("motor_current.csv")
motor_voltage = load_data("motor_voltage.csv")
vehicle_velocity = load_data("vehicle_velocity.csv")
motor_power_measured = motor_current * motor_voltage
motor_power_measured = np.append(motor_power_measured, 0)
vehicle_velocity = np.append(vehicle_velocity, 0)


predicted_sum = calculate_total_motor_power(motor_power_predicted, 0, 18000)[-1]
measured_sum = calculate_total_motor_power(motor_power_measured, 0, 18000)[-1]
percent_error = predicted_sum * 100 / measured_sum
print(f'\n\n\n')
print(f'________ Predicted Total Energy: {predicted_sum}')
print(f'________ Measured Total Energy: {measured_sum}')
print(f'________ Percent Error: {percent_error}')
print(f'\n\n\n')

# ----- Plotting Logic -----

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot motor power (Predicted & Measured) on the primary y-axis
ax1.plot(ticks, motor_power_predicted, linestyle='-', label="Predicted Motor Power", color='blue')
ax1.plot(ticks, motor_power_measured, linestyle='-', label="Measured Motor Power", color='red')
ax1.set_xlabel("Tick")
ax1.set_ylabel("Power (Watts)", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True)

# Create secondary y-axis for vehicle velocity
ax2 = ax1.twinx()
ax2.plot(ticks, vehicle_velocity, linestyle='-', label="Vehicle Velocity", color='green')
ax2.set_ylabel("Vehicle Velocity (km/h)", color='green')
ax2.tick_params(axis='y', labelcolor='green')

ax3 = ax1.twinx()
ax3.plot(ticks, gradient_global, linestyle='-', label="Gradient", color='orange')
ax3.set_ylabel("Gradient", color='orange')
ax3.tick_params(axis='y', labelcolor='orange')

# Add legends for both plots
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Set title
plt.title("Motor Power vs Vehicle Velocity")

# Show the plot
plt.show()


