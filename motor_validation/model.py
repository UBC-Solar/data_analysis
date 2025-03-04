import numpy as np
from physics.environment.gis import GIS
from geopy.distance import geodesic  # For distance calculation
import matplotlib.pyplot as plt
import pandas as pd

from efficiencies import calculate_motor_efficiency, calculate_motor_controller_efficiency
from plotting import plot_power, plot_forces, plot_cornering_data, plot_singe_value
from cornering_stuff import calculate_radii, get_slip_angle_for_tire_force


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

# Temporary constants for testing
vehicle_mass = bs_config["vehicle_mass"]
drag_coefficient = bs_config["drag_coefficient"]
vehicle_frontal_area = bs_config["vehicle_frontal_area"]
road_friction = bs_config["road_friction"]
tire_radius = bs_config["tire_radius"]

def forward_fill_nans(array: np.ndarray):
    for i in range(1, len(array)):
        if np.isnan(array[i]).any():
            array[i] = array[i - 1]
    return array


def calculate_total_motor_power(array, tick_start, tick_end):
    return np.cumsum(array[tick_start:tick_end]) * tick_end


def calculate_cornering_losses(required_speed_kmh, gis_waypoints, tick):
    required_speed_ms = required_speed_kmh / 3.6

    cornering_radii = calculate_radii(gis_waypoints)
    plot_singe_value(cornering_radii)
    centripetal_lateral_force = vehicle_mass * (required_speed_ms ** 2) / cornering_radii
    centripetal_lateral_force = np.clip(centripetal_lateral_force, a_min=0, a_max=10000)

    slip_angles_degrees = get_slip_angle_for_tire_force(centripetal_lateral_force)
    slip_angles_radians = np.radians(slip_angles_degrees)

    slip_distances = np.tan(slip_angles_radians) * required_speed_ms * tick
    cornering_friction_work = slip_distances * centripetal_lateral_force

    plot_cornering_data(slip_distances, slip_angles_degrees, centripetal_lateral_force)

    print("total slip distances: ")
    print(np.sum(slip_distances))
    print("\ntotal cornering_friction_work: ")
    print(np.sum(cornering_friction_work))
    print("\n")

    #   # Check for values above 8000 in centripetal_lateral_force
    # for i, force in enumerate(centripetal_lateral_force):
    #     if force > 8000:
    #         print(f"High centripetal force detected: {force} N")
    #         print(f"Speed: {required_speed_ms[i]} m/s")
    #         print(f"Cornering Radius: {cornering_radii[i]} m")
    #         print("\n \n")
    # Plotting the slip angles
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(slip_angles_degrees, marker='o', linestyle='-', color='b')
    # plt.title('plot')
    # plt.xlabel('index')
    # plt.ylabel('value')
    # plt.grid(True)
    # plt.show()

    CORNERING_COEFFICIENT = 1
    return cornering_friction_work * CORNERING_COEFFICIENT


def calculate_energy_in(required_speed_kmh, gradients, wind_speeds, gis_waypoints, tick):
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

    required_speed_ms = required_speed_kmh / 3.6

    acceleration_ms2 = np.clip(np.gradient(required_speed_ms), a_min=0, a_max=None)
    acceleration_force = acceleration_ms2 * vehicle_mass

    required_angular_speed_rads = required_speed_ms / tire_radius

    drag_forces = DRAG_COEF * 0.5 * AIR_DENSITY * (
            (required_speed_ms + wind_speeds) ** 2) * drag_coefficient * vehicle_frontal_area

    angles = np.arctan(gradients)
    g_forces = vehicle_mass * ACCELERATION_G * np.sin(angles)

    road_friction_array = road_friction * vehicle_mass * ACCELERATION_G * np.cos(angles)

    cornering_force = calculate_cornering_losses(required_speed_kmh, gis_waypoints, tick)

    net_force = road_friction_array + drag_forces + g_forces + acceleration_force + cornering_force

    motor_output_energies = required_angular_speed_rads * net_force * tire_radius * tick
    motor_output_energies = np.clip(motor_output_energies, a_min=0, a_max=None)

    e_m = calculate_motor_efficiency(required_angular_speed_rads, motor_output_energies, tick)
    e_mc = calculate_motor_controller_efficiency(required_angular_speed_rads, motor_output_energies, tick)

    motor_controller_input_energies = motor_output_energies / (e_m * e_mc)

    # Filter out and replace negative energy consumption as 0
    motor_controller_input_energies = np.where(motor_controller_input_energies > 0,
                                               motor_controller_input_energies, 0)

    plot_forces(
        drag_forces,
        g_forces,
        acceleration_force,
        net_force,
        cornering_force,
        required_speed_kmh,
        gradients)

    return motor_controller_input_energies


def run_motor_model() -> (np.ndarray, np.ndarray):
    """
    Adapted from Simulation's motor model
    """
    tick = 1 # seconds

    # from /Simulation/simulation/cache/route/route_data.npz
    route_data = np.load("route_data_FSGP.npz")
    gis_coords = np.load("coords_day3.npy")
    forward_fill_nans(gis_coords)
    gis_indices = np.load("coord_indices_day3.npy")


    # Day 1 index 0 = index 269 of Simulation's gis index 0
    start_index = 269
    # gis_indices = np.roll(gis_indices, start_index)  # np.ndarray[float]
    gis_indices = forward_fill_nans(gis_indices)     # ensure there are no Nans since they mess with indexing
    gis_indices = np.round(gis_indices).astype(int)  # Convert to integers

    # ----- Expected distance estimate -----
    speed_ms = load_data("vehicle_velocity_day1.csv")
    speed_ms = np.append(speed_ms, 0)  # Proper way to append a value to a NumPy array
    speed_kmh = speed_ms * 3.6

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
        gis_coords,
        tick
    )

    return motor_consumed_energy, gradients


# load data from a csv using dataframe
def load_data(path):
    data = pd.read_csv(path)
    values = data["Value"].to_numpy()
    return values


motor_power_predicted, gradient_global = run_motor_model()



motor_current = load_data("motor_current_day1.csv")
motor_voltage = load_data("motor_voltage_day1.csv")
vehicle_velocity = load_data("vehicle_velocity_day1.csv")
motor_power_measured = motor_current * motor_voltage
motor_power_measured = np.append(motor_power_measured, 0)
vehicle_velocity = np.append(vehicle_velocity, 0)


predicted_sum = calculate_total_motor_power(motor_power_predicted, 0, len(motor_power_predicted) - 10)[-1]
measured_sum = calculate_total_motor_power(motor_power_measured, 0, len(motor_power_predicted) - 10)[-1]
percent_error = predicted_sum * 100 / measured_sum

print(f'\n\n\n')
print(f'________ Predicted Total Energy: {predicted_sum}')
print(f'________ Measured Total Energy: {measured_sum}')
print(f'________ Percent Error: {percent_error}')
print(f'\n\n\n')


plot_power(motor_power_predicted, motor_power_measured, vehicle_velocity, gradient_global)


