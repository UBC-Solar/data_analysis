# Formulae from here: https://www.desmos.com/calculator/s5vykogxbr
# Modified to account for BasicMotor efficiency (motor and motor controller)

from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from typing import Callable
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from physics.models.motor import BasicMotor
# from physics.models.battery import basic_battery
from physics.models.constants import ACCELERATION_G, AIR_DENSITY
import numdifftools as nd


# ===============================================================
# PARAMETERS
# Default parameters for the car and FSGP race environment
# ===============================================================

BATT_ENERGY_J = 5000 * 3600  # Wh * J/Wh
MPPT_EFFICIENCY = 0.97
BATTERY_INPUT_EFFICIENCY = 0.95
BATTERY_OUTPUT_EFFICIENCY = 0.95
# THROUGH_BATTERY_FRACTION = 0.6 # account for the fact some array power is used directly and doesn't go through the battery?
FSGP_RACE_TIME_S = 3 * 8 * 3600  # days * race hours/day * seconds/hour
TIRE_RADIUS_M = 0.2032
VEHICLE_MASS_KG = 350
VEHICLE_FRONTAL_AREA = 1.1853
DRAG_COEFFICIENT = 1.166e-01
ROLLING_RESISTANCE_COEFFICIENT = 2.340e-02
ARRAY_POWER_W = 950
LVS_POWER_W = 12 * 1.5
FSGP_TRACK_LENGTH_M = 5070

# ===============================================================
# FUNCTIONS
# Calculate the performance given a set of parameters
# ===============================================================

def find_longest_increasing_slice(arr):
    # Ensure input is a numpy array
    arr = np.asarray(arr)

    # Check if the array is strictly increasing
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            # Return the longest strictly increasing slice from the beginning
            return arr[:i]

    # If the array is strictly increasing
    return arr

def get_motor_mechanical_power(
    speed: NDArray,
    air_density: float,
    drag_coefficient: float,
    vehicle_frontal_area: float,
    vehicle_mass_kg: float,
    acceleration_g: float,
    rolling_resistance_coefficient: float
) -> NDArray:
    """Compute mechanical power required to overcome drag and rolling resistance."""
    drag_power = 0.5 * air_density * (speed ** 3) * drag_coefficient * vehicle_frontal_area
    rolling_resistance_power = (
            vehicle_mass_kg * acceleration_g * rolling_resistance_coefficient * speed
    )
    return drag_power + rolling_resistance_power


def get_motor_electrical_power(
    mechanical_power: NDArray,
    speed: NDArray,
    tire_radius_m: float,
) -> NDArray:
    """Convert mechanical power to electrical power using motor and controller efficiencies."""
    one_second: float = 1.0 # 1W * 1s = 1J (motor model asks for energy & time rather than power)
    angular_speed = speed / tire_radius_m

    motor_efficiency = BasicMotor.calculate_motor_efficiency(
        angular_speed, mechanical_power, one_second
    )
    motor_controller_efficiency = BasicMotor.calculate_motor_controller_efficiency(
        angular_speed, mechanical_power, one_second
    )

    return mechanical_power / (motor_efficiency * motor_controller_efficiency)

def estimate_laps(params: dict) -> float:
    # Speeds to evaluate (m/s)
    speeds_mps = np.linspace(0.0, 30.0, 30)

    # Mechanical power from aerodynamic drag + rolling resistance
    mech_powers_w = get_motor_mechanical_power(
        speed=speeds_mps,
        air_density=params["air_density"],
        drag_coefficient=params["drag_coefficient"],
        vehicle_frontal_area=params["vehicle_frontal_area"],
        vehicle_mass_kg=params["vehicle_mass_kg"],
        acceleration_g=params["acceleration_g"],
        rolling_resistance_coefficient=params["rolling_resistance_coefficient"]
    )

    # Electrical power is after motor/motor controller efficiency losses
    # elec_powers_w = get_motor_electrical_power(
    #     mechanical_power=mech_powers_w,
    #     speed=speeds_mps,
    #     tire_radius_m=params["tire_radius_m"],
    # )
    elec_powers_w = mech_powers_w / (0.90 * 0.95) # TODO: revert

    # Compute available electrical power from battery
    battery_input_w = (params["array_power_w"] - params["lvs_power_w"]) * params["mppt_efficiency"] * params["battery_input_efficiency"]
    available_power_w = battery_input_w * params["battery_output_efficiency"]

    # Interpolate race speed from power/speed data points
    # VERY SUSPICIOUS CODE... XP IS NOT ALWAYS INCREASING
    speed_mps = np.interp(available_power_w, elec_powers_w, speeds_mps)

    print(f"Interpolated speed: {speed_mps} m/s")

    # Laps completed in race time
    lap_count = (
        speed_mps * params["fsgp_race_time_s"] / params["fsgp_track_length_m"]
    )

    return float(lap_count)


if __name__ == "__main__":

    params = {
        "batt_energy_j": BATT_ENERGY_J,
        "fsgp_race_time_s": FSGP_RACE_TIME_S,
        "tire_radius_m": TIRE_RADIUS_M,
        "vehicle_mass_kg": VEHICLE_MASS_KG,
        "vehicle_frontal_area": VEHICLE_FRONTAL_AREA,
        "drag_coefficient": DRAG_COEFFICIENT,
        "rolling_resistance_coefficient": ROLLING_RESISTANCE_COEFFICIENT,
        "array_power_w": ARRAY_POWER_W,
        "battery_input_efficiency": BATTERY_INPUT_EFFICIENCY,
        "battery_output_efficiency": BATTERY_OUTPUT_EFFICIENCY,
        "mppt_efficiency": MPPT_EFFICIENCY,
        "lvs_power_w": LVS_POWER_W,
        "fsgp_track_length_m": FSGP_TRACK_LENGTH_M,
        "air_density": AIR_DENSITY,
        "acceleration_g": ACCELERATION_G,
    }

    # def estimate_laps_vec(vec: list) -> float:
    #     reconstructed_dict = {key: val for key, val in zip(params.keys(), vec)}
    #     return estimate_laps(reconstructed_dict)
    #
    # # compute gradient at the given values
    # grad_func = nd.Gradient(estimate_laps_vec)
    # gradient = grad_func(list(params.values()))
    #
    # # Map gradient back to parameter names
    # grad_dict = dict(zip(params.keys(), gradient))
    #
    # print("\n\nGradients (d_laps / d_param):\n")
    # for key, val in grad_dict.items():
    #     print(f"{key}: {val}")

    masses = np.linspace(0, 500, 500)
    lap_counts = []
    for mass in masses:
        param_copy = deepcopy(params)
        param_copy["vehicle_mass_kg"] = mass
        lap_counts.append(estimate_laps(param_copy))
    plt.plot(masses, lap_counts)
    plt.xlabel("Car mass (kg)")
    plt.ylabel("Estimated Lap Count")
    plt.show()

    diffs = np.diff(lap_counts) # spaced by 1kg so no need to scale
    plt.plot(masses[:-1], diffs)
    plt.xlabel("Car mass (kg)")
    plt.ylabel("Estimated Laps / Kg")
    plt.show()