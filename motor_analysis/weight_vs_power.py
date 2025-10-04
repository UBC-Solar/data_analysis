# Formulae from here: https://www.desmos.com/calculator/s5vykogxbr
# Modified to account for BasicMotor efficiency (motor and motor controller)

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

# Format is aligned for easy pasting from spreadsheet
BATT_ENERGY_J                  = 18000000
MPPT_EFFICIENCY                = 0.97
BATTERY_INPUT_EFFICIENCY       = 0.95
BATTERY_OUTPUT_EFFICIENCY      = 0.95
FSGP_RACE_TIME_S               = 86400
TIRE_RADIUS_M                  = 0.2032
VEHICLE_MASS_KG                = 350
VEHICLE_FRONTAL_AREA           = 1.1853
DRAG_COEFFICIENT               = 0.1166
ROLLING_RESISTANCE_COEFFICIENT = 0.0234
ARRAY_POWER_W                  = 749.04
LVS_POWER_W                    = 18
FSGP_TRACK_LENGTH_M            = 5070

# ===============================================================
# FUNCTIONS
# Calculate the performance given a set of parameters
# ===============================================================


def get_motor_mechanical_power(
    speed: float,
    air_density: float,
    drag_coefficient: float,
    vehicle_frontal_area: float,
    vehicle_mass_kg: float,
    acceleration_g: float,
    rolling_resistance_coefficient: float
) -> float:
    """Compute mechanical power required to overcome drag and rolling resistance."""
    drag_power = 0.5 * air_density * (speed ** 3) * drag_coefficient * vehicle_frontal_area
    rolling_resistance_power = (
            vehicle_mass_kg * acceleration_g * rolling_resistance_coefficient * speed
    )
    return drag_power + rolling_resistance_power


def get_motor_electrical_power(
    speed: float,
    air_density: float,
    drag_coefficient: float,
    vehicle_frontal_area: float,
    vehicle_mass_kg: float,
    acceleration_g: float,
    rolling_resistance_coefficient: float,
    tire_radius_m: float,
) -> float:

    """Convert mechanical power to electrical power using motor and controller efficiencies."""
    mech_power = get_motor_mechanical_power(
        speed, air_density, drag_coefficient, vehicle_frontal_area, vehicle_mass_kg, acceleration_g, rolling_resistance_coefficient
    )

    one_second: float = 1.0 # 1W * 1s = 1J (motor model asks for energy & time rather than power)
    angular_speed = speed / tire_radius_m

    motor_efficiency = BasicMotor.calculate_motor_efficiency(
        np.array([angular_speed]), np.array([mech_power]), one_second
    )
    motor_controller_efficiency = BasicMotor.calculate_motor_controller_efficiency(
        np.array([angular_speed]), np.array([mech_power]), one_second
    )

    return float(mech_power / (motor_efficiency * motor_controller_efficiency))


if __name__ == "__main__":

    params = {
        "speed": 5, # m/s
        "air_density": AIR_DENSITY,
        "drag_coefficient": DRAG_COEFFICIENT,
        "vehicle_frontal_area": VEHICLE_FRONTAL_AREA,
        "vehicle_mass_kg": VEHICLE_MASS_KG,
        "acceleration_g": ACCELERATION_G,
        "rolling_resistance_coefficient": ROLLING_RESISTANCE_COEFFICIENT,
        "tire_radius_m": TIRE_RADIUS_M,
    }

    # vals = list(params.values())
    #
    # def estimate_power_vec(vec: list) -> float:
    #     return get_motor_electrical_power(*vec)
    #
    # grad_func = nd.Gradient(estimate_power_vec)
    # gradient = grad_func(vals)
    # grad_dict = dict(zip(params.keys(), gradient))
    #
    # # Run estimation with the given parameters
    # get_motor_electrical_power(*vals)
    #
    # print("\nGradients [d_laps / d_param]:")
    # for key, val in grad_dict.items():
    #     print(f"  > {key}: {val}")
    #
    # print("\nLap Count Change per 1% Parameter Increase:")
    # for key, val in grad_dict.items():
    #     # Compute how many laps are gained when the param increases by 1% using Euler's Method
    #     one_percent_value_change = val * params[key] * 0.01
    #     print(f"  > {key}: {one_percent_value_change}")

    weights = np.linspace(300, 400, 100)
    powers = []

    for w in weights:
        params["vehicle_mass_kg"] = w
        powers.append(get_motor_electrical_power(*params.values()))

    print(np.diff(powers))

    plt.plot(weights, powers)
    plt.xlabel("weight (kg)")
    plt.ylabel("motor electrical power @ 36 km/h (W)")
    plt.show()
