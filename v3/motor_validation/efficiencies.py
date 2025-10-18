import math
import numpy as np

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