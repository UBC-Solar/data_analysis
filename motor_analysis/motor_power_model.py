import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from physics.models.motor import BasicMotor
from physics.models.constants import ACCELERATION_G, AIR_DENSITY


# ===============================================================
# PARAMETERS
# ===============================================================

# constants
batt_energy_j = 5000 * 3600
fsgp_race_time_s = 3 * 8 * 3600
tire_radius_m = 0.2032
vehicle_mass_kg = 350
vehicle_frontal_area = 1.1853
drag_coefficient = 1.166e-01
rolling_resistance_coefficient = 2.340e-02

# average power estimates
array_power_w = 500
lvs_power_w = 12 * 1.5


def get_motor_mechanical_power(v: NDArray) -> NDArray:
    drag_power = 0.5 * AIR_DENSITY * (v ** 3) * drag_coefficient * vehicle_frontal_area
    rolling_resistance_power = vehicle_mass_kg * ACCELERATION_G * rolling_resistance_coefficient * v

    return drag_power + rolling_resistance_power

def get_motor_electrical_power(mechanical_power: NDArray, v: NDArray) -> NDArray:
    angular_speed = v / tire_radius_m
    one_second = 1 # one watt for one second -> 1J

    motor_efficiency = BasicMotor.calculate_motor_efficiency(
        angular_speed, mechanical_power, one_second
    )
    motor_controller_efficiency = BasicMotor.calculate_motor_controller_efficiency(
        angular_speed, mechanical_power, one_second
    )

    return mechanical_power / (motor_efficiency * motor_controller_efficiency)

def get_efficiency_loss_power(mechanical_power: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
    angular_speed = v / tire_radius_m
    one_second = 1 # one watt for one second -> 1J

    motor_efficiency = BasicMotor.calculate_motor_efficiency(
        angular_speed, mechanical_power, one_second
    )
    motor_controller_efficiency = BasicMotor.calculate_motor_controller_efficiency(
        angular_speed, mechanical_power, one_second
    )

    motor_input_power = mechanical_power / motor_efficiency
    mc_input_power = mechanical_power / (motor_efficiency * motor_controller_efficiency)  # same as electrical power

    motor_power_loss = motor_input_power * (1 - motor_efficiency)
    motor_controller_power_loss = mc_input_power * (1 - motor_controller_efficiency)

    return motor_power_loss, motor_controller_power_loss


if __name__ == '__main__':
    speeds_mps = np.linspace(0, 30, 30)

    mech_powers_w = get_motor_mechanical_power(speeds_mps)
    elec_powers_w = get_motor_electrical_power(mech_powers_w, speeds_mps)

    plt.plot(speeds_mps, mech_powers_w, label='Mechanical Power')
    plt.plot(speeds_mps, elec_powers_w, label='Electrical Power')
    plt.plot(speeds_mps, elec_powers_w - mech_powers_w, label='Motor + MC Efficiency Losses (difference)')

    plt.xlabel('Speed (m/s)')
    plt.ylabel('Power (Watts)')
    plt.title('Motor Power vs Speed')
    plt.legend()
    plt.show()

    motor_efficiency_loss_power, mc_efficiency_loss_power = get_efficiency_loss_power(mech_powers_w, speeds_mps)

    plt.plot(speeds_mps, motor_efficiency_loss_power, label='MC Power Loss')
    plt.plot(speeds_mps, mc_efficiency_loss_power, label='Motor Power Loss')
    plt.plot(speeds_mps, motor_efficiency_loss_power + mc_efficiency_loss_power, label='Motor + MC Efficiency Losses (sum)')

    plt.xlabel('Speed (m/s)')
    plt.ylabel('Power (Watts)')
    plt.title('Efficiency Power Losses vs Speed')
    plt.legend()
    plt.show()