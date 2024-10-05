from math import isnan

import numpy as np
from numpy import ndarray

# ---------- Obtain the charge-voltage curve from the datasheet ----------

# Points extracted from datasheet in ``data_analysis/data/battery_cell_datasheet.png`` using ``plot_points.ipynb``.
# mAh is discharge from full
points_2A = np.array([
    [5.169867060561383, 4.1688311688311686], [15.509601181683795, 4.051948051948052],
    [77.54800590841933, 4.0], [144.75627769571625, 3.9935064935064934],
    [242.9837518463811, 3.9805194805194803], [398.0797636632201, 3.9415584415584415],
    [532.4963072378139, 3.909090909090909], [677.2525849335302, 3.8636363636363633],
    [811.669128508124, 3.831168831168831], [956.4254062038405, 3.779220779220779],
    [1090.8419497784344, 3.7532467532467533], [1261.4475627769566, 3.7012987012987013],
    [1432.0531757754798, 3.662337662337662], [1581.9793205317574, 3.6103896103896105],
    [1731.9054652880352, 3.571428571428571], [1881.8316100443128, 3.5259740259740258],
    [2011.0782865583458, 3.4935064935064934], [2140.3249630723776, 3.4675324675324672],
    [2269.5716395864106, 3.428571428571428], [2403.9881831610046, 3.409090909090909],
    [2548.744460856721, 3.37012987012987], [2698.6706056129988, 3.331168831168831],
    [2817.5775480059087, 3.2857142857142856], [2936.484490398818, 3.233766233766233],
    [3039.881831610044, 3.1363636363636362], [3127.769571639586, 3.0129870129870127],
    [3194.977843426883, 2.9155844155844157], [3246.676514032496, 2.7987012987012987],
    [3282.865583456425, 2.701298701298701], [3319.0546528803548, 2.5974025974025974],
    [3344.9039881831613, 2.5129870129870127]
])

MIN_CELL_VOLTAGE = 2.7
MAX_CELL_VOLTAGE = points_2A[0][1]

points_increasing_voltage = np.flipud(points_2A)
points_usable_energy = points_increasing_voltage[points_increasing_voltage[:, 1] > MIN_CELL_VOLTAGE]

max_discharge_mah = points_usable_energy[0][0]  # further discharging results in < MIN_CELL_VOLTAGE

# get remaining charge instead
curve_points_mah_v = points_usable_energy
curve_points_mah_v[:, 0] = max_discharge_mah - points_usable_energy[:, 0]


# ---------- Interpolate curve at x_vals ----------

num_points = 500
x_vals = np.linspace(0, max_discharge_mah, num_points)

xp = curve_points_mah_v[:, 0]
yp = curve_points_mah_v[:, 1]
points_mah_v = np.interp(x_vals, xp, yp)


# ---------- Integrate the curve to get energy vs. charge ----------

delta_x = max_discharge_mah / num_points
points_mah_wh = np.cumsum(points_mah_v) * delta_x / 1000


# ---- CubicSpline interpolation and composing functions to get energy (Wh) as a function of voltage (V) ----

from scipy.interpolate import CubicSpline
energy_of_charge = CubicSpline(x_vals, points_mah_wh)  # Interpolate energy as a function of charge
charge_of_voltage = CubicSpline(points_mah_v, x_vals)  # Invert datasheet points to get charge as a function of voltage
# Compose the two above functions. The result is energy as a function of voltage

energy_of_voltage = CubicSpline(points_mah_v, energy_of_charge(charge_of_voltage(points_mah_v)), extrapolate=False)


def cell_wh_from_voltage(cell_voltages: ndarray) -> ndarray:
    """
    Determine the usable energy in Wh of a Sanyo NCR18650GA cell given its voltage.
    Can be called on a single value or map an entire ndarray.

    :param cell_voltages: All values must be between 2.702V and 4.168V, otherwise the result is nan
    :type cell_voltages: ndarray
    :return: usable energy in Wh of a Sanyo NCR18650GA cell with v=cell_voltages
    :type: ndarray
    """
    return energy_of_voltage(cell_voltages)


MAX_CELL_ENERGY = points_mah_wh[-1]

def cell_soc_from_voltage(cell_voltages: ndarray) -> ndarray:
    """
    Determine the fraction of usable energy remaining of a Sanyo NCR18650GA cell given its voltage.
    Can be called on a single value or map an entire ndarray.

    :param cell_voltages: All values must be between 2.702V and 4.168V, otherwise the result is nan
    :type cell_voltages: ndarray
    :return: usable energy fraction of a Sanyo NCR18650GA cell with v=cell_voltages
             all values satisfy 0 <= value <= 1
    :type: ndarray
    """
    return energy_of_voltage(cell_voltages) / MAX_CELL_ENERGY

pass