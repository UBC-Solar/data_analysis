import math
import numpy as np
from numpy.typing import NDArray

# include constants for standard testing conditions
AVG_IRRADIANCE = 1000  # (W/m^2)
RATED_POWER = 3.76  # (Watt peaks i.e. Wp from maxeon data sheet
TEMP_CORRECTION = -0.0029  # (%/C) convert from % to /C from data sheet, assumed at 25 C.


# initial abstraction considering only the top shell and array. again, we do not consider individual cells but the array as a whole with considering uniform steady state temperature throughout.
# maxeon cells : individual area is 155cm^2
# 800w for an average solar irradiance


class arrayTemperatureModel():

    def __init__(self, ambient_temperature, irradiance, wind_speed, thermal_loss_coefficient,
                 convective_loss_coefficient, **kwargs):
        self.ambient_temperature = ambient_temperature
        self.irradiance = irradiance
        self.wind_speed = wind_speed
        self.thermal_loss_coefficient = thermal_loss_coefficient
        self.convective_loss_coefficient = convective_loss_coefficient

    # the range of u0  12.5–22 W/m2C and u1 0–6 Ws/m3C

    # temperature rise due to radiation is also caused by heating up of parts of array other than pv module itself
    # heating due to electrical losses

    # initially considering a stationary vehicle, not considering degradation of cell efficiency etc

    # this method uses the Faiman model. later: incorporate wind speed + car speed later.

    def calculate_speed(self, car_speed):
        # vectorially add car_speeds and wind_speeds together
        air_speed = car_speed
        return air_speed

    def calculateArrayTemperature(self):
        temperature_array = self.ambient_temperature + self.irradiance / (
                self.thermal_loss_coefficient + (self.wind_speed * self.convective_loss_coefficient))
        return temperature_array
        # ta(array) = t(ambient) + G/(u(0) + u(convec)*wind speeds)

    def power_output(self):
        temp_array = self.calculateArrayTemperature()

        # using the PVwatts model:
        # power = irradiance/expected_irradiance * power_rating * (1+ temp_correction (T_array - t_ambient))
        power_temperature = (self.irradiance / AVG_IRRADIANCE) * RATED_POWER * (
                1 + TEMP_CORRECTION * (temp_array - self.ambient_temperature))

        return power_temperature

    # another requirement from the DR is the partial derivative of this power output with respect to irradiance and array_temp. this helps us get a sensitivity analysis
    # also need to extend the faiman model for considering multiple layers

    def partial_irradiance(self):
        temp_array = self.calculateArrayTemperature()

        temperature_term = TEMP_CORRECTION * ((temp_array - self.ambient_temperature) + self.irradiance / (
                self.thermal_loss_coefficient + (self.wind_speed * self.convective_loss_coefficient)))
        partial_irradiance = (RATED_POWER / AVG_IRRADIANCE) * (temperature_term + 1)
        return partial_irradiance

    def partial_temperature(self):
        return (self.irradiance / AVG_IRRADIANCE) * RATED_POWER * TEMP_CORRECTION


# SHORTWAVE = 729


def model():
    return arrayTemperatureModel(27, 715, 0, 22.5,
                                 0.8)


