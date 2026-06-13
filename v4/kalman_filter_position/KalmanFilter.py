class KalmanFilter():
    def __init__(self, latitude, longitude, accel_x, accel_y, accel_z, speed):
        self.latitude = latitude
        self.longitude = longitude
        self.accel_x = accel_x
        self.accel_y = accel_y
        self.accel_z = accel_z
        self.speed = speed

    def initialize(self):

    def predict(self):
# this will propagate through errors and the recurrence based on the previous time steps.
