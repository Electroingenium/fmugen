"""imufusion (x-io Fusion): AHRS orientation from gyroscope (deg/s) and accelerometer (g) samples at a
fixed sample period; each step feeds one second of constant samples and returns the quaternion
(w, x, y, z)."""
import numpy as np
import imufusion


class Orientation:
    def __init__(self, sample_period=0.01):
        self.ahrs = imufusion.Ahrs()
        self.ahrs.set_sample_period(sample_period)
        self.n = round(1.0 / sample_period)

    def update(self, gx=0.0, gy=0.0, gz=10.0, ax=0.0, ay=0.0, az=1.0):
        gyr, acc = np.array([gx, gy, gz]), np.array([ax, ay, az])
        for _ in range(self.n):
            self.ahrs.update_no_magnetometer(gyr, acc)
        return self.ahrs.get_quaternion()
