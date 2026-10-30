"""VQF: IMU orientation from gyroscope and accelerometer samples at the sensor's sample time gyrTs;
each step feeds one second of constant samples and returns the 6D orientation quaternion."""
import numpy as np
from vqf import VQF


class Orientation:
    def __init__(self, gyrTs=0.01):
        self.vqf = VQF(gyrTs)
        self.n = round(1.0 / gyrTs)

    def update(self, gx=0.0, gy=0.0, gz=0.1, ax=0.0, ay=0.0, az=9.81):
        gyr, acc = np.array([gx, gy, gz]), np.array([ax, ay, az])
        for _ in range(self.n):
            self.vqf.update(gyr, acc)
        return self.vqf.getQuat6D()
