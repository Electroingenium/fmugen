"""Ruckig: a jerk-limited trajectory for one axis toward target_position, at a fixed control cycle
(delta_time); each step runs the cycles of one communication step and returns the new position,
velocity and acceleration."""
from ruckig import InputParameter, OutputParameter, Result, Ruckig


class Axis:
    def __init__(self, delta_time=0.01, max_velocity=1.0, max_acceleration=2.0, max_jerk=5.0):
        self.otg = Ruckig(1, delta_time)
        self.inp, self.out = InputParameter(1), OutputParameter(1)
        self.inp.max_velocity, self.inp.max_acceleration, self.inp.max_jerk = [max_velocity], [max_acceleration], [max_jerk]
        self.cycles = round(1.0 / delta_time)

    def update(self, target_position=1.0):
        self.inp.target_position = [target_position]
        for _ in range(self.cycles):
            if self.otg.update(self.inp, self.out) == Result.Error:
                raise RuntimeError("ruckig: invalid input")
            self.out.pass_to_input(self.inp)
        return self.out.new_position[0], self.out.new_velocity[0], self.out.new_acceleration[0]
