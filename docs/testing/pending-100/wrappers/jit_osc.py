"""jitcode: a damped oscillator compiled to C from SymPy expressions, integrated one step at a time."""
from jitcode import jitcode, y


class Oscillator:
    def __init__(self, x0=1.0, v0=0.0, c=0.2):
        self.ode = jitcode([y(1), -y(0) - c * y(1)], verbose=False)
        self.ode.set_integrator("dopri5")
        self.ode.set_initial_value([x0, v0], 0.0)

    def step(self, time):
        x = self.ode.integrate(time + 1.0)
        return float(x[0]), float(x[1])
