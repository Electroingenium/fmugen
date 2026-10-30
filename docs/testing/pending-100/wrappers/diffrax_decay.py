"""diffrax: dy/dt = -k y solved with Tsit5 over [0, dt]."""
import diffrax
import jax.numpy as jnp


def _f(t, y, k):
    return -k * y


def decay(y0=1.0, k=0.5, dt=1.0):
    sol = diffrax.diffeqsolve(diffrax.ODETerm(_f), diffrax.Tsit5(), t0=0.0, t1=dt, dt0=0.05, y0=jnp.asarray(y0), args=k)
    return float(sol.ys[-1])
