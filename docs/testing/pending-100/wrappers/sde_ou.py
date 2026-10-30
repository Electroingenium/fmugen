"""sdeint: an Ornstein-Uhlenbeck process dy = -theta y dt + sigma dW (Ito), seeded, over [0, 1]."""
import numpy as np
import sdeint


def ou(y0=1.0, theta=1.0, sigma=0.2, seed=0):
    f = lambda y, t: -theta * y
    G = lambda y, t: np.array([[sigma]])
    y = sdeint.itoint(f, G, np.array([y0]), np.linspace(0.0, 1.0, 11), generator=np.random.default_rng(seed))
    return float(y[-1, 0])
