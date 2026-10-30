"""SALib: Sobol first-order and total indices of the Ishigami function (parameters a, b)."""
import numpy as np
from SALib.analyze import sobol
from SALib.sample import sobol as sample


def ishigami_sobol(a=7.0, b=0.1, n=256):
    p = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[-np.pi, np.pi]] * 3}
    X = sample.sample(p, n, calc_second_order=False, seed=0)
    Y = np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])
    r = sobol.analyze(p, Y, calc_second_order=False, seed=0)
    return r["S1"], r["ST"]
