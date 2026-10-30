"""casadi: a direct-shooting optimal control problem solved with nlpsol (IPOPT): N steps of a double
integrator from (x0, v0) toward the origin; returns the first control and the cost."""
import casadi as ca


class DoubleIntegratorOCP:
    def __init__(self, N=10, dt=0.1):
        U = ca.SX.sym("U", N)
        x0 = ca.SX.sym("x0", 2)
        x, cost = x0, 0
        for k in range(N):
            x = ca.vertcat(x[0] + dt * x[1], x[1] + dt * U[k])
            cost += x[0] ** 2 + x[1] ** 2 + 0.1 * U[k] ** 2
        self.N = N
        self.solver = ca.nlpsol("S", "ipopt", {"x": U, "p": x0, "f": cost},
                                {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})

    def solve(self, x0=1.0, v0=0.0):
        r = self.solver(x0=[0.0] * self.N, p=[x0, v0], lbx=-1.0, ubx=1.0)
        return float(r["x"][0]), float(r["f"])
