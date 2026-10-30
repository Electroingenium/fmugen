"""do-mpc: an MPC for a double integrator (x' = v, v' = u), |u| <= 1, horizon 10, t_step 0.1."""
import numpy as np
import do_mpc


class DoubleIntegratorMPC:
    def __init__(self, t_step=0.1, n_horizon=10):
        m = do_mpc.model.Model("continuous")
        x = m.set_variable("_x", "x")
        v = m.set_variable("_x", "v")
        u = m.set_variable("_u", "u")
        m.set_rhs("x", v)
        m.set_rhs("v", u)
        m.set_expression("cost", x ** 2 + v ** 2)
        m.setup()
        self.mpc = do_mpc.controller.MPC(m)
        self.mpc.settings.n_horizon = n_horizon
        self.mpc.settings.t_step = t_step
        self.mpc.settings.store_full_solution = False
        self.mpc.settings.nlpsol_opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        self.mpc.set_objective(mterm=m.aux["cost"], lterm=m.aux["cost"])
        self.mpc.set_rterm(u=0.1)
        self.mpc.bounds["lower", "_u", "u"] = -1.0
        self.mpc.bounds["upper", "_u", "u"] = 1.0
        self.mpc.setup()
        self.mpc.x0 = np.array([1.0, 0.0])
        self.mpc.set_initial_guess()

    def make_step(self, x=1.0, v=0.0):
        return float(self.mpc.make_step(np.array([[x], [v]]))[0, 0])
