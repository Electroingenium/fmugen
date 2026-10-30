"""emcee: ensemble MCMC of a 1-D Gaussian with mean mu, seeded; each step runs n more steps and
returns the mean of the chain so far."""
import numpy as np
import emcee


def _log_prob(x, mu):
    return -0.5 * float(np.sum((x - mu) ** 2))


class GaussMCMC:
    def __init__(self, mu=1.0, nwalkers=8, seed=0):
        self.s = emcee.EnsembleSampler(nwalkers, 1, _log_prob, args=[mu])
        self.s.random_state = np.random.RandomState(seed).get_state()
        self.state = np.random.default_rng(seed).normal(size=(nwalkers, 1))

    def step(self, n=20):
        self.state = self.s.run_mcmc(self.state, n)
        return float(np.mean(self.s.get_chain()))
