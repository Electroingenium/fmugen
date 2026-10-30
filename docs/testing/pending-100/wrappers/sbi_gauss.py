"""sbi: neural posterior estimation (NPE, a normalizing flow) for theta from a Gaussian simulator
x = theta + noise, trained on 500 seeded simulations when the object is made; each step gives the
posterior mean for the observation x_obs (seeded sampling)."""
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform


class Posterior:
    def __init__(self, n=500, epochs=20):
        torch.manual_seed(0)
        prior = BoxUniform(torch.zeros(2), torch.ones(2))
        theta = prior.sample((n,))
        x = theta + 0.05 * torch.randn_like(theta)
        npe = NPE(prior=prior, show_progress_bars=False)
        npe.append_simulations(theta, x).train(max_num_epochs=epochs, show_train_summary=False)
        self.posterior = npe.build_posterior()

    def mean(self, x0=0.3, x1=0.7):
        torch.manual_seed(0)
        s = self.posterior.sample((200,), x=torch.tensor([x0, x1]), show_progress_bars=False)
        return s.mean(0).numpy()
