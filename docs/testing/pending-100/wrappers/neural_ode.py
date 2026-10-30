"""torchdiffeq: a neural ODE dy/dt = f(y), f a small MLP (seeded weights), integrated from the state y
over one step with dopri5."""
import torch
from torchdiffeq import odeint


class _F(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))

    def forward(self, t, y):
        return self.net(y)


class NeuralODE:
    def __init__(self):
        self.f = _F()
        self.y = torch.tensor([1.0, 0.0])

    def step(self, dt=1.0):
        with torch.no_grad():
            self.y = odeint(self.f, self.y, torch.tensor([0.0, dt]))[-1]
        return self.y.numpy().copy()
