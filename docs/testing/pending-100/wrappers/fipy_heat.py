"""FiPy: the 1-D heat equation on nx finite-volume cells (rod of length 1, left end held at T_left,
right end at 0); each step advances it by dt, returns the cell temperatures."""
from fipy import CellVariable, DiffusionTerm, Grid1D, TransientTerm


class Rod:
    def __init__(self, nx=10, D=0.1):
        self.mesh = Grid1D(nx=nx, dx=1.0 / nx)
        self.T = CellVariable(mesh=self.mesh, value=0.0)
        self.eq = TransientTerm() == DiffusionTerm(coeff=D)

    def step(self, T_left=1.0, dt=1.0):
        self.T.constrain(T_left, self.mesh.facesLeft)
        self.T.constrain(0.0, self.mesh.facesRight)
        for _ in range(10):
            self.eq.solve(var=self.T, dt=dt / 10)
        return self.T.value.copy()
