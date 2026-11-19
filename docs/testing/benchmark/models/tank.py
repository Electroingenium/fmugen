class Tank:
    def __init__(self, area=1.0, level0=0.0):
        self.area = area
        self.level = level0
    def step(self, q, dt=0.1):
        self.level += q * dt / self.area
        return self.level
