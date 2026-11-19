def mix(features, gain=1.0):
    """Weighted sum of a dict of features."""
    return gain * (features["a"] + 2 * features["b"])

class Plant:
    def __init__(self, config):
        self.k = config["k"]; self.y = 0.0
    def step(self, action):
        self.y += self.k * action.thrust
        return self.y
