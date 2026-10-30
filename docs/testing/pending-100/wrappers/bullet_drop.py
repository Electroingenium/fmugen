"""PyBullet in DIRECT mode: a 1 kg box dropped onto a plane under gravity g; each step advances the
physics by one second (240 steps of 1/240 s) and returns the box's height and vertical velocity."""
import pybullet as p


class BoxDrop:
    def __init__(self, height=2.0):
        self.client = p.connect(p.DIRECT)
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client), physicsClientId=self.client)
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], physicsClientId=self.client)
        self.body = p.createMultiBody(1.0, box, basePosition=[0, 0, height], physicsClientId=self.client)

    def step(self, g=9.81):
        p.setGravity(0, 0, -g, physicsClientId=self.client)
        for _ in range(240):
            p.stepSimulation(physicsClientId=self.client)
        (_, _, z), _ = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        (_, _, vz), _ = p.getBaseVelocity(self.body, physicsClientId=self.client)
        return z, vz

    def close(self):
        p.disconnect(self.client)
