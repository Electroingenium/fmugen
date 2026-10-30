"""MuJoCo: a hinged pendulum (capsule of length 1) driven by a motor torque; each step advances the
physics by one second (timestep 0.002) and returns the joint angle and velocity (qpos, qvel)."""
import mujoco

XML = """<mujoco><option timestep="0.002"/><worldbody><body><joint name="hinge" type="hinge" axis="0 1 0"/>
<geom type="capsule" size="0.05" fromto="0 0 0 0 0 -1"/></body></worldbody>
<actuator><motor joint="hinge"/></actuator></mujoco>"""


class Pendulum:
    def __init__(self, angle0=0.5):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0] = angle0
        self.n = round(1.0 / self.model.opt.timestep)

    def step(self, torque=0.0):
        self.data.ctrl[0] = torque
        for _ in range(self.n):
            mujoco.mj_step(self.model, self.data)
        return float(self.data.qpos[0]), float(self.data.qvel[0])
