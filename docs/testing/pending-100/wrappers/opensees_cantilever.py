"""OpenSees (openseespy): an elastic cantilever (one beam element) with a tip load P; returns the tip
deflection and the base moment. openseespy keeps its model in module-level state."""
import openseespy.opensees as ops


def cantilever(P=1000.0, L=2.0, E=200e9, I=8e-6, A=0.01):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, L, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.geomTransf("Linear", 1)
    ops.element("elasticBeamColumn", 1, 1, 2, A, E, I, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, -P, 0.0)
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ops.analyze(1)
    return ops.nodeDisp(2, 2), ops.eleForce(1)[2]
