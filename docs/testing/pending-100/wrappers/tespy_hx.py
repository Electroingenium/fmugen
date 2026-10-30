"""tespy: a water heater (source, simple heat exchanger, sink): heat Q into water of mass flow m at
T_in (°C); returns the outlet temperature."""
from tespy.components import SimpleHeatExchanger, Sink, Source
from tespy.connections import Connection
from tespy.networks import Network


def heater(Q=100e3, m=1.0, T_in=20.0, p=5.0):
    nw = Network(iterinfo=False)
    nw.units.set_defaults(temperature="degC", pressure="bar")
    so, si, hx = Source("in"), Sink("out"), SimpleHeatExchanger("heater")
    c1, c2 = Connection(so, "out1", hx, "in1"), Connection(hx, "out1", si, "in1")
    nw.add_conns(c1, c2)
    c1.set_attr(fluid={"water": 1}, m=m, T=T_in, p=p)
    hx.set_attr(Q=Q, pr=1.0)
    nw.solve("design")
    return float(c2.T.val)
