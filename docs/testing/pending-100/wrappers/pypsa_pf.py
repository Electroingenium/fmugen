"""PyPSA: AC power flow of two buses (a slack generator and a load through one line); returns the load
bus voltage angle and magnitude and the line flow."""
import pypsa


def power_flow(p_load=100.0, q_load=20.0):
    n = pypsa.Network()
    n.add("Bus", "a", v_nom=110.0)
    n.add("Bus", "b", v_nom=110.0)
    n.add("Line", "ab", bus0="a", bus1="b", x=10.0, r=1.0)
    n.add("Generator", "g", bus="a", control="Slack")
    n.add("Load", "l", bus="b", p_set=p_load, q_set=q_load)
    n.pf()
    return float(n.buses_t.v_ang["b"].iloc[0]), float(n.buses_t.v_mag_pu["b"].iloc[0]), float(n.lines_t.p0["ab"].iloc[0])
