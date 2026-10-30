"""PyNite: a simply supported 3-D frame member with a point load at midspan; returns the midspan
deflection and the largest moment."""
from Pynite import FEModel3D


def beam(P=10.0, L=6.0, E=29000.0, Iy=100.0, Iz=150.0, J=250.0, A=20.0):
    m = FEModel3D()
    m.add_node("N1", 0, 0, 0)
    m.add_node("N2", L, 0, 0)
    m.add_material("Steel", E, 11200.0, 0.3, 0.49)
    m.add_section("S", A, Iy, Iz, J)
    m.add_member("M1", "N1", "N2", "Steel", "S")
    m.def_support("N1", True, True, True, True, False, False)
    m.def_support("N2", False, True, True, False, False, False)
    m.add_member_pt_load("M1", "Fy", -P, L / 2)
    m.analyze(check_statics=False, log=False)
    return m.members["M1"].deflection("dy", L / 2), m.members["M1"].max_moment("Mz")
