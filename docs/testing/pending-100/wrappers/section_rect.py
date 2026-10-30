"""sectionproperties: the geometric properties of a d x b rectangle, meshed and computed by finite
elements: area, second moments of area and centroid."""
from sectionproperties.analysis import Section
from sectionproperties.pre.library import rectangular_section


def rectangle(d=100.0, b=50.0, mesh_size=50.0):
    geom = rectangular_section(d=d, b=b).create_mesh(mesh_sizes=[mesh_size])
    sec = Section(geometry=geom)
    sec.calculate_geometric_properties()
    ixx, iyy, ixy = sec.get_ic()
    cx, cy = sec.get_c()
    return sec.get_area(), ixx, iyy, cx, cy
