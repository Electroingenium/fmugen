"""landlab: flow routing and accumulation on a 5x5 raster tilted toward one corner; rainfall is the
water input, outputs are the drainage area and discharge at the outlet."""
import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator


class Hillslope:
    def __init__(self):
        g = RasterModelGrid((5, 5), xy_spacing=10.0)
        z = g.add_zeros("topographic__elevation", at="node")
        z[:] = g.x_of_node + g.y_of_node
        g.set_closed_boundaries_at_grid_edges(True, True, True, True)
        self.outlet = int(g.core_nodes[0])
        g.status_at_node[self.outlet] = g.BC_NODE_IS_FIXED_VALUE
        self.g = g
        self.fa = FlowAccumulator(g, flow_director="D8")

    def run_one_step(self, rainfall=1.0):
        self.g.at_node["water__unit_flux_in"] = np.full(self.g.number_of_nodes, rainfall)
        self.fa.run_one_step()
        return float(self.g.at_node["drainage_area"][self.outlet]), float(self.g.at_node["surface_water__discharge"][self.outlet])
