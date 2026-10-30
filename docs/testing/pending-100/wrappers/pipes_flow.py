"""pandapipes: steady gas flow in pandapipes' meshed delta network, with the sink demand scaled by
`load`; `pipeflow` writes its results onto the network, read back here (junction pressures, pipe
velocities)."""
import pandapipes
from pandapipes import networks


def gas_flow(load=1.0):
    net = networks.gas_meshed_delta()
    net.sink["mdot_kg_per_s"] *= load
    pandapipes.pipeflow(net)
    return net.res_junction["p_bar"].to_numpy(), net.res_pipe["v_mean_m_per_s"].to_numpy()
