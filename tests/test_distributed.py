"""Distributed FMUs (--distributed): a proxy FMU, and a private folder that runs the model and connects to it."""
import re
import subprocess
import sys

import pytest
from fmpy import simulate_fmu

from fmugen.builder import build

for _module in ("toml", "coloredlogs", "colorama"):   # fmugen[distributed]: the private side imports them
    pytest.importorskip(_module)

PORT = re.compile(rb"dispatcher through port:\s*(\d+)")


def _simulate(fmu, private, command, fmi_version=2, output="output", **options):
    """Simulate the proxy, starting the private side with the port the proxy logs (UniFMU picks it)."""
    started = []

    def connect(message):
        found = PORT.search(message if isinstance(message, bytes) else message.encode())
        if found and not started:
            started.append(subprocess.Popen([*command, found.group(1).decode()], cwd=private,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    logger = ((lambda env, instance, status, category, message: connect(message)) if fmi_version == 2 else
              (lambda env, status, category, message: connect(message)))
    try:
        result = simulate_fmu(str(fmu), output=[output], debug_logging=True, logger=logger,
                              **{"stop_time": 1, **options})
    finally:
        for process in started:
            process.wait(timeout=300)   # a vendored first run installs its wheels
    assert started, "the proxy didn't log its port"
    return list(result[output])


@pytest.mark.fmu
def test_composite_distributed(tmp_path):
    from test_composite import closed_loop, write_composite
    fmu, interface = build(write_composite(tmp_path), tmp_path / "loop.fmu", distributed=True)
    result = _simulate(fmu, interface["private_output"], [sys.executable, "main.py"], output="tank.y",
                       stop_time=5.0, output_interval=1.0, start_values={"valve.setpoint": 3.0})
    assert result[1:] == pytest.approx(closed_loop(5, setpoint=3.0))
