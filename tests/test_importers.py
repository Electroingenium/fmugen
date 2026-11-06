"""The same FMUs through other importers, which call FMI functions in their own order; results must match FMPy's."""
import csv
import importlib.util
import os
import shutil
import subprocess
import sys

import pytest
from fmpy import simulate_fmu
from test_runtime import write

from fmugen.builder import build

pytestmark = [pytest.mark.fmu, pytest.mark.importers]

STOP = 3.0
SOURCE = """
class Total:
    def __init__(self, rate=2.0):
        self.rate, self.total = rate, 0.0

    def step(self, u=1.0, dt=1.0):
        self.total += self.rate * u * dt
        return self.total
"""
CONFIG = """
[model]
entry = "total.py:Total"
call = "step"
fmi_version = {fmi}
[experiment]
start_time = 0.0
stop_time = 3.0
step_size = 1.0
[time]
dt = "step_size"
[parameters]
rate = {{ start = 2.0 }}
[inputs]
u = {{ start = 1.0 }}
[outputs]
y = {{ from = "return" }}
"""


CLOCKED_SOURCE = """
class Ticks:
    def __init__(self):
        self.count = 0.0

    def tick(self):
        self.count += 1.0
"""
CLOCKED_CONFIG = """
[model]
entry = "ticks.py:Ticks"
call = false
fmi_version = 3
[experiment]
start_time = 0.0
stop_time = 3.0
step_size = 0.5
[outputs]
count = { from = "attr:count" }
[clocks.sample]
interval = 1.0
call = "tick"
"""


def last_value(path, name):
    """The last value of the column ``name`` (or ``<prefix>.name``) of a result CSV."""
    with open(path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row]
    column = next(c for c in rows[0] if c.strip().strip('"').split(".")[-1] == name)
    return float(rows[-1][column])


def run_omsimulator(fmu, result):
    if not importlib.util.find_spec("OMSimulator"):
        pytest.skip("OMSimulator is not installed")
    return [sys.executable, "-m", "OMSimulator", str(fmu), "--resultFile", str(result), "--stopTime", str(STOP),
            "--stepSize", "1.0", "--stripRoot"]


def run_fmusim(fmu, result):
    exe = os.environ.get("FMUGEN_FMUSIM") or shutil.which("fmusim")
    if not exe:
        pytest.skip("fmusim is not installed (put it on PATH or set FMUGEN_FMUSIM)")
    return [exe, "--interface-type", "cs", "--stop-time", str(STOP), "--output-interval", "1",
            "--output-file", str(result), str(fmu)]


@pytest.mark.parametrize("fmi", [2, 3])
@pytest.mark.parametrize("command", [run_omsimulator, run_fmusim], ids=["omsimulator", "fmusim"])
def test_a_stateful_model_matches_fmpy(tmp_path, command, fmi):
    write(tmp_path, "total.py", SOURCE, CONFIG.format(fmi=fmi))
    fmu = build(tmp_path / "fmugen.toml", tmp_path / "total.fmu")[0]
    expected = simulate_fmu(str(fmu), stop_time=STOP, output_interval=1.0, output=["y"])["y"][-1]
    assert expected == pytest.approx(6.0)
    result = tmp_path / "result.csv"
    done = subprocess.run(command(fmu, result), cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert last_value(result, "y") == pytest.approx(expected)


def test_an_fmi3_periodic_clock_ticks(tmp_path):
    # not OMSimulator, which can't read clocks: docs/limitations/omsimulator.md
    write(tmp_path, "ticks.py", CLOCKED_SOURCE, CLOCKED_CONFIG)
    fmu = build(tmp_path / "fmugen.toml", tmp_path / "ticks.fmu")[0]
    result = tmp_path / "result.csv"
    done = subprocess.run(run_fmusim(fmu, result), cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert last_value(result, "count") >= 3.0, done.stdout
