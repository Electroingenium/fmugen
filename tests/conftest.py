import contextlib
from pathlib import Path

import pytest

from fmugen.builder import build, isolated_imports
from fmugen.templates import model_fmi2, model_fmi3
from fmugen.templates.fmugen_runtime import Status

ROOT = Path(__file__).resolve().parents[1]


class Adapter:
    """Drives a generated FMI 2 adapter (resources/model.py) directly, without UniFMU or zmq."""

    def __init__(self, resources):
        self.logs = []
        self.model = model_fmi2.Model(self._log, resources_dir=resources)
        self.engine = self.model.engine
        self.vr = {v["name"]: v["valueReference"] for v in [*self.engine.variables, *self.engine.clocks.values()]}

    def _log(self, status, category, message):
        self.logs.append((status, category, message))

    def __getattr__(self, name):
        return getattr(self.model, name)

    def get(self, *names):
        status, values = self.engine.get_values([self.vr[n] for n in names])
        assert status == Status.ok, self.logs
        return values[0] if len(names) == 1 else values

    def set(self, name, value):
        return self.engine.set_values([self.vr[name]], value if isinstance(value, list) else [value])

    def initialize(self, start=0.0):
        assert self.model.fmi2SetupExperiment(start, None, None) == Status.ok
        assert self.model.fmi2EnterInitializationMode() == Status.ok
        assert self.model.fmi2ExitInitializationMode() == Status.ok, self.logs


class Adapter3(Adapter):
    """Drives a generated FMI 3 adapter directly."""

    def __init__(self, resources, event_mode_used=False):
        self.logs = []
        self.model = model_fmi3.Model("test", "", str(resources), False, True, event_mode_used, False, [],
                                      self._log, resources_dir=resources)
        self.engine = self.model.engine
        self.vr = {v["name"]: v["valueReference"] for v in [*self.engine.variables, *self.engine.clocks.values()]}

    def initialize(self, start=0.0):
        assert self.model.fmi3EnterInitializationMode(False, 0.0, start, False, 0.0) == Status.ok
        assert self.model.fmi3ExitInitializationMode() == Status.ok, self.logs


@pytest.fixture
def make_fmu(tmp_path):
    """Build a target into an unzipped FMU folder and return the folder."""
    counter = iter(range(100))

    def make(target, **kwargs):
        out = tmp_path / f"fmu{next(counter)}"
        build(target, out, output_format="folder", **kwargs)
        return out
    return make


@pytest.fixture
def adapter():
    """Open the adapter of a built FMU folder; imports are undone afterwards."""
    with contextlib.ExitStack() as stack:
        def open_(fmu_dir, fmi=2, **kwargs):
            stack.enter_context(isolated_imports())
            resources = Path(fmu_dir) / "resources"
            return Adapter3(resources, **kwargs) if fmi == 3 else Adapter(resources)
        yield open_
