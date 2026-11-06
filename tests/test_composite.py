"""Composite FMUs: several fmugen models in one FMU, connected output to input."""
import importlib.util
import textwrap
import zipfile

import pytest
from fmpy import read_model_description, simulate_fmu
from fmpy.validation import validate_fmu

from fmugen.builder import build, init
from fmugen.config import InterfaceError

OK = 0

TANK = """
class Tank:
    def __init__(self, area=2.0):
        self.area, self.level = area, 1.0

    def step(self, inflow: float = 0.0, dt=1.0):
        self.level += (inflow - 0.1 * self.level) / self.area * dt
        return self.level
"""

VALVE = """
def valve(level: float = 1.0, setpoint: float = 2.0, gain: float = 0.5):
    return {"flow": max(0.0, gain * (setpoint - level))}
"""


def write_composite(tmp_path, connections=("valve.flow -> tank.inflow", "tank.level -> valve.level")):
    for name, source in (("tank", TANK), ("valve", VALVE)):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / f"{name}.py").write_text(source)
        init(tmp_path / name / f"{name}.py", force=True)
    lines = ", ".join(f'"{c}"' for c in connections)
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent(f"""
        [composite]
        name = "tank_loop"
        parts = {{ valve = "valve", tank = "tank" }}
        connections = [{lines}]
    """))
    return tmp_path / "fmugen.toml"


def closed_loop(steps, setpoint=2.0):
    """The same loop, run directly in Python."""
    namespace = {}
    exec(TANK + VALVE, namespace)  # noqa: S102 (the test's own model source)
    tank, levels = namespace["Tank"](), []
    for _ in range(steps):
        flow = namespace["valve"](level=tank.level, setpoint=setpoint)["flow"]
        levels.append(tank.step(inflow=flow))
    return levels


@pytest.mark.parametrize("fmi", [2, 3])
def test_composite_closed_loop_matches_the_models(tmp_path, adapter, fmi):
    out = tmp_path / "fmu"
    _, interface = build(write_composite(tmp_path), out, output_format="folder", fmi_version=fmi)
    names = [v["name"] for v in interface["variables"]]
    assert "tank.inflow" not in names and "valve.level" not in names    # fed by connections
    assert {"valve.setpoint", "valve.gain", "tank.area", "tank.y", "valve.flow"} <= set(names)

    fmu = adapter(out, fmi=fmi)
    fmu.initialize()
    step = fmu.fmi3DoStep if fmi == 3 else fmu.fmi2DoStep
    levels = []
    for t in range(5):
        status = step(float(t), 1.0, False)
        assert (status[0] if fmi == 3 else status) == OK, fmu.logs
        levels.append(fmu.get("tank.y"))
    assert levels == closed_loop(5)


def test_composite_state_rollback_and_reset(tmp_path, adapter):
    out = tmp_path / "fmu"
    build(write_composite(tmp_path), out, output_format="folder")
    fmu = adapter(out)
    fmu.initialize()
    for t in range(3):
        assert fmu.fmi2DoStep(float(t), 1.0, False) == OK
    _, state = fmu.fmi2SerializeFmuState()
    assert fmu.fmi2DoStep(3.0, 1.0, False) == OK
    after = fmu.get("tank.y")
    assert fmu.fmi2DeserializeFmuState(state) == OK
    assert fmu.fmi2DoStep(3.0, 1.0, False) == OK and fmu.get("tank.y") == after
    assert fmu.fmi2Reset() == OK
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK and fmu.get("tank.y") == closed_loop(1)[0]


@pytest.mark.fmu
def test_composite_through_unifmu(tmp_path):
    fmu = tmp_path / "loop.fmu"
    build(write_composite(tmp_path), fmu)
    assert validate_fmu(str(fmu)) == []
    assert {v.name for v in read_model_description(str(fmu)).modelVariables} >= {"tank.y", "valve.setpoint"}
    result = simulate_fmu(str(fmu), stop_time=5.0, output_interval=1.0, start_values={"valve.setpoint": 3.0},
                          output=["tank.y"])
    assert list(result["tank.y"][1:]) == pytest.approx(closed_loop(5, setpoint=3.0))


@pytest.mark.parametrize("connections, message", [
    (["valve.flow -> tank.nope"], "has no variable"),
    (["valve.flow -> pump.inflow"], "no part"),
    (["valve.flow -> tank.y"], "not an input"),
    (["valve.flow -> tank.inflow", "valve.flow -> tank.inflow"], "fed twice"),
    (["valve.flow to tank.inflow"], "must look like"),
])
def test_composite_connection_errors(tmp_path, connections, message):
    with pytest.raises(InterfaceError, match=message):
        build(write_composite(tmp_path, connections), tmp_path / "out", output_format="folder")


@pytest.mark.slow
@pytest.mark.fmu
def test_compiled_composite_has_no_sources_and_simulates(tmp_path):
    if not importlib.util.find_spec("PyInstaller"):
        pytest.skip("PyInstaller is not installed")
    fmu = build(write_composite(tmp_path), tmp_path / "loop.fmu", compiler="pyinstaller")[0]
    with zipfile.ZipFile(fmu) as z:
        names = z.namelist()
    assert not any(n.endswith(".py") for n in names)
    assert [n for n in names if n.startswith("resources/") and not n.startswith("resources/dist/")
            and not n.endswith("/")] == ["resources/launch.toml"]
    result = simulate_fmu(str(fmu), stop_time=5.0, output_interval=1.0, start_values={"valve.setpoint": 3.0},
                          output=["tank.y"])
    assert list(result["tank.y"][1:]) == pytest.approx(closed_loop(5, setpoint=3.0))


def test_composite_parts_with_modules_of_the_same_name_are_refused(tmp_path):
    # both parts import their own utils.py: in one process, the second would get the first's
    for name, value in (("a", 1.0), ("b", 2.0)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "utils.py").write_text(f"GAIN = {value}\n")
        (tmp_path / name / f"{name}.py").write_text(f"from utils import GAIN\n\ndef {name}(x: float = 1.0):\n"
                                                    f"    return {{'y': GAIN * x}}\n")
        init(tmp_path / name / f"{name}.py", force=True)
    (tmp_path / "fmugen.toml").write_text('[composite]\nparts = { a = "a", b = "b" }\nconnections = ["a.y -> b.x"]\n')
    with pytest.raises(InterfaceError, match=r"utils \(in a, b\)"):
        build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")


def write_units(tmp_path, source_var, target_var, source_body="return {'y': 2.0}"):
    """Part a's output y to part b's input x, which b returns as out, with the given variable keys."""
    for name, code, config in (
            ("a", f"def a():\n    {source_body}\n",
             f'[model]\nentry = "a.py:a"\n[outputs]\ny = {{ from = "return:y", {source_var} }}\n'),
            ("b", "def b(x=0.0):\n    return {'out': x}\n",
             f'[model]\nentry = "b.py:b"\n[inputs]\nx = {{ {target_var} }}\n[outputs]\nout = {{ from = "return:out" }}\n')):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.py").write_text(code)
        (tmp_path / name / "fmugen.toml").write_text(config)
    (tmp_path / "fmugen.toml").write_text('[composite]\nparts = { a = "a", b = "b" }\nconnections = ["a.y -> b.x"]\n')
    return tmp_path / "fmugen.toml"


@pytest.mark.parametrize("source, target, expected", [
    ('unit = "kW"', 'unit = "W", start = 0.0', 2000.0),
    ('unit = "degC"', 'unit = "K", start = 0.0', 275.15),
    ('unit = "degC", extra = { relativeQuantity = true }', 'unit = "K", start = 0.0', 2.0),
    ('unit = "W"', 'unit = "W", start = 0.0', 2.0),
    ('unit = "kW/m2"', 'unit = "W/m2", start = 0.0', 2000.0),
])
def test_connections_convert_units(tmp_path, adapter, source, target, expected):
    path = write_units(tmp_path, source, target)
    build(path, tmp_path / "out", output_format="folder")
    fmu = adapter(tmp_path / "out")
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK, fmu.logs
    assert fmu.get("b.out") == pytest.approx(expected)


def test_connections_refuse_different_quantities(tmp_path):
    with pytest.raises(InterfaceError, match="don't measure the same quantity"):
        build(write_units(tmp_path, 'unit = "m"', 'unit = "s"'), tmp_path / "out", output_format="folder")


@pytest.mark.parametrize("source, target", [('unit = "furlong"', 'unit = "m", start = 0.0'),
                                            ('unit = "C"', 'unit = "K", start = 0.0'),
                                            ('unit = "kW"', 'start = 0.0')])
def test_units_that_cant_be_checked_pass_unchanged_with_a_note(tmp_path, adapter, capsys, source, target):
    build(write_units(tmp_path, source, target), tmp_path / "out", output_format="folder")
    assert "can't be checked" in capsys.readouterr().out
    fmu = adapter(tmp_path / "out")
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK and fmu.get("b.out") == pytest.approx(2.0)


def test_conversion_applies_to_each_element_of_an_array(tmp_path, adapter):
    path = write_units(tmp_path, 'unit = "kW", dimensions = [2]', 'unit = "W", dimensions = [2], start = 0.0',
                       "return {'y': [1.0, 2.0]}")
    (tmp_path / "b" / "fmugen.toml").write_text(
        (tmp_path / "b" / "fmugen.toml").read_text().replace('out = { from = "return:out" }',
                                                             'out = { from = "return:out", dimensions = [2] }'))
    build(path, tmp_path / "out", output_format="folder", fmi_version=3)
    fmu = adapter(tmp_path / "out", fmi=3)
    fmu.initialize()
    assert fmu.fmi3DoStep(0.0, 1.0, False)[0] == OK, fmu.logs
    assert fmu.engine.get_values([fmu.vr["b.out"]])[1] == pytest.approx([1000.0, 2000.0])


def write_sized(tmp_path, target_dimensions='["m"]'):
    """Part a's array y, sized by its n, into part b's array x, sized by its m unless given."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a.py").write_text("class A:\n    def __init__(self, n=2):\n        self.n = n\n\n"
                                         "    def step(self):\n        return {'y': [1.0] * self.n}\n")
    (tmp_path / "a" / "fmugen.toml").write_text(
        '[model]\nentry = "a.py:A"\ncall = "step"\nfmi_version = 3\n'
        '[structural_parameters]\nn = { start = 2, variability = "tunable", attr = false }\n'
        '[outputs]\ny = { from = "return:y", dimensions = ["n"] }\n')
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "b.py").write_text("class B:\n    def __init__(self, m=2):\n        self.m = m\n\n"
                                         "    def step(self, x):\n        return {'total': sum(x)}\n")
    (tmp_path / "b" / "fmugen.toml").write_text(
        '[model]\nentry = "b.py:B"\ncall = "step"\nfmi_version = 3\n'
        '[structural_parameters]\nm = { start = 2, variability = "tunable", attr = false }\n'
        f'[inputs]\nx = {{ dimensions = {target_dimensions}, start = 0.0 }}\n'
        '[outputs]\ntotal = { from = "return:total" }\n')
    (tmp_path / "fmugen.toml").write_text(
        '[composite]\nfmi_version = 3\nparts = { a = "a", b = "b" }\nconnections = ["a.y -> b.x"]\n')
    return tmp_path / "fmugen.toml"


@pytest.mark.parametrize("target_dimensions", ['["m"]', "[2]"])
def test_arrays_sized_by_structural_parameters_connect(tmp_path, adapter, target_dimensions):
    build(write_sized(tmp_path, target_dimensions), tmp_path / "out", output_format="folder")
    fmu = adapter(tmp_path / "out", fmi=3)
    fmu.initialize()
    assert fmu.fmi3DoStep(0.0, 1.0, False)[0] == OK, fmu.logs
    assert fmu.get("b.total") == pytest.approx(2.0)


def test_sizes_set_together_in_configuration_mode_work(tmp_path, adapter):
    build(write_sized(tmp_path), tmp_path / "out", output_format="folder")
    fmu = adapter(tmp_path / "out", fmi=3)
    assert fmu.fmi3EnterConfigurationMode() == OK
    assert fmu.fmi3SetUInt64([fmu.vr["a.n"], fmu.vr["b.m"]], [3, 3]) == OK
    assert fmu.fmi3ExitConfigurationMode() == OK, fmu.logs
    fmu.initialize()
    assert fmu.fmi3DoStep(0.0, 1.0, False)[0] == OK, fmu.logs
    assert fmu.get("b.total") == pytest.approx(3.0)


def test_different_sizes_are_an_error_naming_the_connection(tmp_path, adapter):
    build(write_sized(tmp_path), tmp_path / "out", output_format="folder")
    fmu = adapter(tmp_path / "out", fmi=3)
    assert fmu.fmi3EnterConfigurationMode() == OK
    assert fmu.fmi3SetUInt64([fmu.vr["a.n"]], [3]) == OK
    assert fmu.fmi3ExitConfigurationMode() != OK
    assert any("a.y -> b.x" in message and "3 and 2" in message for *_, message in fmu.logs)


def test_reconfiguration_checks_sizes_again(tmp_path, adapter):
    build(write_sized(tmp_path), tmp_path / "out", output_format="folder")
    fmu = adapter(tmp_path / "out", fmi=3)
    fmu.initialize()
    assert fmu.fmi3EnterConfigurationMode() == OK
    assert fmu.fmi3SetUInt64([fmu.vr["b.m"]], [5]) == OK
    assert fmu.fmi3ExitConfigurationMode() != OK


@pytest.mark.fmu
def test_unit_conversion_through_unifmu(tmp_path):
    fmu = tmp_path / "units.fmu"
    build(write_units(tmp_path, 'unit = "kW"', 'unit = "W", start = 0.0'), fmu)
    assert validate_fmu(str(fmu)) == []
    result = simulate_fmu(str(fmu), stop_time=1.0, output_interval=1.0, output=["b.out"])
    assert result["b.out"][-1] == pytest.approx(2000.0)


@pytest.mark.parametrize("fmi", [2, 3])
def test_composite_connection_types_are_checked(tmp_path, fmi):
    path = write_composite(tmp_path, ["valve.state -> tank.inflow"])
    (tmp_path / "valve" / "valve.py").write_text(VALVE.replace('return {"flow"', 'return {"state": "open", "flow"'))
    init(tmp_path / "valve" / "valve.py", force=True)
    with pytest.raises(InterfaceError, match="valve.state -> tank.inflow: a String can't be passed to a"):
        build(path, tmp_path / "out", output_format="folder", fmi_version=fmi)
