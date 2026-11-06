"""Composite FMUs with isolate = true: each part in a process of its own, with its own Python if given."""
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fmpy import simulate_fmu
from fmpy.validation import validate_fmu
from test_composite import TANK, VALVE, closed_loop

from fmugen.builder import build, init
from fmugen.compose import Composite
from fmugen.config import InterfaceError
from fmugen.inspect_fmu import inspect_fmu

OK = 0
ROOT = Path(__file__).resolve().parents[1]


def write_loop(tmp_path):
    for name, source in (("tank", TANK), ("valve", VALVE)):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.py").write_text(source)
        init(tmp_path / name / f"{name}.py", force=True)
    (tmp_path / "fmugen.toml").write_text(
        '[composite]\nname = "tank_loop"\nisolate = true\nparts = { valve = "valve", tank = "tank" }\n'
        'connections = ["valve.flow -> tank.inflow", "tank.level -> valve.level"]\n')
    return tmp_path / "fmugen.toml"


@pytest.fixture
def opened(adapter):
    """Open an isolated FMU's adapter; its parts' processes end afterwards."""
    fmus = []

    def open_(folder, **kwargs):
        fmus.append(adapter(folder, **kwargs))
        return fmus[-1]
    yield open_
    for fmu in fmus:
        fmu.engine.close()


@pytest.mark.parametrize("fmi", [2, 3])
def test_results_match_the_models(tmp_path, opened, fmi):
    build(write_loop(tmp_path), tmp_path / "out", output_format="folder", fmi_version=fmi)
    fmu = opened(tmp_path / "out", fmi=fmi)
    fmu.initialize()
    step = fmu.fmi3DoStep if fmi == 3 else fmu.fmi2DoStep
    levels = []
    for t in range(5):
        status = step(float(t), 1.0, False)
        assert (status[0] if fmi == 3 else status) == OK, fmu.logs
        levels.append(fmu.get("tank.y"))
    assert levels == closed_loop(5)


def test_each_part_runs_in_its_own_process(tmp_path, opened):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a.py").write_text("import os\n\ndef a():\n    return {'pid': float(os.getpid())}\n")
    (tmp_path / "a" / "fmugen.toml").write_text('[model]\nentry = "a.py:a"\n[outputs]\npid = { from = "return:pid" }\n')
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "b.py").write_text("import os\n\ndef b():\n    return {'pid': float(os.getpid())}\n")
    (tmp_path / "b" / "fmugen.toml").write_text('[model]\nentry = "b.py:b"\n[outputs]\npid = { from = "return:pid" }\n')
    (tmp_path / "fmugen.toml").write_text('[composite]\nisolate = true\nparts = { a = "a", b = "b" }\n')
    build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")
    fmu = opened(tmp_path / "out")
    fmu.initialize()
    pids = {fmu.get("a.pid"), fmu.get("b.pid"), float(os.getpid())}
    assert len(pids) == 3


def test_modules_of_the_same_name_are_allowed(tmp_path, opened):
    for name, value in (("a", 1.0), ("b", 2.0)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "utils.py").write_text(f"GAIN = {value}\n")
        (tmp_path / name / f"{name}.py").write_text(f"from utils import GAIN\n\ndef {name}(x: float = 1.0):\n"
                                                    f"    return {{'y': GAIN * x}}\n")
        init(tmp_path / name / f"{name}.py", force=True)
    (tmp_path / "fmugen.toml").write_text(
        '[composite]\nisolate = true\nparts = { a = "a", b = "b" }\nconnections = ["a.y -> b.x"]\n')
    build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")
    fmu = opened(tmp_path / "out")
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK, fmu.logs
    assert fmu.get("b.y") == pytest.approx(2.0)


def test_rollback_and_reset(tmp_path, opened):
    build(write_loop(tmp_path), tmp_path / "out", output_format="folder")
    fmu = opened(tmp_path / "out")
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


def test_a_part_that_dies_gives_an_error_status(tmp_path, opened):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a.py").write_text("import os\n\ndef a(x: float = 1.0):\n    if x > 1:\n        os._exit(7)\n"
                                         "    return {'y': x}\n")
    (tmp_path / "a" / "fmugen.toml").write_text(
        '[model]\nentry = "a.py:a"\n[inputs]\nx = { start = 1.0 }\n[outputs]\ny = { from = "return:y" }\n')
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "b.py").write_text("def b(u: float = 0.0):\n    return {'v': u}\n")
    (tmp_path / "b" / "fmugen.toml").write_text(
        '[model]\nentry = "b.py:b"\n[inputs]\nu = { start = 0.0 }\n[outputs]\nv = { from = "return:v" }\n')
    (tmp_path / "fmugen.toml").write_text(
        '[composite]\nisolate = true\nparts = { a = "a", b = "b" }\nconnections = ["a.y -> b.u"]\n')
    build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")
    fmu = opened(tmp_path / "out")
    fmu.initialize()
    assert fmu.set("a.x", 2.0) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) != OK
    assert any("part a" in message and "status 7" in message for *_, message in fmu.logs)
    assert fmu.fmi2DoStep(0.0, 1.0, False) != OK     # and it stays failed, without hanging


def test_a_part_that_fails_to_import_fails_instantiation(tmp_path, adapter):
    build(write_loop(tmp_path), tmp_path / "out", output_format="folder")
    (tmp_path / "out" / "resources" / "parts" / "tank" / "fmugen_model" / "tank.py").write_text("raise ValueError('broken')\n")
    with pytest.raises(RuntimeError, match="broken"):
        adapter(tmp_path / "out")


def test_log_records_reach_the_importer_during_the_call(tmp_path, opened):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a.py").write_text("import logging\n\ndef a():\n    logging.getLogger('a').warning('hello from a')\n"
                                         "    return {'y': 1.0}\n")
    (tmp_path / "a" / "fmugen.toml").write_text('[model]\nentry = "a.py:a"\n[outputs]\ny = { from = "return:y" }\n')
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "b.py").write_text("def b():\n    return {'v': 1.0}\n")
    (tmp_path / "b" / "fmugen.toml").write_text('[model]\nentry = "b.py:b"\n[outputs]\nv = { from = "return:v" }\n')
    (tmp_path / "fmugen.toml").write_text('[composite]\nisolate = true\nparts = { a = "a", b = "b" }\n')
    build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")
    fmu = opened(tmp_path / "out")
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert any("hello from a" in message for *_, message in fmu.logs)


def test_closing_leaves_no_process(tmp_path, adapter):
    build(write_loop(tmp_path), tmp_path / "out", output_format="folder")
    fmu = adapter(tmp_path / "out")
    fmu.initialize()
    processes = [part.process for part in fmu.engine.parts.values()]
    fmu.engine.close()
    assert all(process.poll() is not None for process in processes)


def test_python_needs_isolate(tmp_path):
    write_loop(tmp_path)
    (tmp_path / "fmugen.toml").write_text(
        '[composite]\nparts = { valve = "valve", tank = { path = "tank", python = "py" } }\n')
    with pytest.raises(InterfaceError, match="needs isolate"):
        Composite(tmp_path / "fmugen.toml")


def test_a_python_that_cant_run_is_named(tmp_path):
    write_loop(tmp_path)
    (tmp_path / "fmugen.toml").write_text(
        '[composite]\nisolate = true\nparts = { valve = "valve", tank = { path = "tank", python = "./missing/python" } }\n')
    with pytest.raises(InterfaceError, match="part tank"):
        build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")


def test_inspect_shows_isolated_parts(tmp_path):
    build(write_loop(tmp_path), tmp_path / "out", output_format="folder")
    report = inspect_fmu(tmp_path / "out")
    assert report["isolated"] and report["part_commands"]["tank"] == sys.executable


def test_per_step_overhead_is_reported(tmp_path, adapter, capsys):
    """Not a limit: prints the time per step, recorded in docs/models.md."""
    path = write_loop(tmp_path)
    build(path, tmp_path / "isolated", output_format="folder")
    fmu = adapter(tmp_path / "isolated")
    fmu.initialize()
    start = time.perf_counter()
    for t in range(200):
        assert fmu.fmi2DoStep(float(t), 1.0, False) == OK
    per_step = (time.perf_counter() - start) / 200
    fmu.engine.close()
    with capsys.disabled():
        print(f"\nisolated composite, 2 parts, 2 connections: {per_step * 1000:.2f} ms per step")


# ---------------- through UniFMU, and each kind of FMU ----------------

@pytest.mark.fmu
@pytest.mark.parametrize("fmi", [2, 3])
def test_through_unifmu(tmp_path, fmi):
    fmu = tmp_path / "loop.fmu"
    build(write_loop(tmp_path), fmu, fmi_version=fmi)
    assert validate_fmu(str(fmu)) == []
    result = simulate_fmu(str(fmu), stop_time=5.0, output_interval=1.0, start_values={"valve.setpoint": 3.0},
                          output=["tank.y"])
    assert list(result["tank.y"][1:]) == pytest.approx(closed_loop(5, setpoint=3.0))


@pytest.mark.fmu
@pytest.mark.slow
def test_conflicting_packages_in_two_environments(tmp_path):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is not installed")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    pythons = {}
    for name, version in (("a", 1.0), ("b", 2.0)):
        venv = tmp_path / f"venv_{name}"
        subprocess.run([uv, "venv", "--python", python_version, str(venv)], check=True, capture_output=True)
        subprocess.run([uv, "pip", "install", "--python", str(venv), str(ROOT)], check=True, capture_output=True)
        scripts = venv / ("Scripts" if sys.platform == "win32" else "bin")
        pythons[name] = scripts / "python"
        site = venv / "Lib" / "site-packages" if sys.platform == "win32" else next((venv / "lib").glob("*/site-packages"))
        (site / "conflict.py").write_text(f"VERSION = {version}\n")   # one module, a different version in each
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.py").write_text(f"import conflict\n\ndef {name}():\n    return {{'version': conflict.VERSION}}\n")
        (tmp_path / name / "fmugen.toml").write_text(
            f'[model]\nentry = "{name}.py:{name}"\n[outputs]\nversion = {{ from = "return:version" }}\n')
    (tmp_path / "fmugen.toml").write_text(
        "[composite]\nisolate = true\nparts = { "
        f'a = {{ path = "a", python = "{pythons["a"].as_posix()}" }}, '
        f'b = {{ path = "b", python = "{pythons["b"].as_posix()}" }} }}\n')
    fmu = tmp_path / "both.fmu"
    build(tmp_path / "fmugen.toml", fmu)
    result = simulate_fmu(str(fmu), stop_time=1.0, output_interval=1.0, output=["a.version", "b.version"])
    assert result["a.version"][-1] == 1.0 and result["b.version"][-1] == 2.0


@pytest.mark.fmu
@pytest.mark.slow
def test_vendored(tmp_path, monkeypatch):
    monkeypatch.setenv("FMUGEN_ENV_DIR", str(tmp_path / "envs"))
    fmu = tmp_path / "loop.fmu"
    build(write_loop(tmp_path), fmu, vendor=True)
    result = simulate_fmu(str(fmu), stop_time=5.0, output_interval=1.0, start_values={"valve.setpoint": 3.0},
                          output=["tank.y"])
    assert list(result["tank.y"][1:]) == pytest.approx(closed_loop(5, setpoint=3.0))


@pytest.mark.fmu
@pytest.mark.slow
def test_compiled(tmp_path):
    if not importlib.util.find_spec("PyInstaller"):
        pytest.skip("PyInstaller is not installed")
    fmu = tmp_path / "loop.fmu"
    build(write_loop(tmp_path), fmu, compiler="pyinstaller")
    assert inspect_fmu(fmu)["mode"] == "compile"
    result = simulate_fmu(str(fmu), stop_time=5.0, output_interval=1.0, start_values={"valve.setpoint": 3.0},
                          output=["tank.y"])
    assert list(result["tank.y"][1:]) == pytest.approx(closed_loop(5, setpoint=3.0))


@pytest.mark.fmu
def test_distributed(tmp_path):
    pytest.importorskip("toml")
    pytest.importorskip("coloredlogs")
    pytest.importorskip("colorama")
    from test_distributed import _simulate
    fmu, interface = build(write_loop(tmp_path), tmp_path / "loop.fmu", distributed=True)
    result = _simulate(fmu, interface["private_output"], [sys.executable, "main.py"], output="tank.y",
                       stop_time=5.0, output_interval=1.0, start_values={"valve.setpoint": 3.0})
    assert result[1:] == pytest.approx(closed_loop(5, setpoint=3.0))
