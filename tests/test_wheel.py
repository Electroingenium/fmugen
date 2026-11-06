"""fmugen installed from its wheel, in a clean environment: what a user gets, not the source tree."""
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fmpy import simulate_fmu

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.slow, pytest.mark.fmu]

MODEL = "def gain(x: float = 2.0, k: float = 3.0):\n    return {'y': k * x}\n"


def run(*command, cwd=None):
    result = subprocess.run([str(c) for c in command], cwd=cwd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{command}\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def fmugen(tmp_path_factory):
    """The fmugen command of a new environment holding only the wheel and its dependencies."""
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is not installed")
    folder = tmp_path_factory.mktemp("wheel")
    run(uv, "build", "--wheel", "--out-dir", folder / "dist", cwd=ROOT)
    wheel = next((folder / "dist").glob("fmugen-*.whl"))
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    run(uv, "venv", "--python", python_version, folder / "venv")
    scripts = folder / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
    run(uv, "pip", "install", "--python", folder / "venv", wheel)
    return scripts / "fmugen"


def write_model(folder, name="gain"):
    folder.mkdir()
    (folder / f"{name}.py").write_text(MODEL.replace("gain", name))
    return folder / f"{name}.py"


def test_the_installed_command_reports_its_version_and_schema(fmugen):
    assert run(fmugen, "--version").startswith("fmugen ")
    assert '"title": "fmugen.toml"' in run(fmugen, "schema")


@pytest.mark.parametrize("fmi", [2, 3])
def test_init_build_inspect_simulate(fmugen, tmp_path, fmi):
    model = write_model(tmp_path / "m")
    run(fmugen, "init", model, "--fmi", fmi)
    fmu = tmp_path / "gain.fmu"
    run(fmugen, "build", model.parent, "-o", fmu)
    assert "Mode:         default" in run(fmugen, "inspect", fmu)
    result = simulate_fmu(str(fmu), stop_time=1.0, output_interval=1.0, output=["y"])
    assert result["y"][-1] == pytest.approx(6.0)


def test_vendored(fmugen, tmp_path, monkeypatch):
    monkeypatch.setenv("FMUGEN_ENV_DIR", str(tmp_path / "envs"))
    model = write_model(tmp_path / "m")
    run(fmugen, "init", model)
    fmu = tmp_path / "gain.fmu"
    run(fmugen, "build", model.parent, "-o", fmu, "--vendor")
    result = simulate_fmu(str(fmu), stop_time=1.0, output_interval=1.0, output=["y"])
    assert result["y"][-1] == pytest.approx(6.0)


def test_composite(fmugen, tmp_path):
    for name in ("a", "b"):
        run(fmugen, "init", write_model(tmp_path / name, name))
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent("""
        [composite]
        parts = { a = "a", b = "b" }
        connections = ["a.y -> b.x"]
    """))
    fmu = tmp_path / "both.fmu"
    run(fmugen, "build", tmp_path, "-o", fmu)
    result = simulate_fmu(str(fmu), stop_time=1.0, output_interval=1.0, output=["b.y"])
    assert result["b.y"][-1] == pytest.approx(18.0)
