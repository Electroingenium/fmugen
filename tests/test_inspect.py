"""fmugen inspect: reading an FMU's versions, mode and requirements without running it."""
import json
import shutil
import sys
import zipfile

import pytest
from test_composite import write_composite
from test_runtime import write

from fmugen import UNIFMU_VERSION, __version__
from fmugen.__main__ import main
from fmugen.builder import build
from fmugen.config import InterfaceError
from fmugen.inspect_fmu import THIS_OS, format_report, inspect_fmu

SOURCE = "def heat(T=293.15):\n    return {'q': 0.1 * T}\n"
CONFIG = '[model]\nentry = "heat.py:heat"\nrequirements = ["numpy>=2"]\n[inputs]\nT = { start = 293.15 }\n[outputs]\nq = {}\n'


@pytest.fixture
def folder(tmp_path):
    write(tmp_path, "heat.py", SOURCE, CONFIG)
    return build(tmp_path / "fmugen.toml", tmp_path / "fmu", output_format="folder")[0]


def set_launch(folder, text):
    (folder / "resources" / "launch.toml").write_text(text)


def test_default_folder_and_zip_agree(tmp_path, folder):
    fmu = build(tmp_path / "fmugen.toml", tmp_path / "heat.fmu")[0]
    report = inspect_fmu(folder)
    assert (report["fmugen_version"], report["unifmu_version"]) == (__version__, UNIFMU_VERSION)
    assert report["mode"] == "default" and report["python"] == sys.executable and report["problem"] is None
    assert report["entry"] == "heat:heat" and "numpy>=2" in report["requirements"]
    assert report["variables"] == {"input": 1, "output": 1} and report["can_save_state"] and report["platforms"]
    zipped = inspect_fmu(fmu)
    assert {k: v for k, v in zipped.items() if k not in ("path", "generated")} == \
           {k: v for k, v in report.items() if k not in ("path", "generated")}


def test_a_missing_interpreter_is_the_verdict(folder):
    set_launch(folder, "".join(f'{s} = ["C:/gone/python", "main.py"]\n' for s in ("windows", "linux", "macos")))
    report = inspect_fmu(folder)
    assert "C:/gone/python" in report["problem"] and "--vendor" in report["problem"]
    assert report["problem"] in format_report(report)


def test_vendored(folder):
    set_launch(folder, "".join(f'{s} = ["{sys.executable.replace(chr(92), "/")}", "fmugen_launch.py"]\n'
                               for s in ("windows", "linux", "macos")))
    (folder / "resources" / "wheels").mkdir()
    (folder / "resources" / "wheels" / "numpy-2.whl").write_text("")
    report = inspect_fmu(folder)
    assert report["mode"] == "vendor" and report["wheels"] == ["numpy-2.whl"] and report["problem"] is None


@pytest.mark.parametrize("executable, head, system", [("main.exe", b"MZ", "windows"), ("main", b"\x7fELF", "linux"),
                                                      ("main", b"\xcf\xfa\xed\xfe", "macos")])
def test_compiled_records_less_and_knows_its_os(folder, executable, head, system):
    (folder / "resources" / "interface.json").unlink()
    (folder / "resources" / "requirements.txt").unlink()
    dist = folder / "resources" / "dist" / "main"
    dist.mkdir(parents=True)
    (dist / executable).write_bytes(head + b"rest")
    set_launch(folder, 'windows = ["powershell", "-command", "./dist/main/main.exe"]\n'
                       'linux = ["sh", "-c", "./dist/main/main"]\nmacos = ["sh", "-c", "./dist/main/main"]\n')
    report = inspect_fmu(folder)
    assert report["mode"] == "compile" and report["compiled_for"] == system
    assert report["entry"] == report["requirements"] == "not recorded (compiled FMU)"
    assert (report["problem"] is None) == (system == THIS_OS)


def test_distributed_proxy(folder):
    set_launch(folder, 'location = "Remote"\n')
    (folder / "resources" / "interface.json").unlink()
    (folder / "resources" / "requirements.txt").unlink()
    report = inspect_fmu(folder)
    assert report["mode"] == "distributed" and "private side" in report["problem"]
    assert report["entry"] == "not recorded (in the private folder)"


def test_composite_lists_its_parts(tmp_path):
    out = build(write_composite(tmp_path), tmp_path / "out", output_format="folder")[0]
    report = inspect_fmu(out)
    assert report["parts"] == ["valve", "tank"] and report["connections"] == 2


def test_older_fmu_has_no_recorded_version(folder):
    path = folder / "modelDescription.xml"
    text = path.read_text()
    path.write_text(text.replace(f"fmugen {__version__} + unifmu {UNIFMU_VERSION}", "fmugen + unifmu"))
    report = inspect_fmu(folder)
    assert report["fmugen_version"] == "not recorded" and "versions not recorded" in format_report(report)


def test_not_a_fmugen_fmu(tmp_path, folder):
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    path = other / "modelDescription.xml"
    path.write_text(path.read_text().replace("fmugen ", "SomeTool "))
    with pytest.raises(InterfaceError, match="wasn't built by fmugen"):
        inspect_fmu(other)
    with pytest.raises(InterfaceError, match="no such file"):
        inspect_fmu(tmp_path / "missing.fmu")
    (tmp_path / "bad.fmu").write_text("not a zip")
    with pytest.raises(InterfaceError, match="not an FMU"):
        inspect_fmu(tmp_path / "bad.fmu")
    with zipfile.ZipFile(tmp_path / "empty.fmu", "w"):
        pass
    with pytest.raises(InterfaceError, match="no modelDescription"):
        inspect_fmu(tmp_path / "empty.fmu")


def test_cli(folder, tmp_path, capsys):
    main(["inspect", str(folder)])
    assert "Mode:         default" in capsys.readouterr().out
    main(["inspect", str(folder), "--json"])
    assert json.loads(capsys.readouterr().out)["entry"] == "heat:heat"
    with pytest.raises(SystemExit) as exit_info:
        main(["inspect", str(tmp_path / "missing.fmu")])
    assert exit_info.value.code == 2
