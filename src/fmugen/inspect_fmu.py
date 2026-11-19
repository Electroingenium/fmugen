"""`fmugen inspect`: what a fmugen FMU contains and what it needs to run, without running it.

Only reads the FMU's files, so it needs neither UniFMU nor the model's packages.
"""
import json
import re
import shutil
import sys
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from fmugen.config import InterfaceError

#: This machine's OS, as ``launch.toml`` names it.
THIS_OS = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
#: What a compiled or proxy FMU doesn't hold.
NOT_RECORDED = {"compile": "not recorded (compiled FMU)", "distributed": "not recorded (in the private folder)"}
#: ``generationTool`` with versions: ``fmugen 1.2.0 + unifmu 1.1.0``.
_TOOL = re.compile(r"fmugen (\S+) \+ unifmu (\S+)")


class _Files:
    """The files of an FMU, zipped or as a folder.

    :ivar names: paths relative to the FMU's root, with ``/``.
    """

    def __init__(self, path):
        """Open an FMU.

        :param path: a ``.fmu`` file or a folder.
        :type path: str or pathlib.Path
        :raises InterfaceError: neither, or not a zip archive.
        """
        self.path = Path(path)
        if self.path.is_dir():
            self.names = {p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()}
        elif self.path.is_file():
            try:
                with zipfile.ZipFile(self.path) as z:
                    self.names = {n for n in z.namelist() if not n.endswith("/")}
            except zipfile.BadZipFile as e:
                raise InterfaceError(f"{self.path} is not an FMU: {e}") from e
        else:
            raise InterfaceError(f"{self.path}: no such file or folder")

    def read(self, name, size=-1):
        """The bytes of a file, or None when the FMU doesn't have it.

        :param name: its path in the FMU.
        :type name: str
        :param size: how many bytes (default: all).
        :type size: int
        :rtype: bytes or None
        """
        if name not in self.names:
            return None
        if self.path.is_dir():
            with open(self.path / name, "rb") as f:
                return f.read(size)
        with zipfile.ZipFile(self.path) as z, z.open(name) as f:
            return f.read(size)


def _mode(launch):
    """How the FMU runs, from its ``launch.toml``.

    :param launch: the parsed file.
    :type launch: dict
    :return: ``"default"``, ``"vendor"``, ``"compile"`` or ``"distributed"``.
    :rtype: str
    """
    if "Remote" in str(launch.get("location", "")):
        return "distributed"
    text = " ".join(str(part) for system in ("windows", "linux", "macos") for part in launch.get(system, []))
    if "fmugen_launch.py" in text:
        return "vendor"
    return "compile" if "dist/main" in text else "default"


def _compiled_for(files):
    """The OS a compiled FMU's executable runs on.

    :param files: the FMU.
    :type files: _Files
    :return: ``"windows"``, ``"linux"``, ``"macos"``, or None when there's no executable.
    :rtype: str or None
    """
    if "resources/dist/main/main.exe" in files.names:
        return "windows"
    head = files.read("resources/dist/main/main", 4)
    if head is None:
        return None
    return "linux" if head == b"\x7fELF" else "macos"


def _platforms(files):
    """The FMU's native binaries, as the folders in ``binaries/``.

    :param files: the FMU.
    :type files: _Files
    :rtype: list[str]
    """
    return sorted({n.split("/")[1] for n in files.names if n.startswith("binaries/") and n.count("/") >= 2})


def _variables(root):
    """How many variables of each causality the model description declares, without ``time``.

    :param root: the root of ``modelDescription.xml``.
    :type root: xml.etree.ElementTree.Element
    :rtype: dict[str, int]
    """
    counts = {}
    variables = root.find("ModelVariables")
    for variable in () if variables is None else variables:
        causality = variable.get("causality", "local")
        if causality != "independent":
            counts[causality] = counts.get(causality, 0) + 1
    return counts


def _requirements(text):
    """The requirements in a ``requirements.txt``.

    :param text: its contents.
    :type text: bytes
    :rtype: list[str]
    """
    return [line.strip() for line in text.decode().splitlines() if line.strip() and not line.startswith("#")]


def _run_problem(report, launch):
    """Why this FMU can't run on this machine as it is, or None when nothing says so.

    :param report: the report so far.
    :type report: dict
    :param launch: the parsed ``launch.toml``.
    :type launch: dict
    :rtype: str or None
    """
    mode, command = report["mode"], launch.get(THIS_OS) or []
    if mode == "distributed":
        return "This FMU is only a proxy: it runs once its private side is started and connects to it."
    if mode == "compile":
        built = report["compiled_for"]
        if built and built != THIS_OS:
            return f"This FMU was compiled for {built} and can't run on {THIS_OS}."
        return None
    if not command:
        return f"Its launch.toml has no command for {THIS_OS}."
    python = command[0]
    if not (Path(python).is_file() if ("/" in python or "\\" in python) else shutil.which(python)):
        fix = " Rebuild it with --vendor or --compile." if mode == "default" else ""
        return f"This FMU won't run here: its Python, {python}, doesn't exist on this machine.{fix}"
    return None


def inspect_fmu(path):
    """Describe an FMU built by fmugen.

    :param path: a ``.fmu`` file or a folder.
    :type path: str or pathlib.Path
    :return: the report; what the FMU doesn't hold is a "not recorded" text.
    :rtype: dict
    :raises InterfaceError: it isn't an FMU built by fmugen.
    """
    files = _Files(path)
    description = files.read("modelDescription.xml")
    if description is None:
        raise InterfaceError(f"{path} is not an FMU: it has no modelDescription.xml")
    root = ET.fromstring(description)
    tool = root.get("generationTool", "")
    launch_text = files.read("resources/launch.toml")
    if not tool.startswith("fmugen") or launch_text is None:
        raise InterfaceError(f"{path} wasn't built by fmugen (generationTool: {tool or 'none'})")
    versions = _TOOL.fullmatch(tool)
    launch = tomllib.loads(launch_text.decode())
    mode = _mode(launch)
    missing = NOT_RECORDED.get(mode, "not recorded")
    co_simulation = root.find("CoSimulation")
    capabilities = {} if co_simulation is None else co_simulation.attrib
    report = {
        "path": str(path),
        "name": root.get("modelName"),
        "fmi_version": root.get("fmiVersion"),
        "fmugen_version": versions.group(1) if versions else "not recorded",
        "unifmu_version": versions.group(2) if versions else "not recorded",
        "generated": root.get("generationDateAndTime"),
        "mode": mode,
        "platforms": _platforms(files),
        "variables": _variables(root),
        "can_save_state": "true" in (capabilities.get("canGetAndSetFMUstate"), capabilities.get("canGetAndSetFMUState")),
        "fixed_step": capabilities.get("canHandleVariableCommunicationStepSize") == "false",
    }
    if mode == "default":
        report["python"] = (launch.get(THIS_OS) or [None])[0]
    if mode == "compile":
        report["compiled_for"] = _compiled_for(files)
    if mode == "vendor":
        report["wheels"] = sorted(Path(n).name for n in files.names if n.startswith("resources/wheels/"))
    requirements = files.read("resources/requirements.txt")
    report["requirements"] = _requirements(requirements) if requirements is not None else missing
    interface = files.read("resources/interface.json")
    if interface is None:
        report["entry"] = missing
    else:
        spec = json.loads(interface)
        if spec.get("composite"):
            report["entry"] = "composite"
            report["parts"] = [p["name"] for p in spec["parts"]]
            report["connections"] = len(spec["connections"])
            report["isolated"] = bool(spec.get("isolated"))
            if report["isolated"]:
                report["part_commands"] = {
                    p["name"]: (json.loads(files.read(f"resources/{p['dir']}/launch.json") or b"{}").get(THIS_OS) or ["?"])[0]
                    for p in spec["parts"]}
        else:
            entry = spec["entry"]
            report["entry"] = f"{entry['module']}:{entry['name']}"
            if entry.get("call"):
                report["step_method"] = entry["call"]
    report["problem"] = _run_problem(report, launch)
    return report


def format_report(report):
    """The report as text for a terminal.

    :param report: from :func:`inspect_fmu`.
    :type report: dict
    :rtype: str
    """
    def listed(value):
        return value if isinstance(value, str) else ", ".join(value) or "none"

    variables = ", ".join(f"{n} {c}" for c, n in report["variables"].items()) or "none"
    lines = [f"{report['path']}: FMI {report['fmi_version']} Co-Simulation, model {report['name']}",
             f"Built with:   fmugen {report['fmugen_version']}, UniFMU {report['unifmu_version']}"
             if report["fmugen_version"] != "not recorded" else "Built with:   fmugen (versions not recorded)",
             f"Built on:     {report['generated']}", f"Mode:         {report['mode']}"]
    if "python" in report:
        lines.append(f"Python:       {report['python']}")
    if report.get("compiled_for"):
        lines.append(f"Compiled for: {report['compiled_for']}")
    lines.append(f"Platforms:    {listed(report['platforms'])}")
    lines.append(f"Model:        {report['entry']}" + (f", step method {report['step_method']}" if "step_method" in report else ""))
    if "parts" in report:
        lines.append(f"Parts:        {', '.join(report['parts'])} ({report['connections']} connections)")
        if report["isolated"]:
            lines += [f"  {name} runs with: {command}" for name, command in report["part_commands"].items()]
    lines.append(f"Requires:     {listed(report['requirements'])}")
    if "wheels" in report:
        lines.append(f"Wheels:       {len(report['wheels'])}")
    lines.append(f"Variables:    {variables}")
    lines.append(f"State:        {'can save and restore' if report['can_save_state'] else 'can not save and restore'}")
    if report["fixed_step"]:
        lines.append("Step size:    fixed")
    if report["problem"]:
        lines += ["", report["problem"]]
    return "\n".join(lines)
