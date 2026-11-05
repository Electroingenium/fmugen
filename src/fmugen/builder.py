"""Build and init: packaging Python models into FMUs via UniFMU."""
import atexit
import contextlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from fmugen.arguments import assumptions, guessed_outputs, unfit_outputs
from fmugen.compose import PARTS_DIR, Composite, combine, is_composite
from fmugen.config import (
    MODEL_DIR,
    RESERVED_MODULES,
    Config,
    InterfaceError,
    load_config,
    normalize,
)
from fmugen.description import build_time, stamp_guid, write_model_description
from fmugen.distribute import (
    COMPILERS,
    bundle_hf_models,
    compile_fmu,
    compiled_launch_command,
)
from fmugen.distribute import vendor as vendor_wheels
from fmugen.interface import import_failure, infer_config, parse_target
from fmugen.templates.fmugen_runtime import setup_sys_path
from fmugen.toml_output import render_toml

#: The installed fmugen package folder.
PACKAGE_DIR = Path(__file__).resolve().parent
#: Files copied into every FMU (adapters, runtime, launcher).
TEMPLATES_DIR = PACKAGE_DIR / "templates"
#: FMI version -> the adapter copied to ``resources/model.py``.
ADAPTERS = {2: TEMPLATES_DIR / "model_fmi2.py", 3: TEMPLATES_DIR / "model_fmi3.py"}
#: The runtime's source package, concatenated into one file per FMU.
RUNTIME_DIR = TEMPLATES_DIR / "fmugen_runtime"
#: Runtime modules in dependency order, as concatenated into ``fmugen_runtime.py``.
RUNTIME_SOURCES = [
    RUNTIME_DIR / "_constants.py",
    RUNTIME_DIR / "_helpers.py",
    RUNTIME_DIR / "_engine.py",
    RUNTIME_DIR / "_composite.py",
]
#: The launcher of vendored FMUs (``--vendor``).
LAUNCHER = TEMPLATES_DIR / "fmugen_launch.py"
#: Files never copied into an FMU.
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "venv", ".venv")
#: ``--format`` values.
FORMATS = ("fmu", "folder")

def _concatenate_runtime(destination):
    """Concatenate the split runtime source files into a single ``fmugen_runtime.py`` for the FMU.

    The runtime is developed as a package (``templates/fmugen_runtime/``) but shipped as one
    module, so the FMU needs no fmugen install. The modules' imports of each other are removed
    (their names are all in the one file), later modules lose their module docstring, and
    duplicate import lines are dropped.

    :param destination: the file to write.
    :type destination: str or pathlib.Path
    """
    _INTERNAL_IMPORT = re.compile(
        r"from fmugen\.templates\.fmugen_runtime\._\w+ import\s+"
        r"(?:\(.*?\)|[^\n]+)\n",
        re.DOTALL,
    )
    sections = []
    for source in RUNTIME_SOURCES:
        text = source.read_text(encoding="utf-8")
        text = _INTERNAL_IMPORT.sub("", text)
        sections.append(text)
    first, *rest = sections
    # only module-level imports: one inside a function (``import numpy`` where numpy is optional) is that
    # function's own, and the same line in another function doesn't make it redundant
    seen_imports = set()
    for line in first.splitlines():
        if line.startswith(("import ", "from ")):
            seen_imports.add(line.strip())
    cleaned_rest = []
    for section in rest:
        section = re.sub(r'\A\s*""".*?"""\s*\n', "", section, count=1, flags=re.DOTALL)
        lines = []
        for line in section.splitlines(keepends=True):
            if line.startswith(("import ", "from ")):
                if line.strip() in seen_imports:
                    continue
                seen_imports.add(line.strip())
            lines.append(line)
        cleaned_rest.append("".join(lines))
    content = first + "\n" + "\n".join(cleaned_rest)
    Path(destination).write_text(content, encoding="utf-8")

#: Operating systems a ``launch.toml`` has a command for.
OSES = ("linux", "macos", "windows")
#: Python command on each OS when the FMU runs elsewhere.
DEFAULT_PYTHON = {"linux": "python3", "macos": "python3", "windows": "python"}
#: This machine's OS, in :data:`OSES` names.
CURRENT_OS = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
#: Packages UniFMU's Python backend needs (protobuf pinned to what UniFMU 0.14 generated).
BACKEND_REQUIREMENTS = ["protobuf==5.27.3", "pyzmq", "cloudpickle"]
#: Extra packages of a distributed FMU's private side.
DISTRIBUTED_REQUIREMENTS = ["toml", "coloredlogs", "colorama"]

#: The only UniFMU version fmugen works with.
UNIFMU_VERSION = "0.14.0"
#: Environment variable naming the unifmu executable.
UNIFMU_ENV = "FMUGEN_UNIFMU"
#: How to install UniFMU, for error messages.
UNIFMU_INSTALL = (f"install UniFMU {UNIFMU_VERSION} from "
                  "https://github.com/INTO-CPS-Association/unifmu/releases "
                  f"and put it on PATH, or set {UNIFMU_ENV} to the executable")
#: The unifmu executable whose version was already checked in this process.
_unifmu_checked = None


def find_unifmu():
    """Return the path of the unifmu executable, after checking it is version :data:`UNIFMU_VERSION`.

    :rtype: str
    :raises InterfaceError: not found, can't run, or another version.
    """
    global _unifmu_checked
    exe = os.environ.get(UNIFMU_ENV) or shutil.which("unifmu")
    if not exe:
        raise InterfaceError(f"the unifmu command was not found; {UNIFMU_INSTALL}")
    if _unifmu_checked == exe:
        return exe
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise InterfaceError(f"cannot run {exe}: {e}; {UNIFMU_INSTALL}") from e
    found = out.split()[-1] if out.split() else "?"
    if found != UNIFMU_VERSION:
        raise InterfaceError(f"{exe} is UniFMU {found}, but fmugen needs exactly {UNIFMU_VERSION}; {UNIFMU_INSTALL}")
    _unifmu_checked = exe
    return exe


#: Cache: (exe, *args) -> folder of a ``unifmu generate`` run, removed at exit.
_templates = {}


def _template(args, check):
    """Run ``unifmu *args`` once per process, in a cached folder, and return that folder.

    :param args: the unifmu arguments.
    :type args: list[str]
    :param check: tells whether the folder holds what the command should have written.
    :type check: callable
    :rtype: pathlib.Path
    :raises InterfaceError: the command failed.
    """
    exe = find_unifmu()
    key = (exe, *args)
    if key not in _templates:
        folder = Path(tempfile.mkdtemp(prefix="fmugen-unifmu-"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        result = subprocess.run([exe, *args], capture_output=True, text=True, cwd=folder, check=False)
        if result.returncode != 0 or not check(folder):
            shutil.rmtree(folder, ignore_errors=True)
            raise InterfaceError(f"`unifmu {' '.join(args)}` failed:\n{result.stdout}{result.stderr}")
        _templates[key] = folder
    return _templates[key]


def _link_or_copy(src, dst):
    """Hard-link UniFMU binaries when possible (avoids a virus scanner rescan), else copy.

    A ``copytree`` copy function.

    :param src: the source file.
    :param dst: the destination.
    :return: the destination.
    """
    if "binaries" in Path(src).parts:
        try:
            os.link(src, dst)
            return dst
        except OSError:
            pass
    return shutil.copy2(src, dst)


def generate_boilerplate(fmi_version, dest):
    """Write UniFMU's Python FMU template for ``fmi_version`` (2 or 3) to ``dest``.

    :param fmi_version: 2 or 3.
    :type fmi_version: int
    :param dest: the FMU folder to create.
    :type dest: pathlib.Path
    """
    template = _template(["generate", "python", "out", f"fmi{fmi_version}"],
                         lambda folder: (folder / "out" / "resources").is_dir())
    shutil.copytree(template / "out", dest, copy_function=_link_or_copy)


def generate_distributed_boilerplate(fmi_version, dest, endpoint):
    """Write UniFMU's distributed Python template for ``fmi_version`` into ``dest``.

    :param fmi_version: 2 or 3.
    :type fmi_version: int
    :param dest: the folder to create.
    :type dest: pathlib.Path
    :param endpoint: IP address of the host running the proxy.
    :type endpoint: str
    :return: ``(proxy FMU folder, private folder)``.
    :rtype: tuple[pathlib.Path, pathlib.Path]
    """
    template = _template(["generate-distributed", "python", "x", f"fmi{fmi_version}", "--endpoint", endpoint],
                         lambda folder: (folder / "x_proxy" / "resources" / "launch.toml").is_file()
                         and (folder / "x_private" / "main.py").is_file())
    dest = Path(dest)
    shutil.copytree(template, dest, copy_function=_link_or_copy, dirs_exist_ok=True)
    return dest / "x_proxy", dest / "x_private"


def private_start_command(mode=None):
    """How to start a distributed FMU's private side from its folder.

    :param mode: ``"compile"``, ``"vendor"`` or None.
    :type mode: str or None
    :return: the command, with ``PORT`` for the port the proxy logs.
    :rtype: str
    """
    if mode == "compile":
        return r".\dist\main\main.exe PORT" if CURRENT_OS == "windows" else "./dist/main/main PORT"
    return f"{DEFAULT_PYTHON[CURRENT_OS]} {'fmugen_launch.py' if mode == 'vendor' else 'main.py'} PORT"


def _write_private_readme(private, model_name, endpoint, mode=None):
    """Write the README of a distributed FMU's private side (how to connect it).

    :param private: the private folder.
    :type private: pathlib.Path
    :param model_name: the model's name.
    :type model_name: str
    :param endpoint: the proxy's host.
    :type endpoint: str
    :param mode: ``"compile"``, ``"vendor"`` or None.
    :type mode: str or None
    """
    packages = {None: "\nThe Python running it needs the packages in requirements.txt.\n",
                "vendor": "\nfmugen_launch.py installs the packages from wheels/ on its first run.\n",
                "compile": ""}[mode]
    (private / "README.md").write_text(
        f"# {model_name}: private side of a distributed FMU\n\n"
        f"Built by fmugen with UniFMU {UNIFMU_VERSION}. The FMU the importer loads is only a proxy;\n"
        "this folder runs the model and connects to it over TCP. See fmugen's docs/packaging.md.\n\n"
        "1. Start the simulation with the proxy FMU, with logging on for the `logUnifmuMessages`\n"
        f"   category. The proxy, on the host {endpoint} (see endpoint.toml), logs:\n\n"
        "       ACTION REQUIRED ~ Connect remote backend to dispatcher through port: PORT\n\n"
        "   UniFMU 0.14 picks a new port on every run, and it can't be set.\n"
        f"2. In this folder, run:\n\n       {private_start_command(mode)}\n" + packages,
        encoding="utf-8")


def _write_launch_toml(resources_dir, mode=None):
    """Write ``launch.toml``: how UniFMU starts the backend on each OS.

    Normal FMUs run ``main.py`` with this Python here (``python`` / ``python3`` elsewhere);
    vendored ones run ``fmugen_launch.py``; compiled ones the executable.

    :param resources_dir: the FMU's resources folder.
    :type resources_dir: pathlib.Path
    :param mode: ``"compile"``, ``"vendor"`` or None.
    :type mode: str or None
    """
    if mode == "compile":
        commands = {system: compiled_launch_command(system) for system in OSES}
    elif mode == "vendor":
        commands = {system: [DEFAULT_PYTHON[system], "fmugen_launch.py"] for system in OSES}
    else:
        pythons = {**DEFAULT_PYTHON, CURRENT_OS: sys.executable}
        commands = {system: [pythons[system], "main.py"] for system in OSES}
    (resources_dir / "launch.toml").write_text(
        "".join(f"{system} = {json.dumps(commands[system])}\n" for system in OSES)
    )


def write_interface(interface, path):
    """Write an interface spec as indented JSON (``interface.json``).

    :param interface: the spec.
    :type interface: dict
    :param path: the file.
    :type path: str or pathlib.Path
    """
    Path(path).write_text(json.dumps(interface, indent=2))


@contextlib.contextmanager
def isolated_imports():
    """Context manager: undo ``sys.path`` changes and forget modules loaded from the added paths.

    Importing the model (to infer or normalize it) adds its folders to ``sys.path`` and its
    modules to :data:`sys.modules`. Removing them afterwards lets one process build several
    models with modules of the same name (composite parts, tests) without getting a stale module.
    Installed packages the model imported stay loaded.
    """
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    try:
        yield
    finally:
        added = [Path(p).resolve() for p in sys.path if p and p not in saved_path]
        sys.path[:] = saved_path
        to_forget = []
        # only modules whose file or package path lies under an added folder: those are the model's,
        # and a later import of the same name must load the new copy, not this one
        for name in set(sys.modules) - saved_modules:
            attrs = getattr(sys.modules[name], "__dict__", {})
            try:
                locations = [attrs.get("__file__"), *(attrs.get("__path__") or [])]
            except KeyError:
                locations = [attrs.get("__file__")]
            if any(loc and Path(loc).resolve().is_relative_to(p) for loc in locations for p in added):
                to_forget.append(name)
        for name in to_forget:
            del sys.modules[name]


def resolve_target(target, call=None, fmi_version=None, probe=False):
    """Return a Config for a fmugen.toml, a directory containing one, or a model file.

    A model file's config is inferred in memory (as ``fmugen init`` would).

    :param target: the build target.
    :type target: str or pathlib.Path
    :param call: ``--call`` for a model file.
    :type call: str or None
    :param fmi_version: ``--fmi``.
    :type fmi_version: int or None
    :param probe: ``--probe`` for a model file.
    :type probe: bool or float
    :rtype: Config
    :raises InterfaceError: a missing fmugen.toml or an invalid config.
    """
    path = Path(target)
    if path.is_dir():
        path = path / "fmugen.toml"
    if path.suffix == ".toml":
        if not path.exists():
            raise InterfaceError(f"{path} does not exist; create it with `fmugen init`")
        return load_config(path)
    model_path, _, _ = parse_target(target)
    with isolated_imports():
        data, _ = infer_config(target, call=call, fmi_version=fmi_version, probe=probe)
    return Config(data, model_path.parent if model_path else Path.cwd())


def _copy_sources(config, model_dir):
    """Copy the entry file and ``[model] sources`` into the FMU's model folder.

    :param config: the config.
    :type config: Config
    :param model_dir: ``resources/fmugen_model``.
    :type model_dir: pathlib.Path
    """
    for src, rel in config.source_files():
        dest = model_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, ignore=IGNORE, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)


def _prepare(config, resources, version, model_name, author, capture_output):
    """Copy a model's code into ``resources``, import it and write its interface.json.

    The model is imported from the copy, as the FMU will import it (same module name, same
    ``sys.path``, inside ``[model] cwd``), so a build fails exactly where the FMU would.
    Notes from :func:`normalize` are printed.

    :param config: the config.
    :type config: Config
    :param resources: the FMU's resources folder (or a part's folder).
    :type resources: pathlib.Path
    :param version: the FMI version.
    :type version: int
    :param model_name: ``--name``.
    :type model_name: str or None
    :param author: ``--author``.
    :type author: str or None
    :param capture_output: ``--capture-output``.
    :type capture_output: bool
    :return: the interface spec.
    :rtype: dict
    :raises InterfaceError: the model can't be imported or doesn't fit its config.
    """
    config.check_module_names()
    resources.mkdir(parents=True, exist_ok=True)
    _copy_sources(config, resources / MODEL_DIR)
    module, sys_path = config.entry_import()
    cwd = config.model_cwd()
    if cwd:
        (resources / cwd).mkdir(parents=True, exist_ok=True)
    with isolated_imports(), contextlib.chdir(resources / cwd if cwd else Path.cwd()):
        setup_sys_path({"sys_path": sys_path}, resources)
        try:
            entry_obj = getattr(importlib.import_module(module), config.entry_name)
        except (ImportError, AttributeError) as e:
            hint = (" (install the model's requirements in the environment fmugen runs in)"
                    if config.requirements else "")
            raise InterfaceError(f"cannot import {config.model['entry']}: {e!r}{hint}") from e
        except (Exception, SystemExit) as e:
            raise import_failure(config.model["entry"], e) from e
        interface = normalize(config, entry_obj, module, sys_path, model_name, author, version)
        interface["capture_output"] = bool(capture_output)
        for note in interface.pop("notes"):
            print(f"note: {note}")
    write_interface(interface, resources / "interface.json")
    return interface


def _flat(interface, resources):
    """The parts' variables and constants together (for ``--hf-weights``), for a composite FMU.

    :param interface: the FMU's interface.
    :type interface: dict
    :param resources: the FMU's resources folder.
    :type resources: pathlib.Path
    :return: the interface itself for a single model.
    :rtype: dict
    """
    if not interface.get("composite"):
        return interface
    parts = [json.loads((resources / p["dir"] / "interface.json").read_text()) for p in interface["parts"]]
    return {"variables": [v for p in parts for v in p["variables"]],
            "constants": {f"{i}.{k}": v for i, p in enumerate(parts) for k, v in p.get("constants", {}).items()},
            "call_constants": {f"{i}.{k}": v for i, p in enumerate(parts)
                               for k, v in p.get("call_constants", {}).items()}}


def _write_zip(fmu_dir, output):
    """Zip ``fmu_dir`` reproducibly into an .fmu archive.

    Files in sorted order with a fixed date (``SOURCE_DATE_EPOCH`` when set), and Unix
    permissions set explicitly (executable for binaries) so the archive is the same on every OS.

    :param fmu_dir: the FMU folder.
    :type fmu_dir: pathlib.Path
    :param output: the archive to write.
    :type output: pathlib.Path
    """
    date_time = build_time().timetuple()[:6] if "SOURCE_DATE_EPOCH" in os.environ else (1980, 1, 1, 0, 0, 0)
    date_time = max(date_time, (1980, 1, 1, 0, 0, 0))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in sorted(fmu_dir.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                info = zipfile.ZipInfo(file.relative_to(fmu_dir).as_posix(), date_time)
                info.compress_type = zipfile.ZIP_DEFLATED
                if os.name == "nt":
                    executable = "binaries" in file.relative_to(fmu_dir).parts or file.suffix in (".exe", ".sh")
                else:
                    executable = bool(file.stat().st_mode & 0o111)
                info.external_attr = (0o100755 if executable else 0o100644) << 16
                with open(file, "rb") as src, zipf.open(info, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst)


def _replace(output, write):
    """Call ``write(path)`` on a new path next to ``output``, then put it in place atomically.

    A failed build leaves the previous FMU untouched.

    :param output: the file or folder to replace.
    :type output: pathlib.Path
    :param write: writes the new output to the path it is given.
    :type write: callable
    """
    new = output.with_name(f".{output.name}.new")
    old = output.with_name(f".{output.name}.old")
    for leftover in (new, old):
        if leftover.is_dir():
            shutil.rmtree(leftover)
        else:
            leftover.unlink(missing_ok=True)
    try:
        write(new)
    except BaseException:
        shutil.rmtree(new, ignore_errors=True) if new.is_dir() else new.unlink(missing_ok=True)
        raise
    if output.is_dir():
        output.rename(old)
        new.rename(output)
        shutil.rmtree(old)
    else:
        os.replace(new, output)


def build(target, output, model_name=None, author=None, output_format="fmu", call=None, fmi_version=None,
          vendor=False, platforms=(), python_versions=(), compiler=None, capture_output=False, probe=False,
          hf_weights=None, distributed=False, endpoint=None, private_output=None):
    """Build an FMU (``fmugen build``).

    Steps: resolve the target (config, folder, model file or composite); generate UniFMU's
    template; copy the adapter and the concatenated runtime; write ``requirements.txt``; copy,
    import and normalize each model (:func:`_prepare`) and write ``interface.json`` and
    ``modelDescription.xml``; optionally bundle Hugging Face models, vendor wheels or compile;
    write ``launch.toml``; stamp the GUID; zip or copy into place.

    :param target: fmugen.toml, a folder containing one, ``model.py[:Name]`` or a composite config.
    :type target: str or pathlib.Path
    :param output: the output path, used exactly as given.
    :type output: str or pathlib.Path
    :param model_name: ``--name``.
    :type model_name: str or None
    :param author: ``--author``.
    :type author: str or None
    :param output_format: ``"fmu"`` (zipped) or ``"folder"``.
    :type output_format: str
    :param call: ``--call`` for a model file.
    :type call: str or None
    :param fmi_version: ``--fmi``.
    :type fmi_version: int or None
    :param vendor: ``--vendor``: put wheels of every requirement into the FMU.
    :type vendor: bool
    :param platforms: ``--platform`` tags to vendor wheels for.
    :type platforms: tuple or list
    :param python_versions: ``--python-version`` values to vendor wheels for.
    :type python_versions: tuple or list
    :param compiler: ``--compile`` (a key of ``COMPILERS``), or None.
    :type compiler: str or None
    :param capture_output: ``--capture-output``.
    :type capture_output: bool
    :param probe: ``--probe`` for a model file.
    :type probe: bool or float
    :param hf_weights: ``--hf-weights``; None means on with ``--vendor`` and ``--compile``.
    :type hf_weights: bool or None
    :param distributed: ``--distributed``: a proxy FMU plus a private folder.
    :type distributed: bool
    :param endpoint: ``--endpoint`` (default 127.0.0.1).
    :type endpoint: str or None
    :param private_output: ``--private-output``.
    :type private_output: str or pathlib.Path or None
    :return: ``(output path, interface)``; for a distributed FMU the interface also has
        ``private_output`` and ``private_start``.
    :rtype: tuple
    :raises InterfaceError: invalid options or config.
    :raises FileExistsError: the output path has the wrong kind (file vs folder).
    """
    if (endpoint or private_output) and not distributed:
        raise InterfaceError("--endpoint and --private-output only apply with --distributed")
    if vendor and compiler:
        raise InterfaceError("--vendor and --compile exclude each other: a compiled FMU already contains its packages")
    if (platforms or python_versions) and not vendor:
        raise InterfaceError("--platform and --python-version only apply with --vendor")
    if compiler is not None and compiler not in COMPILERS:
        raise InterfaceError(f"unknown compiler {compiler!r}, expected one of {sorted(COMPILERS)}")
    if output_format not in FORMATS:
        raise ValueError(f"unknown output format {output_format!r}, expected one of {FORMATS}")
    output = Path(output)
    if output_format == "fmu" and output.is_dir():
        raise FileExistsError(f"{output} is a directory; cannot write an .fmu archive there")
    if output_format == "folder" and output.is_file():
        raise FileExistsError(f"{output} is a file; cannot write a folder there")
    if distributed:
        endpoint = endpoint or "127.0.0.1"
        private_output = Path(private_output) if private_output else output.with_name(output.stem + "_private")
        if private_output.resolve() == output.resolve():
            raise FileExistsError(f"{private_output} is the proxy FMU's path; give --private-output another one")
        if private_output.is_file():
            raise FileExistsError(f"{private_output} is a file; cannot write the private folder there")

    composite_path = is_composite(target)
    if composite_path:
        composite = Composite(composite_path)
        version = fmi_version or composite.fmi_version
        configs, base_dir = composite.parts, composite.path.parent
    else:
        config = resolve_target(target, call=call, fmi_version=fmi_version, probe=probe)
        version = config.fmi_version(fmi_version)
        configs, base_dir = {None: config}, config.base_dir
    if version not in ADAPTERS:
        raise InterfaceError(f"unknown FMI version {version!r}, expected one of {sorted(ADAPTERS)}")
    requirements = list(dict.fromkeys(r for c in configs.values() for r in c.requirements))
    backend_requirements = BACKEND_REQUIREMENTS + (DISTRIBUTED_REQUIREMENTS if distributed else [])

    with tempfile.TemporaryDirectory() as tmp:
        if distributed:
            fmu_dir, resources = generate_distributed_boilerplate(version, Path(tmp) / "unifmu", endpoint)
        else:
            fmu_dir = Path(tmp) / "fmu"
            generate_boilerplate(version, fmu_dir)
            resources = fmu_dir / "resources"
        shutil.copy2(ADAPTERS[version], resources / "model.py")
        _concatenate_runtime(resources / "fmugen_runtime.py")
        (resources / "requirements.txt").write_text(
            "# UniFMU backend\n" + "\n".join(backend_requirements) + "\n"
            + ("# model requirements\n" + "\n".join(requirements) + "\n" if requirements else "")
        )
        if composite_path:
            parts = list(configs)
            interfaces = {
                name: _prepare(cfg, resources / PARTS_DIR / name, version, None, None,
                               capture_output and name == parts[-1])
                for name, cfg in configs.items()
            }
            interface = combine(composite, interfaces, version, model_name, author)
        else:
            interface = _prepare(config, resources, version, model_name, author, capture_output)
        write_interface(interface, resources / "interface.json")

        write_model_description(interface, fmu_dir / "modelDescription.xml")
        if hf_weights if hf_weights is not None else (vendor or compiler):
            for repo_id in bundle_hf_models(_flat(interface, resources), base_dir, resources):
                print(f"note: bundled the Hugging Face model {repo_id!r}; the FMU loads it offline")
        if vendor:
            shutil.copy2(LAUNCHER, resources / "fmugen_launch.py")
            vendor_wheels([*backend_requirements, *requirements], resources / "wheels",
                          platforms, python_versions)
        if compiler:
            compile_fmu(compiler, resources, interface, Path(tmp) / "compile", requirements)
        mode = "vendor" if vendor else "compile" if compiler else None
        if distributed:
            _write_private_readme(resources, interface["model_name"], endpoint, mode)
            missing = [r for r in DISTRIBUTED_REQUIREMENTS if not importlib.util.find_spec(r)]
            if missing and not mode:
                print(f"note: the private side's main.py imports {', '.join(missing)}, which this environment "
                      "lacks: pip install fmugen[distributed] where it runs")
        else:
            _write_launch_toml(resources, mode)

        stamp_guid(fmu_dir)

        output.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "fmu":
            _replace(output, lambda path: _write_zip(fmu_dir, path))
        else:
            _replace(output, lambda path: shutil.copytree(fmu_dir, path, ignore=IGNORE))
        if distributed:
            private_output.parent.mkdir(parents=True, exist_ok=True)
            _replace(private_output, lambda path: shutil.copytree(resources, path, ignore=IGNORE))
            interface = {**interface, "private_output": private_output, "private_start": private_start_command(mode)}

    return output, interface


#: Warning and config note when init finds no outputs.
NO_OUTPUTS = "no outputs found: add [outputs] by hand (e.g. attributes or properties of the model object)"


def init(target, output=None, call=None, force=False, fmi_version=None, starts=None, setup=None, kind=None,
         create=None, probe=False, converts=None, reads=None):
    """Write the inferred config for a model (``fmugen init``).

    Warnings about assumptions are printed to stderr.

    :param target: ``model.py[:Name]`` or ``package.module:Name``.
    :type target: str
    :param output: the config path (default: fmugen.toml next to the model); ``"-"`` prints it.
    :type output: str or None
    :param call: ``--call``.
    :type call: str or None
    :param force: overwrite an existing config.
    :type force: bool
    :param fmi_version: ``--fmi``.
    :type fmi_version: int or None
    :param starts: ``--start`` values.
    :type starts: dict or None
    :param setup: ``--setup`` calls.
    :type setup: list[str] or None
    :param kind: ``--kind``.
    :type kind: str or None
    :param create: ``--create``.
    :type create: str or None
    :param probe: ``--probe`` (a number: its timeout).
    :type probe: bool or float
    :param converts: ``--convert`` options.
    :type converts: dict or None
    :param reads: ``--read`` options: ``{output name: "attr:..." or "return[:...]"}``.
    :type reads: dict or None
    :return: ``(written path or None when printed, config dict)``.
    :rtype: tuple
    :raises FileExistsError: the config exists and ``force`` is false.
    """
    model_path, _, _ = parse_target(target)
    default_dir = model_path.parent if model_path else Path.cwd()
    to_stdout = output == "-"
    output = default_dir / "fmugen.toml" if (output is None or to_stdout) else Path(output)
    if output.exists() and not force and not to_stdout:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite it, or -o - to print it")
    with isolated_imports():
        data, comments = infer_config(target, call=call, config_dir=output.parent, fmi_version=fmi_version,
                                      starts=starts, setup=setup, kind=kind, create=create, probe=probe,
                                      converts=converts, reads=reads)
    warnings = _init_warnings(data, comments, probe)
    if model_path and model_path.stem in RESERVED_MODULES:
        warnings.insert(0, f"{model_path.name} clashes with a UniFMU backend module, so build will refuse it; "
                           "rename the file")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    text = render_toml(data, comments)
    if to_stdout:
        sys.stdout.write(text)
        return None, data
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output, data


def _init_warnings(data, comments, probe):
    """All warnings ``fmugen init`` prints: assumptions, missing outputs, guessed types, outputs left out.

    :param data: the inferred config.
    :type data: dict
    :param comments: the comments; a note is added when there are no outputs.
    :type comments: dict
    :param probe: whether the model was probed (changes the advice).
    :type probe: bool
    :rtype: list[str]
    """
    warnings = assumptions(comments) + list(comments.get(("init", "data_arguments"), []))
    outputs = data.get("outputs", {})
    if not outputs:
        warnings.append(NO_OUTPUTS)
        comments.setdefault(("outputs", None), NO_OUTPUTS)
    guessed = guessed_outputs(data, comments)
    if guessed:
        warnings.append(f"{_names(guessed)}: type not shown by the code, assumed a single Real; "
                        + ("set type = ... if that's wrong" if probe else
                           "it may be an array or a tuple, which fails the step: run init --probe, "
                           "or set type = ... / dimensions = [...]"))
    by_reason = {}
    for name, why in unfit_outputs(comments).items():
        by_reason.setdefault(why, []).append(name)
    for why, names in by_reason.items():
        warnings.append(f"{_names(names)}: left commented out in the config, since {why}")
    return warnings


def _names(names):
    """Up to eight names joined with commas, then how many more.

    :param names: the names.
    :type names: list[str]
    :rtype: str
    """
    return ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")
