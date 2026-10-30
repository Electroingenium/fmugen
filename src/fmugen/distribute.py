"""Making an FMU run on other machines: vendored wheels (--vendor) or a frozen executable (--compile).

By default an FMU runs with the virtual environment fmugen was installed in, so it only
works where it was built. Two ways to ship it elsewhere:

- `vendor()`: wheels of every requirement go into resources/wheels/. The FMU starts
  through fmugen_launch.py, which builds a cached virtual environment from them on its
  first run, offline. The target needs a Python interpreter.
- `compile_fmu()`: PyInstaller or Nuitka freezes the backend, the adapter, the runtime
  and the model, with the interpreter and every package, into resources/dist/main/.
  No .py files are shipped and the target needs no Python, but the executable only runs
  on the OS (and architecture) it was built on.
"""
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fmugen.config import MODEL_DIR, InterfaceError
from fmugen.arguments import _dotted_names
from fmugen.static import packages_distributions

COMPILERS = {"pyinstaller": "PyInstaller", "nuitka": "nuitka"}   # --compile value -> module
EXE_DIR = "dist/main"   # resources/dist/main/: the frozen backend
EXE_NAME = "main"
# What stays in resources/ after compiling; everything else is inside the executable
KEEP_AFTER_COMPILE = {"dist", "launch.toml", "hf_cache"}
HF_CACHE = "hf_cache"   # Hugging Face models inside the FMU, in resources/
#: Text shaped like a Hugging Face repo id: ``owner/name``.
HF_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
# "data/input.csv" is a file the model reads, not a Hugging Face repo, even when it doesn't exist at build time
DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".toml", ".xml", ".dat", ".xlsx", ".xls",
                 ".parquet", ".feather", ".npy", ".npz", ".h5", ".hdf5", ".nc", ".mat", ".pkl", ".pickle",
                 ".zip", ".gz", ".png", ".jpg", ".wav", ".mos", ".epw", ".fmu"}


# ---------------- --vendor ----------------

def _pip():
    """Command prefix that runs pip (the environment's, else pip through uvx).

    :rtype: list[str]
    :raises InterfaceError: neither is available (pip is missing from uv venvs).
    """
    if importlib.util.find_spec("pip"):
        return [sys.executable, "-m", "pip"]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "pip"]
    raise InterfaceError("--vendor needs pip: install it in this environment (pip is missing from uv venvs), "
                         "or install uv")


def hf_models(interface, base_dir):
    """Hugging Face model ids (``"amazon/chronos-bolt-tiny"``) among the model's string start values and constants.

    Text that is a path of a file in the project (relative to the config folder or ``[model]
    cwd``) or has a data file suffix is not a repo id.

    :param interface: the interface spec (or the flattened one of a composite).
    :type interface: dict
    :param base_dir: the config folder.
    :type base_dir: str or pathlib.Path
    :rtype: list[str]
    """
    texts = [v["start"] for v in interface["variables"] if isinstance(v.get("start"), str)]
    for value in _constants(interface):
        if isinstance(value, dict) and "python" in value:
            try:
                texts.append(ast.literal_eval(value["python"]))
            except (ValueError, SyntaxError):
                continue
        elif isinstance(value, str):
            texts.append(value)
    # a relative file path also looks like "owner/name": the model opens those from the config's folder
    # or, with [model] cwd, from that folder
    folders = [Path(base_dir)]
    if interface.get("cwd"):
        folders.append(Path(base_dir).joinpath(*Path(interface["cwd"]).parts[1:]))   # without fmugen_model/
    return sorted({t for t in texts if isinstance(t, str) and HF_REPO_ID.match(t)
                   and Path(t).suffix.lower() not in DATA_SUFFIXES
                   and not any((folder / t).exists() for folder in folders)})


def bundle_hf_models(interface, base_dir, resources):
    """Download the Hugging Face models the model loads into ``resources/hf_cache``, so the FMU runs offline.

    Repo ids found by :func:`hf_models` are downloaded with ``snapshot_download``. Constants
    computed by a call (huggingface_sb3's ``load_from_hub(...)``) are called once with
    huggingface_hub's cache pointed inside the FMU, so the same call finds its file there
    offline. At runtime the engine sets ``HF_HUB_CACHE`` and the offline variables. Does nothing
    without huggingface_hub installed.

    :param interface: the interface spec (or the flattened one of a composite).
    :type interface: dict
    :param base_dir: the config folder.
    :type base_dir: pathlib.Path
    :param resources: the FMU's resources folder.
    :type resources: pathlib.Path
    :return: the model ids that were bundled.
    :rtype: list[str]
    """
    ids = hf_models(interface, base_dir)
    calls = [value["call"] for value in _constants(interface) if isinstance(value, dict) and "call" in value]
    if not ids and not calls:
        return []
    try:
        from huggingface_hub import constants, snapshot_download
    except ImportError:
        return []
    cache = resources / HF_CACHE
    bundled = []
    for repo_id in ids:
        try:
            snapshot_download(repo_id, cache_dir=cache)
        except Exception as e:
            print(f"note: not bundling {repo_id!r} from Hugging Face ({type(e).__name__}): "
                  "the FMU will download it on first use if it is a model")
            continue
        bundled.append(repo_id)
    # constants computed by a call, e.g. huggingface_sb3's load_from_hub(repo_id=..., filename=...):
    # make the call once with the cache inside the FMU, so the same call finds its file there offline
    from fmugen.templates.fmugen_runtime import call_reference
    previous = constants.HF_HUB_CACHE
    constants.HF_HUB_CACHE = str(cache)
    try:
        for text in calls:
            before = set(cache.glob("models--*")) if cache.is_dir() else set()
            try:
                call_reference(text)
            except Exception as e:
                print(f"note: could not run {text!r} while bundling Hugging Face files ({type(e).__name__})")
                continue
            bundled += [p.name.removeprefix("models--").replace("--", "/")
                        for p in sorted(set(cache.glob("models--*")) - before) if cache.is_dir()]
    finally:
        constants.HF_HUB_CACHE = previous
    return bundled


def _constants(interface):
    """All constant values of an interface (``constants`` and ``call_constants``).

    :param interface: the interface spec.
    :type interface: dict
    :rtype: list
    """
    return [v for table in ("constants", "call_constants") for v in interface.get(table, {}).values()]


def vendor(requirements, wheels_dir, platforms=(), python_versions=()):
    """Put wheels for ``requirements`` (and their dependencies) into ``wheels_dir``.

    Without targets, wheels are made for this machine and Python; packages published only
    as source are built into wheels here, so the target needs no compiler. With targets,
    each platform and Python version is downloaded separately, and only published wheels
    can be used.

    :param requirements: pip requirement strings.
    :type requirements: list[str]
    :param wheels_dir: the folder to fill (``resources/wheels``).
    :type wheels_dir: pathlib.Path
    :param platforms: pip platform tags (``--platform``).
    :type platforms: tuple or list
    :param python_versions: Python versions (``--python-version``).
    :type python_versions: tuple or list
    :raises InterfaceError: pip failed.
    """
    pip = [*_pip(), "--disable-pip-version-check"]
    if not platforms and not python_versions:
        commands = [[*pip, "wheel", "--quiet", "--wheel-dir", str(wheels_dir), *requirements]]
    else:
        commands = []
        for platform in platforms or [None]:
            for version in python_versions or [None]:
                cmd = [*pip, "download", "--quiet", "--only-binary=:all:", "--dest", str(wheels_dir)]
                cmd += ["--platform", platform] if platform else []
                cmd += ["--python-version", version] if version else []
                commands.append(cmd + list(requirements))
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise InterfaceError(f"vendoring {requirements} failed ({' '.join(cmd[len(pip):])}):\n"
                                 f"{result.stdout}{result.stderr}")


# ---------------- --compile ----------------

def compile_fmu(compiler, resources, interface, work, requirements=()):
    """Freeze ``resources/main.py`` and everything it runs into ``resources/dist/main/``; drop the sources.

    Besides what the imports reach, the model's packages and their dependencies are collected
    whole (data files, binaries, metadata), and the direct ones with all their submodules:
    libraries read files next to their code, find plugins through entry points, and import
    modules by name, which no freezer follows. Modules the runtime imports by name
    (:func:`_modules`) are hidden imports; interface.json and the model's data files are added
    as data. Afterwards only the executable, ``launch.toml``, ``hf_cache`` and the data folders
    of ``[model] cwd`` (without code) remain.

    :param compiler: ``"pyinstaller"`` or ``"nuitka"``.
    :type compiler: str
    :param resources: the FMU's resources folder.
    :type resources: pathlib.Path
    :param interface: the FMU's interface spec.
    :type interface: dict
    :param work: a scratch folder for the freezer.
    :type work: pathlib.Path
    :param requirements: the model's requirements (their packages are collected too).
    :type requirements: tuple or list
    :raises InterfaceError: the freezer isn't installed or failed.
    """
    module = COMPILERS[compiler]
    if not importlib.util.find_spec(module):
        raise InterfaceError(f"--compile {compiler} needs {module} in this environment: "
                             f"pip install {compiler} (or pip install fmugen[{compiler}])")
    resources, work = Path(resources), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    models = _models(resources, interface)
    paths = [resources, *(folder / p for folder, model in models for p in model["sys_path"])]
    modules = sorted({m for _, model in models for m in _modules(model)})
    data = [(resources / "interface.json", ".")]
    for folder, model in models:
        if folder != resources:   # a part of a composite FMU reads its interface.json from parts/<name>/
            data.append((folder / "interface.json", folder.relative_to(resources).as_posix()))
        data += _model_data(folder, model)
    if (resources / "endpoint.toml").exists():   # a distributed FMU's private side reads it next to main
        data.append((resources / "endpoint.toml", "."))
    local = {"model", "fmugen_runtime"}
    direct, packages, distributions, imports = _packages([m for m in modules if m not in local], requirements)
    paths += _editable_paths({m.split(".")[0] for m in modules if m not in local} | set(packages))

    if compiler == "pyinstaller":
        cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--log-level", "WARN", "--name", EXE_NAME,
               "--distpath", str(work / "dist"), "--workpath", str(work / "build"), "--specpath", str(work)]
        cmd += [arg for p in paths for arg in ("--paths", str(p))]
        cmd += [arg for m in modules for arg in ("--hidden-import", m)]
        cmd += [arg for src, dest in data for arg in ("--add-data", f"{src}{os.pathsep}{dest}")]
        cmd += [arg for p in packages for arg in ("--collect-data", p, "--collect-binaries", p)]
        cmd += [arg for src, dest in _loaded_libraries(packages) for arg in ("--add-binary", f"{src}{os.pathsep}{dest}")]
        cmd += [arg for p in direct for arg in ("--collect-submodules", p)]
        cmd += [arg for p in imports for arg in ("--hidden-import", p)]
        if "numpy" in packages:   # numpy 2's numpy.core aliases: what older pickles name (np.load of .npz)
            cmd += ["--collect-submodules", "numpy.core"]
        cmd += [arg for src, dest in _plugin_dirs(direct) for arg in ("--add-data", f"{src}{os.pathsep}{dest}")]
        cmd += [arg for d in distributions for arg in ("--copy-metadata", d)]
        hooks = _source_hooks(packages, work / "hooks")
        cmd += ["--additional-hooks-dir", str(hooks)] if hooks else []
        built = work / "dist" / EXE_NAME
        env = None
    else:
        cmd = [sys.executable, "-m", "nuitka", "--standalone", "--assume-yes-for-downloads", "--quiet",
               f"--output-dir={work}", f"--output-filename={EXE_NAME}"]
        cmd += [f"--include-module={m}" for m in modules]
        cmd += [f"--include-data-files={src}={Path(dest, src.name).as_posix()}" for src, dest in data]
        cmd += _nuitka_package_args(packages, direct)
        cmd += [f"--include-module={p}" for p in imports]
        cmd += ["--include-module=numpy.core.multiarray"] if "numpy" in packages else []
        cmd += [f"--include-data-dir={src}={dest}" for src, dest in _plugin_dirs(direct)]
        cmd += [f"--include-distribution-metadata={d}" for d in distributions]
        built = work / "main.dist"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))}
    cmd.append(str(resources / "main.py"))

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work, env=env)
    if result.returncode != 0 or not built.is_dir():
        # warnings (Nuitka writes hundreds about duplicate data files) would push the error out of the tail
        lines = [line for line in (result.stdout + result.stderr).splitlines()
                 if line.strip() and ":WARNING:" not in line and not line.startswith(("WARNING:", "INFO:"))]
        why = (f"exit status {result.returncode}" if result.returncode != 0
               else f"it exited normally, but {built} wasn't made")
        raise InterfaceError(f"--compile {compiler} failed ({why}):\n" + "\n".join(lines)[-4000:])

    if compiler == "pyinstaller" and (built / "_internal").is_dir():
        _add_package_folders(packages, built / "_internal")
    keep = {folder / MODEL_DIR for folder, model in models if model.get("cwd")}
    _remove_compiled(resources, keep, top=True)
    shutil.move(str(built), str(resources / EXE_DIR))


def _models(resources, interface):
    """``[(folder, interface)]`` of the models in the FMU: itself, or the parts of a composite FMU.

    :param resources: the FMU's resources folder.
    :type resources: pathlib.Path
    :param interface: the FMU's interface.
    :type interface: dict
    :rtype: list[tuple]
    """
    if not interface.get("composite"):
        return [(resources, interface)]
    return [(resources / p["dir"], json.loads((resources / p["dir"] / "interface.json").read_text()))
            for p in interface["parts"]]


def _remove_compiled(folder, keep, top=False):
    """Remove what is now inside the executable.

    Except the folders the models read data files from (``[model] cwd``), without their code,
    and the folders leading to them.

    :param folder: the folder to clean.
    :type folder: pathlib.Path
    :param keep: data folders to keep.
    :type keep: set[pathlib.Path]
    :param top: whether this is ``resources/`` (where :data:`KEEP_AFTER_COMPILE` stays).
    :type top: bool
    """
    for item in folder.iterdir():
        if item in keep:
            _remove_code(item)
        elif top and item.name in KEEP_AFTER_COMPILE:
            continue
        elif item.is_dir() and any(k.is_relative_to(item) for k in keep):
            _remove_compiled(item, keep)
        else:
            shutil.rmtree(item) if item.is_dir() else item.unlink()


def _is_package(name):
    """Whether the top-level import ``name`` is a package (a folder), not a single module.

    matplotlib's pylab.py or an extension .pyd are modules. :func:`importlib.util.find_spec`
    doesn't import a top-level name. A module that sets ``__path__ = []`` (six, for six.moves)
    has no folder either.

    :param name: the import name.
    :type name: str
    :rtype: bool
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return spec is not None and bool(spec.submodule_search_locations)


def _nuitka_package_args(packages, direct):
    """Nuitka's options for collecting ``packages`` (data files) and ``direct`` (all submodules).

    Nuitka refuses package options for a single module ("'pylab' is a module not a package"): a
    module has no data files of its own, and is included as a module.

    :param packages: packages whose data files are collected.
    :type packages: list[str]
    :param direct: packages whose submodules are all included.
    :type direct: list[str]
    :rtype: list[str]
    """
    args = [f"--include-package-data={p}" for p in packages if _is_package(p)]
    args += [f"--include-package={p}" if _is_package(p) else f"--include-module={p}" for p in direct]
    return args


def _remove_code(folder):
    """Delete the Python files of a folder (they are inside the executable), keeping its data files.

    :param folder: the folder.
    :type folder: pathlib.Path
    """
    for file in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if file.is_dir() and file.name == "__pycache__":
            shutil.rmtree(file)
        elif file.is_file() and file.suffix in (".py", ".pyc"):
            file.unlink()


def compiled_launch_command(system):
    """The ``launch.toml`` entry for the frozen backend (relative paths need a shell to resolve).

    :param system: ``"windows"``, ``"linux"`` or ``"macos"``.
    :type system: str
    :rtype: list[str]
    """
    if system == "windows":
        return ["powershell", "-command", f"./{EXE_DIR}/{EXE_NAME}.exe"]
    # unzipping the FMU drops execute bits: also those of executables the model's packages run
    # (gekko's solver), not only the backend's
    exe = f"{EXE_DIR}/{EXE_NAME}"
    return ["sh", "-c", f"chmod -R u+x {EXE_DIR} && ./{exe}"]


def _modules(interface):
    """Modules the runtime imports by name, which a freezer can't find by itself.

    The adapter, runtime, cloudpickle, the entry's module, and the modules named in setup calls,
    enum types, clock calls and constants (``ref`` and ``call``, including dotted names in a
    call's arguments).

    :param interface: one model's interface.
    :type interface: dict
    :rtype: list[str]
    """
    modules = {"model", "fmugen_runtime", "cloudpickle", interface["entry"]["module"]}
    refs = [step["call"] for step in interface.get("setup", [])]
    refs += [td["enum"] for td in interface.get("type_definitions", {}).values() if td.get("enum")]
    refs += [c["call"] for c in interface.get("clocks", []) if c.get("call")]
    step = interface["entry"].get("call")
    if isinstance(step, str) and ":" in step:   # "module:function(self, ...)": the module, and dotted names in its arguments
        refs += [step.partition("(")[0], *(name for name in _dotted_names(f"f({step.partition('(')[2]})")
                                            if name.split(".")[0] != "self")]
    for table in ("constants", "call_constants"):
        for value in interface.get(table, {}).values():
            if isinstance(value, dict) and "ref" in value:
                refs.append(value["ref"])
            elif isinstance(value, dict) and isinstance(value.get("call"), str):
                # "module:fn(numpy.linspace(...))": the module, and the dotted names in the arguments
                call = value["call"]
                refs += [call.partition("(")[0], *_dotted_names(f"f({call.partition('(')[2]})")]
    for step in interface.get("setup", []):
        refs += [a["ref"] for a in [*step.get("args", []), *step.get("kwargs", {}).values()]
                 if isinstance(a, dict) and "ref" in a]
    for ref in refs:
        if ":" in ref:
            modules.add(ref.partition(":")[0])
        elif "." in ref:
            modules.add(ref.split(".")[0])
    return sorted(modules)


#: Distributions never collected into a compiled FMU: fmugen itself and the build tools.
NOT_COLLECTED = {"fmugen", "pyinstaller", "pyinstaller-hooks-contrib", "nuitka", "pip", "setuptools", "wheel"}
#: Packages that import their own submodules by name, so they are collected with all of them even when
#: they are only a dependency: pyomo's plugins (OpenHENS), tensorly's backends (neuraloperator), ray's
#: _private modules (neuralforecast).
IMPORTS_OWN_SUBMODULES = {"pyomo", "tensorly", "ray"}


def _packages(modules, requirements=()):
    """Installed packages the frozen model needs, for the freezer to collect whole.

    ``direct`` are the import names of ``modules`` (the entry's, setup's and constants'
    packages) and of the ``requirements``, whose submodules may be imported dynamically (by
    name, or by unpickling a model that was saved with them); ``packages`` adds those of every
    distribution they depend on, whose data files and binaries a freezer doesn't find by
    following imports; ``distributions`` are all those distributions, for their metadata
    (version lookups, entry points); ``imports`` are the packages the direct ones depend on
    (with their installed optional dependencies), which they may import by name. Dependencies
    are walked through :mod:`importlib.metadata`.

    :param modules: modules the runtime imports by name.
    :type modules: list[str]
    :param requirements: the model's requirement strings.
    :type requirements: tuple or list
    :return: ``(direct, packages, distributions, imports)``, each sorted.
    :rtype: tuple[list[str], list[str], list[str], list[str]]
    """
    import importlib.metadata as md
    by_top = packages_distributions()
    tops_of = {}
    for top, names in by_top.items():
        for name in names:
            tops_of.setdefault(_canonical(name), set()).add(top)
    direct = {m.split(".")[0] for m in modules} - set(sys.stdlib_module_names)
    direct = {top for top in direct if top in by_top}
    queue = [_canonical(n) for top in sorted(direct) for n in by_top[top]]
    required = [_canonical(re.match(r"[A-Za-z0-9._-]+", r).group()) for r in requirements if r.strip()]
    direct |= {top for name in required for top in tops_of.get(name, ())
               if top.isidentifier() and top not in sys.stdlib_module_names}
    queue += required
    # The model's own distributions: their dependencies, and their optional ones when installed
    # here, are imported too, since libraries import them by name when needed (pybamm's anytree).
    roots = set(queue)
    seen, imported = set(), set()
    while queue:
        name = queue.pop()
        if name in seen or name in NOT_COLLECTED:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        seen.add(name)
        extras = (dist.metadata.get_all("Provides-Extra") or []) if name in roots else []
        deps = [dep for req in dist.requires or [] for dep in [_dependency(req, extras)] if dep]
        imported |= set(deps) if name in roots else set()
        queue += deps
    usable = lambda top: top.isidentifier() and top not in sys.stdlib_module_names
    imports = {top for name in imported & seen for top in tops_of.get(name, ()) if usable(top)}
    packages = {top for name in seen for top in tops_of.get(name, ()) if usable(top)}
    direct |= packages & IMPORTS_OWN_SUBMODULES
    return sorted(direct), sorted(packages | direct), sorted(seen), sorted(imports - direct)


def _editable_paths(tops):
    """Folders to add to the freezer's search path for packages installed in editable mode.

    ``pip install -e`` with setuptools installs an import hook (a finder in a ``.pth`` file)
    instead of putting the project's folder on ``sys.path``, and freezers don't follow it: the
    package would be missing from the frozen FMU (the debutanizer and OpenHENS projects).

    :param tops: top-level module names the model imports.
    :type tops: collections.abc.Iterable[str]
    :return: the folders holding those of them that aren't found through ``sys.path``.
    :rtype: list[pathlib.Path]
    """
    on_path = {Path(p).resolve() for p in sys.path if p}
    folders = []
    for top in sorted(tops):
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
            continue
        location = Path(spec.origin).resolve().parent
        folder = location.parent if spec.submodule_search_locations else location
        if folder not in on_path and folder not in folders:
            folders.append(folder)
    return folders


#: Libraries that read the source of code built on them: inspect.getsource (transformers),
#: torch.jit.script when a module is imported (mace, norse).
NEEDS_SOURCE = {"transformers", "torch"}
#: Packages that use TorchScript without declaring torch as a dependency (matgl's torch_geometric).
SOURCE_ALWAYS = {"torch_geometric"}


def _source_hooks(packages, hooks_dir):
    """PyInstaller hooks that also ship the .py files of packages built on :data:`NEEDS_SOURCE`.

    transformers' docstring decorators read the source of the model classes they decorate.
    PyInstaller's own hook does this for transformers itself. A package with a hook of its own
    is left alone.

    :param packages: the collected packages.
    :type packages: list[str]
    :param hooks_dir: where to write the hooks.
    :type hooks_dir: pathlib.Path
    :return: ``hooks_dir``, or None if no package needs it.
    :rtype: pathlib.Path or None
    """
    import importlib.metadata as md
    by_top = packages_distributions()
    existing = {p.name for d in _hook_dirs() for p in Path(d).glob("hook-*.py")}
    names = []
    for top in packages:
        if f"hook-{top}.py" in existing or top in NEEDS_SOURCE:
            continue
        if top in SOURCE_ALWAYS:
            names.append(top)
            continue
        for dist in by_top.get(top, []):
            try:
                requires = md.distribution(dist).requires or []
            except md.PackageNotFoundError:
                continue
            if any(_dependency(r) in NEEDS_SOURCE for r in requires):
                names.append(top)
                break
    if not names:
        return None
    hooks_dir.mkdir(parents=True, exist_ok=True)
    for top in names:
        (hooks_dir / f"hook-{top}.py").write_text("module_collection_mode = 'pyz+py'\n")
    return hooks_dir


def _hook_dirs():
    """PyInstaller's and pyinstaller-hooks-contrib's hook folders (empty if not installed).

    :rtype: list[str]
    """
    dirs = []
    for name in ("PyInstaller.hooks", "_pyinstaller_hooks_contrib.stdhooks"):
        spec = importlib.util.find_spec(name)
        dirs += list(spec.submodule_search_locations or []) if spec else []
    return dirs


def _plugin_dirs(packages):
    """Folders of ``packages`` that hold .py files but aren't packages (no ``__init__.py``).

    Code a library loads by path (pyromat's registry), so it ships as files.

    :param packages: package names.
    :type packages: list[str]
    :return: ``[(folder, destination inside the executable)]``.
    :rtype: list[tuple]
    """
    found = []
    for name in packages:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            root = Path(location)
            for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
                if folder.name in ("__pycache__", "tests", "test", "doc", "docs") or (folder / "__init__.py").exists():
                    continue
                if any(folder.glob("*.py")) and all((parent / "__init__.py").exists()
                                                    for parent in folder.parents if parent.is_relative_to(root)):
                    found.append((folder, Path(name, folder.relative_to(root)).as_posix()))
    return found


def _package_folders(packages):
    """``[(package folder, its path relative to the package's parent)]`` of each of ``packages``.

    :param packages: package names.
    :type packages: list[str]
    :return: one pair per folder of each package (a namespace package may have several).
    :rtype: list[tuple]
    """
    found = []
    for name in packages:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):   # ValueError: imported already, without a spec (humanfriendly)
            continue
        if spec is not None and spec.submodule_search_locations:
            found += [(Path(location), Path(name)) for location in spec.submodule_search_locations]
    return found


def _loaded_libraries(packages):
    """Native libraries inside ``packages`` that PyInstaller's ``--collect-binaries`` skips.

    It collects ``lib*.so`` and imported extension modules; a library loaded by path, like
    torchvision's ``_C_stable.so`` (``torch.ops.load_library``), is neither, and without it
    ``import torchvision`` fails in the frozen FMU (timm, ultralytics).

    :param packages: package names.
    :type packages: list[str]
    :return: ``[(file, destination folder inside the executable)]``.
    :rtype: list[tuple]
    """
    found = []
    for root, dest in _package_folders(packages):
        for path in sorted(root.rglob("*")):
            name = path.name
            native = name.endswith((".so", ".pyd", ".dylib")) or ".so." in name
            if native and not name.startswith("lib") and path.is_file():
                found.append((path, (dest / path.parent.relative_to(root)).as_posix()))
    return found


def _add_package_folders(packages, internal):
    """Create each package's folder tree in the frozen app's ``_internal``, without files.

    PyInstaller keeps modules in an archive, so a package folder holding only code doesn't exist on
    disk, and a path built from ``__file__`` through it fails on Linux and macOS
    (chgnet's ``chgnet/model/../pretrained/...``) even though the data file is there.

    :param packages: package names.
    :type packages: list[str]
    :param internal: the frozen app's ``_internal`` folder.
    :type internal: pathlib.Path
    """
    for root, dest in _package_folders(packages):
        for folder in [root, *(p for p in root.rglob("*") if p.is_dir() and p.name != "__pycache__")]:
            made = internal / dest / folder.relative_to(root)
            made.mkdir(parents=True, exist_ok=True)
            if not any(made.iterdir()):   # the FMU is a zip, which keeps no empty folders
                (made / ".keep").touch()


def _canonical(name):
    """A distribution name in canonical form (PEP 503: lower case, runs of ``-_.`` as ``-``).

    :param name: the name.
    :type name: str
    :rtype: str
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _dependency(requirement, extras=()):
    """The distribution a Requires-Dist line needs here, or None.

    None for another platform, or an extra not in ``extras``. An optional dependency that isn't
    installed is skipped by the caller. Markers are evaluated with ``packaging`` when installed.

    :param requirement: the Requires-Dist line.
    :type requirement: str
    :param extras: the extras that count as requested.
    :type extras: tuple or list
    :return: the canonical distribution name, or None.
    :rtype: str or None
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:   # without packaging: extras only when asked, the rest always
        if "extra" in requirement.partition(";")[2] and not extras:
            return None
        return _canonical(re.match(r"[A-Za-z0-9._-]+", requirement).group())
    req = Requirement(requirement)
    if req.marker and not any(req.marker.evaluate({"extra": e}) for e in ["", *extras]):
        return None
    return _canonical(req.name)


def _model_data(resources, interface):
    """Non-Python files of the model, placed where the model's modules will look for them.

    Each file goes relative to the deepest ``sys_path`` folder it lies in, since inside the
    executable the modules' ``__file__`` points there.

    :param resources: the model's resources folder.
    :type resources: pathlib.Path
    :param interface: the model's interface.
    :type interface: dict
    :return: ``[(file, destination folder)]``.
    :rtype: list[tuple]
    """
    roots = sorted((resources / p for p in interface["sys_path"]), key=lambda p: len(p.parts), reverse=True)
    data = []
    model_dir = resources / MODEL_DIR
    if not model_dir.is_dir():
        return data
    for file in model_dir.rglob("*"):
        if file.is_dir() or file.suffix in (".py", ".pyc") or "__pycache__" in file.parts:
            continue
        root = next((r for r in roots if file.is_relative_to(r)), None)
        if root is not None:
            data.append((file, file.parent.relative_to(root).as_posix() or "."))
    return data
