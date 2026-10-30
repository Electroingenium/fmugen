"""fmugen.toml: loading, validation and normalisation into the interface spec.

The config describes how an unmodified Python model maps onto an FMU: which callable
to use, which of its arguments and attributes are FMU variables, and FMU-level
settings such as the default experiment. See docs/config.md for the full reference.

`load_config()` only reads and validates the file. `normalize()` needs the imported
entry object (to tell a function from a class and to check argument names) and
returns the interface spec that is written to resources/interface.json and read by
the runtime engine (templates/fmugen_runtime/) and the XML writer (description.py).
The spec has the same shape for FMI 2 and FMI 3, except for the type names and the
FMI 3-only parts (arrays, structural parameters, clocks, events).
"""
import datetime
import inspect
import keyword
import sys
import tomllib
from pathlib import Path, PurePosixPath

from fmugen.templates.fmugen_runtime import SPEC_VERSION

#: FMI versions fmugen can build (``[model] fmi_version`` / ``--fmi``).
FMI_VERSIONS = (2, 3)
#: Floating-point type names accepted in the config (``Real`` is FMI 2's name).
FLOAT_TYPES = ("Real", "Float32", "Float64")
#: Integer type names accepted in the config (``Integer`` is FMI 2's 32-bit type).
INT_TYPES = ("Integer", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64")
#: Every ``type`` a variable may have in the config.
FMI_TYPES = (*FLOAT_TYPES, *INT_TYPES, "Boolean", "String", "Binary", "Enumeration")
#: Config type name -> type name in the spec, per FMI version (missing: not available).
#: FMI 2 has one float and one 32-bit integer type: the other float and integer types become those;
#: FMI 3 ``Real`` / ``Integer`` mean ``Float64`` / ``Int32``.
TYPE_NAMES = {
    # FMI 2 has one float and one 32-bit integer type: the other float and integer types become those
    2: {**{t: "Real" for t in FLOAT_TYPES}, **{t: "Integer" for t in INT_TYPES},
        "Boolean": "Boolean", "String": "String", "Enumeration": "Enumeration"},
    3: {"Real": "Float64", "Integer": "Int32", **{t: t for t in FMI_TYPES if t not in ("Real", "Integer")}},
}
#: Start value of a variable that needs one but has no ``start``, by type.
DEFAULT_START = {
    **{t: 0.0 for t in FLOAT_TYPES}, **{t: 0 for t in INT_TYPES},
    "Boolean": False, "String": "", "Binary": "", "Enumeration": 1,
}

#: Folder of the user's code inside the FMU's ``resources/`` folder.
MODEL_DIR = "fmugen_model"   # the user's code, inside the FMU's resources/ folder

#: Module names already used by the UniFMU backend and fmugen in ``resources/``;
#: a model module with one of these names would shadow them.
RESERVED_MODULES = {"model", "backend", "main", "abstract_backend", "schemas", "fmugen_runtime", "fmugen_launch"}

#: Config section -> FMI causality, in valueReference order.
SECTIONS = {
    "structural_parameters": "structuralParameter",
    "parameters": "parameter",
    "calculated_parameters": "calculatedParameter",
    "inputs": "input",
    "states": "local",
    "outputs": "output",
    "locals": "local",
}

#: Keys allowed in ``[model]``.
MODEL_KEYS = {
    "entry", "call", "name", "description", "author", "sources", "requirements",
    "constants", "call_constants", "init_call", "terminate", "fmi_version", "setup", "kind", "create",
    "save_state", "cwd", "globals", "discard_on", "integrate", "substeps", "annotations", "call_on_class",
}
#: Keys allowed in ``[experiment]`` (the default experiment).
EXPERIMENT_KEYS = {"start_time", "stop_time", "step_size", "tolerance", "fixed_step"}
#: What a ``[time]`` argument can receive.
TIME_SOURCES = ("time", "step_size", "end_time", "start_time", "stop_time", "tolerance")
#: Built-in ``[model] integrate`` methods, besides ``"scipy:<solve_ivp method>"``.
INTEGRATORS = ("euler", "rk4")   # [model] integrate, besides "scipy:<solve_ivp method>"
#: ``scipy.integrate.solve_ivp`` methods allowed as ``integrate = "scipy:<method>"``.
SCIPY_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")
#: FMI 2 Integer is 32-bit: these types are narrowed to it (with a note).
WIDER_THAN_FMI2_INTEGER = ("Int64", "UInt32", "UInt64")
#: Keys allowed in every variable section.
COMMON_VAR_KEYS = {
    "type", "start", "unit", "description", "variability", "initial",
    "min", "max", "nominal", "quantity", "items", "enum", "dimensions", "numpy", "convert", "epoch", "extra",
}
#: Variable section -> keys allowed in its variables (common keys plus the bindings that fit
#: the section: ``to`` for values passed in, ``from`` for values read out, ``next`` / ``der`` for states).
VAR_KEYS = {
    "structural_parameters": COMMON_VAR_KEYS | {"to", "attr"},
    "parameters": COMMON_VAR_KEYS | {"to", "attr"},
    "calculated_parameters": COMMON_VAR_KEYS | {"from"},
    "inputs": COMMON_VAR_KEYS | {"to", "clocks"},
    "states": COMMON_VAR_KEYS | {"to", "next", "der"},
    "outputs": COMMON_VAR_KEYS | {"from", "depends_on", "clocks"},
    "locals": COMMON_VAR_KEYS | {"from", "clocks"},
}
#: Keys allowed in a ``[clocks.<name>]`` table.
CLOCK_KEYS = {"causality", "interval_variability", "interval", "shift", "interval_from", "call", "from", "description"}
#: FMI 3 clock ``intervalVariability`` values.
INTERVAL_VARIABILITIES = ("constant", "fixed", "tunable", "changing", "countdown", "triggered")
#: Keys allowed in ``[events]``.
EVENT_KEYS = {"terminate", "next_event_time"}
#: Top-level tables allowed in fmugen.toml.
TOP_KEYS = {"model", "experiment", "time", "clocks", "events", *SECTIONS}
#: Message suffix for features that need FMI 3.
FMI3_ONLY = "requires FMI 3 ([model] fmi_version = 3 or fmugen build --fmi 3)"

#: FMI causality -> the variabilities it allows (defaults are chosen in ``_variable``).
VARIABILITIES = {
    "structuralParameter": ("fixed", "tunable"),
    "parameter": ("fixed", "tunable"),
    "calculatedParameter": ("fixed", "tunable"),
    "input": ("discrete", "continuous"),
    "output": ("constant", "discrete", "continuous"),
    "local": ("constant", "discrete", "continuous"),
}
#: FMI causality -> the ``initial`` values it allows.
INITIALS = {
    "structuralParameter": ("exact",),
    "parameter": ("exact",),
    "calculatedParameter": ("approx", "calculated"),
    "input": (),
    "output": ("exact", "calculated"),
    "local": ("exact", "approx", "calculated"),
}


class InterfaceError(ValueError):
    """A problem with the config or the model's interface, reported to the user without a traceback.

    The CLI prints its message and exits with an error; ``fmugen init`` turns some of them into
    warnings.
    """
    pass


class Config:
    """A validated fmugen.toml (or an equivalent dict produced by inference).

    Construction checks every table for unknown keys, that ``[model] entry`` exists and that
    ``fmi_version`` and ``cwd`` are valid. It does not import the model: that, and the checks that
    need the imported entry, are done by :func:`normalize`.

    :ivar data: the whole config as a dict.
    :ivar base_dir: the folder of fmugen.toml; all relative paths are relative to it.
    :ivar model: the ``[model]`` table.
    :ivar experiment: the ``[experiment]`` table.
    :ivar entry_target: the part of ``entry`` before ``:`` (a ``.py`` path or a module name).
    :ivar entry_name: the class or function name after ``:``.
    :ivar entry_is_file: whether ``entry_target`` is a ``.py`` file.
    :ivar requirements: ``[model] requirements`` (pip requirement strings).
    """

    def __init__(self, data, base_dir):
        """Validate the config's structure.

        :param data: the parsed TOML (or the dict from :func:`fmugen.interface.infer_config`).
        :type data: dict
        :param base_dir: the folder relative paths refer to.
        :type base_dir: str or pathlib.Path
        :raises InterfaceError: unknown keys, a missing or invalid entry, a bad ``fmi_version``,
            a variable that isn't a table, or a ``cwd`` that isn't a folder.
        """
        self.data = data
        self.base_dir = Path(base_dir).resolve()
        _check_keys(data, TOP_KEYS, "fmugen.toml")
        self.model = data.get("model", {})
        _check_keys(self.model, MODEL_KEYS, "[model]")
        if "entry" not in self.model:
            raise InterfaceError("[model] entry is required, e.g. entry = \"model.py:simulate\"")
        if self.model.get("fmi_version", 2) not in FMI_VERSIONS:
            raise InterfaceError(f"[model] fmi_version must be one of {FMI_VERSIONS}")
        self.experiment = data.get("experiment", {})
        _check_keys(self.experiment, EXPERIMENT_KEYS, "[experiment]")
        for section, keys in VAR_KEYS.items():
            for name, info in data.get(section, {}).items():
                if not isinstance(info, dict):
                    raise InterfaceError(f"[{section}] {name} must be a table, e.g. {name} = {{ start = 0.0 }}")
                _check_keys(info, keys, f"[{section}] {name}")
        for name, info in data.get("clocks", {}).items():
            _check_keys(info, CLOCK_KEYS, f"[clocks.{name}]")
        _check_keys(data.get("events", {}), EVENT_KEYS, "[events]")

        self.entry_target, self.entry_name = parse_entry(self.model["entry"])
        self.entry_is_file = self.entry_target.endswith(".py")
        self.requirements = list(self.model.get("requirements", []))
        if "cwd" in self.model:
            cwd = self._inside(str(self.model["cwd"]), "cwd")
            if not cwd.is_dir():
                raise InterfaceError(f"[model] cwd {self.model['cwd']!r} is not a folder (relative to {self.base_dir})")

    def fmi_version(self, override=None):
        """The FMI version to build.

        :param override: ``--fmi`` from the command line, which wins over ``[model] fmi_version``.
        :type override: int or None
        :return: 2 or 3 (2 by default).
        :rtype: int
        """
        return override or self.model.get("fmi_version", 2)

    # ---- files ----

    def entry_file(self):
        """The absolute path of the entry file, or None when the entry is a module name.

        :rtype: pathlib.Path or None
        :raises InterfaceError: the path points outside the config folder.
        """
        return self._inside(self.entry_target, "entry") if self.entry_is_file else None

    def model_cwd(self):
        """Where model code runs, relative to the FMU's ``resources/`` folder (``[model] cwd``), or None.

        :rtype: str or None
        """
        if "cwd" not in self.model:
            return None
        rel = self._inside(str(self.model["cwd"]), "cwd").relative_to(self.base_dir).as_posix()
        return (PurePosixPath(MODEL_DIR) / rel).as_posix()

    def source_files(self):
        """The files and folders to copy into the FMU: the entry file and ``[model] sources``.

        :return: ``[(absolute source path, path relative to the config dir)]``.
        :rtype: list[tuple[pathlib.Path, pathlib.PurePosixPath]]
        :raises InterfaceError: a source that doesn't exist or lies outside the config folder.
        """
        paths = [self.entry_target] if self.entry_is_file else []
        paths += self.model.get("sources", [])
        files = []
        for rel in paths:
            src = self._inside(rel, "sources")
            if not src.exists():
                raise InterfaceError(f"{rel} does not exist (relative to {self.base_dir})")
            files.append((src, PurePosixPath(src.relative_to(self.base_dir).as_posix())))
        return files

    def _inside(self, rel, what):
        """Resolve a relative path and check it stays inside the config folder.

        :param rel: a path relative to :attr:`base_dir`.
        :type rel: str
        :param what: the config key, for the error message.
        :type what: str
        :return: the absolute path.
        :rtype: pathlib.Path
        :raises InterfaceError: the path escapes the config folder (e.g. ``../x``).
        """
        path = (self.base_dir / rel).resolve()
        if not path.is_relative_to(self.base_dir):
            raise InterfaceError(f"{what} {rel!r} must be inside {self.base_dir}")
        return path

    def entry_import(self):
        """How the runtime imports the entry: its module name and the ``sys.path`` folders it needs.

        For a file inside packages (folders with ``__init__.py``) the module is imported with its
        package path, so relative imports work; the file's own folder is also put on ``sys.path`` for
        flat imports.

        :return: ``(module name, sys.path entries relative to resources/)``.
        :rtype: tuple[str, list[str]]
        :raises InterfaceError: the module's top-level name is one of :data:`RESERVED_MODULES`.
        """
        sys_path = []
        if not self.entry_is_file:
            module = self.entry_target
        else:
            rel = PurePosixPath(self.entry_file().relative_to(self.base_dir).as_posix())
            dirs = rel.parts[:-1]
            # Import as a package module when the file sits in a package (supports relative
            # imports); the file's own directory is also on sys.path for flat imports.
            root = len(dirs)
            while root > 0 and (self.base_dir.joinpath(*dirs[:root]) / "__init__.py").exists():
                root -= 1
            module = ".".join(dirs[root:] + (rel.stem,))
            for path in (PurePosixPath(MODEL_DIR, *dirs), PurePosixPath(MODEL_DIR, *dirs[:root])):
                if str(path) not in sys_path:
                    sys_path.append(str(path))
        if module.split(".")[0] in RESERVED_MODULES:
            raise InterfaceError(
                f"model module {module!r} clashes with a UniFMU backend module "
                f"({', '.join(sorted(RESERVED_MODULES))}); rename the file"
            )
        return module, sys_path

    def local_modules(self):
        """Top-level module names the model's own files are imported as.

        Its entry file and the ``.py`` files and packages of its sources, under each folder put on
        ``sys.path``. Installed packages aren't included.

        :return: module names.
        :rtype: set[str]
        """
        module, sys_path = self.entry_import()
        names = {module.split(".")[0]} if self.entry_is_file else set()
        files = []
        for src, _ in self.source_files():
            files += sorted(src.rglob("*.py")) if src.is_dir() else [src] if src.suffix == ".py" else []
        for path in sys_path:
            root = self.base_dir.joinpath(*PurePosixPath(path).relative_to(MODEL_DIR).parts)
            for file in files:
                if file.is_relative_to(root):
                    rel = file.relative_to(root)
                    names.add(rel.stem if len(rel.parts) == 1 else rel.parts[0])
        return names

    def check_module_names(self):
        """Refuse model files named like a standard library module (json.py, logging.py, queue.py).

        The FMU's backend has already imported the real one, so importing the model's name gives the
        standard library's module, or the model's module replaces it for the backend.

        :raises InterfaceError: some model modules clash with :data:`sys.stdlib_module_names`.
        """
        clashes = sorted(self.local_modules() & set(sys.stdlib_module_names))
        if clashes:
            raise InterfaceError(stdlib_clash(clashes))


def stdlib_clash(names):
    """The error message for model modules named like standard library modules.

    :param names: the clashing module names.
    :type names: list[str]
    :return: the message.
    :rtype: str
    """
    return (f"the model's module(s) {', '.join(names)} have the name of a Python standard library module, "
            "which the FMU's backend also imports: rename the file(s)")


def load_config(path):
    """Read and validate an fmugen.toml (no model import).

    :param path: the config file.
    :type path: str or pathlib.Path
    :return: the validated config; relative paths are relative to its folder.
    :rtype: Config
    :raises InterfaceError: invalid TOML or an invalid config.
    """
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise InterfaceError(f"{path}: {e}") from e
    return Config(data, path.parent)


def parse_entry(entry):
    """Split ``[model] entry`` into target and name.

    :param entry: ``"path/to/file.py:Name"`` or ``"package.module:Name"``.
    :type entry: str
    :return: ``(target, name)``.
    :rtype: tuple[str, str]
    :raises InterfaceError: no ``:``, an empty target, or a name that isn't an identifier.
    """
    target, sep, name = entry.rpartition(":")
    if not sep or not target or not name.isidentifier():
        raise InterfaceError(
            f"entry {entry!r} must look like 'path/to/file.py:Name' or 'package.module:Name'"
        )
    return target, name


def parse_binding(text, kinds, what):
    """Parse a binding such as ``"arg:x"``, ``"attr:state.T"`` or ``"return"``.

    These are the ``to``, ``from``, ``next``, ``der`` and event values of the config; the result
    is the ``{"kind", "name"}`` form stored in ``interface.json`` (see ``docs/interface.md``,
    Bindings).

    :param text: the binding text.
    :type text: str
    :param kinds: the kinds allowed here (e.g. ``("return", "attr")``).
    :type kinds: tuple[str, ...]
    :param what: where it comes from, for error messages.
    :type what: str
    :return: ``{"kind": kind, "name": name or None}``.
    :rtype: dict
    :raises InterfaceError: an unknown kind, or a missing name (only ``return`` may omit it).
    """
    kind, _, name = text.partition(":")
    if kind not in kinds:
        raise InterfaceError(f"{what}: {text!r} must start with one of {', '.join(k + ':' for k in kinds)}")
    if kind != "return" and not name:
        raise InterfaceError(f"{what}: {text!r} needs a name after '{kind}:'")
    return {"kind": kind, "name": name or None}


def normalize(config, entry_obj, module_name, sys_path, model_name=None, author=None, fmi_version=None):
    """Return the interface spec for a validated Config and its imported entry object.

    This is where the config meets the real model: the entry is classified (function, class,
    factory function, class called like a function), every argument name in the config is
    checked against the real signatures, variables get FMI types, defaults, bindings and value
    references, and the FMI-version rules are applied. The result is written as
    ``resources/interface.json`` (fields: ``docs/interface.md``) and used for
    ``modelDescription.xml``.

    :param config: the validated config.
    :type config: Config
    :param entry_obj: the imported entry (class or function).
    :param module_name: the module the runtime imports the entry from.
    :type module_name: str
    :param sys_path: ``sys.path`` entries relative to ``resources/``.
    :type sys_path: list[str]
    :param model_name: ``--name``, overriding ``[model] name``.
    :type model_name: str or None
    :param author: ``--author``, overriding ``[model] author``.
    :type author: str or None
    :param fmi_version: ``--fmi``, overriding ``[model] fmi_version``.
    :type fmi_version: int or None
    :return: the interface spec; its ``notes`` are warnings for the user.
    :rtype: dict
    :raises InterfaceError: anything in the config that doesn't fit the model or the FMI version.
    """
    model = config.model
    version = config.fmi_version(fmi_version)
    if version not in FMI_VERSIONS:
        raise InterfaceError(f"unknown FMI version {version!r}, expected one of {FMI_VERSIONS}")
    notes = []
    if model.get("kind", "auto") not in ("auto", "function"):
        raise InterfaceError('[model] kind must be "function" (call a class like a function on every step)')
    # kind = "function": a class whose constructor does the work is called like a function on every step
    is_class = inspect.isclass(entry_obj) and model.get("kind") != "function"
    if not is_class and not callable(entry_obj):
        raise InterfaceError(f"{config.model['entry']} is neither a function nor a class")
    # A function entry with a call method is a factory: called once with the parameters, like a
    # constructor; `call` runs on what it returns (e.g. silero_vad:load_silero_vad, call = "__call__").
    factory = not inspect.isclass(entry_obj) and isinstance(model.get("call"), str)
    if factory and model.get("kind") == "function":
        raise InterfaceError('[model] kind = "function" and call exclude each other for a function entry')
    is_class = is_class or factory
    kind = "class" if is_class else "function"
    create = model.get("create")   # a classmethod that builds the object, e.g. "from_pretrained"
    if create is not None:
        if not is_class or factory:
            raise InterfaceError("[model] create only applies to classes")
        if not isinstance(create, str) or not callable(getattr(entry_obj, create, None)):
            raise InterfaceError(f"[model] create: {entry_obj.__name__} has no classmethod {create!r}")
        constructor = getattr(entry_obj, create)
    elif factory:
        constructor = entry_obj
    else:
        constructor = entry_obj.__init__ if is_class else None

    # Signatures of what the runtime will call: the constructor (init_sig, for init:/initpos:
    # bindings and constants) and the step callable (call_sig, for arg:/pos: and [time]). They
    # let every argument name in the config be checked now instead of failing inside the FMU.
    call = model.get("call")
    step_call = call is not False   # call = false: nothing runs on doStep, only clocks run code
    if not step_call and not config.data.get("clocks"):
        raise InterfaceError("[model] call = false needs [clocks] (otherwise the model never runs)")
    if is_class and not step_call:
        call = None
        init_sig = _signature(constructor)
        call_sig = None
    elif isinstance(call, str) and ":" in call:   # "module:function(self, ...)": a function taking the object
        _object_call(call, is_class, bool(model.get("call_on_class")))
        init_sig = _signature(constructor)
        call_sig = None   # its leading arguments are given in the call text, so names can't be checked here
    elif factory:
        init_sig = _signature(constructor)
        call_sig = None   # the method belongs to the object the factory returns, known only at runtime
    elif is_class:
        if call is None:
            if not any("__call__" in vars(k) for k in entry_obj.__mro__[:-1]):
                raise InterfaceError(f"{entry_obj.__name__} is a class: set [model] call to the method run on each step")
            call = "__call__"
        if not callable(getattr(entry_obj, call, None)):
            raise InterfaceError(f"{entry_obj.__name__} has no method {call!r}")
        init_sig = _signature(constructor)
        call_sig = _signature(getattr(entry_obj, call))
    else:
        for key in ("call", "terminate", "call_constants"):
            if key in model and not (key == "call" and call is False):
                raise InterfaceError(f"[model] {key} only applies to classes; {model['entry']} is a function")
        init_sig = None
        call_sig = _signature(entry_obj) if step_call else None

    if step_call and (inspect.isgeneratorfunction(entry_obj) if not is_class
                      else call and inspect.isgeneratorfunction(getattr(entry_obj, call, None))):
        raise InterfaceError(f"{model['entry']} is a generator (it uses yield), which fmugen can't step yet; "
                             "a factory that returns an object with a step method works ([model] call)")
    integrate = _integrate(model, step_call)
    repeat = _repeat(model, step_call, integrate)
    discard_on = _discard_on(model.get("discard_on", []))
    if discard_on and _save_state(model) is False:
        raise InterfaceError("[model] discard_on needs the FMU state (save_state not false): a rejected step is "
                             "undone by restoring the state saved before it")

    def check_arg(sig, arg, what):
        """Raise unless ``arg`` is an argument of the signature (when the signature is known).

        :param sig: ``(inspect.Signature, qualified name)`` from :func:`_signature`, or None.
        :type sig: tuple or None
        :param arg: the argument name.
        :type arg: str
        :param what: the config key, for the error message.
        :type what: str
        :raises InterfaceError: the function has no such argument (and no ``**kwargs``).
        """
        if sig is not None and not _accepts(sig, arg):
            raise InterfaceError(f"{what}: {arg!r} is not an argument of {sig[1]}")

    constants = model.get("constants", {})
    call_constants = model.get("call_constants", {})
    # constants named "0", "1", ... are positional arguments (for *args); checked with the variables' positions
    for name in constants:
        if not name.isdigit():
            check_arg(init_sig if is_class else call_sig, name, f"[model] constants.{name}")
    for name in call_constants:
        if not name.isdigit():
            check_arg(call_sig, name, f"[model] call_constants.{name}")
    for table, values in (("constants", constants), ("call_constants", call_constants)):
        for name, value in values.items():
            if isinstance(value, dict) and "call" in value:   # computed when the FMU initializes
                from fmugen.templates.fmugen_runtime import parse_call_text
                try:
                    parse_call_text(str(value["call"]))
                except (ValueError, SyntaxError) as e:
                    raise InterfaceError(f"[model] {table}.{name}: {e}") from e

    time_args, time_epochs = {}, {}
    for arg, source in config.data.get("time", {}).items():
        if isinstance(source, dict):   # { source = "time", epoch = "2026-06-21T00:00:00+00:00" }: a datetime
            unknown = set(source) - {"source", "epoch"}
            if unknown:
                raise InterfaceError(f"[time] {arg}: unknown keys {sorted(unknown)}")
            if "epoch" in source:
                try:
                    datetime.datetime.fromisoformat(str(source["epoch"]))
                except ValueError as e:
                    raise InterfaceError(f"[time] {arg}: epoch {source['epoch']!r} is not an ISO 8601 date-time") from e
                time_epochs[arg] = str(source["epoch"])
            source = source.get("source", "time")
        if source not in TIME_SOURCES:
            raise InterfaceError(f"[time] {arg} = {source!r}: expected one of {TIME_SOURCES}")
        if arg in time_epochs and source == "step_size":
            raise InterfaceError(f"[time] {arg}: epoch applies to \"time\" and \"end_time\", not \"step_size\"")
        check_arg(call_sig, arg, f"[time] {arg}")
        time_args[arg] = source

    clock_names = set(config.data.get("clocks", {}))
    structural = set(config.data.get("structural_parameters", {}))
    if version == 2:
        for key in ("structural_parameters", "clocks"):
            if config.data.get(key):
                raise InterfaceError(f"[{key}] {FMI3_ONLY}")
        if config.data.get("events"):
            notes.append("[events] is ignored for FMI 2: an FMI 2 FMU cannot signal events or ask to stop")

    clocks = [_clock(name, info, entry_obj, is_class, time_args)
              for name, info in config.data.get("clocks", {}).items()]
    clock_sigs = {c["name"]: c.pop("_signature") for c in clocks}

    variables, type_definitions = [], {}
    for section, causality in SECTIONS.items():
        for name, info in config.data.get(section, {}).items():
            var = _variable(section, causality, name, info, is_class, type_definitions,
                            version, structural, clock_names, integrate is not None)
            if version == 2 and info.get("type") in WIDER_THAN_FMI2_INTEGER:
                notes.append(f"[{section}] {name}: FMI 2 has no {info['type']}; it is a 32-bit Integer "
                             "in this FMU, so values outside -2147483648..2147483647 are an error")
            to = var.get("to")
            if to and to["kind"] in ("init", "arg"):
                clock = (var.get("clocks") or [None])[0]
                sig = init_sig if to["kind"] == "init" else (clock_sigs[clock] if clock else call_sig)
                from fmugen.templates.fmugen_runtime import split_item
                check_arg(sig, split_item(to["name"])[0], f"[{section}] {name}")
            variables.append(var)

    if not variables:
        raise InterfaceError(
            "the config defines no FMU variables: add [inputs]/[parameters] and [outputs] "
            "(the model's arguments and results could not be inferred)"
        )
    if integrate is not None:
        states = [v for v in variables if "der" in v]
        if not states:
            raise InterfaceError("[model] integrate needs [states]: the variables the model returns the derivatives of")
        if len(states) > 1 and any(v["der"] == {"kind": "return", "name": None} for v in states):
            raise InterfaceError("[model] integrate with several states: give each one der = \"return:...\" "
                                 "(which part of the result is its derivative)")
    if not any("from" in v for v in variables):
        notes.append("the FMU has no outputs: add [outputs] to read results from the model")
    # Positional bindings (variables with to = "pos:N" plus constants named "0", "1", ...) are
    # assembled into one argument list per callable at runtime, so each list must be 0..n-1.
    for pos_kind, clock_name in [("initpos", None), *(("pos", c) for c in [None, *clock_names])]:
        positions = [int(v["to"]["name"]) for v in variables
                     if v.get("to", {}).get("kind") == pos_kind and (v.get("clocks") or [None])[0] == clock_name]
        if clock_name is None:
            table = call_constants if is_class and pos_kind == "pos" else constants
            positions += [int(k) for k in table if k.isdigit()]
        positions.sort()
        if positions != list(range(len(positions))):
            where = "the constructor's" if pos_kind == "initpos" else "the call's"
            raise InterfaceError(f"{where} positional arguments (to = \"pos:N\") must be numbered 0, 1, 2, ... "
                                 f"without gaps; got {positions}")

    names = [v["name"] for v in variables]
    duplicates = sorted({n for n in names if names.count(n) > 1} | (set(names) & clock_names))
    if duplicates:
        raise InterfaceError(f"duplicate variable or clock names: {duplicates}")
    for clock in clocks:
        clocked_inputs = [v["name"] for v in variables if v.get("clocks") == [clock["name"]] and "to" in v]
        if clocked_inputs and clock["causality"] == "input" and not clock.get("call"):
            raise InterfaceError(f"[clocks.{clock['name']}] has clocked inputs {clocked_inputs} but no call")
    if version == 3:
        if "time" in names:
            raise InterfaceError("FMI 3 reserves the variable name 'time' for the independent variable; rename it")
        variables.append({"name": "time", "causality": "independent", "variability": "continuous",
                          "type": "Float64", "description": "Simulation time"})
    # Value references follow the section order of SECTIONS; clocks come after all variables.
    vr = 0
    for var in variables:
        var["valueReference"] = vr
        vr += element_count(var) if version == 2 else 1   # FMI 2: an array takes one reference per element
    for vr, clock in enumerate(clocks, start=vr):
        clock["valueReference"] = vr

    inputs = {v["name"] for v in variables if v["causality"] == "input"}
    for var in variables:
        for dep in var.get("depends_on", []):
            if dep not in inputs:
                raise InterfaceError(f"[outputs] {var['name']}: depends_on {dep!r} is not an input")

    events = {}
    if version == 3:
        kinds = ("return", "attr") if is_class else ("return",)
        for key, text in config.data.get("events", {}).items():
            events[key] = parse_binding(text, kinds, f"[events] {key}")

    experiment = dict(config.experiment)
    if experiment.get("fixed_step") and "step_size" not in experiment:
        raise InterfaceError("[experiment] fixed_step = true requires step_size")

    init_call = model.get("init_call", not is_class and step_call)
    if ("init_call" not in model and init_call and "step_size" in time_args.values()
            and "step_size" not in experiment):
        # without a step size the init call would pass 0 as the step size (a model dividing by it fails)
        init_call = False
        notes.append("the function is not called at initialization: it takes the step size, and [experiment] "
                     "has no step_size to pass; its outputs keep their start values until the first step")

    doc = inspect.getdoc(entry_obj) or inspect.getdoc(inspect.getmodule(entry_obj)) or ""
    annotations = model.get("annotations")
    if annotations is not None:
        _check_annotations(annotations)
    return {
        "spec_version": SPEC_VERSION,
        "fmi_version": version,
        "model_name": model_name or model.get("name") or config.entry_name,
        "description": model.get("description", summary(doc)),
        "author": author if author is not None else model.get("author", ""),
        "entry": {
            "module": module_name,
            "name": config.entry_name,
            "kind": kind,
            "create": create,
            "factory": factory,
            "call": call if is_class and step_call else None,
            "call_on_class": bool(model.get("call_on_class")) and is_class,
            "step": step_call,
            "init_call": init_call,
            "terminate": model.get("terminate"),
        },
        "sys_path": sys_path,
        "cwd": config.model_cwd(),
        "globals": _globals(model.get("globals", [])),
        "setup": _setup(model.get("setup", []), is_class),
        "constants": constants,
        "call_constants": call_constants,
        "time_args": time_args,
        "time_epochs": time_epochs,
        "experiment": experiment,
        "type_definitions": type_definitions,
        "variables": variables,
        "clocks": clocks,
        "events": events,
        "has_event_mode": bool(clocks or events),
        "can_get_and_set_state": _save_state(model) is not False,
        "save_state": _save_state(model),
        "discard_on": discard_on,
        "integrate": integrate,
        **({"repeat": repeat} if repeat else {}),
        "annotations": annotations,
        "notes": notes,
    }


def summary(doc, limit=200):
    """The first line of a docstring, at most ``limit`` characters.

    :param doc: the docstring (may be empty).
    :type doc: str
    :param limit: maximum length; longer lines are cut and end with an ellipsis.
    :type limit: int
    :return: the summary line, or ``""``.
    :rtype: str
    """
    line = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    return line if len(line) <= limit else line[:limit - 1].rstrip() + "…"


def _integrate(model, step_call):
    """Validate ``[model] integrate``: the model returns derivatives, and the FMU integrates its states.

    :param model: the ``[model]`` table.
    :type model: dict
    :param step_call: whether the model is called on each step (``call`` isn't false).
    :type step_call: bool
    :return: ``{"method", "substeps"}``, or None without ``integrate``.
    :rtype: dict or None
    :raises InterfaceError: an unknown method, ``substeps`` with scipy, or ``call = false``.
    """
    method = model.get("integrate")
    substeps = model.get("substeps", 1)
    if method is None:
        return None   # substeps without integrate: see _repeat
    if not step_call:
        raise InterfaceError("[model] integrate and call = false exclude each other")
    if not isinstance(method, str) or not (method in INTEGRATORS or method.startswith("scipy:")):
        raise InterfaceError(f'[model] integrate must be one of {INTEGRATORS} or "scipy:<method>" '
                             f'(a scipy.integrate.solve_ivp method, e.g. "scipy:RK45"); got {method!r}')
    if method.startswith("scipy:"):
        if "substeps" in model:
            raise InterfaceError("[model] substeps only applies to euler and rk4: scipy chooses its own steps")
        name = method.partition(":")[2]
        if name not in SCIPY_METHODS:
            raise InterfaceError(f"[model] integrate: {name!r} is not a solve_ivp method ({', '.join(SCIPY_METHODS)})")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise InterfaceError("[model] substeps must be a positive integer")
    return {"method": method, "substeps": substeps}


def _repeat(model, step_call, integrate):
    """``[model] substeps`` without ``integrate``: how many times the step runs per communication step.

    Each run gets an equal part of the step (its time and step size), for models stepped in a
    loop of fixed small steps (an IMU filter at its sample rate, a physics engine's timestep).

    :param model: the ``[model]`` table.
    :type model: dict
    :param step_call: whether the model is called on each step (``call`` isn't false).
    :type step_call: bool
    :param integrate: the validated ``integrate``, or None.
    :type integrate: dict or None
    :return: the number of runs, or None (one run, the default).
    :rtype: int or None
    :raises InterfaceError: not a positive integer, or with ``call = false``.
    """
    if "substeps" not in model or integrate is not None:
        return None
    substeps = model["substeps"]
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise InterfaceError("[model] substeps must be a positive integer")
    if not step_call:
        raise InterfaceError("[model] substeps and call = false exclude each other: nothing runs on a step")
    return substeps


def _object_call(call, is_class, call_on_class):
    """Validate ``[model] call = "module:function(self, ...)"``: a function taking the model object.

    :param call: the call text.
    :type call: str
    :param is_class: whether the entry is a class (or a factory).
    :type is_class: bool
    :param call_on_class: ``[model] call_on_class``.
    :type call_on_class: bool
    :raises InterfaceError: not a class, with ``call_on_class``, invalid arguments, or a target that
        can't be imported or isn't callable.
    """
    from fmugen.templates.fmugen_runtime import parse_call_text, resolve_reference
    if not is_class:
        raise InterfaceError(f"[model] call {call!r}: a function call on the model object only applies to classes")
    if call_on_class:
        raise InterfaceError(f"[model] call {call!r} and call_on_class exclude each other")
    try:
        target, _, _ = parse_call_text(call)
    except (ValueError, SyntaxError) as e:
        raise InterfaceError(f"[model] call: {e}") from e
    try:
        fn = resolve_reference(target)
    except Exception as e:
        raise InterfaceError(f"[model] call: cannot import {target!r}: {e!r}") from e
    if not callable(fn):
        raise InterfaceError(f"[model] call: {target!r} is not callable")


def _discard_on(refs):
    """Validate ``[model] discard_on = ["module:Exception", ...]``: exceptions and warnings that reject a step.

    Each reference is imported now, at build time, with the runtime's own
    :func:`resolve_reference`, so a typo fails the build rather than the simulation.

    :param refs: the references.
    :type refs: list[str]
    :return: the same references.
    :rtype: list[str]
    :raises InterfaceError: not a list of strings, not importable, or not an exception class.
    """
    if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
        raise InterfaceError('[model] discard_on must be a list like ["scipy.integrate:IntegrationWarning"]')
    from fmugen.templates.fmugen_runtime import resolve_reference
    for ref in refs:
        try:
            cls = resolve_reference(ref)
        except Exception as e:
            raise InterfaceError(f"[model] discard_on: cannot import {ref!r}: {e!r}") from e
        if not (isinstance(cls, type) and issubclass(cls, Exception)):
            raise InterfaceError(f"[model] discard_on: {ref!r} is not an exception or warning class")
    return list(refs)


def _check_annotations(text):
    """Validate ``[model] annotations``: XML put inside the modelDescription's ``<Annotations>``.

    Parsed with :mod:`xml.etree.ElementTree` wrapped in an ``<Annotations>`` element, so the text
    may hold several ``<Tool>`` elements and is checked exactly as it will be embedded.

    :param text: the XML text.
    :raises InterfaceError: not a string, invalid XML, or children other than
        ``<Tool name="...">``.
    """
    import xml.etree.ElementTree as ET
    if not isinstance(text, str):
        raise InterfaceError('[model] annotations must be XML text, e.g. \'<Tool name="mytool"><x/></Tool>\'')
    try:
        root = ET.fromstring(f"<Annotations>{text}</Annotations>")
    except ET.ParseError as e:
        raise InterfaceError(f"[model] annotations is not valid XML: {e}") from e
    if not len(root) or any(child.tag != "Tool" or "name" not in child.attrib for child in root):
        raise InterfaceError('[model] annotations must be <Tool name="..."> elements')


def _save_state(model):
    """Validate ``[model] save_state``: how the FMU saves and restores its state.

    true: the whole model object (pickle, else cloudpickle); false: not supported; a list of
    attribute paths: only those attributes. fmugen init picks one by trying to pickle the object.

    :param model: the ``[model]`` table.
    :type model: dict
    :return: True, False or the list of attribute paths.
    :rtype: bool or list[str]
    :raises InterfaceError: any other value.
    """
    value = model.get("save_state", True)
    if isinstance(value, bool):
        return value
    if isinstance(value, list) and value and all(
            isinstance(p, str) and all(part.isidentifier() for part in p.split(".")) for p in value):
        return list(value)
    raise InterfaceError('[model] save_state must be true, false, or a list of attribute names like ["_state"]')


def parse_call(text):
    """Parse a ``[model] setup`` string into ``{call, args, kwargs}``.

    Accepts ``"module:function(arg, ..., key=arg, ...)"`` or ``"method(...)"``. The argument list
    is parsed with :func:`ast.parse` (as the call ``_(...)``) and never executed. Arguments are
    Python literals (simple ones kept as values, others as ``{ python = ... }``) or dotted
    references to importable objects (e.g. ``psychrolib.SI``, kept as ``{ ref = ... }``) or
    calls of those (e.g. ``numpy.array([1.0, 2.0])``, kept as ``{ call = ... }``, the same as a call
    constant); references are resolved and calls made when the FMU initializes.

    :param text: the setup call text.
    :type text: str
    :return: ``{"call": target, "args": [...], "kwargs": {...}}``.
    :rtype: dict
    :raises InterfaceError: a syntax error, a missing ``)``, ``**kwargs``, or an argument that is
        neither a literal, a name nor a call of a name.
    """
    import ast

    def argument(node):
        """Convert one argument node to its interface form.

        :param node: the argument's AST node.
        :type node: ast.expr
        :return: a plain value, ``{"python": source}`` or ``{"ref": dotted name}``.
        :raises InterfaceError: the node is neither a literal nor a (dotted) name.
        """
        try:
            value = ast.literal_eval(node)
            return value if isinstance(value, (bool, int, float, str)) else {"python": ast.unparse(node)}
        except ValueError:
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)):
                # a call of an importable object, e.g. numpy.array([...]): a call constant
                return {"call": ast.unparse(node)}
            if not isinstance(node, (ast.Attribute, ast.Name)):
                raise InterfaceError(f"[model] setup {text!r}: arguments must be literals, names or calls of names")
            return {"ref": ast.unparse(node)}

    call, paren, rest = text.partition("(")
    args, kwargs = [], {}
    if paren:
        if not rest.endswith(")"):
            raise InterfaceError(f"[model] setup {text!r}: missing ')'")
        try:
            parsed = ast.parse(f"_({rest}", mode="eval").body
        except SyntaxError as e:
            raise InterfaceError(f"[model] setup {text!r}: {e.msg}") from e
        if not (isinstance(parsed, ast.Call) and isinstance(parsed.func, ast.Name) and parsed.func.id == "_"):
            raise InterfaceError(f"[model] setup {text!r}: only one call is supported, nothing after its ')'")
        args = [argument(node) for node in parsed.args]
        for keyword in parsed.keywords:
            if keyword.arg is None:
                raise InterfaceError(f"[model] setup {text!r}: **kwargs is not supported")
            kwargs[keyword.arg] = argument(keyword.value)
    return {"call": call.strip(), "args": args, "kwargs": kwargs}


def _setup(steps, is_class):
    """Validate ``[model] setup``: calls run when the FMU initializes, before the model is used.

    :param steps: one entry or a list of them: ``"module:function"``, ``"method"`` (classes),
        a call string with arguments, or ``{ call = ..., args = [...], kwargs = {...} }``.
    :type steps: str or dict or list
    :param is_class: whether the model is a class (only then are bare method names allowed).
    :type is_class: bool
    :return: ``[{"call", "args", "kwargs"}]``.
    :rtype: list[dict]
    :raises InterfaceError: an entry of another form.
    """
    if isinstance(steps, (str, dict)):
        steps = [steps]
    normalized = []
    for step in steps:
        if isinstance(step, str):
            step = parse_call(step)
        if not isinstance(step, dict) or "call" not in step:
            raise InterfaceError("[model] setup entries are \"module:function\", \"method\", "
                                 "or { call = ..., args = [...], kwargs = {...} }")
        _check_keys(step, {"call", "args", "kwargs"}, "[model] setup")
        if ":" not in step["call"] and not is_class:
            raise InterfaceError(f"[model] setup {step['call']!r}: use \"module:function\" (the model is a function)")
        normalized.append({"call": step["call"], "args": list(step.get("args", [])),
                           "kwargs": dict(step.get("kwargs", {}))})
    return normalized


def _clock(name, info, entry_obj, is_class, time_args):
    """Validate one ``[clocks.<name>]`` table (FMI 3) and build its spec entry.

    Input clocks are ticked by the importer and may run a ``call`` (a method, a function of the
    model's module, or ``"module:function"``); periodic ones (constant, fixed, tunable) need an
    ``interval``, changing/countdown ones an ``interval_from``. Output clocks are ticked by the
    model and need ``from``.

    :param name: the clock name.
    :type name: str
    :param info: the clock's table.
    :type info: dict
    :param entry_obj: the imported entry (to look up ``call``).
    :param is_class: whether the entry is a class.
    :type is_class: bool
    :param time_args: the ``[time]`` arguments; those the clock's call accepts are passed to it.
    :type time_args: dict
    :return: the clock entry, plus a temporary ``_signature`` key removed by :func:`normalize`.
    :rtype: dict
    :raises InterfaceError: an invalid combination of keys or a ``call`` that doesn't exist.
    """
    where = f"[clocks.{name}]"
    if not name.isidentifier() or keyword.iskeyword(name):
        raise InterfaceError(f"{where}: clock names must be valid Python identifiers")
    causality = info.get("causality", "input")
    if causality not in ("input", "output"):
        raise InterfaceError(f"{where}: causality must be \"input\" or \"output\"")
    variability = info.get("interval_variability", "constant" if "interval" in info else "triggered")
    if variability not in INTERVAL_VARIABILITIES:
        raise InterfaceError(f"{where}: interval_variability must be one of {INTERVAL_VARIABILITIES}")
    clock = {"name": name, "causality": causality, "interval_variability": variability, "_signature": None}
    if info.get("description"):
        clock["description"] = str(info["description"])
    kinds = ("return", "attr") if is_class else ("return",)

    if causality == "output":
        if variability != "triggered":
            raise InterfaceError(f"{where}: output clocks are ticked by the model, so they must be \"triggered\"")
        if "from" not in info:
            raise InterfaceError(f"{where}: output clocks need from = \"...\" (a value that is true when it ticks)")
        for key in ("call", "interval", "shift", "interval_from"):
            if key in info:
                raise InterfaceError(f"{where}: {key} only applies to input clocks")
        clock["from"] = parse_binding(info["from"], kinds, where)
        return clock

    if "from" in info:
        raise InterfaceError(f"{where}: from only applies to output clocks")
    if variability in ("constant", "fixed", "tunable"):
        if "interval" not in info:
            raise InterfaceError(f"{where}: a {variability} clock needs interval = <seconds>")
        clock["interval"] = float(info["interval"])
        clock["shift"] = float(info.get("shift", 0.0))
    elif "interval" in info or "shift" in info:
        raise InterfaceError(f"{where}: interval and shift only apply to constant, fixed and tunable clocks")
    if variability in ("changing", "countdown"):
        if "interval_from" not in info:
            raise InterfaceError(f"{where}: a {variability} clock needs interval_from = \"...\" (the next interval)")
        clock["interval_from"] = parse_binding(info["interval_from"], kinds, where)
    elif "interval_from" in info:
        raise InterfaceError(f"{where}: interval_from only applies to changing and countdown clocks")

    if info.get("call"):
        call = info["call"]
        if ":" in call:
            from fmugen.templates.fmugen_runtime import resolve_reference
            try:
                target = resolve_reference(call)
            except Exception as e:
                raise InterfaceError(f"{where}: cannot import {call!r}: {e!r}") from e
        elif is_class:
            target = getattr(entry_obj, call, None)
        else:
            target = getattr(inspect.getmodule(entry_obj), call, None)
        if not callable(target):
            raise InterfaceError(f"{where}: call {call!r} is not a method or function of the model")
        clock["call"] = call
        clock["_signature"] = signature = _signature(target)
        clock["time_args"] = [arg for arg in time_args if signature is None or _accepts(signature, arg)]
    return clock


def _globals(refs):
    """Validate ``[model] globals = ["package.module:NAME", ...]``: module-level state of the model.

    Each one is looked up now, so a typo fails the build.

    :param refs: the references.
    :type refs: list[str]
    :return: the same references.
    :rtype: list[str]
    :raises InterfaceError: a malformed or missing reference.
    """
    if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
        raise InterfaceError('[model] globals must be a list like ["package.module:NAME"]')
    from fmugen.templates.fmugen_runtime import split_global
    for ref in refs:
        module, sep, name = ref.partition(":")
        if not sep or not module or not name.isidentifier():
            raise InterfaceError(f"[model] globals: {ref!r} must look like 'package.module:NAME'")
        try:
            split_global(ref)
        except Exception as e:
            raise InterfaceError(f"[model] globals: cannot find {ref!r}: {e!r}") from e
    return list(refs)


def element_count(var):
    """Number of elements of a fixed-size array variable (1 for a scalar).

    Used for FMI 2 value references, where every array element takes one. Only valid for
    integer dimensions (FMI 2 has no structural parameters).

    :param var: a spec variable.
    :type var: dict
    :rtype: int
    """
    n = 1
    for d in var.get("dimensions", ()):
        n *= d
    return n


def _variable(section, causality, name, info, is_class, type_definitions, version=2, structural=(), clocks=(),
              integrate=False):
    """Validate one variable of a config section and build its spec entry.

    Decides the FMI type, dimensions, clocks, enumeration type, variability, start and
    ``initial`` (with the FMI defaults for its causality), attributes (unit, min/max, extra) and
    its bindings to the user's code: ``to`` for values passed in (default ``init:`` for a class's
    parameters, else ``arg:``), ``from`` for values read out (default ``attr:`` for classes,
    ``return:`` for functions), ``next`` / ``der`` for states, ``attr`` for tunable parameters.

    :param section: the config section (``inputs``, ``parameters``, ...).
    :type section: str
    :param causality: the FMI causality of that section.
    :type causality: str
    :param name: the variable name (outputs and locals may be dotted).
    :type name: str
    :param info: the variable's table.
    :type info: dict
    :param is_class: whether the model is a class.
    :type is_class: bool
    :param type_definitions: enumeration types found so far; updated in place.
    :type type_definitions: dict
    :param version: the FMI version.
    :type version: int
    :param structural: names of structural parameters (allowed as dimensions).
    :type structural: set[str] or tuple
    :param clocks: names of the config's clocks.
    :type clocks: set[str] or tuple
    :param integrate: whether ``[model] integrate`` is used (states then have ``der``).
    :type integrate: bool
    :return: the spec variable (without ``valueReference``).
    :rtype: dict
    :raises InterfaceError: anything invalid about the variable.
    """
    where = f"[{section}] {name}"
    parts = name.split(".") if section in ("outputs", "locals") else [name]   # outputs may be "zone.T"
    if not all(p.isidentifier() and not keyword.iskeyword(p) for p in parts):
        raise InterfaceError(f"{where}: variable names must be valid Python identifiers"
                             + (" or dotted ones (zone.T)" if section in ("outputs", "locals") else ""))

    config_type = _type(where, info, section)
    fmi_type = TYPE_NAMES[version].get(config_type)
    if fmi_type is None:
        raise InterfaceError(f"{where}: type {config_type} {FMI3_ONLY}")
    var = {"name": name, "causality": causality, "type": fmi_type}
    is_float = fmi_type in FLOAT_TYPES

    if "dimensions" in info:   # FMI 2 has no arrays: one scalar variable per element, x[1], x[2], ...
        dims = info["dimensions"]
        if section == "structural_parameters":
            raise InterfaceError(f"{where}: structural parameters are scalars")
        if not isinstance(dims, list) or not dims:
            raise InterfaceError(f"{where}: dimensions must be a list like [3] or [\"n\", 2]")
        for d in dims:
            if isinstance(d, str):
                if d not in structural:
                    raise InterfaceError(f"{where}: dimension {d!r} is not a structural parameter")
            elif not isinstance(d, int) or isinstance(d, bool) or d < 1:
                raise InterfaceError(f"{where}: dimension {d!r} must be a positive integer or a structural parameter")
        var["dimensions"] = list(dims)
        if info.get("numpy"):
            var["numpy"] = True
    elif "numpy" in info:
        raise InterfaceError(f"{where}: numpy only applies to arrays (set dimensions)")
    if "convert" in info:
        # a function applied to the value before the model gets it, e.g. "torch:tensor"
        if section not in ("structural_parameters", "parameters", "inputs", "states"):
            raise InterfaceError(f"{where}: convert only applies to values passed to the model")
        from fmugen.templates.fmugen_runtime import (
            resolve_reference,  # same resolution as at runtime
        )
        if info["convert"] == "pint":   # a pint quantity in the variable's unit
            if not info.get("unit"):
                raise InterfaceError(f"{where}: convert = \"pint\" needs a unit")
            reference = "pint:get_application_registry"
        elif info["convert"] == "series":   # a one-value pandas Series indexed by the date epoch + t
            if "epoch" not in info:
                raise InterfaceError(f"{where}: convert = \"series\" needs an epoch (the date at time 0)")
            if "dimensions" in info:
                raise InterfaceError(f"{where}: convert = \"series\" only applies to single values, not arrays")
            try:
                datetime.datetime.fromisoformat(str(info["epoch"]))
            except ValueError as e:
                raise InterfaceError(f"{where}: epoch {info['epoch']!r} is not an ISO 8601 date-time") from e
            var["epoch"] = str(info["epoch"])
            reference = "pandas:Series"
        else:
            reference = str(info["convert"])
        try:
            converter = resolve_reference(reference)
        except Exception as e:
            raise InterfaceError(f"{where}: cannot import convert {info['convert']!r}: {e!r}") from e
        if not callable(converter):
            raise InterfaceError(f"{where}: convert {info['convert']!r} is not callable")
        var["convert"] = str(info["convert"])
    elif "epoch" in info:
        raise InterfaceError(f"{where}: epoch only applies with convert = \"series\"")

    if "clocks" in info:
        if version == 2:
            raise InterfaceError(f"{where}: clocked variables {FMI3_ONLY}")
        names = [info["clocks"]] if isinstance(info["clocks"], str) else list(info["clocks"])
        if len(names) != 1 or names[0] not in clocks:
            raise InterfaceError(f"{where}: clocks must name exactly one clock from [clocks]")
        var["clocks"] = names

    if fmi_type == "Enumeration":
        type_name, items = _enumeration(where, name, info)
        type_definitions[type_name] = {"items": items, "enum": info.get("enum")}
        var["declared_type"] = type_name
        if isinstance(info.get("start"), str):
            if info["start"] not in items:
                raise InterfaceError(f"{where}: start {info['start']!r} is not one of {items}")
            info = {**info, "start": items.index(info["start"]) + 1}

    # variability
    default_variability = {
        "parameter": "fixed", "calculatedParameter": "fixed", "structuralParameter": "fixed",
    }.get(causality, "continuous" if is_float and "clocks" not in var else "discrete")
    variability = info.get("variability", default_variability)
    if variability not in VARIABILITIES[causality]:
        raise InterfaceError(f"{where}: variability must be one of {VARIABILITIES[causality]}")
    if variability == "continuous" and not is_float:
        raise InterfaceError(f"{where}: only floating-point variables can be continuous")
    var["variability"] = variability

    # start / initial
    has_start = "start" in info
    if section in ("structural_parameters", "parameters", "inputs", "states"):
        initial = "exact" if section != "inputs" else None
        var["start"] = info.get("start", DEFAULT_START[fmi_type])
    elif section == "outputs":
        initial = "exact" if has_start else "calculated"
    else:
        initial = "approx" if has_start else "calculated"
    initial = info.get("initial", initial)
    if section == "states" and initial != "exact":
        raise InterfaceError(f"{where}: states always have initial = \"exact\"")
    if initial is not None and initial not in INITIALS[causality]:
        raise InterfaceError(f"{where}: initial must be one of {INITIALS[causality]}")
    if initial == "calculated" and has_start:
        raise InterfaceError(f"{where}: a start value requires initial = \"exact\" or \"approx\"")
    if initial in ("exact", "approx"):
        var["start"] = info.get("start", DEFAULT_START[fmi_type])
    if initial is not None:
        var["initial"] = initial
    if "start" in var:
        start = var["start"]
        if isinstance(start, list):
            if "dimensions" not in var:
                raise InterfaceError(f"{where}: a list start needs dimensions")
            var["start"] = [_check_value(where, fmi_type, x) for x in start]
        else:
            var["start"] = _check_value(where, fmi_type, start)

    for key in ("unit", "description", "quantity"):
        if info.get(key):
            var[key] = str(info[key])
    for key in ("min", "max", "nominal"):
        if key in info:
            if fmi_type not in (*FLOAT_TYPES, *INT_TYPES, "Enumeration") or (key == "nominal" and not is_float):
                raise InterfaceError(f"{where}: {key} is not allowed for {fmi_type} variables")
            var[key] = info[key]
    _check_range(where, var)
    if "extra" in info:
        var["extra"] = _extra(where, info["extra"])

    # bindings to the user's code
    if section in ("structural_parameters", "parameters", "inputs", "states"):
        default = "init" if (section in ("parameters", "structural_parameters") and is_class) else "arg"
        kinds = ("init", "arg", "pos", "attr", "call") if is_class else ("arg", "pos")
        var["to"] = parse_binding(info.get("to", f"{default}:{name}"), kinds, where)
        if var["to"]["kind"] == "pos" and not var["to"]["name"].isdigit():
            raise InterfaceError(f"{where}: pos needs an argument index, e.g. to = \"pos:0\"")
        if var["to"]["kind"] == "pos" and default == "init":   # a class's parameter: a constructor position
            var["to"]["kind"] = "initpos"
        if "clocks" in var and var["to"]["kind"] == "init":
            raise InterfaceError(f"{where}: a clocked input is passed to its clock's call, not the constructor")
        kinds = ("return", "attr", "get") if is_class else ("return",)
        if section == "states" and integrate:   # the model returns the derivative; the FMU integrates it
            if "next" in info:
                raise InterfaceError(f"{where}: with [model] integrate, states have der = \"...\" (the "
                                     "derivative), not next")
            var["der"] = parse_binding(info.get("der", "return"), kinds, where)
        elif section == "states":
            if "der" in info:
                raise InterfaceError(f"{where}: der only applies with [model] integrate")
            if "next" not in info:
                raise InterfaceError(f"{where}: states need next = \"...\" (where the next value comes from)")
            var["next"] = parse_binding(info["next"], kinds, where)
    else:
        default = f"attr:{name}" if is_class else f"return:{name}"
        var["from"] = parse_binding(info.get("from", default), ("return", "attr", "get") if is_class else ("return",),
                                    where)
        if section == "calculated_parameters" and var["from"]["kind"] != "attr":
            raise InterfaceError(f"{where}: calculated parameters are read from an attribute (from = \"attr:...\")")

    if section in ("parameters", "structural_parameters"):
        attr = info.get("attr")
        if attr is not None and not is_class:
            raise InterfaceError(f"{where}: attr only applies to classes")
        if variability == "tunable" and is_class:
            if var["to"]["kind"] == "attr" or var["to"]["kind"] == "init" and attr is None:
                attr = var["to"]["name"]
            if var["to"]["kind"] == "initpos" and attr is None:
                raise InterfaceError(f"{where}: a tunable parameter passed by position needs attr = \"...\", the "
                                     "attribute of the model object that holds it (else a new value can't reach it)")
            if attr:
                var["attr"] = attr

    if section == "outputs" and "depends_on" in info:
        var["depends_on"] = list(info["depends_on"])
    return var


def _check_range(where, var):
    """Check min <= max, and the start value (each element of an array) between them.

    :param where: the variable, for error messages.
    :type where: str
    :param var: the spec variable.
    :type var: dict
    :raises InterfaceError: a non-numeric bound, min > max, or a start out of range.
    """
    for key in ("min", "max", "nominal"):
        if key in var and (isinstance(var[key], bool) or not isinstance(var[key], (int, float))):
            raise InterfaceError(f"{where}: {key} must be a number, got {var[key]!r}")
    low, high = var.get("min"), var.get("max")
    if low is not None and high is not None and low > high:
        raise InterfaceError(f"{where}: min {low} is larger than max {high}")
    start = var.get("start")
    for value in start if isinstance(start, list) else [start]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if low is not None and value < low or high is not None and value > high:
            raise InterfaceError(f"{where}: start {value} is outside min..max ({low}..{high})")


#: XML attributes fmugen writes from the variable's own keys, which ``extra`` can't set.
RESERVED_ATTRIBUTES = {"name", "valueReference", "causality", "variability", "initial", "start", "description",
                       "declaredType", "unit", "min", "max", "nominal", "quantity"}


def _extra(where, extra):
    """Validate ``extra = { displayUnit = "degC", relativeQuantity = true }``: more XML attributes.

    :param where: the variable, for error messages.
    :type where: str
    :param extra: the table.
    :return: attribute name -> XML text (booleans as ``true`` / ``false``).
    :rtype: dict[str, str]
    :raises InterfaceError: not a table, an invalid or reserved name, or a non-scalar value.
    """
    if not isinstance(extra, dict):
        raise InterfaceError(f'{where}: extra must be a table, e.g. extra = {{ displayUnit = "degC" }}')
    result = {}
    for key, value in extra.items():
        if not key.isidentifier():
            raise InterfaceError(f"{where}: extra: {key!r} is not an XML attribute name")
        if key in RESERVED_ATTRIBUTES:
            raise InterfaceError(f"{where}: extra: set {key} as a key of the variable, not in extra")
        if isinstance(value, (dict, list)):
            raise InterfaceError(f"{where}: extra.{key} must be a single value")
        result[key] = "true" if value is True else "false" if value is False else str(value)
    return result


def _type(where, info, section=None):
    """The config type of a variable: given, or guessed from its other keys.

    ``items`` / ``enum`` mean Enumeration; structural parameters default to UInt64; otherwise the
    type of ``start`` (its first element for arrays), else Real.

    :param where: the variable, for error messages.
    :type where: str
    :param info: the variable's table.
    :type info: dict
    :param section: the config section.
    :type section: str or None
    :return: a name from :data:`FMI_TYPES`.
    :rtype: str
    :raises InterfaceError: an unknown ``type``.
    """
    if "type" in info:
        if info["type"] not in FMI_TYPES:
            raise InterfaceError(f"{where}: unknown type {info['type']!r}, expected one of {FMI_TYPES}")
        return info["type"]
    if "items" in info or "enum" in info:
        return "Enumeration"
    if section == "structural_parameters":
        return "UInt64"
    start = info.get("start")
    if isinstance(start, list) and start:
        start = start[0]
    # bool before int: bool is a subclass of int
    if isinstance(start, bool):
        return "Boolean"
    if isinstance(start, int):
        return "Integer"
    if isinstance(start, str):
        return "String"
    return "Real"


def _enumeration(where, name, info):
    """The enumeration type of a variable: from a Python Enum class (``enum``) or ``items``.

    :param where: the variable, for error messages.
    :type where: str
    :param name: the variable name (gives ``<name>_type`` for ``items``).
    :type name: str
    :param info: the variable's table.
    :type info: dict
    :return: ``(type name, item names)``; FMI value 1 is the first item.
    :rtype: tuple[str, list[str]]
    :raises InterfaceError: the enum can't be imported, or neither key is given.
    """
    if "enum" in info:
        from fmugen.templates.fmugen_runtime import (
            resolve_reference,  # same resolution as at runtime
        )
        try:
            enum = resolve_reference(info["enum"])
        except Exception as e:
            raise InterfaceError(f"{where}: cannot import enum {info['enum']!r}: {e!r}") from e
        return enum.__name__, [member.name for member in enum]
    if not info.get("items"):
        raise InterfaceError(f"{where}: Enumeration variables need items = [...] or enum = \"module:Class\"")
    return f"{name}_type", [str(item) for item in info["items"]]


def _check_value(where, fmi_type, value):
    """Check a start value against its FMI type, using the runtime's :func:`coerce`.

    :param where: the variable, for error messages.
    :type where: str
    :param fmi_type: the spec type.
    :type fmi_type: str
    :param value: the start value from the config.
    :return: the value (as float for float types).
    :raises InterfaceError: the wrong Python type, or out of the type's range.
    """
    from fmugen.templates.fmugen_runtime import coerce
    if fmi_type in FLOAT_TYPES:
        expected = (int, float)
    else:
        expected = {"Boolean": (bool,), "String": (str,), "Binary": (str,)}.get(fmi_type, (int,))
    if isinstance(value, bool) and fmi_type != "Boolean" or not isinstance(value, expected):
        hint = f" (use {str(bool(value)).lower()})" if fmi_type == "Boolean" and value in (0, 1) else ""
        raise InterfaceError(f"{where}: start {value!r} is not a valid {fmi_type}{hint}")
    try:
        coerce(fmi_type, value)  # integer ranges, hex for Binary
    except ValueError as e:
        raise InterfaceError(f"{where}: start {value!r}: {e}") from e
    return float(value) if fmi_type in FLOAT_TYPES else value


def _signature(fn):
    """The signature of a callable, with its qualified name for messages.

    :param fn: a function, method or class.
    :return: ``(inspect.Signature, qualified name)``, or None when it has none (some builtins
        and C extensions); argument checks are then skipped.
    :rtype: tuple or None
    """
    try:
        return inspect.signature(fn), getattr(fn, "__qualname__", repr(fn))
    except (TypeError, ValueError):
        return None


def _accepts(sig, arg):
    """Whether a function accepts ``arg`` by keyword.

    :param sig: ``(inspect.Signature, name)`` from :func:`_signature`.
    :type sig: tuple
    :param arg: the argument name.
    :type arg: str
    :return: True if it has ``**kwargs`` or a keyword-capable parameter of that name.
    :rtype: bool
    """
    signature, _ = sig
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    p = params.get(arg)
    return p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _check_keys(table, allowed, where):
    """Raise if a config table has keys outside ``allowed``.

    :param table: the table.
    :param allowed: the allowed keys.
    :type allowed: set[str]
    :param where: the table's name, for the error message.
    :type where: str
    :raises InterfaceError: not a table, or unknown keys.
    """
    if not isinstance(table, dict):
        raise InterfaceError(f"{where} must be a table")
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise InterfaceError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")
