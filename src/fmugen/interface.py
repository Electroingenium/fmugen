"""Infer a fmugen.toml from an ordinary Python model.

`infer_config()` imports the model and inspects the entry function or class: its
signatures give the parameters and inputs, and its source code (static.py) what it
returns and which attributes it sets. The model is not called. With probe=True
(`fmugen init --probe`) it is also called once with the start values, which finds
what reading the code can't: array sizes, exact types, results built at runtime, and
whether the object can be pickled. The result is a config dict (the same shape as
fmugen.toml) plus comments explaining each guess; `render_toml()` writes it out for the
user to review. `fmugen build model.py` runs the same inference in memory.
"""
import ast
import contextlib
import enum
import importlib
import importlib.util
import inspect
import os
import sys
import threading
import traceback
import types
from pathlib import Path

from fmugen import static
from fmugen.config import InterfaceError, stdlib_clash
from fmugen.toml_output import render_toml
from fmugen.arguments import (
    SCALARS, POSITIONAL_STARTS, START_ERRORS, NO_SIGNATURE, PROBED,
    _number_like, _array_info, _dotted_names, _arguments, _apply_converts,
    _assign_starts, assumptions, guessed_outputs, unfit_outputs, class_called,
)
from fmugen.probe import (
    ConstantError, _run_setup, _probe_call, _probe_class,
    _outputs_from_return, _add_returned, _add_attributes, _detect_states,
    _taken, _split_varargs, _varargs_by_keyword,
)

#: Step method names tried in order when a class is given without ``--call``.
CALL_NAMES = ("step", "do_step", "update", "__call__")

def load_module(model_path):
    """Import a model file by path, as the module named after the file.

    Uses :func:`importlib.util.spec_from_file_location` instead of a plain import so the exact
    file is loaded even when another module of that name is on the path. The module is put in
    :data:`sys.modules` before it runs (as a normal import does), so the model's own imports of
    itself and pickling of its classes work; its folder goes on ``sys.path`` for sibling imports.

    :param model_path: the ``.py`` file.
    :type model_path: str or pathlib.Path
    :return: the executed module.
    :rtype: types.ModuleType
    :raises InterfaceError: the file is named like a standard library module (registering it
        would replace the real module in this process too).
    :raises Exception: whatever the module's top-level code raises.
    """
    model_path = Path(model_path).resolve()
    if model_path.stem in sys.stdlib_module_names:   # registering it would replace the real module here too
        raise InterfaceError(stdlib_clash([model_path.stem]))
    if str(model_path.parent) not in sys.path:
        sys.path.insert(0, str(model_path.parent))
    spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
    module = importlib.util.module_from_spec(spec)
    # registered before it runs, as `import` does: the module's own code (dataclasses, pickling,
    # `from model import X` in sibling files) looks itself up in sys.modules
    sys.modules[model_path.stem] = module
    spec.loader.exec_module(module)
    return module


def import_failure(target, error):
    """An InterfaceError for a model whose top-level code raised while it was imported.

    A script that solves or plots when imported, a missing data file: the message names the
    innermost line of the model's own code (from the traceback, skipping importlib and fmugen
    frames) and where the error was raised if that's elsewhere.

    :param target: the model target, for the message.
    :type target: str
    :param error: the exception.
    :type error: BaseException
    :rtype: InterfaceError
    """
    frames = traceback.extract_tb(error.__traceback__)
    # the innermost line of the model's own code, and where the error was raised if that's elsewhere
    ours = [f for f in frames if "importlib" not in f.filename and "fmugen" not in Path(f.filename).parts]
    where = [ours[0]] if ours else []
    if frames and (not where or frames[-1] is not where[0]):
        where.append(frames[-1])
    at = ", in ".join(f"{Path(f.filename).name}:{f.lineno}" for f in where)
    return InterfaceError(f"importing {target} failed: {type(error).__name__}: {error}"
                          + (f" (at {at})" if at else "")
                          + "; importing a module runs its top-level code")


def parse_target(target):
    """Split a model target into file, module and name.

    :param target: ``'path/model.py[:Name]'`` or ``'package.module:Name'``.
    :type target: str or pathlib.Path
    :return: ``(path or None, module name or None, Name or None)``.
    :rtype: tuple
    :raises InterfaceError: a module target without ``:Name``.
    """
    target = str(target)
    path, sep, name = target.rpartition(":")
    if not sep or not name.isidentifier() or (not path.endswith(".py") and "/" in name):
        path, name = target, None
    if path.endswith(".py"):
        return Path(path), None, name
    if name is None:
        raise InterfaceError(f"{target!r}: give a .py file, or 'package.module:Name' for an installed module")
    return None, path, name


def infer_config(target, call=None, config_dir=None, fmi_version=None, starts=None, setup=None, kind=None,
                 create=None, probe=False, converts=None):
    """Infer a config for a model target (``fmugen init``, and ``fmugen build model.py``).

    With fmi_version=3, list/tuple/numpy values become array variables and bytes become Binary.
    The model is imported (and probed) in ``config_dir``, as it will run inside the FMU with
    ``[model] cwd``, so files it opens by a relative path are found.

    :param target: ``'path/model.py[:Name]'`` or ``'package.module:Name'``.
    :type target: str
    :param call: the step method of a class (``--call``); for a function, makes it a factory.
    :type call: str or None
    :param config_dir: where the config will be written (default: the model's folder).
    :type config_dir: str or pathlib.Path or None
    :param fmi_version: the target FMI version.
    :type fmi_version: int or None
    :param starts: ``{argument: value}``: probe/start values for arguments, e.g. ones without a default.
    :type starts: dict or None
    :param setup: ``"module:function(args)"`` / ``"method"`` calls run before the probe.
    :type setup: list[str] or None
    :param kind: ``"function"`` to call a class like a function.
    :type kind: str or None
    :param create: a classmethod that builds the object (e.g. ``"from_pretrained"``).
    :type create: str or None
    :param probe: also call the model once (setup, construction, one step); otherwise it is
        never called. A number is the probe's time limit in seconds (fmugen exits when it
        passes); True: no limit.
    :type probe: bool or float
    :param converts: ``{argument: "module:function" or "numpy"}``: how arguments are passed in.
    :type converts: dict or None
    :return: ``(config dict, {(section, name): comment})``.
    :rtype: tuple[dict, dict]
    :raises InterfaceError: the model can't be imported, the target is ambiguous, or the options
        don't fit it.
    """
    path, _module_name, name = parse_target(target)
    config_dir = Path(config_dir or (path.parent if path else ".")).resolve()
    if path and not path.resolve().is_relative_to(config_dir):
        raise InterfaceError(
            f"the config must be in the folder of {path.name} or a folder above it, since it names the model "
            f"by a path relative to itself: write it to {path.resolve().parent / 'fmugen.toml'} "
            f"(not {config_dir})")
    if path:
        target = f"{path.resolve()}" + (f":{name}" if name else "")
    with contextlib.chdir(config_dir):
        if probe is True or not probe:
            return _infer_config(target, call, config_dir, fmi_version, starts, setup, kind, create, probe, converts)
        return _with_timeout(float(probe), _infer_config, target, call, config_dir, fmi_version, starts, setup, kind,
                             create, True, converts)


def _with_timeout(timeout, fn, *args):
    """Call ``fn(*args)``, stopping the process if it takes longer than ``timeout`` seconds.

    The model runs in this thread, as without a limit (some libraries only work there), and its
    code can't be interrupted: a watchdog :class:`threading.Timer` prints why and calls
    :func:`os._exit` instead (``sys.exit`` would only end the watchdog thread).

    :param timeout: the limit in seconds.
    :type timeout: float
    :param fn: the function.
    :type fn: callable
    :param args: its arguments.
    :return: what ``fn`` returned.
    """
    def abort():
        """Print why the probe stopped and exit with code 2."""
        sys.stderr.write(f"fmugen: error: the probe did not finish in {timeout:g} s: the model may wait for "
                         "input or a device, or be slow on the start values. Give a larger --probe-timeout, "
                         "other --start values, or run init without --probe\n")
        sys.stderr.flush()
        os._exit(2)

    watchdog = threading.Timer(timeout, abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        return fn(*args)
    finally:
        watchdog.cancel()


def _infer_config(target, call, config_dir, fmi_version, starts, setup, kind, create, probe, converts):
    """The inference itself, inside ``config_dir`` (see :func:`infer_config`).

    Imports the model, picks the entry, infers a function or class model, adds data files,
    globals and requirements, renames clashing names, adapts the types to the FMI version, and
    finally checks that every ``--start`` and ``--convert`` name was used.

    :param target: the model target (absolute file path when it's a file).
    :type target: str
    :param call: see :func:`infer_config`.
    :type call: str or None
    :param config_dir: the config folder.
    :type config_dir: pathlib.Path
    :param fmi_version: see :func:`infer_config`.
    :type fmi_version: int or None
    :param starts: see :func:`infer_config`.
    :type starts: dict or None
    :param setup: see :func:`infer_config`.
    :type setup: list[str] or None
    :param kind: see :func:`infer_config`.
    :type kind: str or None
    :param create: see :func:`infer_config`.
    :type create: str or None
    :param probe: whether to probe.
    :type probe: bool
    :param converts: see :func:`infer_config`.
    :type converts: dict or None
    :return: ``(config dict, comments)``.
    :rtype: tuple[dict, dict]
    :raises InterfaceError: see :func:`infer_config`.
    """
    arrays = True   # FMI 2 writes arrays as one scalar per element; see _fmi2_types
    starts = dict(starts or {})
    if fmi_version != 3:
        for name, value in starts.items():
            if isinstance(value, bytes):
                raise InterfaceError(f"--start {name}: bytes need --fmi 3 (FMI 2 has no Binary variables)")
    path, module_name, name = parse_target(target)
    try:
        module = load_module(path) if path else importlib.import_module(module_name)
    except ImportError as e:
        raise InterfaceError(f"cannot import {target}: {e} (is it installed in this environment?)") from e
    except (Exception, SystemExit) as e:
        raise import_failure(target, e) from e
    entry = _pick_entry(module, name)

    # the name it was given: an alias (PyCO2SYS:sys) or a lambda (rdkit's Descriptors.MolWt) has
    # another __name__, which may name a different object or none
    entry_name = name or entry.__name__
    if path:
        rel = Path(path).resolve().relative_to(config_dir).as_posix()
        entry_ref = f"{rel}:{entry_name}"
    else:
        entry_ref = f"{module_name}:{entry_name}"

    data = {"model": {"entry": entry_ref}}
    if fmi_version:
        data["model"]["fmi_version"] = fmi_version
    if path:
        sources = _local_sources(Path(path).resolve(), config_dir)
        if sources:
            data["model"]["sources"] = sources
    if setup:
        data["model"]["setup"] = list(setup)
    if kind == "function":
        data["model"]["kind"] = "function"
    comments = {}
    options = {"arrays": arrays, "starts": starts, "probe": probe, "converts": dict(converts or {})}
    if inspect.isclass(entry) and kind != "function":
        _infer_class(entry, call, data, comments, create=create, **options)
    elif call and kind != "function":
        # a function with --call is a factory: called once, --call runs on what it returns
        if create:
            raise InterfaceError("--create only applies to classes; a function with --call is already a factory")
        _infer_class(None, call, data, comments, factory=entry, **options)
    else:
        if call or create:
            raise InterfaceError("--call and --create only apply to classes")
        _infer_function(entry, data, comments, **options)
    _add_data_files(data, comments, config_dir, Path(path).resolve() if path else None)
    _add_globals(data, comments, module, config_dir if path else None)
    _add_requirements(data, comments, module_name, config_dir, Path(path).resolve() if path else None)
    _rename_duplicates(data, comments)
    # FMI 2 too: importers name the simulation time `time` in their results (FMPy), hiding the variable
    _rename_reserved(data, comments, is_class="call" in data["model"] or inspect.isclass(entry))
    if fmi_version == 3:
        _wide_integers(data, comments)
    else:
        _fmi2_types(data, comments)
    variables = {n for sec in ("parameters", "inputs", "states") for n in data.get(sec, {})}
    unknown = set(converts or {}) - variables
    if unknown:
        raise InterfaceError(f"--convert names that are not arguments of the model: {sorted(unknown)}")
    # a call: start for the method's copy of a shared name is stored under the argument's own name
    prefix = f"{data['model'].get('call', 'call').strip('_')}_"
    unused = set(starts) - variables - set(comments.get(POSITIONAL_STARTS, {})) \
        - {n for table in ("constants", "call_constants") for n in data["model"].get(table, {})} \
        - {prefix + n for n in data["model"].get("call_constants", {})} \
        - {str(info.get("to", "")).partition(":")[2].partition("[")[0]
           for sec in ("parameters", "inputs", "states") for info in data.get(sec, {}).values()} \
        - {str(info["to"]).partition(":")[2] for sec in ("parameters", "inputs", "states")
           for info in data.get(sec, {}).values() if "." in str(info.get("to", ""))}
    errors = comments.get(START_ERRORS, {})
    explained = sorted(n for n in unused if n in errors)
    if explained:
        by_message = {}
        for name in explained:
            by_message.setdefault(errors[name], []).append(name)
        raise InterfaceError("; ".join(f"--start {', '.join(names)}: {message}" for message, names in by_message.items()))
    if unused and comments.get(NO_SIGNATURE):
        unreadable = list(dict.fromkeys(comments[NO_SIGNATURE]))
        raise InterfaceError(
            f"--start {', '.join(sorted(unused))}: cannot tell which arguments these are, since "
            f"{' and '.join(unreadable)} {'has' if len(unreadable) == 1 else 'have'} no signature (e.g. a C built-in); "
            "write the config by hand, with to = \"pos:N\" for arguments passed by position (docs/config.md)")
    if unused:
        raise InterfaceError(f"--start names that are not arguments of the model: {sorted(unused)}")
    return data, comments


#: FMI 3 types found by inference -> FMI 2 type (None: the default Real).
FMI2_TYPES = {"Float32": None, "Float64": None, **{t: "Integer" for t in (
    "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64")}}


def _fmi2_types(data, comments):
    """Adapt an inferred config to FMI 2.

    FMI 2 has only Real, Integer, Boolean and String: map the FMI 3 types found, and leave out
    what FMI 2 can't hold (bytes, arrays sized by structural parameters).

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    """
    for section in ("parameters", "inputs", "states", "outputs", "locals"):
        variables = data.get(section, {})
        for name, info in list(variables.items()):
            dims = info.get("dimensions", [])
            if info.get("type") == "Binary" or any(isinstance(d, str) for d in dims):
                del variables[name]
                comments[(section, f"#{name}")] = f"{name}: bytes and resizable arrays need --fmi 3"
            elif info.get("type") in FMI2_TYPES:
                if FMI2_TYPES[info["type"]] is None:
                    del info["type"]
                else:
                    info["type"] = FMI2_TYPES[info["type"]]
        if section in data and not variables:
            del data[section]
    data.pop("structural_parameters", None)


def _wide_integers(data, comments):
    """FMI 3: an integer start beyond Int32 gets a 64-bit type, which ``build`` would otherwise refuse.

    E.g. numpy's ``default_rng(seed=12345678901234567890)``: UInt64.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    """
    for section in ("parameters", "inputs", "states"):
        for name, info in data.get(section, {}).items():
            start = info.get("start")
            if "type" in info or not isinstance(start, int) or isinstance(start, bool) or -2**31 <= start < 2**31:
                continue
            if -2**63 <= start < 2**63:
                info["type"] = "Int64"
            elif 0 <= start < 2**64:
                info["type"] = "UInt64"
            else:
                continue
            comments.setdefault((section, name), f"{info['type']}: the start doesn't fit in an Int32")


def _rename_reserved(data, comments, is_class):
    """A variable for the model's own ``time`` becomes ``model_time``.

    FMI 3 reserves ``time``; in FMI 2, importers' results name the simulation time ``time`` (FMPy's
    result array), which hides a variable of that name (nengo's ``Simulator.time``). Bound to the
    model's name with ``to`` / ``from``.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    :param is_class: whether the model is a class (outputs then come from attributes).
    :type is_class: bool
    """
    for section in ("parameters", "inputs", "states", "outputs", "locals"):
        variables = data.get(section, {})
        if "time" not in variables:
            continue
        new = next(n for n in ("model_time", f"model_time_{section}") if n not in _taken(data))
        info = variables.pop("time")
        if section in ("outputs", "locals"):
            info.setdefault("from", "attr:time" if is_class else "return:time")
        else:
            info.setdefault("to", "init:time" if section == "parameters" and is_class else "arg:time")
        variables[new] = info
        comments.pop((section, "time"), None)
        comments[(section, new)] = "the model's `time` (FMI 3 reserves that name, and importers use it for the simulation time)"


def _rename_duplicates(data, comments):
    """Rename step method arguments that have the same name as a constructor argument.

    A constructor argument and a step method argument with the same name (radioactivedecay's
    ``Inventory(..., units).decay(..., units)``) can't both be called NAME: the method's becomes
    ``<method>_NAME``, bound to it with ``to``.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    """
    method = data["model"].get("call", "call").strip("_")
    parameters = data.get("parameters", {})
    for section in ("inputs", "states"):
        variables = data.get(section, {})
        for name in [n for n in variables if n in parameters]:
            new = f"{method}_{name}"
            if new in _taken(data):
                continue
            info = variables.pop(name)
            info.setdefault("to", f"arg:{name}")
            variables[new] = info
            comments[(section, new)] = f"{method}()'s {name} (the constructor also has a {name})"
            comments.pop((section, name), None)



def _local_sources(entry_file, config_dir):
    """Files and directories under ``config_dir`` that the model imported (besides the entry file).

    Found from :data:`sys.modules` after the import. A module inside a subdirectory contributes
    the whole top-level directory, so packages and flat sibling imports are copied together.
    ``__file__`` is read from the module's ``__dict__``, not with getattr: lazy modules import on
    attribute access.

    :param entry_file: the entry file.
    :type entry_file: pathlib.Path
    :param config_dir: the config folder.
    :type config_dir: pathlib.Path
    :return: paths relative to ``config_dir`` for ``[model] sources``.
    :rtype: list[str]
    """
    sources = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__dict__", {}).get("__file__")  # no getattr: lazy modules import on access
        if not file:
            continue
        file = Path(file).resolve()
        if not file.is_relative_to(config_dir) or "site-packages" in file.parts or file == entry_file \
                or _in_venv(file, config_dir):
            continue
        rel = file.relative_to(config_dir)
        sources.add(rel.parts[0] if len(rel.parts) > 1 else rel.as_posix())
    entry_rel = entry_file.relative_to(config_dir)
    if len(entry_rel.parts) > 1:
        sources.add(entry_rel.parts[0])
    return sorted(sources)


def _in_venv(file, root):
    """Whether ``file`` is inside a virtual environment under ``root`` (a folder with ``pyvenv.cfg``).

    A venv next to the model (``cftoc/.venv``) isn't part of it: its packages are not the model's imports.

    :param file: a file under ``root``.
    :type file: pathlib.Path
    :param root: the source folder.
    :type root: pathlib.Path
    :rtype: bool
    """
    return any((root / folder / "pyvenv.cfg").is_file() for folder in file.relative_to(root).parents[:-1])


def _add_data_files(data, comments, config_dir, entry_file):
    """Add files next to the config that the model opens by a relative path.

    Found as string start values (a file name argument), string constants, or string literals
    in the entry file (parsed with :func:`ast.parse`): copied with ``sources``, and ``cwd = "."``
    so the model finds them inside the FMU.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    :param config_dir: the config folder.
    :type config_dir: pathlib.Path
    :param entry_file: the entry file, or None for a module target.
    :type entry_file: pathlib.Path or None
    """
    texts = [info["start"] for section in ("parameters", "inputs") for info in data.get(section, {}).values()
             if isinstance(info.get("start"), str)]
    texts += [c["python"] for table in ("constants", "call_constants")
              for c in data["model"].get(table, {}).values() if isinstance(c, dict) and "python" in c]
    if entry_file is not None:
        try:
            tree = ast.parse(entry_file.read_text(encoding="utf-8"))
            texts += [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        except (OSError, SyntaxError, UnicodeDecodeError):
            pass
    sources = data["model"].get("sources", [])
    found = []
    for text in dict.fromkeys(t.strip("'\"") for t in texts):
        if not text or len(text) > 255 or "\n" in text or Path(text).is_absolute():
            continue
        try:
            file = (config_dir / text).resolve()
            is_file = file.is_file()
        except (OSError, ValueError):
            continue
        if not is_file or not file.is_relative_to(config_dir) or file.suffix == ".py" or file == entry_file:
            continue
        rel = file.relative_to(config_dir).as_posix()
        if not any(rel == s or rel.startswith(f"{s}/") for s in sources + found):
            found.append(rel)
    if found:
        data["model"]["sources"] = sources + found
        data["model"]["cwd"] = "."
        comments[("model", "cwd")] = (f"the model opens {', '.join(found)} by a relative path: "
                                      "they are copied into the FMU and the model runs in that folder")


def _add_globals(data, comments, entry_module, local_dir=None):
    """Add ``[model] globals``: module-level variables the model's own package rebinds while it runs.

    Starting from the entry module, follows the modules of the same package (or of the user's
    project) that it uses, as modules or through their functions and classes, and asks
    :func:`static.written_globals` which of their globals are rebound. Only numbers, strings and
    arrays are kept. They are put back on reset and saved with the FMU state.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    :param entry_module: the model's module.
    :type entry_module: types.ModuleType
    :param local_dir: the user's project folder (its modules count as the model's own), or None.
    :type local_dir: pathlib.Path or None
    """
    top = entry_module.__name__.split(".")[0]
    def local(m):   # a module of the user's project (next to the config), not an installed one
        """Whether a module is a file of the user's project (next to the config), not an installed one.

        :param m: the module.
        :type m: types.ModuleType
        :rtype: bool
        """
        file = getattr(m, "__dict__", {}).get("__file__")
        return (local_dir is not None and file and Path(file).resolve().is_relative_to(local_dir)
                and "site-packages" not in Path(file).parts)

    def ours(m):
        """Whether a module belongs to the model: its package, or the user's project.

        :param m: the module.
        :type m: types.ModuleType
        :rtype: bool
        """
        return m.__name__ == top or m.__name__.startswith(top + ".") or local(m)

    # the entry module, and the package's modules it uses (as modules, or by their functions and classes)
    reached, todo = {entry_module.__name__: entry_module}, [entry_module]
    while todo:
        for value in list(vars(todo.pop()).values()):
            used = value if isinstance(value, types.ModuleType) else sys.modules.get(getattr(value, "__module__", None) or "")
            if isinstance(used, types.ModuleType) and used.__name__ not in reached and ours(used):
                reached[used.__name__] = used
                todo.append(used)
    found = []
    for module in reached.values():
        for name in static.written_globals(module):
            value = module.__dict__.get(name)
            if isinstance(value, (bool, int, float, str)) or _number_like(value) or _array_info(value) is not None:
                found.append(f"{module.__name__}:{name}")
    if found:
        data["model"]["globals"] = sorted(set(found))
        comments[("model", "globals")] = ("module-level state the model changes while it runs: "
                                          "put back on reset and saved with the FMU state")


def _add_requirements(data, comments, module_name, config_dir, entry_file):
    """Add ``[model] requirements``: the installed distributions that the model needs, pinned.

    The distributions (with their versions here) that provide the entry's package, the modules
    the local model files import (read with :func:`ast.parse`, not by importing), and modules
    named in setup, computed constants, convert and enum. :func:`importlib.util.find_spec`
    locates each module without importing it; ``packages_distributions`` maps it to its
    distribution. ``--vendor`` installs these.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    :param module_name: the entry's module for a module target, else None.
    :type module_name: str or None
    :param config_dir: the config folder.
    :type config_dir: pathlib.Path
    :param entry_file: the entry file, or None.
    :type entry_file: pathlib.Path or None
    """
    import importlib.metadata
    tops = set()
    if module_name:
        tops.add(module_name.split(".")[0])
    files = [entry_file] if entry_file else []
    for source in data["model"].get("sources", []):
        source = config_dir / source
        files += sorted(f for f in source.rglob("*.py") if not _in_venv(f, source)) if source.is_dir()             else [source] if source.suffix == ".py" else []
    for file in files:
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                tops |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops.add(node.module.split(".")[0])
    references = [step if isinstance(step, str) else step.get("call", "") for step in data["model"].get("setup", [])]
    for table in ("constants", "call_constants"):
        references += [v["call"] for v in data["model"].get(table, {}).values() if isinstance(v, dict) and "call" in v]
    references += [info[key] for section in ("parameters", "inputs", "states")
                   for info in data.get(section, {}).values() for key in ("convert", "enum") if key in info]
    for ref in references:
        if ref == "series":   # convert = "series": a pandas Series
            tops.add("pandas")
        if ":" in ref:   # module:function(numpy.linspace(...)): the module, and the dotted names in the arguments
            tops.add(ref.partition(":")[0].split(".")[0])
            tops |= {name.split(".")[0] for name in _dotted_names(ref.partition("(")[2])}
    if any(info.get("numpy") for section in ("parameters", "inputs", "states") for info in data.get(section, {}).values()):
        tops.add("numpy")
    distributions = static.packages_distributions()
    found, unknown = {}, []
    for top in sorted(tops):
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):   # __main__ has no spec under a console script
            continue
        if top in sys.stdlib_module_names or top == "fmugen" or spec is None:
            continue
        origin = Path(spec.origin).resolve() if spec.origin else None
        if origin is not None and origin.is_relative_to(config_dir) and "site-packages" not in origin.parts:
            continue   # one of the model's own files
        names = distributions.get(top, [])
        if len(set(names)) != 1:
            unknown.append(top)
            continue
        dist = importlib.metadata.distribution(names[0])
        found[dist.metadata["Name"]] = f"{dist.metadata['Name']}=={dist.version}"
    if found:
        data["model"]["requirements"] = sorted(found.values(), key=str.lower)
        comments[("model", "requirements")] = "the versions installed here; installed by --vendor"
    if unknown:
        comments[("model", "#requirements")] = (f"add the PyPI packages that provide {', '.join(unknown)} to "
                                                "requirements (they could not be told from what is installed)")



def _pick_entry(module, name):
    """The model's function or class in a module.

    Without a name, the one public function or class defined in the module itself (enums and
    exceptions don't count); for a package's ``__init__`` that only re-exports, from ``__all__``.

    :param module: the imported module.
    :type module: types.ModuleType
    :param name: the name given in the target, or None.
    :type name: str or None
    :return: the entry object.
    :raises InterfaceError: the name doesn't exist, or there isn't exactly one candidate.
    """
    if name:
        if not hasattr(module, name):
            raise InterfaceError(f"{module.__name__} has no {name!r}")
        return getattr(module, name)
    own = [
        obj for attr, obj in vars(module).items()
        if not attr.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == module.__name__
        and not (inspect.isclass(obj) and issubclass(obj, (enum.Enum, BaseException)))
    ]
    if not own:   # a package's __init__ that only re-exports (from .core import Model): its __all__
        own = [obj for attr in getattr(module, "__all__", ())
               if (inspect.isfunction(obj := getattr(module, attr, None)) or inspect.isclass(obj))
               and not (inspect.isclass(obj) and issubclass(obj, (enum.Enum, BaseException)))]
    if len(own) != 1:
        found = ", ".join(o.__name__ for o in own) or "none"
        raise InterfaceError(
            f"cannot tell which function or class is the model (found: {found}); "
            f"use {module.__name__}.py:Name"
        )
    return own[0]


# ---------------- functions ----------------

def _infer_function(fn, data, comments, arrays=False, starts=None, probe=False, converts=None):
    """Infer a function model: its arguments are inputs, its return value the outputs.

    Also handles a class with ``kind = "function"``: the new object is the result and its
    constructor sets the outputs.

    :param fn: the function (or class).
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    :param arrays: whether arrays may be variables (always True here; FMI 2 splits them later).
    :type arrays: bool
    :param starts: ``--start`` values.
    :type starts: dict or None
    :param probe: whether to call the model once (``--probe``).
    :type probe: bool
    :param converts: ``--convert`` options.
    :type converts: dict or None
    """
    _arguments(fn, "inputs", data, comments, arrays=arrays, starts=starts)
    _apply_converts(data, converts)
    returned_names, seen = set(), set()
    if probe:
        try:
            _run_setup(data)
            result = _probe_call(fn, data, comments)
        except ConstantError:
            raise
        except Exception as e:
            comments[("outputs", None)] = f"probe call failed ({e!r}); outputs below are read from the code"
        else:
            comments[PROBED] = True
            seen = _outputs_from_return(result, data, comments, default_name="y", is_class=False, arrays=arrays)
            returned_names |= seen
            if result is None:
                _suggest_object_step(data, comments)
    if inspect.isclass(fn):   # kind = "function": the new object is the result; its constructor sets the outputs
        kinds = static.class_attributes(fn, "__init__")[0]
        public = [a for a, k in kinds.items() if not a.startswith("_") and k not in ("other", "none")
                  and (k != "array" or arrays)]
        read = static.Returned(public, {a: f"return:{a}" for a in public}, {a: kinds[a] for a in public})
        if not public and static.function_node(fn.__init__) is None:
            read = static.Returned(unknown="no Python source to read")
    else:
        read = static.returned(fn)
    _add_returned(read, data, comments, is_class=False, arrays=arrays, probe=probe, seen=seen)
    _detect_states(data, comments, returned=returned_names | set(read.names), attrs=set())


def _suggest_object_step(data, comments):
    """A function that returned None and took a computed object writes its results onto it (pandapower's
    ``runpp(net)``): suggest the config that makes that object the model, whose attributes can be read.

    :param data: the config being inferred.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    """
    objects = {n: v["call"] for n, v in data["model"].get("constants", {}).items()
               if isinstance(v, dict) and "call" in v and not n.isdigit()}
    if len(objects) != 1:
        return
    (name, made), = objects.items()
    factory = made.split("(", 1)[0]
    function = data["model"]["entry"]
    comments[("outputs", None)] = (
        f"{function.split(':')[-1]}() returned None: its results are probably written onto {name}. To read them, "
        f"make {name} the model: fmugen init {factory} --call \"{function}(self)\" --probe")


# ---------------- classes ----------------

def _infer_class(cls, call, data, comments, arrays=False, starts=None, create=None, factory=None, probe=False,
                 converts=None):
    """Infer a class model: constructor arguments are parameters, step method arguments inputs.

    Outputs come from what the step method returns and the attributes it sets (and, with
    ``--probe``, from what changed on the object). With ``factory`` (a function), the object
    comes from calling it instead; its class is found from the factory's code when possible.

    :param cls: the class, or None for a factory.
    :type cls: type or None
    :param call: the step method (``--call``), or None to pick one.
    :type call: str or None
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    :param arrays: whether arrays may be variables.
    :type arrays: bool
    :param starts: ``--start`` values.
    :type starts: dict or None
    :param create: a classmethod that builds the object, or None.
    :type create: str or None
    :param factory: a factory function, or None.
    :type factory: callable or None
    :param probe: whether to probe.
    :type probe: bool
    :param converts: ``--convert`` options.
    :type converts: dict or None
    :raises InterfaceError: a missing ``create`` or step method.
    """
    if factory and cls is None:
        cls = static.factory_class(factory)   # from its return annotation; else known only by calling it
    method_name = call or _pick_method(cls)
    object_step = ":" in method_name   # "module:function(self, ...)": a function taking the object (pandapipes)
    if object_step:
        from fmugen.config import _object_call
        _object_call(method_name, is_class=True, call_on_class=False)
    if create:
        if not callable(getattr(cls, create, None)):
            raise InterfaceError(f"{cls.__name__} has no classmethod {create!r}")
        data["model"]["create"] = create
    if factory:
        constructor = factory
        # The step method's arguments may be known only once the object exists: names the factory
        # declares go to it, everything else to the step method (even if the factory takes **kwargs).
        try:
            declared = inspect.signature(factory).parameters
        except (TypeError, ValueError):   # a C built-in: _arguments reports it
            declared = {}
        init_starts = {k: v for k, v in (starts or {}).items() if k in declared}
        call_starts = {k: v for k, v in (starts or {}).items() if k not in declared}
    else:
        constructor = getattr(cls, create) if create else cls.__init__   # a bound classmethod has no cls argument
        init_starts, call_starts = _assign_starts(constructor, None if object_step else getattr(cls, method_name, None),
                                                  starts or {})
        init_starts, call_starts = _split_varargs(cls, create, method_name, init_starts, call_starts, data,
                                                  comments, probe)
    by_keyword = not factory and _varargs_by_keyword(getattr(cls, create) if create else cls, init_starts, data,
                                                     probe)
    _arguments(constructor, "parameters", data, comments, skip_self=not (create or factory), allow_time=False,
               arrays=arrays, starts=init_starts, positional=False, by_keyword=by_keyword)
    _apply_converts(data, converts)

    if method_name != "__call__" or call:
        data["model"]["call"] = method_name
    if not factory and not object_step and not callable(getattr(cls, method_name, None)):
        raise InterfaceError(f"{cls.__name__} has no method {method_name!r}")
    on_class = not factory and not object_step and class_called(cls, method_name)
    if on_class:
        data["model"]["call_on_class"] = True
        comments[("model", "call_on_class")] = (f"{method_name}() has no self: it's called as {cls.__name__}."
                                                f"{method_name}(...); remove this if it's a normal method")

    seen_returns, seen_attrs, obj = set(), set(), None
    if factory is not None and not probe and cls is not None and not object_step and not hasattr(cls, method_name):
        cls = None   # annotated with a base class (tirex's PretrainedModel): the real class is known only by running it
    if probe:
        obj, seen_returns, seen_attrs = _probe_class(cls, method_name, data, comments, arrays, call_starts,
                                                     create, factory, converts)
        if obj is not None and (object_step or cls is not None and not hasattr(cls, method_name)):
            # a factory annotated with a base class (tirex's load_model -> PretrainedModel): read the
            # class of the object it returned, which has the method
            cls = type(obj)
    elif object_step:
        comments[("outputs", None)] = (f"{method_name} is a function taking the object: what it changes is known only "
                                       "by running it; add outputs by hand (e.g. attr: paths), or run init --probe")
    elif cls is None:
        # getattr: a callable object (a library's placeholder for a missing optional dependency) has no __name__
        comments[("inputs", None)] = (f"the object {getattr(factory, '__name__', type(factory).__name__)}() returns is known only by running it: add the "
                                      f"inputs and outputs of {method_name}() by hand, or run init --probe")
    else:
        method = inspect.getattr_static(cls, method_name)
        _arguments(getattr(cls, method_name), "inputs", data, comments,
                   skip_self=not isinstance(method, (staticmethod, classmethod)) and not on_class,
                   constants_key="call_constants",
                   arrays=arrays, starts=call_starts)
        _apply_converts(data, converts)
        copies = static.init_copies(cls) if not factory and not create else set()
        for name in data.get("parameters", {}):
            if name in copies:
                comments.setdefault(("parameters", name),
                                    "kept as an attribute: add variability = \"tunable\" if the step reads it")
        comments[("model", "#save_state")] = (
            "save_state = false  # not checked without --probe: set it if the model holds a device, file, "
            "socket or ONNX session")
    if obj is None and cls is None or object_step:
        return   # a factory whose object type isn't known without running it; a step function's results are on obj

    if cls is not None:
        in_step, in_init = static.class_attributes(cls, method_name)
        attributes = {a: static._merge([in_step.get(a), in_init.get(a)]) for a in {*in_step, *in_init}}
        read = static.returned(getattr(cls, method_name), attributes=attributes)
    else:
        read = static.Returned()
    _add_returned(read, data, comments, is_class=True, arrays=arrays, probe=probe, seen=seen_returns)
    if cls is not None:
        _add_attributes(cls, method_name, data, comments, arrays, probe, seen_attrs)
    attrs = set(seen_attrs) | (set(dir(obj)) if obj is not None else set())
    if cls is not None:
        attrs |= set(static.assigned_attributes(cls, method_name)) | set(static.assigned_attributes(cls, "__init__"))
        attrs |= set(static.own_properties(cls))
    _detect_states(data, comments, returned=seen_returns | set(read.names), attrs=attrs)






def _pick_method(cls):
    """The step method of a class: the first of :data:`CALL_NAMES` it has, else its only public method.

    :param cls: the class.
    :type cls: type
    :rtype: str
    :raises InterfaceError: none of these, and not exactly one public method.
    """
    for name in CALL_NAMES:
        if name == "__call__":
            if any("__call__" in vars(k) for k in cls.__mro__[:-1]):
                return name
        elif callable(getattr(cls, name, None)):
            return name
    public = [n for n, v in vars(cls).items() if not n.startswith("_") and inspect.isfunction(v)]
    if len(public) == 1:
        return public[0]
    raise InterfaceError(
        f"cannot tell which method of {cls.__name__} runs a step (public methods: {', '.join(public) or 'none'}); "
        "pass --call METHOD"
    )

