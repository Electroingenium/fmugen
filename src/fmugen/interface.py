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
import copy
import enum
import importlib
import importlib.util
import inspect
import os
import re
import sys
import threading
import traceback
import types
import typing
from pathlib import Path

from fmugen import static
from fmugen.config import InterfaceError, stdlib_clash
from fmugen.toml_output import render_toml
from fmugen.arguments import (
    SCALARS, POSITIONAL_STARTS, START_ERRORS, NO_SIGNATURE, PROBED,
    _number_like, _array_info, _dotted_names, _arguments, _apply_converts,
    _assign_starts, assumptions, guessed_outputs, unfit_outputs, class_called, _only_varargs, _time_source,
)
from fmugen.probe import (
    ConstantError, _run_setup, _probe_call, _probe_class, _forwarded,
    _outputs_from_return, _add_returned, _add_attributes, _detect_states,
    _taken, _split_varargs, _varargs_by_keyword, MADE_UP_TEXT,
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
    with contextlib.chdir(config_dir), _source_reads() as reads:
        if probe is True or not probe:
            data, comments = _infer_config(target, call, config_dir, fmi_version, starts, setup, kind, create, probe,
                                           converts)
        else:
            data, comments = _with_timeout(float(probe), _infer_config, target, call, config_dir, fmi_version, starts,
                                           setup, kind, create, True, converts)
    if reads:
        shown = ", ".join(sorted(reads)[:3]) + (f" and {len(reads) - 3} more" if len(reads) > 3 else "")
        comments[("model", "#compile")] = (f"--compile won't work: the model's libraries read their own source files "
                                           f"by path ({shown}), and a compiled FMU has none; build without --compile, "
                                           "or with --vendor")
    return data, comments


@contextlib.contextmanager
def _source_reads():
    """Record ``.py`` files of installed packages read as files (not imported), while the model is imported and probed.

    py-pde loads ``pde/backends/*/config.py`` with :func:`importlib.util.spec_from_file_location`; others
    ``open()`` them. A compiled FMU (``--compile``) contains no ``.py`` files, so those reads fail there.
    Regular imports don't go through either, so they aren't recorded.

    :return: a set, filled with paths relative to their ``site-packages`` folder.
    """
    import builtins
    import importlib.util
    import sysconfig
    roots = {Path(sysconfig.get_paths()[k]).resolve() for k in ("purelib", "platlib")}
    own = Path(__file__).resolve().parent
    reads = set()

    def note(path):
        try:
            p = Path(path).resolve()
        except (TypeError, ValueError, OSError):
            return
        if p.suffix != ".py" or p.is_relative_to(own):
            return
        root = next((r for r in roots if p.is_relative_to(r)), None)
        if root is not None:
            reads.add(p.relative_to(root).as_posix())

    real_open, real_spec = builtins.open, importlib.util.spec_from_file_location

    def watched_open(file, *args, **kwargs):
        # not the source lines of tracebacks, warnings and inspect (linecache, through tokenize.open), nor
        # numba's cache keys (a hash of the source file): both do without the file, and those compiled FMUs
        # work (librosa, pythermalcomfort)
        caller = Path(sys._getframe(1).f_code.co_filename)
        if isinstance(file, (str, os.PathLike)) and caller.name not in ("tokenize.py", "linecache.py", "inspect.py") \
                and "numba" not in caller.parts:
            note(file)
        return real_open(file, *args, **kwargs)

    def watched_spec(name, location=None, *args, **kwargs):
        if location is not None:
            note(location)
        return real_spec(name, location, *args, **kwargs)

    builtins.open, importlib.util.spec_from_file_location = watched_open, watched_spec
    try:
        yield reads
    finally:
        builtins.open, importlib.util.spec_from_file_location = real_open, real_spec


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


def _infer_config(target, call, config_dir, fmi_version, starts, setup, kind, create, probe, converts, examples=True):
    """:func:`_infer_config_once`; then, while the probe fails on a made-up argument that has to be a dict or an
    object with some attributes (:func:`_shaped_start`), again with it made up so.

    river's ``learn_one(x, y)`` takes a dict of features, simglucose's ``step(action)`` an object with ``CHO`` and
    ``insulin``, a hyperparameter function a dict with its keys: each error names what's missing, so this
    goes on until the probe works or an error says nothing new.
    """
    data, comments = _infer_config_once(target, call, config_dir, fmi_version, starts, setup, kind, create, probe,
                                        converts, examples)
    if not probe:
        return data, comments
    if _probe_failed(comments) and not setup:
        # a set-up function of the module the error asks for: psychrolib's "The system of units has not been
        # defined": SetUnitSystem(psychrolib.UnitSystem.SI)
        for step in _module_setups(data["model"].get("entry", ""), comments[("outputs", None)]):
            try:
                trial = _infer_config_once(target, call, config_dir, fmi_version, starts, [step], kind, create,
                                           probe, converts, examples)
            except InterfaceError:
                continue
            if not _probe_failed(trial[1]):
                data, comments, setup = *trial, [step]
                comments[("model", "setup")] = ("the probe failed without it (the error asks for a set-up); "
                                                "check the argument")
                break
    starts, shaped = dict(starts or {}), {}
    for _ in range(MAX_SHAPED_ROUNDS):
        if not _probe_failed(comments):
            break
        error = comments[("outputs", None)]
        further = None   # the first that fails further on, taken when none works
        for added, how in _shaped_starts(data, comments, starts, shaped):
            try:
                trial = _infer_config_once(target, call, config_dir, fmi_version, {**starts, **added}, setup, kind,
                                           create, probe, converts, examples)
            except InterfaceError:
                continue
            # it works, or fails further on (a count of arguments: only when building works then)
            if trial[1].get(("outputs", None)) != error and not (
                    "arguments the error asks for" in how and str(trial[1].get(("outputs", None))).startswith("probe construction failed")):
                if not _probe_failed(trial[1]):
                    further = trial, added, how
                    break
                further = further or (trial, added, how)
        if further is None:
            break
        (data, comments), added, how = further
        starts.update(added)
        shaped.update(dict.fromkeys(added, how))
    for name, how in shaped.items():
        for section in ("parameters", "inputs"):
            for key in [k for k in comments if k[0] == section and (k[1] == name or str(k[1]).startswith(name + "."))]:
                comments[key] = how
        for table in ("constants", "call_constants"):
            if name in data["model"].get(table, {}):
                comments[("constant", f"{table}.{name}")] = how
    return data, comments


#: First words of a module's set-up function: ``SetUnitSystem``, ``init_backend``, ``configure``.
SETUP_WORDS = ("set", "init", "initialize", "configure", "setup", "use", "enable", "select", "load")


def _module_setups(entry_ref, error):
    """``module:function(args)`` set-ups of the entry's module that the error seems to ask for: a function named
    :data:`SETUP_WORDS` + words the error has (psychrolib's ``SetUnitSystem`` for "The system of units has not
    been defined"), its arguments made up: the SI member of an Enum they're annotated with, else the first."""
    module_name = entry_ref.partition(":")[0]
    if not module_name or module_name.endswith(".py"):
        return []
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return []
    said = {w[:4] for w in re.findall(r"[a-z]{4,}", str(error).lower())}
    found = []
    for attr, fn in vars(module).items():
        words = [w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", attr)]
        if attr.startswith("_") or not callable(fn) or inspect.isclass(fn) or not words \
                or words[0] not in SETUP_WORDS or not any(w[:4] in said for w in words[1:] if len(w) >= 4):
            continue
        try:
            params = [p for p in inspect.signature(fn).parameters.values()
                      if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            continue
        made = []
        for p in params:
            kind = p.annotation
            if isinstance(kind, str):
                kind = getattr(module, kind, None)
            if inspect.isclass(kind) and issubclass(kind, enum.Enum) and len(kind):
                member = next((m for m in kind if m.name.upper() in ("SI", "METRIC")), next(iter(kind)))
                made.append(f"{kind.__module__}.{kind.__qualname__}.{member.name}")
            else:
                break
        if len(made) == len(params):
            found.append(f"{module_name}:{attr}({', '.join(made)})")
    return found[:3]


#: How many arguments, or attributes and keys of one, :func:`_infer_config` makes up from the errors, at most.
MAX_SHAPED_ROUNDS = 12
#: "'float' object has no attribute 'CHO'": the argument is an object with that attribute.
MISSING_ATTRIBUTE = re.compile(r"'(?:float|int|types\.SimpleNamespace)' object has no attribute '(\w+)'")
#: Attributes of arrays and tables: the argument is one of those, not an object made up here.
ARRAY_ATTRIBUTES = {"shape", "ndim", "dim", "dtype", "size", "T", "astype", "copy", "reshape", "numpy", "values",
                    "index", "columns", "iloc", "loc", "flatten", "ravel", "tolist", "detach", "to", "cpu",
                    "device", "real", "imag", "is_integer", "unit", "units", "magnitude", "value"}
#: Attributes of text: the argument is a string.
STRING_ATTRIBUTES = {"strip", "lower", "upper", "split", "startswith", "endswith", "encode", "replace", "casefold",
                     "isdigit", "isalpha", "rstrip", "lstrip", "partition", "format"}
#: An error asking for a dict: river's "'float' object is not an instance of 'dict'".
DICT_ASKED = re.compile(r"instance of '?dict|must be (?:a )?dict|expected (?:a )?dict|"
                        r"'(?:float|int)' object has no attribute '(?:items|keys|get)'|not a mapping", re.I)
#: "KeyError('n_estimators')": the argument is a dict with that key.
MISSING_KEY = re.compile(r"KeyError\('(\w+)'\)")


def _shaped_starts(data, comments, starts, shaped):
    """``({name: start}, comment)`` to try for the made-up arguments, from the probe's error (:func:`_infer_config`):
    a dict of two features, a dict with the key it lacks, an object (:class:`types.SimpleNamespace`) with the
    attributes it lacks; for each made-up argument in turn, the inputs first (those already made up so first).

    :param shaped: name -> its comment, of those made up so far.
    :rtype: list[tuple]
    """
    error = comments[("outputs", None)]
    made = [n for n in shaped] + [
        n for section in ("inputs", "parameters") for n, info in data.get(section, {}).items()
        if n not in shaped and n not in starts and str(comments.get((section, n), "")).startswith("no default")
        and "enum" not in info] + [
        n for table in ("constants", "call_constants") for n in data["model"].get(table, {})
        if n not in shaped and n not in starts and any(str(comments.get((k, f"{table}.{n}"), "")).startswith(
            "no default") for k in ("model", "constant"))]
    tries = []
    if (match := NONE_ATTRIBUTE.search(error)) and importlib.util.find_spec("ase") is not None:
        # an ASE calculator's calculate(atoms=None): mace's "'NoneType' object has no attribute 'symbols'"
        for key in list(comments):   # the None defaults: commented out, as ("model", "call_constants.atoms")
            table, _, name = str(key[1]).partition(".") if key[0] == "model" else ("", "", "")
            if table in ("constants", "call_constants") and name and name not in starts \
                    and name not in data["model"].get(table, {}) \
                    and (name == "atoms" or match.group(1) in ATOMS_ATTRIBUTES and "atom" in name):
                tries.append(({name: {"call": MADE_UP_ATOMS}}, "made up as a water molecule (ase): the probe failed "
                                                               "without it; give your own"))
    if NONE_SIZED.search(error):
        # a list left None the code takes the length of: pykan's KAN(width=None), "'NoneType' has no len()"
        for key in list(comments):
            table, _, name = str(key[1]).partition(".") if key[0] == "model" else ("", "", "")
            if table in ("constants", "call_constants") and name and name not in starts \
                    and name not in data["model"].get(table, {}) and comments[key] == {"python": "None"} \
                    and not FUNCTION_NAME.search(name):
                tries.append(({name: [1, 1]}, "made up as [1, 1]: the probe failed with it None; check it"))
    for key in list(comments):
        # a function left None: numdifftools' Derivative(fun=None), whose call asserts it's given
        table, _, name = str(key[1]).partition(".") if key[0] == "model" else ("", "", "")
        if table in ("constants", "call_constants") and FUNCTION_NAME.search(name) and name not in starts \
                and name not in data["model"].get(table, {}) and "numpy" in sys.modules:
            tries.append(({name: {"call": "numpy:sin"}}, "made up as numpy.sin: the probe failed without it; give "
                                                          "your own function"))
    words = set(re.findall(r"[a-z]+", error.lower()))
    for name in sorted(dict.fromkeys(made), key=lambda n: not set(n.lower().split("_")) & words)[:1]:
        # the names the code that raised compares the argument with: deepxde's "Could not interpret activation
        # function identifier: 1.0" from activations.get, whose dict has "elu", "relu", ... "tanh"
        if starts.get(name) is None:
            for text in _raising_literals(data["model"].get("entry", ""), error):
                tries.append(({name: text}, f"made up as {text!r}, a name the code that failed on a number "
                                            "compares it with; check it"))
    classes = {}   # name -> the first package class made up for it
    for name in dict.fromkeys(made):
        current = starts.get(name)
        if (match := MISSING_KEY.search(error)) and (current is None or isinstance(current, dict)
                                                       and "call" not in current):
            keys = {**(current or {}), match.group(1): 1.0}
            tries.append(({name: keys}, f"made up as a dict with {', '.join(keys)}: the probe failed without them; "
                                      "check it"))
        elif DICT_ASKED.search(error) and current is None:
            tries.append(({name: {"a": 1.0, "b": 1.0}}, "made up as a dict of two features: the probe failed with a "
                                                      "number; set your own"))
        elif (match := MISSING_ATTRIBUTE.search(error)) and match.group(1) in STRING_ATTRIBUTES and current is None:
            # thermo's Chemical(ID): "'float' object has no attribute 'strip'", a name: the examples' if any
            text = _example_value(data["model"].get("entry", ""), name)
            text = text if isinstance(text, str) else MADE_UP_TEXT
            tries.append(({name: text}, f"made up as {text!r}: the probe failed with a number where it needs text; "
                                        "check it"))
        elif (match := MISSING_ATTRIBUTE.search(error)) and match.group(1) not in ARRAY_ATTRIBUTES \
                and (current is None or name in shaped and "SimpleNamespace" in str(current)):
            fields = re.findall(r"(\w+)=1\.0", str(current or "")) + [match.group(1)]
            record = _package_record(data["model"].get("entry", ""), [f for f in fields if not f.startswith("_")])
            if record:   # simglucose's step(action): its Action namedtuple, whose _replace() it calls
                tries.append(({name: {"call": record}}, f"made up as {record.replace(':', '.')}: the probe failed "
                                                        "with a number; check it"))
            if not match.group(1).startswith("_"):
                tries.append(({name: {"call": f"types:SimpleNamespace({', '.join(f'{k}=1.0' for k in fields)})"}},
                              f"made up as an object with {', '.join(fields)}: the probe failed with a number; "
                              "check it"))
        if current is None and "astropy" in sys.modules and "Quantity" in _described(
                data["model"].get("entry", ""), data["model"].get("call")).get(name, ""):
            # hapsira's propagate(value : ~astropy.units.Quantity): a time, else a length or a plain quantity
            for unit in MADE_UP_UNITS:
                tries.append(({name: {"call": f"astropy.units:Quantity(1.0, {unit!r})"}},
                              f"made up as an astropy quantity, 1 {unit or '(no unit)'}: its docstring says it's a "
                              "Quantity; check the value and the unit"))
        if current is None and SERIES_ASKED.search(error):   # pastas' Model(oseries): a dated pandas Series
            tries.append(({name: {"call": MADE_UP_SERIES}}, "made up as a dated pandas Series of 24 days: the probe "
                                                            "failed with a number; set your own"))
        elif current is None and (NETWORK_NAME.search(name) or NETWORK_ASKED.search(error)) \
                and "torch" in sys.modules:
            # skorch's NeuralNetRegressor(module), torchdyn's NeuralODE(vector_field): a small network, 4 features
            for layer in ("torch.nn:Linear(4, 1)", "torch.nn:Linear(4, 4)"):
                tries.append(({name: {"call": layer}}, f"made up as {layer.replace(':', '.')}: the probe failed with "
                                                       "a number; give your own network"))
        elif current is None and (found := _package_class(data["model"].get("entry", ""), name, error,
                                                           data["model"].get("call"))):
            # nengo's Simulator(network): the package's Network(); harold's lqr(G) "should be a Transfer or a
            # State", pygfunction's "borehole1 : Borehole object": built with its arguments made up as 1.0
            for built in found:
                tries.append(({name: {"call": built}}, f"made up as {built.replace(':', '.')}: the probe failed "
                                                       "with a number; check it"))
            classes[name] = found[0]
        elif current is None and NOT_CALLABLE.search(error) and "numpy" in sys.modules:
            # numdifftools' Derivative(fun), a callback: a function of an array
            tries.append(({name: {"call": "numpy:sin"}}, "made up as numpy.sin: the probe failed with a number; "
                                                          "give your own function"))
        elif current is None and PRNG_ASKED.search(error) and re.search(r"key|rng", name) and "jax" in sys.modules:
            # equinox's MLP(key): a jax random key
            tries.append(({name: {"call": "jax.random:PRNGKey(0)"}}, "made up as jax.random.PRNGKey(0): the probe "
                                                                      "failed with a number; check it"))
    if "tuple index out of range" in error and error.startswith("probe call failed") and not shaped:
        # a step taking only (*inputs), given none: neuromancer's Block.forward(*inputs), a batch of features
        name = _varargs_step(data["model"].get("entry", ""), data["model"].get("call"))
        if name and name not in starts:
            made_up = {"call": "torch:ones(1, 1)"} if "torch" in sys.modules else 1.0
            tries.append(({name: made_up}, f"made up as {made_up['call'].replace(':', '.') if isinstance(made_up, dict) else made_up}, "
                                           "passed by position: the step takes (*inputs); give your own"))
    if (match := TORCHSCRIPT_DECLARATION.search(error)) and not any(n in starts for n in shaped):
        # a TorchScript module's forward(), whose arguments only its error declares (silero's "Tensor x, int sr")
        added = {}
        for part in match.group(1).split(","):
            kind, _, name = part.strip().rpartition(" ")
            if name and name != "self" and name not in starts and kind in TORCHSCRIPT_MADE_UP:
                added[name] = TORCHSCRIPT_MADE_UP[kind]
        if added:
            tries.append((added, "made up from the TorchScript declaration the error gives: check it"))
    if match := ALLOWED_VALUES.search(error):
        # "Supported sampling rates: [8000, 16000]": a made-up number as each value it lists
        values = [float(v) if "." in v else int(v) for v in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))][:2]
        numbers = [n for n in dict.fromkeys([*shaped, *made]) if isinstance(starts.get(n, 1.0), (int, float))]
        for name in numbers[:3]:
            for value in values:
                tries.append(({name: value}, f"made up as {value}, a value the error lists: check it"))
    if error.startswith("probe call failed") and not shaped and not data.get("parameters"):
        # a C class built with no arguments whose step then fails: its stub's other __init__ (cv2's
        # KalmanFilter(dynamParams: int, measureParams: int), "One or more matrix operands are empty")
        added = {n: v for n, v in _stub_arguments(data["model"].get("entry", "")).items() if n not in starts}
        if added:
            tries.append((added, "made up from the constructor in the package's type stub (.pyi): check it"))
    split = list(dict.fromkeys(
        str(info.get("to", ""))[5:].partition(".")[0] for info in data.get("parameters", {}).values()
        if str(info.get("to", "")).startswith("init:") and "." in str(info.get("to", ""))))
    if error.startswith("probe construction failed") and split and not any(n in starts for n in split):
        # arguments built from the fields of their annotation, which is a descriptor, not what they take:
        # astropy's FlatLambdaCDM(H0: Parameter, Om0: Parameter) takes numbers
        tries.append((dict.fromkeys(split, 1.0), "made up as 1.0: the probe failed building it from the fields of "
                                                 "its annotation; check it"))
    for name, source in data.get("time", {}).items():
        # passed the simulation time, 0 at the probe: pysindy's fit(x, t), a time step that must be positive
        if source == "time" and name not in starts and name not in shaped:
            tries.append(({name: 1.0}, "made up as 1.0, a fixed value: the probe failed passing the simulation "
                                       "time (0); to pass it, remove this and add [time] " f'{name} = "time"'))
    factories = {n: _annotated_factories(data["model"].get("entry", ""), data["model"].get("call"), n)
                 for n in dict.fromkeys(made) if starts.get(n) is None}
    factories = {n: found for n, found in factories.items() if found}
    if factories:
        # pyfluids' with_state(first_input: Input, second_input: Input): Input.density(1.0), Input.enthalpy(1.0),
        # a different factory for each argument of the same class
        used, chosen = {}, {}
        for name, found in factories.items():
            index = used.get(found[0], 0)
            chosen[name] = {"call": found[min(index, len(found) - 1)]}
            used[found[0]] = index + 1
        tries.insert(0, (chosen, "made up with a factory of the class it's annotated with, from 1.0: check it"))
    if len(classes) > 1:   # pygfunction's finite_line_source(time, alpha, borehole1, borehole2): both at once
        tries.insert(0, ({name: {"call": built} for name, built in classes.items()},
                         "made up as a class of the package the docs or the error name: the probe failed with a "
                         "number; check it"))
    if (match := MISSING_KEY.search(error)) and match.group(1) not in starts and match.group(1) not in shaped:
        # a key of **kwargs (surfaces' hyperparameter functions): an Integer, which also passes as a number
        tries.append(({match.group(1): 1}, "made up as 1: the step looks it up in its keyword arguments; check it"))
    if (match := ARGUMENTS_EXPECTED.search(error)) and error.startswith("probe construction failed") \
            and not shaped and not any(n in starts for n in data.get("parameters", {})):
        # a (*args) constructor: control's StateSpace "Expected 1, 4, or 5 arguments", named as its docstring does
        names = _vararg_names(data["model"].get("entry", ""))
        for count in sorted({int(n) for n in re.findall(r"\d+", match.group(1)) if 0 < int(n) <= 8}):
            chosen = (names if len(names) >= count else [f"arg{i + 1}" for i in range(count)])[:count]
            if not any(n in starts for n in chosen):
                tries.append((dict.fromkeys(chosen, 1.0), f"made up as 1.0, one of {count} arguments the error asks "
                                                         "for; check it"))
    if error.startswith("probe construction failed") and not shaped and not data.get("parameters") \
            and not ARGUMENTS_EXPECTED.search(error) and _takes_only_varargs(data["model"].get("entry", "")):
        # a (*args) constructor that says nothing of its arguments (nashpy's Game(*args): payoff matrices)
        name = (_vararg_names(data["model"]["entry"]) or ["arg1"])[0]
        if name not in starts:
            tries.append(({name: [[1.0, 0.0], [0.0, 1.0]]}, "made up as a 2 x 2 identity matrix, the argument of a "
                                                              "(*args) constructor: check it"))
    if error.startswith("probe construction failed") and not shaped:
        # an empty-list default the error gives a length for: impedance's Randles(initial_guess=[]), "circuit length (5)"
        lengths = [int(n) for n in re.findall(r"\((\d+)\)", error) if 1 < int(n) <= 100]
        for name in _empty_list_defaults(data["model"].get("entry", "")):
            if name not in starts:
                for length in dict.fromkeys(reversed(lengths)):
                    tries.append(({name: [1.0] * length}, f"made up as {length} ones: the probe failed with an "
                                                          "empty list, and the error asks for that many; check it"))
    return tries[:4] if not any("a name the code that failed" in how for _, how in tries) else tries[:10]


#: The message inside a probe failure: "probe construction failed (TypeError('Could not interpret ...: 1.0'))".
FAILURE_MESSAGE = re.compile(r"probe (?:construction|call) failed \(\w+\(['\"](.*?)(?:['\"]\)\)|$)", re.S)
#: A string literal that may be a name an argument is compared with: "tanh", "Glorot normal", "fan_in".
NAME_LITERAL = re.compile(r"^[A-Za-z][\w .-]{0,30}$")


def _raising_literals(entry_ref, error):
    """The string literals of the package function whose source has the start of the error's message (its text
    before a ":" or "{"): the names it accepts, in source order, at most 8; [] when none is found."""
    match = FAILURE_MESSAGE.search(str(error))
    head = re.split(r"[:{]", match.group(1))[0].strip() if match else ""
    path = entry_ref.partition(":")[0]
    top = path.partition(".")[0]
    if len(head) < 12 or not top:
        return []
    if path.endswith(".py"):   # a model file: that file
        entry = _entry_object(entry_ref)
        modules = [sys.modules.get(getattr(entry, "__module__", None)) or types.SimpleNamespace(__file__=path)]
    else:
        modules = [m for n, m in sorted(sys.modules.items()) if m is not None and (n == top or n.startswith(top + "."))]
    for module in modules:
        try:
            source = Path(module.__file__).read_text(encoding="utf-8")
        except Exception:
            continue
        if head not in source:
            continue
        line = source[:source.index(head)].count("\n") + 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.lineno <= line <= (n.end_lineno or 0)]
        if not functions:
            continue
        function = min(functions, key=lambda n: (n.end_lineno or 0) - n.lineno)
        # those it compares with or looks up: dict keys, ==/in operands, set/list/tuple items ("LAAF" of a
        # startswith() isn't a name)
        nodes = []
        for node in ast.walk(function):
            if isinstance(node, ast.Dict):
                nodes += [k for k in node.keys if k is not None]
            elif isinstance(node, ast.Compare):
                nodes += [node.left, *node.comparators]
            elif isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                nodes += node.elts
        nodes.sort(key=lambda n: (n.lineno, n.col_offset))
        texts = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and NAME_LITERAL.match(n.value) and head not in n.value]
        for node in ast.walk(function):   # a module dict it looks up: deepxde's INITIALIZER_DICT[identifier]
            table = getattr(module, node.value.id, None) if isinstance(node, ast.Subscript) \
                and isinstance(node.value, ast.Name) else None
            if isinstance(table, dict):
                texts += [k for k in table if isinstance(k, str) and NAME_LITERAL.match(k)]
        return list(dict.fromkeys(texts))[:8]
    return []


#: A TorchScript module's error declaring its forward(): "Declaration: forward(... self, Tensor x, int sr) -> Tensor".
TORCHSCRIPT_DECLARATION = re.compile(r"is missing value for argument '\w+'\. Declaration: \w+\(([^)]*)\)")
#: Made-up values by TorchScript type: a batch of 512 samples for a tensor.
TORCHSCRIPT_MADE_UP = {"Tensor": {"call": "torch:ones(1, 512)"}, "int": 1, "float": 1.0, "bool": False}
#: An error listing the values a number may take: silero's "Supported sampling rates: [8000, 16000]".
ALLOWED_VALUES = re.compile(r"(?:supported|one of|allowed|valid|must be in)[^\[\]()]{0,40}[\[(]((?:\s*-?\d+(?:\.\d+)?\s*,)+"
                            r"\s*-?\d+(?:\.\d+)?\s*)[\])]", re.I)


def _example_value(entry_ref, name):
    """The value the package's examples pass to an argument ``name`` (:func:`fmugen.examples.find_values`), or
    None."""
    from fmugen.examples import find_values
    entry = _entry_object(entry_ref)
    module_name = entry_ref.partition(":")[0]
    if entry is None or module_name.endswith(".py"):
        return None
    try:
        return find_values(entry, importlib.import_module(module_name)).get(name, (None,))[0]
    except Exception:
        return None


def _stub_arguments(entry_ref):
    """``{argument: 1 or 1.0}``: the required arguments of the ``__init__`` with the most of them that the package's
    type stub (``.pyi`` next to the module, or the package's ``__init__.pyi``) declares for the class
    ``entry_ref`` names, when the class has no readable signature (a C class); else {}."""
    cls = _entry_object(entry_ref)
    if not inspect.isclass(cls):
        return {}
    try:
        inspect.signature(cls)
        return {}
    except (TypeError, ValueError):
        pass
    stubs = []
    for module_name in dict.fromkeys([cls.__module__, cls.__module__.partition(".")[0]]):
        file = getattr(sys.modules.get(module_name), "__file__", None)
        if file:
            stubs += [Path(file).with_suffix(".pyi"), Path(file).parent / "__init__.pyi"]
    for stub in dict.fromkeys(stubs):
        try:
            tree = ast.parse(stub.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == cls.__name__):
                continue
            best = {}
            for init in [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]:
                args = init.args.args[1:]
                required = args[:len(args) - len(init.args.defaults)]
                made = {a.arg: 1 if isinstance(a.annotation, ast.Name) and a.annotation.id in ("int", "bool") else 1.0
                        for a in required}
                if len(made) > len(best):
                    best = made
            return best
    return {}


def _annotated_factories(entry_ref, call, name):
    """``module:Class.factory(1.0)`` for each classmethod or staticmethod returning the class argument ``name`` of
    the step (or the constructor) is annotated with, taking one to three numbers, when the class itself can't be
    built from numbers (pyfluids' ``Input(coolprop_key, value)``: ``Input.pressure(value)``); else []."""
    entry = _entry_object(entry_ref)
    found = []
    for fn in (getattr(entry, call, None) if call else None, getattr(entry, "__init__", None), entry):
        try:
            hint = typing.get_type_hints(fn).get(name)
        except Exception:
            continue
        if inspect.isclass(hint) and hint.__module__ not in ("builtins", "typing") and not issubclass(hint, enum.Enum):
            break
    else:
        return []
    for attr in sorted(vars(hint)):
        raw = vars(hint)[attr]
        if attr.startswith("_") or not isinstance(raw, (classmethod, staticmethod)):
            continue
        method = getattr(hint, attr)
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            continue
        params = [p for p in signature.parameters.values() if p.default is p.empty]
        returns = signature.return_annotation
        if not 0 < len(params) <= 3 or returns not in (hint, hint.__name__, "Self", f"'{hint.__name__}'") \
                or any(p.annotation not in (p.empty, float, int, "float", "int") for p in params):
            continue
        found.append(f"{hint.__module__}:{hint.__qualname__}.{attr}({', '.join(['1.0'] * len(params))})")
    return found


def _package_record(entry_ref, fields):
    """``module:Record(a=1.0, ...)``: a namedtuple or dataclass of the model's package with all of ``fields``
    (the fewest fields first), every field 1.0; or None."""
    top = entry_ref.partition(":")[0].partition(".")[0]
    if not top or top.endswith(".py") or not fields:
        return None
    found = []
    for module_name, module in sorted(sys.modules.items()):
        if module is None or (module_name != top and not module_name.startswith(top + ".")):
            continue
        for attr, value in list(vars(module).items()):
            if not inspect.isclass(value) or attr.startswith("_"):
                continue
            names = getattr(value, "_fields", None) if issubclass(value, tuple) else (
                list(getattr(value, "__dataclass_fields__", {})) or None)
            if names and set(fields) <= set(names) and len(names) <= 12:
                found.append((len(names), f"{module_name}:{attr}({', '.join(f'{n}=1.0' for n in names)})"))
    return min(found)[1] if found else None


def _varargs_step(entry_ref, call):
    """The name of the ``*args`` of the step method that takes only those (``inputs``), or None."""
    cls = _entry_object(entry_ref)
    method = getattr(cls, "forward" if call in (None, "__call__") and hasattr(cls, "forward") else call or "", None)
    try:
        params = [p for p in inspect.signature(method).parameters.values() if p.name != "self"]
    except (TypeError, ValueError):
        return None
    if params and all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params) \
            and params[0].kind is params[0].VAR_POSITIONAL:
        return params[0].name
    return None


def _empty_list_defaults(entry_ref):
    """The constructor arguments of the class ``entry_ref`` names whose default is an empty list or tuple, its base
    classes' too (passed through ``**kwargs``: impedance's ``Randles`` to ``BaseCircuit(initial_guess=[])``)."""
    cls = _entry_object(entry_ref)
    if cls is None:
        return []
    names = []
    for klass in getattr(cls, "__mro__", ())[:-1]:
        try:
            params = inspect.signature(klass.__init__).parameters.values()
        except (TypeError, ValueError):
            continue
        names += [p.name for p in params if isinstance(p.default, (list, tuple)) and not p.default]
    return list(dict.fromkeys(names))


def _entry_object(entry_ref):
    """The object a config's ``entry`` names (``module:Name``, or ``file.py:Name`` relative to the config folder),
    or None."""
    module, _, name = entry_ref.partition(":")
    try:
        obj = load_module(Path(module).resolve()) if module.endswith(".py") else importlib.import_module(module)
        for part in name.split("."):
            obj = getattr(obj, part)
    except Exception:
        return None
    return obj


def _takes_only_varargs(entry_ref):
    """Whether the class ``entry_ref`` names is built with ``(*args)`` alone (``**kwargs`` too)."""
    cls = _entry_object(entry_ref)
    try:
        params = list(inspect.signature(cls).parameters.values())
    except Exception:
        return False
    return any(p.kind is p.VAR_POSITIONAL for p in params) \
        and all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)


#: Names of arguments that are a torch network.
NETWORK_NAME = re.compile(r"^(module|net|network|vector_field|model_fn)$")
#: An error asking for a network or a callable: torchdyn's "unsupported callable".
NETWORK_ASKED = re.compile(r"\bcallable\b|nn\.Module|\bModule\b|has no attribute 'forward'|has no attribute 'parameters'")
#: "object of type 'NoneType' has no len()": a list left None is needed.
NONE_SIZED = re.compile(r"'NoneType' has no len|'NoneType' object is not (?:iterable|subscriptable)")
#: "'NoneType' object has no attribute 'pbc'": an argument left None is needed.
NONE_ATTRIBUTE = re.compile(r"'NoneType' object has no attribute '(\w+)'")
#: Attributes of an ASE ``Atoms``, which calculators read from their ``atoms``.
ATOMS_ATTRIBUTES = {"symbols", "pbc", "positions", "cell", "numbers", "get_positions", "get_atomic_numbers",
                    "get_chemical_symbols", "get_cell", "get_pbc"}
#: A made-up ASE ``Atoms``: a water molecule in a 10 Å box.
MADE_UP_ATOMS = "ase.build:molecule('H2O', vacuum=5.0)"
#: An error about a time series: pastas' "The dtype of the values of the series 'None' is not float".
SERIES_ASKED = re.compile(r"\bseries\b|DatetimeIndex|PeriodIndex", re.I)
#: A made-up dated series (:data:`SERIES_ASKED`): 24 daily values going up and down.
MADE_UP_SERIES = f"pandas:Series({[1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5] * 3}, " \
                 "index=pandas.date_range('2020-01-01', periods=24))"


#: "'float' object is not callable": the argument is a function.
NOT_CALLABLE = re.compile(r"'(?:float|int)' object is not callable|must be (?:a )?callable|is not callable")
#: Names of arguments that are a function: numdifftools' ``fun``, scipy-like ``func``, ``callback``.
FUNCTION_NAME = re.compile(r"^(fun|func|function|fn|f|callback|objective)$")
#: An error asking for a jax random key: equinox's "invalid PRNG key data".
PRNG_ASKED = re.compile(r"PRNG ?key", re.I)
#: A class name in an error or a docstring's type line: "a Transfer or a State", "Borehole object".
CLASS_WORD = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)*)\b")


def _package_class(entry_ref, name, error="", call=None):
    """``module:Class(...)`` building each class of the model's package that argument ``name`` may be:
    named as the argument (``network``: nengo's ``Network``), in the error ("should be a Transfer or a State":
    harold), or in the argument's docstring ("Borehole object": pygfunction). Built with no arguments, else with
    its required ones made up as 1.0."""
    top = entry_ref.partition(":")[0].partition(".")[0]
    if not top:
        return []
    wanted = [name.replace("_", "").lower()] if len(name.replace("_", "")) >= 4 else []
    described = _described(entry_ref, call)
    for text in (str(error).partition("probe ")[2], described.get(name, "")):
        wanted += [w.lower() for w in CLASS_WORD.findall(text) if len(w) >= 4]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return []
    found = {}
    for module_name, module in sorted(sys.modules.items()):
        if module is None or (module_name != top and not module_name.startswith(top + ".")) \
                or any(p.startswith("_") for p in module_name.split(".")):
            continue
        for attr, value in list(vars(module).items()):
            if attr.lower() in wanted and attr.lower() not in found and inspect.isclass(value) \
                    and value.__module__.partition(".")[0] == top:
                built = _built_with_made_up(value)
                if built is not None:
                    found[attr.lower()] = f"{module_name}:{attr}({built})"
    return [found[w] for w in wanted if w in found]


#: Units an astropy quantity is made up in, in turn: a time, a length, none.
MADE_UP_UNITS = ("s", "m", "")


def _described(entry_ref, call):
    """``{argument: docstring text}`` of the entry, its ``__init__`` and its step method ``call``."""
    entry = _entry_object(entry_ref)
    return docstring_parameters(*[o for o in (entry, getattr(entry, "__init__", None),
                                              getattr(entry, call, None) if call else None) if o is not None])


def _built_with_made_up(cls):
    """The arguments text that builds ``cls``: none, or its required ones as 1.0, with those defaulting to None
    before them too (harold's ``State(a, b=None, c=None, d=None)``: a system, not a static gain); None when
    neither works."""
    try:
        cls()
        return ""
    except Exception:
        pass
    try:
        params = [p for p in inspect.signature(cls).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return None
    required = sum(p.default is p.empty for p in params)
    unset = required
    while unset < len(params) and params[unset].default is None:
        unset += 1
    for count in dict.fromkeys((unset, required)):
        if 0 < count <= 8:
            try:
                cls(*[1.0] * count)
            except Exception:
                continue
            return ", ".join(["1.0"] * count)
    return None


#: "Expected 1, 4, or 5 arguments; received 0.": a ``(*args)`` constructor's counts.
ARGUMENTS_EXPECTED = re.compile(r"[Ee]xpected ((?:\d+,? (?:or )?)*\d+) (?:positional )?arguments?")


def _vararg_names(entry_ref):
    """The arguments a ``(*args)`` class's docstring names, in order (``A, B, C, D : array_like``), or []."""
    cls = _entry_object(entry_ref)
    if cls is None:
        return []
    names = []
    for entry in docstring_parameters(cls, getattr(cls, "__init__", None)):
        names += [n.strip() for n in entry.split(",") if n.strip().isidentifier()]
    return list(dict.fromkeys(names))


def _infer_config_once(target, call, config_dir, fmi_version, starts, setup, kind, create, probe, converts,
                       examples=True):
    """The inference itself, inside ``config_dir`` (see :func:`infer_config`).

    Imports the model, picks the entry, infers a function or class model, adds data files,
    globals and requirements, renames clashing names, adapts the types to the FMI version, and
    finally checks that every ``--start`` and ``--convert`` name was used.

    First with what the package's own examples show (:func:`_with_usage`), for the options not given;
    when that fails (or its probe does), without them.

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
    :param examples: whether to read the package's examples.
    :type examples: bool
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
    if examples and kind != "function":
        found = _with_usage(entry, module, path, call, create, setup, starts, probe)
        if found is not None:
            usage, taken, options = found
            try:
                data, comments = _infer_config_once(target, options["call"], config_dir, fmi_version, options["starts"],
                                               options["setup"], kind, options["create"], probe, converts,
                                               examples=False)
            except InterfaceError:
                pass
            else:
                if not (probe and (_probe_failed(comments) or not data.get("outputs"))):
                    _note_usage(usage, taken, options, call, create, setup, data, comments)
                    return data, comments

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
    if inspect.isclass(entry) and kind != "function" and not call and not create and _results_only(entry):
        kind = data["model"]["kind"] = "function"
        comments[("model", "kind")] = (f"{entry.__name__} has no methods to run: building it computes everything "
                                       "(its properties are the results), so it's built with the inputs every step")
    options = {"arrays": arrays, "starts": starts, "probe": probe, "converts": dict(converts or {})}
    if inspect.isclass(entry) and kind != "function":
        _infer_class(entry, call, data, comments, create=create, **options)
    elif call and kind != "function":
        # a function with --call is a factory: called once, --call runs on what it returns
        if create:
            raise InterfaceError("--create only applies to classes; a function with --call is already a factory")
        _infer_class(None, call, data, comments, factory=entry, **options)
    elif probe and kind != "function" and not create and (left := _returns_callable(entry, data, starts)) is not None:
        # the --start names it doesn't declare are the arguments of the model it returns (silero's load_silero_vad,
        # transformers' pipeline(task, model, **kwargs)(inputs)); with none, it returns a function (sympy's lambdify)
        _infer_class(None, "__call__", data, comments, factory=entry, **options)
        data["model"]["call"] = "__call__"
        comments[("model", "call")] = (
            f"{entry_name}() returns the model: {', '.join(left)} are arguments of calling it, not of the function"
            if left else f"{entry_name}() returns a function: it's called on every step")
    elif probe and kind != "function" and not create and (found := _returned_method(entry, data, starts)):
        # the --start names it doesn't take are arguments of a method of what it returns (joblib's load: predict(X))
        method, left, how = found
        _infer_class(None, method, data, comments, factory=entry, **options)
        comments[("model", "call")] = (f"{entry_name}() returns the model: {', '.join(left)} are arguments of its "
                                       f"{method}(), {how}")
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
    step = data["model"].get("call")
    _add_docstring_units(data, comments, entry,
                         getattr(entry, "__init__", None) if inspect.isclass(entry) else None,
                         getattr(entry, create, None) if inspect.isclass(entry) and create else None,
                         getattr(entry, step, None) if inspect.isclass(entry) and step and ":" not in step else None)
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


def _probe_failed(comments):
    """Whether the probe failed to build the object or to call the step (its comment on the outputs says so)."""
    note = comments.get(("outputs", None))
    return isinstance(note, str) and note.startswith(("probe call failed", "probe construction failed"))


def _with_usage(entry, module, path, call, create, setup, starts, probe):
    """The options a use of the model in its package's examples adds to those given (:func:`examples.find_usage`).

    Only what wasn't given: the step method and classmethod, the set-up calls (when the step is the
    example's), the arguments' values. Values computed by a call (``numpy.array(...)``) only with
    ``--probe``, and only those that compute here.

    :return: ``(usage, {name: value taken}, {"call", "create", "setup", "starts"})``, or None when the
        examples add nothing.
    :rtype: tuple or None
    """
    from fmugen import examples
    from fmugen.templates.fmugen_runtime import resolve_constant
    try:
        usage = examples.find_usage(entry, module, Path(path).resolve() if path else None)
        values = examples.find_values(entry, module, Path(path).resolve() if path else None)
    except Exception:
        return None
    if usage is None and not values:
        return None
    is_class = inspect.isclass(entry)
    usage = usage or examples.Usage("")
    starts = dict(starts or {})
    chosen = call is not None and call == usage.call   # --call names the example's step: its arguments are wanted
    if usage.call in STEP_LIKE and not chosen \
            and any(callable(getattr(entry, m, None)) for m in STEP_LIKE[:STEP_LIKE.index(usage.call)]):
        usage.call, usage.step, usage.setup = None, {}, []   # pastas' solve() in its README: simulate() runs it
    if usage.call and not chosen and _named(usage.call, SETUP_IF_STEP + ("optimize", "calibrate")) \
            and any(callable(getattr(entry, m, None)) for m in STEP_LIKE):
        usage.call, usage.step, usage.setup = None, {}, []   # training (GPy's optimize()): predict() runs it
    new_call = call or (usage.call if is_class else None)
    new_create = create or (usage.create if is_class else None)
    taken = {}

    def take(args):
        for name, value in args.items():
            if name in starts or name in taken or any(k.startswith(f"{name}.") for k in starts) or _time_source(name):
                continue   # a time argument: the simulation's, not the example's
            if isinstance(value, dict) and "expr" in value:
                value = _computed(value["expr"]) if probe else None
                if value is None:
                    continue
            elif isinstance(value, dict):
                if not probe:
                    continue
                try:
                    resolve_constant(value)
                except Exception:   # arithmetic in its arguments, which a config call can't hold: its numbers
                    module, _, rest = str(value.get("call", "")).partition(":")
                    value = _computed(f"{module}.{rest}") if rest else None
                    if value is None:
                        continue
            taken[name] = value

    if not is_class or (new_create or None) == usage.create:
        take(usage.init)
    same_step = is_class and usage.call is not None and new_call == usage.call
    if same_step:
        take(usage.step)
    elif is_class and new_call:   # another step: its arguments named as the example's (py-pde's state)
        take({n: v for n, v in usage.step.items() if n in _required(getattr(entry, new_call, None))})
    # an argument no use of the model shows: as the examples pass it to anything else (openap's FuelFlow("A320"), ac)
    needed = _required(getattr(entry, new_create, None) if is_class and new_create else entry)
    if is_class and new_call:
        needed += _required(getattr(entry, new_call, None))
    for name in needed:
        if name in values and name not in taken:
            before = len(taken)
            take({name: values[name][0]})
            if len(taken) > before:
                usage.sources[name] = values[name][1]
    new_setup = list(setup) if setup else (list(usage.setup) if same_step and usage.setup else None)
    if not taken and new_call == call and new_create == create and new_setup == (list(setup) if setup else None):
        return None
    options = {"call": new_call, "create": new_create, "setup": new_setup, "starts": {**starts, **taken}}
    return usage, taken, options


#: The most numbers an example's computed value may have to become a start value (:func:`_computed`).
MAX_COMPUTED = 2000


def _computed(text):
    """An example's expression over imports (``numpy.stack((3 * numpy.exp(t), ...))``) computed, as a start value:
    a number or a (nested) list of numbers, rounded to 6 significant digits; or None (other results, too many
    numbers, or it fails). Only with ``--probe``, like the calls of examples."""
    names = {}
    for top in set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)\.", text)):   # numpy of numpy.exp
        try:
            names[top] = importlib.import_module(top)
        except Exception:
            continue
    try:
        value = eval(compile(text, "<example>", "eval"), {"__builtins__": {}}, names)
        value = value.tolist() if hasattr(value, "tolist") else value
    except Exception:
        return None
    count = 0

    def numbers(v):
        nonlocal count
        if isinstance(v, bool) or not isinstance(v, (int, float, list, tuple)):
            return None
        if isinstance(v, (list, tuple)):
            items = [numbers(x) for x in v]
            return None if not items or any(x is None for x in items) else items
        count += 1
        return float(f"{v:.6g}") if isinstance(v, float) else v

    result = numbers(value)
    return result if result is not None and count <= MAX_COMPUTED else None


def _required(fn):
    """The names of ``fn``'s arguments with no default (not ``self``, ``*args`` or ``**kwargs``), or []."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p.name for p in params if p.default is p.empty and p.name not in ("self", "cls")
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def _note_usage(usage, taken, options, call, create, setup, data, comments):
    """Comment what came from the package's examples (:func:`_with_usage`), to check."""
    where = usage.source
    if options["call"] and not call:
        comments[("model", "call")] = f"{options['call']}() as in {where}"
    if options["create"] and not create:
        comments[("model", "create")] = f"{data['model'].get('entry', '')}.{options['create']}(...) builds it, as in {where}"
    if options["setup"] and not setup:
        dropped = f"; left out: {', '.join(f'{n}()' for n in usage.dropped)} (arguments that aren't literals)" \
                if usage.dropped else ""
        comments[("model", "setup")] = f"as in {where}: check it{dropped}"
    for name in taken:
        where = usage.sources.get(name, usage.source)
        for section in ("parameters", "inputs"):
            if name in data.get(section, {}):
                comments[(section, name)] = f"start from the example in {where}: check it"
        for table in ("constants", "call_constants"):
            if name in data["model"].get(table, {}):
                comments[("constant", f"{table}.{name}")] = f"from the example in {where}: check it"


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
    # a constructor constant too (CFTOC's --const x0): the method's x0 is then given as <method>_x0
    parameters = {**data["model"].get("constants", {}), **data.get("parameters", {})}
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

    Found as string start values (a file name argument), string constants (``--const``), or string
    literals in the entry file (parsed with :func:`ast.parse`): copied with ``sources``, and
    ``cwd = "."`` so the model finds them inside the FMU.

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
    texts += [c["python"] if isinstance(c, dict) else c for table in ("constants", "call_constants")
              for c in data["model"].get(table, {}).values()
              if isinstance(c, str) or (isinstance(c, dict) and "python" in c)]
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
        files += sorted(f for f in source.rglob("*.py") if not _in_venv(f, source)) if source.is_dir() \
                else [source] if source.suffix == ".py" else []
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
    _arguments(fn, "inputs", data, comments, arrays=arrays, starts=starts, probe=probe)
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

#: How many of the ranked step methods the probe tries, when none was picked surely.
PROBED_CANDIDATES = 3
#: Classmethods that build an object without arguments from the user, tried when the constructor fails
#: in the probe (with ``from_`` ones taking no required argument).
FACTORY_NAMES = ("default", "defaults", "create", "example", "sample", "demo", "standard", "new")


def _infer_class(cls, call, data, comments, arrays=False, starts=None, create=None, factory=None, probe=False,
                 converts=None):
    """Infer a class model (:func:`_infer_class_once`); with ``--probe``, try other choices when the first fails.

    When the step method was only guessed (:func:`_rank_methods`) and the probe's call fails or gives no
    outputs, the next candidates are probed, and the first that works and gives outputs is kept. When
    building the object fails, the classmethods that build one with no arguments from the user
    (:data:`FACTORY_NAMES`) are tried as ``create``.

    :raises InterfaceError: see :func:`_infer_class_once`.
    """
    if not probe or factory or cls is None:
        _infer_class_once(cls, call, data, comments, arrays, starts, create, factory, probe, converts)
        return
    base = copy.deepcopy((data, comments))

    def attempt(method, maker):
        d, c = copy.deepcopy(base)
        _infer_class_once(cls, method, d, c, arrays, starts, maker, None, True, converts)
        return d, c

    best = attempt(call, create)
    if not create and _construction_failed(best[1]):
        for maker in _factories(cls):
            try:
                tried = attempt(call, maker)
            except InterfaceError:
                continue
            if not _construction_failed(tried[1]):
                tried[1][("model", "create")] = (f"{cls.__name__}.{maker}() builds it: {cls.__name__}(...) failed in "
                                                 "the probe with the made-up values")
                best, create = tried, maker
                break
    picked = best[0]["model"].get("call", "__call__")
    why = best[1].get(("model", "call"))
    note = best[1].get(("outputs", None))
    # a step that needs a set-up (osqp's solve() before setup()), or fails on an attribute the object doesn't have
    # yet (tabnet's predict() before fit(): no network): only a run-like method working may replace it
    needs_set_up = isinstance(note, str) and ("seems to need" in note or f"'{cls.__name__}' object has no attribute" in note)
    if not call and isinstance(why, str) and why.startswith(f"{picked}() guessed") and not _works(best):
        constructor = getattr(cls, create) if create else cls.__init__
        left = _leftover_starts(constructor, starts)
        ranked = _rank_methods(cls, starts, constructor, _methods_taking(cls, left) if left else [])
        failed = [picked]
        others = [m for m in ranked if m != picked and not _named(m, NOT_STEP)   # pyproj's to_json()
                  and not isinstance(inspect.getattr_static(cls, m, None), (staticmethod, classmethod))
                  and (not needs_set_up or m in STEP_LIKE)]   # statsforecast's forecast(), not tabnet's predict_func()
        for method in others[:PROBED_CANDIDATES - 1]:
            try:
                tried = attempt(method, create)
            except InterfaceError:
                continue
            if _works(tried):
                tried[1][("model", "call")] = (
                    f"{method}() picked by probing: {', '.join(f'{m}()' for m in failed)} failed or gave no outputs "
                    f"(candidates, best first: {', '.join(ranked[:8])}); pass --call METHOD to choose another")
                if ("model", "create") in best[1] and create:
                    tried[1][("model", "create")] = best[1][("model", "create")]
                best = tried
                break
            failed.append(method)
    data.clear()
    data.update(best[0])
    comments.clear()
    comments.update(best[1])


def _works(result):
    """Whether a probe's config has outputs and its step worked."""
    data, comments = result
    return bool(data.get("outputs")) and not _probe_failed(comments)


def _construction_failed(comments):
    """Whether the probe failed to build the object."""
    note = comments.get(("outputs", None))
    return isinstance(note, str) and note.startswith("probe construction failed")


def _factories(cls):
    """Classmethods of ``cls`` that may build it with no arguments from the user: :data:`FACTORY_NAMES` ones and
    ``from_...`` ones, that have no required argument."""
    found = []
    for name in dir(cls):
        if not isinstance(inspect.getattr_static(cls, name, None), classmethod):
            continue
        if name not in FACTORY_NAMES and not name.startswith("from_"):
            continue
        try:
            params = inspect.signature(getattr(cls, name)).parameters.values()
        except (TypeError, ValueError):
            continue
        if all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
            found.append(name)
    return sorted(found, key=lambda n: (n not in FACTORY_NAMES, n))[:4]


def _infer_class_once(cls, call, data, comments, arrays=False, starts=None, create=None, factory=None, probe=False,
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
    if not create and not factory and cls is not None:
        create, why = _pick_create(cls, starts)
        if create:
            comments[("model", "create")] = why
    if call:
        method_name = call
    else:
        method_name, why = _pick_method(cls, starts, getattr(cls, create, None) if create else cls.__init__)
        if why:
            comments[("model", "call")] = why
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
               arrays=arrays, starts=init_starts, positional=False, by_keyword=by_keyword, probe=probe)
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
        _arguments(_forwarded(getattr(cls, method_name), method_name, cls), "inputs", data, comments,
                   skip_self=not isinstance(method, (staticmethod, classmethod)) and not on_class,
                   constants_key="call_constants",
                   arrays=arrays, starts=call_starts)   # without --probe: suggestions only
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






def _pick_method(cls, starts=None, constructor=None):
    """The step method of a class: the first of :data:`CALL_NAMES` it has (``update`` only when it has no
    :data:`STEP_LIKE` method: osqp's ``update`` changes settings, ``solve`` runs it), else the only method the
    ``--start`` names fit (:func:`_methods_taking`), else its only public method, else the best of
    :func:`_rank_methods`: asked at a prompt in a terminal (:func:`_ask_method`), else picked with a warning
    listing the candidates.

    The ``--start`` names the constructor doesn't take must be arguments of the method, and cover its
    required ones: JOS3's ``times`` and ``dtime`` fit only ``simulate``.

    :param cls: the class.
    :type cls: type
    :param starts: the ``--start`` / ``--const`` values.
    :type starts: dict or None
    :param constructor: the constructor (or ``--create`` classmethod), for the names it takes.
    :return: ``(method, comment)``; the comment says why, when the ``--start`` names chose it.
    :rtype: tuple[str, str or None]
    :raises InterfaceError: the class has no public method.
    """
    has_step = any(callable(getattr(cls, m, None)) for m in STEP_LIKE)
    left = _leftover_starts(constructor, starts)
    fitting = _methods_taking(cls, left) if left else []
    for name in CALL_NAMES:
        if name == "update" and has_step:
            continue
        if fitting and callable(getattr(cls, name, None)) and not _fits(getattr(cls, name), left):
            continue   # nengo's step() takes no steps=10; run_steps(steps) does
        if name == "__call__":
            if any("__call__" in vars(k) for k in cls.__mro__[:-1]):
                return name, None
        elif callable(getattr(cls, name, None)):
            return name, None
    if any("__next__" in vars(k) for k in cls.__mro__[:-1]) and not has_step:
        return "__next__", (f"__next__(): {cls.__name__} is an iterator, each next() a step "
                            "(pass --call METHOD to choose another)")
    public = [n for n, v in vars(cls).items() if not n.startswith("_") and inspect.isfunction(v)]
    if len(fitting) == 1:   # before the only public method: skfem's MeshTri defines element_finder, refined is inherited
        return fitting[0], (f"{fitting[0]}() from the --start names: {', '.join(left)} are its arguments, not the "
                            "constructor's, and no other method takes them")
    picked = _method_from_starts(cls, starts, constructor)
    if picked:
        return picked
    if len(public) == 1:
        return public[0], None
    ranked = _rank_methods(cls, starts, constructor, fitting)
    if not ranked:
        raise InterfaceError(f"cannot tell which method of {cls.__name__} runs a step (no public methods); "
                             "pass --call METHOD")
    shown = ", ".join(ranked[:8]) + (", …" if len(ranked) > 8 else "")
    chosen = _ask_method(cls, ranked)
    if chosen:
        return chosen, f"{chosen}() chosen at the prompt (candidates, best first: {shown})"
    sys.stderr.write(f"fmugen: warning: it's not clear which method of {cls.__name__} runs a step; picked "
                     f"{ranked[0]}() (candidates, best first: {shown}); pass --call METHOD to choose another\n")
    return ranked[0], (f"{ranked[0]}() guessed: it's not clear which method runs a step (candidates, best first: "
                       f"{shown}); pass --call METHOD to choose another")


def _ask_method(cls, ranked):
    """Ask which method runs a step, when ``init`` runs in a terminal: the candidates numbered, best first.

    Enter takes the first; a number or a method name (any public one) picks another.

    :param cls: the class.
    :type cls: type
    :param ranked: the candidates, best first (:func:`_rank_methods`).
    :type ranked: list[str]
    :return: the method, or None when not in a terminal (stdin or stderr redirected) or at end of input.
    :rtype: str or None
    """
    if not (sys.stdin and sys.stdin.isatty() and sys.stderr.isatty()):
        return None
    shown = ranked[:9]
    lines = [f"Which method of {cls.__name__} runs a step? (pass --call METHOD to skip this)"]
    lines += [f"  {i}. {m}" for i, m in enumerate(shown, 1)]
    if len(ranked) > len(shown):
        lines.append(f"  (or type another: {', '.join(ranked[len(shown):])})")
    sys.stderr.write("\n".join(lines) + "\n")
    while True:
        sys.stderr.write(f"Method [1: {shown[0]}]: ")
        sys.stderr.flush()
        try:
            answer = input().strip()
        except (EOFError, KeyboardInterrupt):
            sys.stderr.write("\n")
            raise InterfaceError(f"no step method chosen for {cls.__name__}; pass --call METHOD") from None
        if not answer:
            return shown[0]
        if answer.isdigit() and 1 <= int(answer) <= len(shown):
            return shown[int(answer) - 1]
        if answer in ranked:
            return answer
        sys.stderr.write(f"  not one of the candidates: type 1–{len(shown)} or a method name\n")


#: Method names that usually run a model, best first: picked when nothing else tells the step method
#: (the --call of the tested models: sklearn-like predict, solve, run, forecast, ...).
STEP_LIKE = ("predict", "forecast", "simulate", "solve", "run", "run_steps", "compute", "propagate", "forward",
             "evaluate", "transcribe", "encode", "calculate", "process", "apply")
#: Method names (or prefixes, ending in _) that set a model up when the class also has a :data:`STEP_LIKE`
#: method (sklearn's fit before predict, osqp's update before solve); without one, fit and train run the
#: model (lifelines' KaplanMeierFitter.fit), after the :data:`STEP_LIKE` names.
SETUP_IF_STEP = ("fit", "train", "update", "fit_", "train_")
#: Method names (or prefixes, ending in _) that set a model up, build it (anastruct's add_element), save, show,
#: copy or close it: a step last.
NOT_STEP = ("close", "stop", "shutdown", "terminate", "release", "dispose", "disconnect", "cleanup", "clear", "reset", "load", "save", "copy", "plot", "show", "summary", "describe", "score",
            "get_params", "set_params", "to_", "from_", "set_", "plot_", "save_", "load_", "print_", "add_", "remove_")


def _rank_methods(cls, starts, constructor, fitting=()):
    """Public methods of ``cls`` that may run a step, most likely first, for when nothing picks one surely.

    Ranked by (only when the constructor doesn't take some names): whether it takes any of those names
    and they cover its required arguments, else whether it takes some of them (or its signature can't be
    read), else none (a partial set of names still points at the step: ``solve(a, b)`` given only ``a``
    before ``summary()``); whether they cover its required arguments; whether it fits them (:func:`_methods_taking`), whether its name is a
    :data:`NOT_STEP` one (or a :data:`SETUP_IF_STEP` one, when a :data:`STEP_LIKE` method is there too),
    its place in :data:`STEP_LIKE` (a name starting with one of them, ``solve_energy``, right after it;
    fit and train after them, when none is there), whether it's a
    variant of another candidate (:func:`_variant_of`: ``certain_timezone_at`` of ``timezone_at``),
    how many of the names it takes, whether the class itself defines it, and its name.

    :param cls: the class.
    :type cls: type
    :param starts: the ``--start`` / ``--const`` values.
    :type starts: dict or None
    :param constructor: the constructor (or ``--create`` classmethod).
    :param fitting: the methods the leftover names fit.
    :rtype: list[str]
    """
    leftover = {n.partition(".")[0] for n in _leftover_starts(constructor, starts)}
    left = leftover or {n.partition(".")[0] for n in starts or {}}
    names = [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m, None))
             and not isinstance(inspect.getattr_static(cls, m, None), (property, type, classmethod))]
    has_step = any(m in STEP_LIKE for m in names)
    keys = {}
    for m in names:
        try:
            params = [p for p in inspect.signature(getattr(cls, m)).parameters.values()
                      if p.name not in ("self", "cls") and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
            required = {p.name for p in params if p.default is p.empty}
            taken = len(left & {p.name for p in params})
            covered = required <= left
            mine = len(leftover & {p.name for p in params})
            tier = 0 if mine and covered else 1 if mine else 2
        except (TypeError, ValueError):   # a C method: may take anything
            taken, covered, tier = 0, True, 1
        not_step = _named(m, NOT_STEP) or has_step and _named(m, SETUP_IF_STEP)
        first = m.split("_")[0]
        place = (STEP_LIKE.index(m) if m in STEP_LIKE else STEP_LIKE.index(first) + 0.5 if first in STEP_LIKE
                 else len(STEP_LIKE) if m in ("fit", "train") else len(STEP_LIKE) + 1)
        keys[m] = (tier if leftover else 0, bool(left) and not covered, m not in fitting, not_step, place,
                   any(_variant_of(m, n) for n in names), -taken, m not in vars(cls), m)
    return sorted(keys, key=keys.get)


def _named(name, names):
    """Whether ``name`` is one of ``names``, or starts with one of those ending in ``_`` (``to_``)."""
    return name in names or any(name.startswith(p) for p in names if p.endswith("_"))


def _variant_of(name, other):
    """Whether method ``name`` is a qualified variant of method ``other``: ``other`` as whole words of it,
    maybe plural (``certain_timezone_at`` of ``timezone_at``, ``cumulative_decays`` of ``decay``).

    :rtype: bool
    """
    if name == other:
        return False
    words, part = name.split("_"), other.split("_")
    for plural in ("", "s"):
        want = part[:-1] + [part[-1] + plural]
        if any(words[i:i + len(want)] == want for i in range(len(words) - len(want) + 1)):
            return True
    return False


#: A numpydoc section header: the name, then a line of dashes.
_SECTION = re.compile(r"^(\w[\w ]*)\n-{3,}\n", re.M)
#: A numpydoc entry: ``name : type`` at the section's indentation (or ``name`` alone).
_ENTRY = re.compile(r"^(\w+)\s*(?::\s*(.*))?$")
#: A unit in square brackets: ``Temperature, [K]``.
_BRACKET_UNIT = re.compile(r"\[([^\[\]]+)\]")


def docstring_units(*objects):
    """Units of arguments and results from numpydoc docstrings: ``T : float`` described as ``Temperature, [K]``.

    The first bracketed text of an entry (its type line, then its description) that :func:`units.parse`
    recognises is its unit; ``[-]`` and other notes are not. fluids, ht, chemicals and thermo write
    their docstrings this way.

    :param objects: functions, methods or classes whose docstrings are read (later ones don't override).
    :return: ``({argument: unit}, {result: unit})`` from the Parameters and Returns sections.
    :rtype: tuple[dict, dict]
    """
    from fmugen import units
    found = ({}, {})
    for obj in objects:
        doc = inspect.getdoc(obj) if obj is not None else None
        if not doc:
            continue
        headers = list(_SECTION.finditer(doc))
        for i, header in enumerate(headers):
            kind = {"Parameters": 0, "Other Parameters": 0, "Returns": 1}.get(header.group(1).strip())
            if kind is None:
                continue
            body = doc[header.end():headers[i + 1].start() if i + 1 < len(headers) else len(doc)]
            entries, name = {}, None
            for line in body.splitlines():
                entry = _ENTRY.match(line) if line and not line[0].isspace() else None
                if entry:   # "name : type"; its description follows, indented
                    name = entry.group(1)
                    entries[name] = [entry.group(2) or ""]
                elif name:
                    entries[name].append(line.strip())
            for name, text in entries.items():
                if name in found[kind]:
                    continue
                unit = next((u.strip() for u in _BRACKET_UNIT.findall(" ".join(text))   # [1]: a reference, not a unit
                             if re.search(r"[^\W\d_]", u) and units.parse(u.strip()) is not None), None)
                if unit:
                    found[kind][name] = unit
    return found


#: A Google-style entry: ``name (type): text``.
_GOOGLE_ENTRY = re.compile(r"^\s+(\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")
#: A Sphinx field: ``:param name: text`` (or ``:param type name:``).
_SPHINX_PARAM = re.compile(r"^:param\s+(?:[\w.\[\], ]+\s+)?(\w+)\s*:\s*(.*)$")


def docstring_parameters(*objects):
    """The text describing each argument in numpydoc, Google or Sphinx docstrings: its type line and description.

    :param objects: functions, methods or classes whose docstrings are read (later ones don't override).
    :return: ``{argument: text}``.
    :rtype: dict
    """
    found = {}
    for obj in objects:
        doc = inspect.getdoc(obj) if obj is not None else None
        if not doc:
            continue
        entries, name = {}, None
        headers = list(_SECTION.finditer(doc))
        for i, header in enumerate(headers):   # numpydoc
            if header.group(1).strip() not in ("Parameters", "Other Parameters"):
                continue
            body = doc[header.end():headers[i + 1].start() if i + 1 < len(headers) else len(doc)]
            for line in body.splitlines():
                entry = _ENTRY.match(line) if line and not line[0].isspace() else None
                if entry:
                    name = entry.group(1)
                    entries.setdefault(name, []).append(entry.group(2) or "")
                elif name:
                    entries[name].append(line.strip())
        section, name = False, None
        for line in doc.splitlines():   # Google and Sphinx
            stripped = line.strip()
            sphinx = _SPHINX_PARAM.match(stripped)
            if sphinx:
                name = sphinx.group(1)
                entries.setdefault(name, []).append(sphinx.group(2))
                continue
            if stripped in ("Args:", "Arguments:", "Parameters:", "Keyword Args:", "Keyword Arguments:"):
                section, name = True, None
                continue
            if section and line and not line[0].isspace():
                section = False
            google = _GOOGLE_ENTRY.match(line) if section else None
            if google and len(line) - len(line.lstrip()) <= 4:
                name = google.group(1)
                entries.setdefault(name, []).append(f"{google.group(2) or ''} {google.group(3)}")
            elif name and stripped and not stripped.startswith(":") and (section or not _ENTRY.match(line)):
                entries[name].append(stripped)
            elif not stripped and not section:
                name = None
        for name, text in entries.items():
            found.setdefault(name, " ".join(text))
    return found


#: Words of a numpydoc type line that mean an array.
_ARRAY_WORDS = re.compile(r"\b(array_like|array-like|ndarray|array|arrays|matrix|tensor|sequence of|list of|"
                          r"iterable of|vector)\b", re.I)
#: A shape in a numpydoc entry: ``shape (4,)``, ``of shape (n_samples, n_features)``, ``shape=(n, 3)``.
_SHAPE = re.compile(r"shape\s*(?:of\s*|=\s*|:\s*)?[(\[]([^)\]]*)[)\]]", re.I)
#: The size made up for a dimension a docstring names but doesn't fix (``n_samples``).
MADE_UP_SIZE = 3


def docstring_arrays(fn):
    """Arguments a numpydoc docstring describes as arrays, with their shapes when it gives them.

    ``q : array_like, shape (4,)`` gives ``[4]``; ``X : ndarray of shape (n_samples, n_features)``
    gives ``[3, 3]`` (named dimensions are :data:`MADE_UP_SIZE`); ``x : array_like`` gives ``None``.
    For a constructor, the class's docstring is read too (numpydoc puts its parameters there).

    :param fn: a function, method or constructor.
    :return: ``{argument: shape or None}``.
    :rtype: dict
    """
    docs = [inspect.getdoc(fn)]
    if getattr(fn, "__name__", "") == "__init__":
        owner = sys.modules.get(getattr(fn, "__module__", ""), None)
        for part in getattr(fn, "__qualname__", "").split(".")[:-1]:
            owner = getattr(owner, part, None)
        docs.append(inspect.getdoc(owner) if isinstance(owner, type) else None)
    found = {}
    for doc in docs:
        if not doc:
            continue
        headers = list(_SECTION.finditer(doc))
        for i, header in enumerate(headers):
            if header.group(1).strip() not in ("Parameters", "Other Parameters"):
                continue
            body = doc[header.end():headers[i + 1].start() if i + 1 < len(headers) else len(doc)]
            entries, name = {}, None
            for line in body.splitlines():
                entry = _ENTRY.match(line) if line and not line[0].isspace() else None
                if entry:
                    name = entry.group(1)
                    entries[name] = [entry.group(2) or "", ""]
                elif name:
                    entries[name][1] += " " + line.strip()
            for name, (kind, text) in entries.items():
                if name in found or not _ARRAY_WORDS.search(kind):
                    continue   # only the type line says what it is: "a list of options" in a description doesn't
                shape = _SHAPE.search(f"{kind} {text}")
                dims = [d.strip() for d in shape.group(1).split(",") if d.strip()] if shape else []
                found[name] = [int(d) if d.isdigit() else MADE_UP_SIZE for d in dims] or None
    return found


def _add_docstring_units(data, comments, *objects):
    """Give variables without a unit the one their docstring states (:func:`docstring_units`), with a comment.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text``; updated in place.
    :type comments: dict
    :param objects: the entry and the callables whose docstrings describe it.
    """
    arguments, results = docstring_units(*objects)
    if not arguments and not results:
        return
    single = next(iter(results.values())) if len(results) == 1 else None
    for section in ("parameters", "structural_parameters", "inputs", "states", "outputs", "locals"):
        for name, info in data.get(section, {}).items():
            if "unit" in info or info.get("type") in ("Boolean", "String", "Binary") or "enum" in info \
                    or isinstance(info.get("start"), (bool, str)):
                continue   # only numbers have units (pythermalcomfort's limit_inputs mentions °C)
            if section in ("outputs", "locals"):
                source = str(info.get("from", f"return:{name}"))
                key = source.partition(":")[2].split(".")[-1] if ":" in source else name
                unit = results.get(name) or results.get(key) or (single if source == "return" else None)
            else:
                target = str(info.get("to", "")).partition(":")[2].split(".")[0].split("[")[0] or name
                unit = arguments.get(name) or arguments.get(target)
            if unit:
                info["unit"] = unit
                note = comments.get((section, name))
                comments[(section, name)] = f"{note}; unit from the docstring" if note else "unit from the docstring"


def _results_only(cls):
    """Whether a class has no method a step could run: no ``__call__``, no public instance method (inherited
    or C ones too), only properties, attributes, classmethods and staticmethods (ambiance's ``Atmosphere(h)``).

    :param cls: the class.
    :type cls: type
    :rtype: bool
    """
    if any("__call__" in vars(k) or "__next__" in vars(k) for k in cls.__mro__[:-1]):
        return False
    for name in dir(cls):
        if name.startswith("_"):
            continue
        raw = inspect.getattr_static(cls, name, None)
        if isinstance(raw, (property, classmethod, staticmethod, type)):
            continue
        if callable(raw):   # a Python function or a C method
            return False
    return True


def _pick_create(cls, starts):
    """The classmethod that builds the object, when the constructor can't take the ``--start`` / ``--const`` names.

    The constructor can't when it doesn't declare one of them (and takes no ``**kwargs``), or when it
    requires an argument they don't give. Then the only public classmethod whose required arguments
    are among them, and that takes some of them, builds it: stable-baselines3's ``PPO.load(path)``,
    chronos' ``from_pretrained(pretrained_model_name_or_path)``. Checked against the recorded --create
    of 79 tested classes: 7 found, none wrong.

    :param cls: the class.
    :type cls: type
    :param starts: the ``--start`` / ``--const`` values.
    :type starts: dict or None
    :return: ``(classmethod name, comment)``, or ``(None, None)``.
    :rtype: tuple
    """
    if not starts:
        return None, None
    names = {n.partition(".")[0] for n in starts}
    try:
        init = inspect.signature(cls.__init__).parameters if cls.__init__ is not object.__init__ else {}
    except (TypeError, ValueError):
        return None, None
    declared = {n for n, p in init.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and n != "self"}
    takes_any = any(p.kind is p.VAR_KEYWORD for p in init.values())
    # names a public method takes, or a method's copy of a constructor argument (calculate_optimal_actuation_x0),
    # are for the step: they don't say the constructor can't build it
    for_steps = set()
    for m in dir(cls):
        if m.startswith("_") or isinstance(inspect.getattr_static(cls, m, None), (classmethod, staticmethod, property)):
            continue
        try:
            for_steps |= set(inspect.signature(getattr(cls, m)).parameters)
        except (TypeError, ValueError):
            continue
        for_steps |= {n for n in names if n.startswith(f"{m.strip('_')}_")}
    blocked = set() if takes_any else names - declared - for_steps
    missing = {n for n in declared if init[n].default is init[n].empty} - names
    if not blocked and not missing:
        return None, None
    fits = []
    for m in dir(cls):
        if m.startswith("_") or not isinstance(inspect.getattr_static(cls, m, None), classmethod):
            continue
        try:
            params = inspect.signature(getattr(cls, m)).parameters.values()
        except (TypeError, ValueError):
            continue
        own = {p.name for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        kwargs = any(p.kind is p.VAR_KEYWORD for p in params)
        required = {p.name for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.default is p.empty}
        if required <= names and (names & own or kwargs) and (not blocked or blocked & own or kwargs):
            fits.append(m)
    if len(fits) == 1:
        why = (f"{', '.join(sorted(blocked))} not arguments of {cls.__name__}()" if blocked
               else f"{cls.__name__}() requires {', '.join(sorted(missing))}")
        return fits[0], f"{cls.__name__}.{fits[0]}(...) builds it: {why}, and it's the only classmethod that fits"
    return None, None


def _method_from_starts(cls, starts, constructor):
    """The step method the ``--start`` / ``--const`` names point at, when the constructor may take some of them too.

    The names the constructor doesn't require are left for the step. First, the only method whose
    required arguments they cover, that takes at least one of them, and that leaves out only names
    the constructor also takes (ahrs' ``Madgwick(gyr=None, acc=None, **kwargs)``: q, gyr and acc fit
    ``updateIMU(q, gyr, acc)``; ``updateMARG`` also requires mag). Else, when every public method has
    a readable signature, the one taking the most of them, if there's only one (pyquaternion's
    ``rotate(vector)``). Checked against the recorded --call of 81 tested models: no wrong pick.

    :param cls: the class.
    :type cls: type
    :param starts: the ``--start`` / ``--const`` values.
    :type starts: dict or None
    :param constructor: the constructor (or ``--create`` classmethod).
    :return: ``(method, comment)``, or None.
    :rtype: tuple or None
    """
    if not starts:
        return None
    try:
        ctor = inspect.signature(constructor).parameters if constructor is not None \
            and constructor is not object.__init__ else {}
    except (TypeError, ValueError):
        return None
    ctor_all = {n for n, p in ctor.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    ctor_required = {n for n, p in ctor.items() if n in ctor_all and p.default is p.empty and n != "self"}
    left = {n.partition(".")[0] for n in starts} - ctor_required
    if not left:
        return None
    methods = {}
    for m in dir(cls):
        if m.startswith("_") or not callable(getattr(cls, m, None)) \
                or isinstance(inspect.getattr_static(cls, m, None), (property, type)):
            continue
        try:
            params = [p for p in inspect.signature(getattr(cls, m)).parameters.values()
                      if p.name not in ("self", "cls") and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        except (TypeError, ValueError):
            params = None   # a C method: its arguments can't be compared
        methods[m] = params
    fits = []
    for m, params in methods.items():
        if params is None:
            continue
        names = {p.name for p in params}
        required = {p.name for p in params if p.default is p.empty}
        if required <= left and left & names and left - names <= ctor_all:
            fits.append(m)
    if len(fits) == 1:
        return fits[0], (f"{fits[0]}() from the --start names: its required arguments are among them, and no other "
                         "method's are")
    if any(params is None for params in methods.values()):
        return None   # a method we can't read might fit as well (sgp4's C method sgp4 next to sgp4_array)
    scored = {m: len(left & {p.name for p in params}) for m, params in methods.items()
              if {p.name for p in params if p.default is p.empty} <= left}
    best = max(scored.values(), default=0)
    top = [m for m, s in scored.items() if s == best]
    if best and len(top) == 1:
        return top[0], (f"{top[0]}() from the --start names: it takes more of them than any other method, and its "
                        "required arguments are among them")
    return None


def _undeclared_starts(fn, starts):
    """The ``--start`` / ``--const`` names ``fn`` doesn't declare, even when it takes ``**kwargs``: for a
    factory, those may be arguments of the model it returns (transformers' ``pipeline(task, model, **kwargs)``
    and its ``__call__(inputs)``).

    :param fn: the factory function.
    :param starts: the values.
    :type starts: dict or None
    :rtype: list[str]
    """
    if not starts:
        return []
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return []
    named = {n for n, p in params.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    return [n for n in starts if n.partition(".")[0] not in named]


def _leftover_starts(fn, starts):
    """The ``--start`` / ``--const`` names ``fn`` doesn't declare (left for the step), or ``[]``.

    ``[]`` when ``fn`` takes ``**kwargs`` or has no signature: then any name may be its own.

    :param fn: the constructor or factory, or None.
    :param starts: the values.
    :type starts: dict or None
    :rtype: list[str]
    """
    if fn is None or not starts:
        return []
    if fn is object.__init__:   # a class without its own constructor takes nothing
        return [n for n in starts]
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return []
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return []
    return [n for n in starts if n.partition(".")[0] not in params]


def _fits(fn, names):
    """Whether ``names`` are all arguments of ``fn`` and cover its required ones.

    :param fn: a function or method.
    :param names: argument names.
    :type names: list[str]
    :rtype: bool
    """
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    except (TypeError, ValueError):
        return False
    declared = {p.name for p in params}
    required = {p.name for p in params if p.default is p.empty}
    wanted = {n.partition(".")[0] for n in names}
    return wanted <= declared and required <= wanted


def _methods_taking(cls, names):
    """Public methods of ``cls`` (inherited too) that ``names`` fit (:func:`_fits`).

    :param cls: the class.
    :type cls: type
    :param names: the ``--start`` names left for the step.
    :type names: list[str]
    :rtype: list[str]
    """
    return [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m, None))
            and _fits(getattr(cls, m), names)]


def _returned_method(fn, data, starts):
    """The method of the object ``fn`` returns that the ``--start`` names it doesn't take fit, for a factory
    function: joblib's ``load(filename)`` and the estimator's ``predict(X)``, tirex's ``load_model`` and its
    ``forecast(context, prediction_length)``.

    Calls ``fn`` once with the values it takes, after the setup calls: only with ``--probe``. The only method
    the names fit (:func:`_methods_taking`), else the one :func:`_method_from_starts` points at, else the first
    of those they fit by :data:`STEP_LIKE`.

    :param fn: the entry function.
    :param data: the config being inferred (for its setup calls).
    :type data: dict
    :param starts: the ``--start`` / ``--const`` values.
    :type starts: dict
    :return: ``(method, names left for it, why)``, or None.
    :rtype: tuple or None
    """
    left = _undeclared_starts(fn, starts)
    if not left:
        return None
    from fmugen.templates.fmugen_runtime import resolve_constant
    try:
        _run_setup(data)
        model = fn(**{n: resolve_constant(v) for n, v in starts.items() if n not in left})
    except Exception:
        return None
    if model is None or inspect.isclass(model) or isinstance(model, (types.FunctionType, types.MethodType,
                                                                      types.ModuleType, *SCALARS)):
        return None
    cls = type(model)
    fitting = _methods_taking(cls, left)
    if len(fitting) == 1:
        return fitting[0], left, "the only method they fit"
    picked = _method_from_starts(cls, {n: starts[n] for n in left}, None)
    if picked:
        return picked[0], left, "the method they point at"
    if fitting:
        def place(m):
            first = m.split("_")[0]
            return (_named(m, NOT_STEP), STEP_LIKE.index(m) if m in STEP_LIKE else
                    STEP_LIKE.index(first) + 0.5 if first in STEP_LIKE else len(STEP_LIKE), m)

        best = min(fitting, key=place)
        return best, left, f"the first of those they fit ({', '.join(sorted(fitting, key=place)[:6])}); pass --call to choose another"
    return None


def _returns_callable(fn, data, starts):
    """Whether ``fn`` is a factory of a callable model: the ``--start`` names it doesn't take fit the
    ``__call__`` of what it returns (silero's ``load_silero_vad()(x, sr)``).

    Calls ``fn`` once with the values it takes, after the setup calls: only with ``--probe``.

    :param fn: the entry function.
    :param data: the config being inferred (for its setup calls).
    :type data: dict
    :param starts: the ``--start`` / ``--const`` values.
    :type starts: dict
    :return: the names left for ``__call__`` (``[]``: none, and it returns a function), or None.
    :rtype: list[str] or None
    """
    left = _undeclared_starts(fn, starts)
    from fmugen.templates.fmugen_runtime import resolve_constant
    try:
        _run_setup(data)
        model = fn(**{n: resolve_constant(v) for n, v in starts.items() if n not in left})
    except Exception:
        return None
    if inspect.isclass(model) or not callable(model):
        return None
    function = isinstance(model, (types.FunctionType, types.MethodType))
    if not left:   # nothing left for it: only a function it makes (lambdify), not any callable object it returns
        return [] if function and model is not fn else None
    call = model if function else type(model).__call__
    if not _fits(call, left) and _only_varargs(call) and callable(getattr(type(model), "forward", None)):
        call = type(model).forward   # a torch module: __call__(*args, **kwargs) runs forward(x, ...)
    return left if _fits(call, left) else None

