"""Argument processing and type/value utilities for interface inference."""
import ast
import dataclasses
import datetime
import enum
import inspect
import types
import json
import keyword
import math
import re
import sys
import textwrap
import typing

from fmugen import static
from fmugen.config import InterfaceError
from fmugen.toml_output import _key, _value

#: Argument names (lower case, without ``_``) taken as ``[time]`` arguments, and their source.
TIME_NAMES = {"t": "time", "time": "time", "currenttime": "time", "tcurrent": "time", "tnow": "time",
              "dt": "step_size", "h": "step_size", "stepsize": "step_size", "timestep": "step_size",
              "until": "end_time", "endtime": "end_time", "tend": "end_time", "tnext": "end_time"}


def _time_source(name):
    """The ``[time]`` source for an argument named like a time (currentTime, step_size, until), or None.

    :param name: the argument name.
    :type name: str
    :rtype: str or None
    """
    return TIME_NAMES.get(name.lower().replace("_", ""))


def _horizon(p):
    """Whether a parameter named ``h`` is a count (a forecast horizon), not a step size.

    ``h`` is a step size in ODE code, but forecasters (neuralforecast's ``predict(h=None)``,
    statsforecast's ``forecast(h)``) use it for the number of values to forecast. It's taken as one
    when it's annotated as an integer (``int``, ``Optional[int]``) or defaults to an integer or None.

    :param p: the parameter.
    :type p: inspect.Parameter
    :rtype: bool
    """
    if p.name != "h":
        return False
    annotation = p.annotation
    if isinstance(annotation, str):
        annotation_ints = re.fullmatch(r"(typing\.)?(Optional\[int\]|int( \| None)?|None \| int)", annotation)
    else:
        annotation_ints = annotation is int or set(typing.get_args(annotation)) in ({int}, {int, type(None)})
    default = p.default
    return bool(annotation_ints) or default is None or isinstance(default, int) and not isinstance(default, bool)

#: Python types that are FMI scalar values.
SCALARS = (bool, int, float, str)

def _dotted_names(text):
    """Dotted names (numpy.linspace) in a Python expression, or ``[]`` if it doesn't parse.

    Parsed with :func:`ast.parse` only, never evaluated; used to find the packages a
    ``call:`` start needs.

    :param text: the expression.
    :type text: str
    :rtype: list[str]
    """
    try:
        tree = ast.parse(text.rstrip().removesuffix(")") or "0", mode="eval")
    except SyntaxError:
        return []
    return [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Name)
            or isinstance(n, ast.Attribute) and isinstance(n.value, (ast.Name, ast.Attribute))]

DATETIME_BY_NAME = ("init", "date-time by name")   # comments key (not rendered): [time] args only named like one
POSITIONAL_STARTS = ("init", "positional starts")   # comments key (not rendered): {--start name: *args position}
START_ERRORS = ("init", "start errors")   # comments key (not rendered): {--start name: why it can't be used}
NO_SIGNATURE = ("init", "no signature")   # comments key (not rendered): callables whose arguments can't be read


def _start_error(comments, name, message):
    """Record why a ``--start`` value can't be used (reported by init).

    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :param name: the ``--start`` name.
    :type name: str
    :param message: the reason.
    :type message: str
    """
    comments.setdefault(START_ERRORS, {})[name] = message


GUESSED = ("init", "guessed")   # comments key (not rendered): outputs typed Real only by default


def _guessed(comments, name):
    """Record an output whose type the code doesn't show: written as Real, which may be wrong.

    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :param name: the output.
    :type name: str
    """
    comments.setdefault(GUESSED, []).append(name)


def guessed_outputs(data, comments):
    """Outputs still in the config whose type is a guess (see :func:`_guessed`).

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :rtype: list[str]
    """
    return [n for n in comments.get(GUESSED, []) if n in data.get("outputs", {})
            and not {"type", "dimensions", "enum"} & set(data["outputs"][n])]


UNFIT = ("init", "unfit")   # comments key (not rendered): {name: why init left it out}
PROBED = ("init", "probed")   # comments key (not rendered): the probe's step call worked


def _unfit(comments, section, name, why, lines):
    """Write an output (or local) the probe showed can't work as it is commented out, with why.

    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :param section: the config section.
    :type section: str
    :param name: the variable.
    :type name: str
    :param why: the reason, shown above it.
    :type why: str
    :param lines: the commented-out TOML lines.
    :type lines: list[str]
    """
    comments[(section, f"#{name}#why")] = f"{name}: {why}"
    for i, line in enumerate(lines):
        comments[(section, f"#{name}#{i}")] = line
    comments.setdefault(UNFIT, {})[name] = why


def unfit_outputs(comments):
    """``{name: why}`` for outputs init left commented out because the probe showed they'd fail.

    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :rtype: dict
    """
    return dict(comments.get(UNFIT, {}))


#: Comments on a variable that record an assumption the user should check: (text in the comment,
#: what `fmugen init` prints after the variable names). Read from the final comments, so a guess the
#: probe later confirmed or replaced isn't reported.
ASSUMPTIONS = (
    ("(its annotation, with no arguments)", "objects built with no arguments from their annotation: give real "
                                           "ones with --const if the model needs them"),
    ("its docstring's default", "no default in the code: the start value is the docstring's, check it"),
    ("its docstring's unit", "no default in the code: the start value is a typical one in its docstring's unit, "
                             "check it"),
    ("passes the code's checks", "no default in the code: the start value is one the code's checks accept, "
                                 "check it"),
    ("start from the example in", "start values taken from the package's examples: check them"),
    ("no default in the code", "no default in the code: the start value is made up, check it"),
    ("no default: check the start value", "no default in the code: the start value is made up, check it"),
    ("Integer because the default is an int", "typed Integer because the default is an int; "
                                              "write a float start if it is Real"),
    ("an int in the probe; Real", 'an int in the probe, typed Real; set type = "Integer" if it is a count'),
    ("taken as a date-time from its name", "taken as a date-time from the name only: check [time], "
                                           "or make it an input"),
    ("check the epoch", "a date-time: the epoch is set to today, check it"),
    ("epoch in UTC", "a date-time epoch taken as UTC (the probe failed without a time zone)"),
    ("passed by position to *args", "passed by position to *args in the order of --start: check the order"),
    ("passed through **kwargs", "passed through **kwargs: check the model uses them"),
    ("can also return None", "the code can also return None, which fails the step: check it, or run init --probe"),
    ("so a fixed value; to pass the simulation time", "named like a time but given by --start, so a fixed "
                                                     "value: add [time] NAME = ... to pass the simulation time"),
    ("the probe failed with lists", "passed as arrays of another kind, since the probe failed with lists"),
    ("the model returns a new", "taken as states: the probe saw the model return a new value under the input's "
                                "own name"),
    ("fed back from", "taken as states from their names (x_prev fed back from x / x_next)"),
    ("set its dimensions and uncomment", "arrays left commented out: set their dimensions and uncomment"),
    ("(type unknown): uncomment", "properties of unknown type left commented out: uncomment the numbers"),
    ("need --fmi 3", "left out: bytes and resizable arrays need --fmi 3"),
)
#: Section notes (and [model] comments) that are warnings as they are.
NOTE_WARNINGS = ("probe call failed", "probe construction failed", "outputs could not be read",
                 "cannot inspect the signature", "is known only by running it", "skipped ",
                 "can't be pickled", "both take *args", "results that can be None")


def assumptions(comments):
    """Warnings for what init assumed and only wrote as comments in the config (see :data:`ASSUMPTIONS`).

    Variables with the same assumption are grouped into one line.

    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :return: the warning lines printed by ``fmugen init``.
    :rtype: list[str]
    """
    grouped, notes = {}, []
    for key, text in comments.items():
        if key[0] == "init" or not isinstance(text, str):
            continue
        section, name = key
        if name is None or (section == "model" and name == "save_state"):
            if any(text.startswith(n) or n in text for n in NOTE_WARNINGS):
                notes.append(f"[{section}] {text}")
            continue
        name = str(name)
        if name in ("#requirements", "#compile"):
            notes.append(f"[model] {text}")
            continue
        if name.count("#") > 1 or name == "#save_state":   # unfit outputs: reported on their own
            continue
        for fragment, warning in ASSUMPTIONS:
            if fragment in text:
                grouped.setdefault(warning, []).append(name.lstrip("#"))
                break
    warnings = notes
    for warning, names in grouped.items():
        names = list(dict.fromkeys(names))
        shown = ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")
        warnings.append(f"{shown}: {warning}")
    return warnings


def _misfit(value):
    """Why a value seen in the probe can't be read into an FMI variable, or None.

    Covers None (type unknown) and complex numbers (FMI has none).

    :param value: the probe value.
    :rtype: str or None
    """
    value = _scalar_of(value)
    if value is None:
        return "None in the probe, so its type is unknown; a None at run time fails the step"
    if isinstance(value, complex) or "complex" in str(getattr(value, "dtype", "")):
        return "complex in the probe, and FMI has no complex type: uncomment its real and imaginary parts"
    return None


def _unfit_value(comments, section, name, value, source, arrays=True):
    """Comment out ``name`` if its probe value is None or complex.

    A complex value is offered as two outputs, its real and imaginary parts.

    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :param section: the config section.
    :type section: str
    :param name: the variable.
    :type name: str
    :param value: the probe value.
    :param source: where it is read from: ``"return"``, ``"return:key"`` or ``"attr:name"``.
    :type source: str
    :param arrays: whether arrays and bytes may be FMU variables (FMI 3).
    :type arrays: bool
    :return: True if it did.
    :rtype: bool
    """
    why = _misfit(value)
    if why is None:
        return False
    if "complex" in why:
        info = _array_info(value.real) if getattr(value, "ndim", 0) else None
        dims = f", dimensions = {info['dimensions']}" if info else ""
        sep = "." if ":" in source else ":"
        lines = [f'{_key(f"{name}_{part}")} = {{ from = "{source}{sep}{part}"{dims} }}' for part in ("real", "imag")]
        if dims and not arrays:
            why += " (an array: needs init --fmi 3)"
    else:
        lines = [f'{_key(name)} = {{ from = "{source}", type = ... }}']
    _unfit(comments, section, name, why, lines)
    return True


def _apply_converts(data, converts):
    """Apply ``--convert NAME=module:function`` (or ``NAME=numpy``): how an argument is passed to the model.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param converts: ``{NAME: converter}``, or None.
    :type converts: dict or None
    """
    for name, converter in (converts or {}).items():
        for section in ("parameters", "inputs", "states"):
            info = data.get(section, {}).get(name)
            if info is None:
                continue
            info.pop("numpy", None), info.pop("convert", None)
            if converter == "numpy":
                info["numpy"] = True
            else:
                info["convert"] = converter


def _assign_starts(init, method, starts):
    """Split ``--start`` values between constructor and step method arguments.

    A name goes to the argument that has no default (it needs a value); on a tie, to the
    step method. Names neither declares go to whichever accepts ``**kwargs`` (method first).
    For a name both declare, ``<method>_NAME`` (the variable name init gives the method's one)
    goes to the method, and then NAME to the constructor: x0 and calculate_optimal_actuation_x0.
    A ``--const`` NAME both declare always goes to the constructor.

    :param init: the constructor (or factory), or None.
    :type init: callable or None
    :param method: the step method, or None.
    :type method: callable or None
    :param starts: the ``--start`` values.
    :type starts: dict
    :return: ``(constructor starts, method starts)``; names nobody takes are left out.
    :rtype: tuple[dict, dict]
    """
    def params(fn):
        """The parameters of ``fn`` from :func:`inspect.signature`; ``(*args)`` for a C built-in without one.

        :param fn: a callable, or None.
        :rtype: collections.abc.Mapping
        """
        try:
            return inspect.signature(fn).parameters if fn is not None else {}
        except (TypeError, ValueError):
            return NO_SIGNATURE_PARAMETERS

    init_p, call_p = params(init), params(method)
    init_starts, call_starts = {}, {}
    # the method's copy of a shared name, given by the name _rename_duplicates gives its variable
    prefix = f"{getattr(method, '__name__', 'call').strip('_')}_"
    for name, value in list(starts.items()):
        shared = name[len(prefix):] if name.startswith(prefix) and name not in init_p and name not in call_p else None
        if shared and shared.partition(".")[0] in init_p and shared.partition(".")[0] in call_p:
            call_starts[shared] = value
    for name, value in starts.items():
        base = name.partition(".")[0]   # ARG.FIELD goes where ARG goes
        in_init, in_call = init_p.get(base), call_p.get(base)
        if name.startswith(prefix) and name[len(prefix):] in call_starts and name not in init_p and name not in call_p:
            continue   # placed above
        if in_init and in_call and name in call_starts:
            target = init_starts   # the method's copy was given by its own name
        elif in_init and in_call and isinstance(value, dict) and len(value) == 1 and {*value} <= {"call", "python"}:
            target = init_starts   # a --const: NAME is the constructor's, <method>_NAME the method's
        elif in_init and in_call:
            target = init_starts if (in_init.default is in_init.empty and in_call.default is not in_call.empty) \
                else call_starts
        elif in_init or in_call:
            target = init_starts if in_init else call_starts
        elif any(p.kind is p.VAR_KEYWORD for p in call_p.values()):
            target = call_starts
        elif any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in init_p.values()):
            target = init_starts
        else:
            continue  # reported as unused
        target[name] = value
    return init_starts, call_starts


def class_called(cls, name):
    """Whether method ``name`` of ``cls`` is written to be called on the class, without an instance.

    A plain function in the class body whose first parameter isn't ``self``:
    ``def building(df, ...)``, called as ``Model.building(df, ...)``. Called on
    an instance, it would get the instance as ``df``. A ``staticmethod`` doesn't need this.

    :param cls: the model class, or None.
    :param name: the step method.
    :type name: str
    :rtype: bool
    """
    raw = inspect.getattr_static(cls, name, None) if cls is not None else None
    if not isinstance(raw, types.FunctionType):
        return False
    try:
        params = list(inspect.signature(raw).parameters.values())
    except (TypeError, ValueError):   # no readable signature: a normal method, as for C built-ins
        return False
    if not params:
        return True
    first = params[0]
    if first.name == "self" or first.kind not in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD):
        return False
    # A method whose instance isn't called self: a function defined outside the class and attached to it
    # (pypsa's Network.pf = pf, written as pf(n, ...)), or one whose first parameter is annotated with the class.
    if "." not in raw.__qualname__.replace(".<locals>.", ""):
        return False
    annotation = first.annotation
    if isinstance(annotation, str):
        return annotation.strip("'\"").rpartition(".")[2] not in {k.__name__ for k in inspect.getmro(cls)}
    return not (inspect.isclass(annotation) and issubclass(cls, annotation))


def _keywordless(fn):
    """Whether ``fn`` is a ufunc, which takes its arguments only by position.

    numpy ufuncs and numba ``@vectorize`` functions (``DUFunc``; psychrolib's with numba installed)
    report named parameters but reject keyword arguments.

    :param fn: a callable.
    :rtype: bool
    """
    return hasattr(fn, "nin") and type(fn).__module__.split(".")[0] in ("numpy", "numba")


def _given_constant(value):
    """Whether a ``--start`` / ``--const`` value is a ``--const`` one: ``{"call": ...}`` or ``{"python": ...}``.

    :param value: a value of the starts.
    :rtype: bool
    """
    return isinstance(value, dict) and len(value) == 1 and ("call" in value or "python" in value)


def _const_entry(value):
    """How a ``--const`` value is written in the constants table.

    A computed one as ``{call = ...}``; a literal as the plain TOML value when TOML has one (a string,
    number, bool, or a list of them: ``model = "yolo11n.pt"``), else as ``{python = repr}``.

    :param value: ``{"call": ...}`` or ``{"python": repr}``.
    :type value: dict
    :rtype: object
    """
    if "call" in value:
        return dict(value)
    literal = ast.literal_eval(value["python"])

    def plain(v):
        return isinstance(v, (str, bool, int, float)) or (isinstance(v, list) and all(plain(x) for x in v))

    return literal if plain(literal) and not (isinstance(literal, float) and math.isnan(literal)) else dict(value)


#: The parameters assumed for a callable without a signature (a C built-in): positional arguments only.
NO_SIGNATURE_PARAMETERS = inspect.Signature([inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL)]).parameters


def _only_varargs(fn, skip_self=True):
    """Whether ``fn`` declares no named arguments, only ``*args`` (and maybe ``**kwargs``).

    A callable without a signature (a C built-in) counts: its arguments can only be given by position.

    :param fn: the callable.
    :param skip_self: ignore the first parameter (an unbound method's self).
    :type skip_self: bool
    :rtype: bool
    """
    try:
        params = list(inspect.signature(fn).parameters.values())[1 if skip_self else 0:]
    except (TypeError, ValueError):
        return True
    return any(p.kind is p.VAR_POSITIONAL for p in params) \
        and all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)


# ---------------- shared ----------------

def _arguments(fn, section, data, comments, skip_self=False, allow_time=True, constants_key="constants",
               arrays=False, starts=None, positional=True, by_keyword=False, probe=False):
    """Sort the arguments of ``fn`` into variables of ``section`` and time arguments.

    Reads the signature with :func:`inspect.signature`. Each parameter becomes, in order of
    precedence: a ``[time]`` argument (named like a time, or a date-time), a call constant
    (``--start NAME=call:...``), item or field variables (tuple/dict starts, dataclass-like
    objects), or a variable typed from its default, ``--start`` value or annotation.
    Positional-only parameters get ``to = "pos:N"``. Arguments whose default is not an FMI value
    (None, tuples, objects) keep their code default; they are listed as commented-out constants
    so the user can override them. ``--start`` values for ``*args`` / ``**kwargs`` become
    positional or keyword variables.

    :param fn: the function, constructor or step method.
    :param section: the config section to fill (``parameters`` or ``inputs``).
    :type section: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :param skip_self: drop the first parameter (self).
    :type skip_self: bool
    :param allow_time: whether arguments may become ``[time]`` arguments (not for constructors).
    :type allow_time: bool
    :param constants_key: ``"constants"`` or ``"call_constants"``.
    :type constants_key: str
    :param arrays: whether arrays and bytes may be FMU variables (FMI 3).
    :type arrays: bool
    :param starts: ``--start`` values for these arguments.
    :type starts: dict or None
    :param positional: whether positional-only parameters are included.
    :type positional: bool
    :param by_keyword: give ``--start`` values for ``*args, **kwargs`` as keywords, not by position
        (the probe found the callable needs them so).
    :type by_keyword: bool
    :param probe: whether ``--probe`` is in effect: an argument with no default whose annotation is
        a class is then built with no arguments (:func:`_default_instance`).
    :type probe: bool
    :raises InterfaceError: an array ``--start`` for FMI 2, or an unusable value for ``*args``.
    """
    variables = data.setdefault(section, {})
    time_args = data.get("time", {})
    starts = starts or {}
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):  # e.g. some C extensions: --start values go by position, as to *args
        comments.setdefault(NO_SIGNATURE, []).append(getattr(fn, "__qualname__", repr(fn)))
        comments.setdefault((section, None), (   # the probe's split between constructor and step says more
            f"cannot inspect the signature of {getattr(fn, '__qualname__', fn)}: --start values are passed by "
            "position, in the order given; add any others by hand (to = \"pos:N\")" if starts else
            f"cannot inspect the signature of {getattr(fn, '__qualname__', fn)}; add the variables by hand "
            "(to = \"pos:N\" for positional arguments), or give them with --start in order"))
        params = list(NO_SIGNATURE_PARAMETERS.values())
        skip_self = False
    if skip_self and params:
        params = params[1:]
    params = [_unwrap_descriptor(p) for p in params]
    keywordless = _keywordless(fn)
    position = 0
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.kind is p.POSITIONAL_ONLY and not positional:
            continue
        if allow_time and _time_source(p.name) and p.name not in starts and not _horizon(p):
            time_args[p.name] = _time_source(p.name)
            continue
        if allow_time and _time_source(p.name) and isinstance(starts.get(p.name), (int, float)):
            comments[(section, p.name)] = (f"from --start, so a fixed value; to pass the simulation time instead, "
                                           f"remove it and add [time] {p.name} = \"{_time_source(p.name)}\"")
        how = _is_datetime(fn, p) if allow_time and p.name not in starts else None
        if how == "name":
            comments.setdefault(DATETIME_BY_NAME, []).append(p.name)
        if how:
            today = datetime.datetime.now(datetime.UTC).date().isoformat()
            time_args[p.name] = {"source": "end_time", "epoch": f"{today}T00:00:00"}   # naive: some models need it
            comments[("time", p.name)] = (
                "a date-time argument: the epoch plus the simulation time; check the epoch" if how == "annotation"
                else f"taken as a date-time from its name {p.name!r} (not annotated): check it, or replace this "
                     "with an input; the epoch plus the simulation time")
            continue
        if _given_constant(starts.get(p.name)):   # --const: a fixed value, not a variable
            data["model"].setdefault(constants_key, {})[p.name] = _const_entry(starts[p.name])
            continue
        if p.name in starts and _split_items(p, starts[p.name], variables, section, comments, arrays):
            continue
        if p.name not in starts and _split_fields(fn, p, variables, section, comments, starts, arrays):
            continue
        if p.name not in starts and p.default is p.empty and p.kind is not p.POSITIONAL_ONLY \
                and _default_instance(fn, p, data, comments, constants_key, probe):
            continue
        for name in starts:
            if name.startswith(f"{p.name}."):
                _start_error(comments, name, f"cannot tell the class of {p.name} (no default and no dataclass, "
                                             "pydantic, attrs or namedtuple annotation) to build it from fields; "
                                             f"give the whole argument instead, e.g. --const {p.name}=call:module:Class(...)")
        if p.name in starts and not arrays and isinstance(starts[p.name], (list, tuple, bytes)):
            raise InterfaceError(f"--start {p.name}: arrays and bytes need --fmi 3 (FMI 2 has only scalar variables)")
        has_default = p.default is not p.empty or p.name in starts
        default = starts.get(p.name, None if p.default is p.empty else p.default)
        before = len(variables)
        if not has_default and arrays and probe:   # tried by the probe if 1.0 fails (gsw's array_like takes numbers)
            shape = _made_up_array(fn, p)
            if shape:
                comments[_shape_key(section, p.name)] = shape
        if not has_default or isinstance(default, SCALARS):
            info = _type_info(default, p.annotation)
            if p.name in starts:
                comments.setdefault((section, p.name), "start from --start")
            elif not has_default:
                informed = _informed_start(fn, p) if not isinstance(info["start"], (bool, str)) else None
                if informed:
                    value, why = informed
                    info = {"start": int(value) if isinstance(info["start"], int) and float(value).is_integer()
                            else float(value)}
                    comments[(section, p.name)] = f"no default in the code: {why}; check the start value"
                else:
                    comments[(section, p.name)] = MADE_UP
            elif type(default) is int and static.unwrap_optional(p.annotation) is not int:
                comments[(section, p.name)] = "Integer because the default is an int; write a float start for Real"
            variables[p.name] = info
        elif isinstance(default, enum.Enum):
            variables[p.name] = {
                "enum": f"{type(default).__module__}:{type(default).__qualname__}",
                "start": default.name,
            }
        elif arrays and isinstance(default, bytes):
            variables[p.name] = {"type": "Binary", "start": default.hex()}
        elif arrays and _array_info(default):
            variables[p.name] = _array_info(default, with_start=True)
        elif p.name in starts:   # --start with a value that can't be an FMU variable
            raise InterfaceError(f"--start {p.name}: {default!r} can't be an FMU variable (a number, string, bool "
                                 f"or array); give a fixed value with --const {p.name}=...")
        else:
            comments[("model", f"{constants_key}.{p.name}")] = _constant(default)
        if len(variables) > before and "dimensions" in variables[p.name]:
            converter = _converter(p.annotation)
            if converter == "numpy":
                variables[p.name]["numpy"] = True
            elif converter:
                variables[p.name]["convert"] = converter
                comments.setdefault((section, p.name), f"passed as {converter.replace(':', '.')}(...)")
        if (p.kind is p.POSITIONAL_ONLY or keywordless) and len(variables) > before:
            variables[p.name]["to"] = f"pos:{position}"
            position += 1
    # not *args/**kwargs' own names: MOMENT's from_pretrained(..., **model_kwargs) takes a model_kwargs keyword
    names = {p.name for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    extra = [name for name in starts
             if name not in names and "." not in name   # ARG.FIELD: a field of an object argument, not a keyword
             and name not in time_args]
    only_varargs = any(p.kind is p.VAR_POSITIONAL for p in params) \
        and all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)
    # by position: for a constructor, for computed constants, and when there's no **kwargs to take them
    by_position = only_varargs and not by_keyword and (section == "parameters" or not any(p.kind is p.VAR_KEYWORD for p in params)
                                    or any(_given_constant(starts[n]) for n in extra))
    for name in extra:
        computed = _given_constant(starts[name])   # --const: a constant
        if by_position:
            # A callable that only declares (*args, **kwargs), e.g. StateSpace(A, B, C, D):
            # the --start values go by position, in the order given. Constants are named "0", "1", ...
            if computed:
                data["model"].setdefault(constants_key, {})[str(position)] = _const_entry(starts[name])
                comments[("model", f"{constants_key}.{position}")] = f"--const {name}, passed by position to *args"
                comments.setdefault(POSITIONAL_STARTS, {})[name] = position
            else:
                variables[name] = _extra_info(name, starts[name], arrays)
                variables[name]["to"] = f"pos:{position}"
                comments[(section, name)] = "start from --start, passed by position to *args: check the order"
            position += 1
        elif any(p.kind is p.VAR_KEYWORD for p in params):  # **kwargs: --start values become keyword arguments
            if computed:
                data["model"].setdefault(constants_key, {})[name] = _const_entry(starts[name])
                comments[("model", f"{constants_key}.{name}")] = "--const, passed through **kwargs"
            else:
                variables[name] = _extra_info(name, starts[name], arrays)
                comments[(section, name)] = "start from --start (passed through **kwargs)"
    if time_args and allow_time:
        data["time"] = time_args
    if not variables:
        data.pop(section)


def _unwrap_descriptor(p):
    """Replace a descriptor default by the value it stands for.

    A default that is a descriptor (astropy's ``Parameter(default=3.04, ...)`` in FlatLambdaCDM's
    signature) stands for the class's own handling of the argument: its ``default`` is the value.

    :param p: a signature parameter.
    :type p: inspect.Parameter
    :rtype: inspect.Parameter
    """
    default = p.default
    if p.default is p.empty or callable(default) \
            or not hasattr(type(default), "__get__") or not hasattr(default, "default"):
        return p
    return p.replace(default=default.default)


def _extra_info(name, value, arrays):
    """The variable for a ``--start`` value that no declared argument takes (``*args``, ``**kwargs``).

    :param name: the ``--start`` name.
    :type name: str
    :param value: its value.
    :param arrays: whether arrays and bytes may be FMU variables (FMI 3).
    :type arrays: bool
    :rtype: dict
    :raises InterfaceError: not an FMI value.
    """
    if isinstance(value, SCALARS):
        return _type_info(value)
    if arrays and _array_info(value):
        return _array_info(value, with_start=True)
    raise InterfaceError(f"--start {name}: {value!r} can't be an FMU variable (a number, string, bool or array); "
                         f"give a fixed value with --const {name}=...")


#: Annotation text -> how an array argument of that type is passed (``convert``, or ``numpy``).
ARRAY_CONVERTERS = {"torch.Tensor": "torch:tensor", "ndarray": "numpy", "jax.Array": "jax.numpy:asarray",
                    "tf.Tensor": "tensorflow:constant"}


def _converter(annotation):
    """How an array argument must be passed, from its annotation.

    :param annotation: the parameter's annotation (or ``inspect.Parameter.empty``).
    :return: ``"numpy"``, ``"module:function"``, or None.
    :rtype: str or None
    """
    if annotation is inspect.Parameter.empty:
        return None
    text = annotation if isinstance(annotation, str) else repr(annotation)
    return next((conv for name, conv in ARRAY_CONVERTERS.items() if name in text), None)


def _split_items(p, value, variables, section, comments, arrays):
    """Make one variable per item of a tuple (of arrays) or dict ``--start`` value.

    Bound with ``to = "arg:NAME[i]"`` / ``"arg:NAME[key]"`` (``init:`` for parameters); the
    runtime puts the argument back together (:func:`assemble_items`).

    :param p: the parameter.
    :type p: inspect.Parameter
    :param value: its ``--start`` value.
    :param variables: the section's variables; updated in place.
    :type variables: dict
    :param section: the config section (``parameters`` or ``inputs``).
    :type section: str
    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :param arrays: whether arrays and bytes may be FMU variables (FMI 3).
    :type arrays: bool
    :return: whether it was split.
    :rtype: bool
    """
    import typing
    if isinstance(value, tuple) and len(value) > 1 and not (arrays and _array_info(value)):
        keys = list(range(len(value)))
        names = [f"{p.name}_{i}" for i in keys]
        hints = typing.get_args(p.annotation) if p.annotation is not p.empty else ()
    elif isinstance(value, dict) and value and all(isinstance(k, str) and k.isidentifier() for k in value):
        keys = list(value)
        names = keys
        hints = ()
    else:
        return False
    items = [value[k] for k in keys]
    if not all(isinstance(item, SCALARS) or (arrays and _array_info(item)) for item in items):
        return False
    for i, (key, name, item) in enumerate(zip(keys, names, items)):
        info = _array_info(item, with_start=True) if (arrays and _array_info(item)) else _type_info(item)
        if "dimensions" in info:
            from fmugen.templates.fmugen_runtime import flatten
            flat = flatten(list(item))
            if all(isinstance(x, int) and not isinstance(x, bool) for x in flat):   # e.g. atomic numbers, indices
                info.update(type="Int32", start=flat)
        info["to"] = f"{'init' if section == 'parameters' else 'arg'}:{p.name}[{key}]"
        converter = _converter(hints[i]) if i < len(hints) else None
        if converter == "numpy":
            info["numpy"] = True
        elif converter and "dimensions" in info:
            info["convert"] = converter
        variables[name] = info
        comments[(section, name)] = f"item {key!r} of the argument {p.name}"
    return True


def _object_fields(cls):
    """The fields of a dataclass, pydantic model, attrs class or namedtuple.

    Field types come from :func:`typing.get_type_hints` (resolving string annotations);
    default factories are called to get a default.

    :param cls: the class.
    :return: ``[(field, type hint, default or inspect.Parameter.empty)]``, else None.
    :rtype: list[tuple] or None
    """
    if not isinstance(cls, type):
        return None
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}
    empty = inspect.Parameter.empty
    if dataclasses.is_dataclass(cls):
        missing = dataclasses.MISSING
        return [(f.name, hints.get(f.name), empty if f.default is missing and f.default_factory is missing
                 else (f.default if f.default is not missing else f.default_factory()))
                for f in dataclasses.fields(cls) if f.init]
    if hasattr(cls, "model_fields") and isinstance(cls.model_fields, dict):   # pydantic 2
        return [(name, f.annotation, empty if f.is_required() else f.get_default(call_default_factory=True))
                for name, f in cls.model_fields.items()]
    if hasattr(cls, "__attrs_attrs__"):
        import attrs
        return [(a.name, hints.get(a.name, a.type), empty if a.default is attrs.NOTHING
                 else (a.default.factory() if isinstance(a.default, attrs.Factory) else a.default))
                for a in cls.__attrs_attrs__ if a.init]
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):   # a namedtuple
        defaults = getattr(cls, "_field_defaults", {})
        return [(name, hints.get(name), defaults.get(name, empty)) for name in cls._fields]
    return None


def _split_fields(fn, p, variables, section, comments, starts, arrays):
    """Make one variable per field of an object argument.

    An argument that is a dataclass / pydantic / attrs object (by its default or its annotation)
    becomes one variable per field, NAME_FIELD, bound with ``to = "arg:NAME.FIELD"``. Fields that
    aren't FMI values keep the default's value; without a default, they must have their own
    default.

    :param fn: the function (for its type hints).
    :param p: the parameter.
    :type p: inspect.Parameter
    :param variables: the section's variables; updated in place.
    :type variables: dict
    :param section: the config section (``parameters`` or ``inputs``).
    :type section: str
    :param comments: ``(section, key) -> text`` for :func:`render_toml`, plus internal ``("init", ...)`` keys.
    :type comments: dict
    :param starts: ``--start`` values (``NAME.FIELD`` ones give field starts).
    :type starts: dict
    :param arrays: whether arrays and bytes may be FMU variables (FMI 3).
    :type arrays: bool
    :return: whether it was split.
    :rtype: bool
    """
    if p.default is None:
        return False   # None means "not given" to the model: it stays a constant, as for any argument
    default = None if p.default is p.empty else p.default
    try:
        cls = typing.get_type_hints(fn).get(p.name)
    except Exception:
        cls = p.annotation if isinstance(p.annotation, type) else None
    cls = type(default) if default is not None else static.unwrap_optional(cls)
    fields = _object_fields(cls)
    if not fields:
        return False
    found = {}
    for field, hint, field_default in fields:
        value = getattr(default, field) if default is not None else starts.get(f"{p.name}.{field}", field_default)
        if value is p.empty:   # a required field without a start: a variable if its annotation is a number
            if static.unwrap_optional(hint) not in (float, int, bool, str):
                return False   # can't be built from FMI values
            value = None
        elif not isinstance(value, SCALARS) and not (arrays and _array_info(value)):
            continue   # None or not an FMI value: the default's value (or the field's default) is kept
        info = _array_info(value, with_start=True) if value is not None and not isinstance(value, SCALARS)             else _type_info(value, static.unwrap_optional(hint))
        if isinstance(info.get("start"), int) and not isinstance(info["start"], bool)                 and static.unwrap_optional(hint) is float:
            info["start"] = float(info["start"])
        info["to"] = f"{'init' if section == 'parameters' else 'arg'}:{p.name}.{field}"   # a constructor's
        found[f"{p.name}_{field}"] = info
        if value is None:
            comments[(section, f"{p.name}_{field}")] = f"field {field!r} of {p.name} ({cls.__name__}); no default: check the start value"
        else:
            comments[(section, f"{p.name}_{field}")] = f"field {field!r} of {p.name} ({cls.__name__})"
    if not found:
        return False
    names = {field for field, _, _ in fields}
    for name in starts:
        base, _, field = name.partition(".")
        if base == p.name and field not in names:
            _start_error(comments, name, f"{p.name} ({cls.__name__}) has no field {field!r}; "
                                         f"its fields: {', '.join(sorted(names))}")
    variables.update(found)
    return True


#: Annotations that aren't objects to build: FMI values, containers, arrays.
NOT_BUILT = (bool, int, float, complex, str, bytes, bytearray, list, tuple, dict, set, frozenset, type(None))


def _default_instance(fn, p, data, comments, constants_key, probe):
    """A constant ``{ call = "module:Class()" }`` for an argument with no default whose annotation is a class.

    For model inputs that are objects the library builds (ase's ``atoms: Atoms``): with ``--probe``,
    the class is built once with no arguments, and the constant is kept if that works, with a comment
    saying it's a default instance to replace with ``--const``. Without ``--probe`` nothing is called,
    so it's written commented out, as a suggestion.

    :param fn: the function (for its type hints).
    :param p: the parameter (no default, no ``--start``).
    :type p: inspect.Parameter
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param constants_key: ``"constants"`` or ``"call_constants"``.
    :type constants_key: str
    :param probe: whether ``--probe`` is in effect.
    :type probe: bool
    :return: whether the argument was handled.
    :rtype: bool
    """
    try:
        cls = typing.get_type_hints(fn).get(p.name)
    except Exception:
        cls = p.annotation if isinstance(p.annotation, type) else None
    cls = static.unwrap_optional(cls)
    if not isinstance(cls, type) or issubclass(cls, NOT_BUILT) or cls.__module__ in ("builtins", "typing") \
            or "<" in cls.__qualname__ or cls.__name__ in ("ndarray", "Tensor", "Array"):   # arrays: a --start
        return False
    ref = f"{cls.__module__}:{cls.__qualname__}()"
    if not probe:   # nothing is called: a suggestion, commented out
        comments[("model", f"{constants_key}.{p.name}")] = {"call": ref}
        comments[("constant", f"{constants_key}.{p.name}")] = (
            f"no default in the code: uncomment to use {cls.__name__}() (its annotation, with no arguments), "
            f"or give one with --const {p.name}=call:...; init --probe checks that it builds")
        return True
    try:
        cls()
    except Exception:
        return False
    data["model"].setdefault(constants_key, {})[p.name] = {"call": ref}
    comments[("constant", f"{constants_key}.{p.name}")] = (
        f"no default in the code: {cls.__name__}() (its annotation, with no arguments); "
        f"give your own with --const {p.name}=call:...")
    return True


#: The comment of an argument with no default (and no --start): its start is made up.
MADE_UP = "no default in the code: check the start value"
def _shape_key(section, name):
    """The comments key (not rendered) of the shape of a made-up argument its annotation or docstring
    says is an array, for the probe to try when the call fails with 1.0."""
    return ("init", f"made up shape {section}.{name}")


def _made_up_array(fn, p):
    """The shape of an argument with no default that is an array by its annotation or numpydoc docstring.

    The docstring's shape, else a 1-D array of :data:`fmugen.interface.MADE_UP_SIZE` elements.

    :param fn: the function the argument is of.
    :param p: the parameter.
    :type p: inspect.Parameter
    :return: the shape, or None when it isn't an array.
    :rtype: list[int] or None
    """
    from fmugen.interface import MADE_UP_SIZE, docstring_arrays
    try:
        documented = docstring_arrays(fn)
    except Exception:   # an odd docstring: no shape from it
        documented = {}
    if p.name not in documented and not static._is_array_hint(static.unwrap_optional(p.annotation)
                                                            if p.annotation is not p.empty else None):
        return None
    return documented.get(p.name) or [MADE_UP_SIZE]


#: A documented default: ``default 1e-6``, ``defaults to 0.5``, ``(default: 3)``, ``default=300``.
_DOCUMENTED_DEFAULT = re.compile(r"\bdefaults?(?:\s+value)?(?:\s+is|\s+to|\s+of)?\s*[:=]?\s*`*([-+]?(?:\d+\.?\d*|\.\d+)"
                                 r"(?:[eE][-+]?\d+)?)(?![\w.])", re.I)
#: A typical value for a quantity in a unit a docstring gives, where 1.0 is far from any real one.
TYPICAL_BY_UNIT = {"K": 293.15, "degC": 20.0, "°C": 20.0, "Pa": 101325.0, "kPa": 101.325, "MPa": 0.101325,
                   "bar": 1.01325, "mbar": 1013.25, "hPa": 1013.25, "psi": 14.696}
#: Values tried, in order, against the checks the code makes on an argument (``if x < 10: raise``).
CHECKED_VALUES = (1.0, 0.5, 2.0, 10.0, 0.1, 100.0, 1000.0, 0.0, -1.0, 1e4, 1e5, 1e6)


def _informed_start(fn, p):
    """A start for an argument with no default from what the code says of it, rather than 1.0.

    In order: a default its docstring gives (``tol : float, default 1e-6``); a typical value of its
    docstring unit when that unit is one where 1.0 is far from real values (293.15 for ``[K]``); a
    value the checks in the code accept, when they refuse 1.0 (``if n_harmonics < 3: raise``).

    :param fn: the function the argument is of.
    :param p: the parameter.
    :type p: inspect.Parameter
    :return: ``(value, why)``, or None.
    :rtype: tuple or None
    """
    from fmugen.interface import docstring_parameters, docstring_units
    owner = _owner_class(fn)
    try:
        text = docstring_parameters(fn, owner).get(p.name, "")
    except Exception:
        text = ""
    match = _DOCUMENTED_DEFAULT.search(text)
    if match:
        return float(match.group(1)), f"its docstring's default, {match.group(1)}"
    try:
        unit = docstring_units(fn, owner)[0].get(p.name)
    except Exception:
        unit = None
    if unit in TYPICAL_BY_UNIT:
        return TYPICAL_BY_UNIT[unit], f"a typical value in {unit}, its docstring's unit"
    checks = _value_checks(fn, p.name)
    if checks and not _passes(checks, p.name, 1.0):
        value = next((v for v in CHECKED_VALUES[1:] if _passes(checks, p.name, v)), None)
        if value is not None:
            return value, f"{value:g} passes the code's checks on it, 1.0 doesn't"
    return None


def _owner_class(fn):
    """The class of a constructor or method (numpydoc puts a constructor's parameters on the class), or None."""
    qualname = getattr(fn, "__qualname__", "") or ""
    owner = sys.modules.get(getattr(fn, "__module__", None) or "")
    for part in qualname.split(".")[:-1]:
        owner = getattr(owner, part, None)
    return owner if inspect.isclass(owner) else None


def _value_checks(fn, name):
    """The checks the code of ``fn`` makes on argument ``name``: ``(test, raises)`` of ``if test: raise``
    (raises True) and ``assert test`` (False), whose test only compares ``name`` with numbers.

    :rtype: list[tuple]
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return []

    def number(s):
        if isinstance(s, ast.UnaryOp) and isinstance(s.op, (ast.USub, ast.UAdd)):
            s = s.operand
        return isinstance(s, ast.Constant) and isinstance(s.value, (int, float)) and not isinstance(s.value, bool)

    def simple(test):
        if isinstance(test, ast.BoolOp):
            return all(simple(v) for v in test.values)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return simple(test.operand)
        if not isinstance(test, ast.Compare):
            return False
        sides = [test.left, *test.comparators]
        return any(isinstance(s, ast.Name) and s.id == name for s in sides) \
            and all((isinstance(s, ast.Name) and s.id == name) or number(s) for s in sides)

    checks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and simple(node.test) and any(isinstance(b, ast.Raise) for b in node.body):
            checks.append((node.test, True))
        elif isinstance(node, ast.Assert) and simple(node.test):
            checks.append((node.test, False))
    return checks


def _passes(checks, name, value):
    """Whether ``value`` for ``name`` passes the checks of :func:`_value_checks`."""
    for test, raises in checks:
        try:
            result = bool(eval(compile(ast.Expression(test), "<check>", "eval"), {"__builtins__": {}}, {name: value}))
        except Exception:
            return False
        if result == raises:
            return False
    return True


def _type_info(value, annotation=inspect.Parameter.empty):
    """The ``start`` of a scalar variable from its default and annotation.

    Without a default, the annotation's one (float by default; ``False`` and ``""`` for bool and str):
    one, not zero, as models divide by it or check it's positive; an int default for a float
    annotation becomes a float.

    :param value: the default, or None.
    :param annotation: the parameter's annotation.
    :rtype: dict
    """
    annotation = static.unwrap_optional(annotation)
    if value is None:
        hint = {float: 1.0, int: 1, bool: False, str: ""}.get(annotation, 1.0)
        return {"start": hint}
    if isinstance(value, int) and not isinstance(value, bool) and annotation is float:
        return {"start": float(value)}
    if isinstance(value, SCALARS):
        return {"start": value}
    return {"start": float(value)}


def _constant(value):
    """How a non-FMI default would be written as a constant (for a commented-out example).

    ``{python = repr}`` when :func:`ast.literal_eval` gives the value back, else
    ``{ref = "module:qualname"}`` for a named object.

    :param value: the default.
    :rtype: dict or None
    """
    text = repr(value)
    try:
        if ast.literal_eval(text) == value:
            return {"python": text}
    except (ValueError, SyntaxError):
        pass
    module, qualname = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
    if module and qualname and "<" not in qualname:
        return {"ref": f"{module}:{qualname}"}
    return None

#: Argument names taken as date-times when they aren't annotated.
DATETIME_NAMES = ("when", "date", "datetime", "dateandtime", "date_time", "timestamp", "utc_time")


def _is_datetime(fn, p):
    """Whether an argument takes a date-time.

    :param fn: the function (for its type hints).
    :param p: the parameter.
    :type p: inspect.Parameter
    :return: ``"annotation"`` (datetime, Optional[datetime]), ``"name"`` (an unannotated argument
        named like one, without a non-None default), or None.
    :rtype: str or None
    """
    try:
        hint = typing.get_type_hints(fn).get(p.name, p.annotation)
    except Exception:
        hint = p.annotation
    hint = static.unwrap_optional(hint)
    if isinstance(hint, type) and issubclass(hint, datetime.date):
        return "annotation"
    if isinstance(hint, str) and "datetime" in hint:
        return "annotation"
    if hint is not p.empty and hint is not None:
        return None   # annotated as something else
    if p.name.lower() in DATETIME_NAMES and (p.default is p.empty or p.default is None):
        return "name"
    return None


def _typed(info, value, arrays=False):
    """Add the type (and unit, dimensions) of an output seen in the probe.

    Ints become Real (a probe often just returns 0); ``numpy`` / ``convert`` are dropped, since
    they only matter for values passed into the model.

    :param info: the variable's table; updated and returned.
    :type info: dict
    :param value: the probe value.
    :param arrays: whether arrays and bytes may be FMU variables (FMI 3).
    :type arrays: bool
    :rtype: dict
    """
    if _unit_of(value):
        info["unit"] = _unit_of(value)
    value = _scalar_of(value)
    if arrays and isinstance(value, bytes):
        info["type"] = "Binary"
    elif arrays and _array_info(value):
        # numpy = true only matters for values passed *into* the model
        info.update({k: v for k, v in _array_info(value).items() if k not in ("numpy", "convert")})
    elif type(value).__name__ == "float32":
        info["type"] = "Float32" if arrays else "Real"
    elif isinstance(value, enum.Enum):
        info["enum"] = f"{type(value).__module__}:{type(value).__qualname__}"
    elif isinstance(value, (bool, str)):
        info["type"] = _type_name(value)
    return info

def _valid_name(name):
    """A dotted variable name for a result key.

    Other characters become ``_`` (``"N-14"`` -> ``"N_14"``), and a part that starts with a digit
    or is a keyword gets a leading ``_``.

    :param name: the key.
    :rtype: str
    """
    parts = []
    for part in str(name).split("."):
        part = re.sub(r"\W", "_", part) or "_"
        if part[0].isdigit() or keyword.iskeyword(part):
            part = f"_{part}"
        parts.append(part)
    return ".".join(parts)


def _dotted_name(name):
    """Whether every dot-separated part of ``name`` is an identifier.

    :param name: the name.
    :type name: str
    :rtype: bool
    """
    return all(part.isidentifier() for part in name.split("."))

#: Suffixes of input names that mean a previous value (x_prev): candidates for states.
STATE_SUFFIXES = ("_prev", "_old", "_last")
#: Prefixes of input names that mean a previous value (prev_x).
STATE_PREFIXES = ("prev_", "last_", "old_")


def _state_base(name):
    """'x' for an input named like a previous value (x_prev, x_old, x_last, prev_x, last_x, old_x), else None.

    :param name: the input name.
    :type name: str
    :rtype: str or None
    """
    for suffix in STATE_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    for prefix in STATE_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return None

#: numpy scalar type names, kept as they are (not unwrapped with .item()).
NUMPY_SCALARS = ("float64", "float32", "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16",
                 "uint8", "bool_")
#: numpy/torch dtype name -> FMI type of an array's elements.
NUMPY_TYPES = {"float32": "Float32", "float64": "Real", "int8": "Int8", "uint8": "UInt8", "int16": "Int16",
               "uint16": "UInt16", "int32": "Int32", "uint32": "UInt32", "int64": "Int64", "uint64": "UInt64",
               "bool": "Boolean"}


def _scalar_of(value):
    """The scalar inside a 0-dimensional array or tensor (numpy, torch, ...) or a quantity, else the value itself.

    numpy scalars (np.float32, ...) also have ndim 0, but are kept as they are, keeping their type.

    :param value: the value.
    """
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        value = value.magnitude
    elif hasattr(value, "unit") and hasattr(value, "to_value"):   # an astropy Quantity
        value = value.value
    # numpy scalars (np.float32, ...) also have ndim 0, but are handled as they are, keeping their type
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item") and type(value).__name__ not in NUMPY_SCALARS:
        try:
            return value.item()
        except Exception:
            return value
    return value


def _fmi_value(value, arrays=False):
    """Whether a value can be an FMI variable.

    :param value: the value.
    :param arrays: whether arrays and bytes may be FMU variables (FMI 3).
    :type arrays: bool
    :rtype: bool
    """
    value = _scalar_of(value)
    if isinstance(value, (*SCALARS, enum.Enum)) or type(value).__name__ in NUMPY_SCALARS or _number_like(value):
        return True
    return arrays and (isinstance(value, bytes) or _array_info(value) is not None)


def _number_like(value):
    """Decimal, Fraction, numpy.float16, ...: anything that converts to a float or an int and isn't an array.

    :param value: the value.
    :rtype: bool
    """
    if getattr(value, "ndim", 0) or isinstance(value, (str, bytes, complex)):
        return False
    if not (hasattr(type(value), "__float__") or hasattr(type(value), "__index__")):
        return False
    try:   # uncertainties' AffineScalarFunc has __float__, but it raises
        float(value)
    except Exception:
        return False
    return True


def _unit_of(value):
    """The unit of a pint or astropy quantity, written compactly ("m/s").

    :param value: the value.
    :return: the unit, or None for anything else or dimensionless.
    :rtype: str or None
    """
    if hasattr(value, "unit") and hasattr(value, "to_value"):   # an astropy Quantity
        unit = value.unit.to_string() if hasattr(value.unit, "to_string") else str(value.unit)
        return unit or None
    units = getattr(value, "units", None)
    if units is None or not hasattr(value, "magnitude"):
        return None
    try:
        return f"{units:~C}" or None
    except Exception:
        return str(units) or None


def _array_info(value, with_start=False):
    """The array keys of a variable for a rectangular list/tuple/numpy array/tensor of numbers.

    :param value: the value.
    :param with_start: also give the flattened ``start``.
    :type with_start: bool
    :return: ``{dimensions, type?, numpy? / convert?, start?}``, else None (also for empty arrays).
    :rtype: dict or None
    """
    # not a class: numpy.float64 (a dtype default) has shape and ndim as descriptors
    if (not isinstance(value, type) and hasattr(value, "shape") and hasattr(value, "tolist")
            and getattr(value, "ndim", 0) > 0):
        from fmugen.templates.fmugen_runtime import flatten
        shape, flat = [int(n) for n in value.shape], flatten(value.tolist())
        fmi_type = NUMPY_TYPES.get(str(value.dtype).rpartition(".")[2])   # "float32", "torch.float32", ...
        if str(value.dtype) == "object" and flat and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in flat):
            fmi_type = "Real"   # numbers in an object array (facenet-pytorch's MTCNN.detect boxes)
        if fmi_type is None:
            return None
        info = {"dimensions": shape}
        library = type(value).__module__.split(".")[0]
        if library == "numpy":
            info["numpy"] = True
        elif library == "torch":
            info["convert"] = "torch:tensor"
    elif isinstance(value, (list, tuple)) and value:
        shape = _shape(value)
        if shape is None:
            return None
        from fmugen.templates.fmugen_runtime import flatten
        flat = flatten(value)
        if all(isinstance(x, bool) for x in flat):
            fmi_type = "Boolean"
        elif all(isinstance(x, int) and not isinstance(x, bool) for x in flat):
            # sizes such as layer_sizes=[1, 16, 16, 1]: as floats, torch refuses them (deepxde, pytorch-tcn)
            fmi_type = "Integer"
        elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in flat):
            fmi_type = "Real"
        else:
            return None
        info = {"dimensions": shape}
    else:
        return None
    if 0 in shape:
        return None
    if fmi_type != "Real":
        info["type"] = fmi_type
    if with_start:
        info["start"] = [float(x) for x in flat] if fmi_type in ("Real", "Float32") else flat
    return info


def _shape(value):
    """The shape of nested lists/tuples.

    :param value: the value.
    :return: the dimensions (``[]`` for a scalar), or None when not rectangular.
    :rtype: list[int] or None
    """
    if not isinstance(value, (list, tuple)):
        return []
    shapes = [_shape(item) for item in value]
    if not shapes or any(s is None or s != shapes[0] for s in shapes):
        return None
    return [len(value), *shapes[0]]


def _same(a, b):
    """Whether two probe values are equal (arrays compared as lists, NaN equal to NaN).

    :param a: one value.
    :param b: the other.
    :rtype: bool
    """
    try:
        if type(a) is type(b) and type(a).__module__.split(".")[0] == "pandas" and hasattr(a, "equals"):
            return bool(a.equals(b))   # tables: == compares element-wise
        if hasattr(a, "tolist"):
            a = a.tolist()
        if hasattr(b, "tolist"):
            b = b.tolist()
        return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
    except Exception:
        return False


def _type_name(value):
    """The config type name of a Python scalar.

    :param value: the value.
    :return: ``"Boolean"``, ``"Integer"``, ``"String"`` or ``"Real"``.
    :rtype: str
    """
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, int):
        return "Integer"
    if isinstance(value, str):
        return "String"
    return "Real"

