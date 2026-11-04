"""Calling the model in the probe, and retrying with other argument forms when a call fails."""
import datetime
import importlib
import inspect
import itertools
import math
import re
import sys

from fmugen import static
from fmugen.arguments import DATETIME_BY_NAME, MADE_UP, _only_varargs, _shape_key
from fmugen.probe._arguments import _owner, _probe_args, _probe_constants, _step_method
from fmugen.probe._base import ConstantError, _run_setup, _time_kwargs


#: The made-up training data :func:`_fit_made_up` tries: samples, features.
MADE_UP_TRAINING = (64, 4)   # more samples than a batch (pyod trains in batches of 32)


def _made_up_training():
    """The made-up training data :func:`_fit_made_up` tries, in order: ``(arguments, description)``.

    :data:`MADE_UP_TRAINING` samples of deterministic random numbers: ``X`` with its features, as float64
    then float32, with ``y`` as a vector, a column, then class labels 0/1; ``X`` alone (pyod's detectors);
    with pandas loaded, a table of one column ``y`` and a table with dates ``ds`` and values ``y`` (prophet);
    with darts loaded, a ``TimeSeries`` of the values.

    :rtype: list[tuple]
    """
    try:
        import numpy
    except ImportError:
        return []
    samples, features = MADE_UP_TRAINING
    rng = numpy.random.default_rng(0)
    X64, values = rng.random((samples, features)), rng.random(samples)
    arrays = f"{samples} samples, {features} features"
    tries = []
    for X in (X64, X64.astype("float32")):
        for y in (values.astype(X.dtype), values.astype(X.dtype).reshape(-1, 1), (values > 0.5).astype(int)):
            tries.append(((X, y), arrays))
    tries.append(((X64,), arrays))   # last: fit(X, y=None) of catboost needs y all the same
    pandas = sys.modules.get("pandas")
    if pandas is not None:
        dates = pandas.date_range("2020-01-01", periods=samples, freq="D")
        tries.append(((pandas.DataFrame({"ds": dates, "y": values}),), f"a table of {samples} days, ds and y"))
    darts = sys.modules.get("darts")
    if darts is not None and hasattr(darts, "TimeSeries"):
        try:
            series = darts.TimeSeries.from_values(values.astype("float32"))
        except Exception:
            series = None
        if series is not None:
            tries.append(((series,), f"a TimeSeries of {samples} values"))
    return tries


def _build_set_up(build, data):
    """Build the model object with the probe's parameters and run ``[model] setup`` on it, without retries.

    :param build: the class, ``--create`` classmethod or factory.
    :param data: the config.
    :type data: dict
    :return: the object.
    """
    args, kwargs = _probe_args(data, ("parameters",), build, "constants")
    obj = build(*args, **_probe_constants(data, "constants"), **kwargs)
    _run_setup(data, obj)
    return obj


#: Converters tried in order when a probe call fails with list inputs.
ARRAY_RETRIES = ("numpy", "torch:tensor")


def _probe_call(fn, data, comments, constants_key="constants"):
    """Call ``fn`` with the probe inputs (:func:`_probe_dated`), dropping time arguments that aren't the clock.

    When the call fails and ``[time]`` binds an argument whose default is None (ssqueezepy's
    ``ssq_cwt(x, ..., t=None)``, the signal's time vector), the call is made again without those
    arguments; if that works, they keep their default instead.

    :param fn: the function or bound step method.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param constants_key: the constants table passed with the call.
    :type constants_key: str
    :return: what ``fn`` returned.
    :raises Exception: what the call raised.
    """
    try:
        return _probe_dated(fn, data, comments, constants_key)
    except Exception as error:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            raise error from None
        dropped = {arg: source for arg, source in data.get("time", {}).items()
                   if isinstance(source, str) and arg in params and params[arg].default is None}
        if not dropped:
            raise
        for arg in dropped:
            del data["time"][arg]
        try:
            result = _probe_dated(fn, data, comments, constants_key)
        except Exception:
            data.setdefault("time", {}).update(dropped)
            raise error from None
        if not data["time"]:
            del data["time"]
        for arg, source in dropped.items():
            comments.pop(("time", arg), None)
            comments[("model", f"{constants_key}.{arg}")] = {"python": "None"}
        comments.setdefault(("model", "entry"), (
            f"not bound to the simulation time ([time]): {', '.join(dropped)} (default None); the probe failed "
            f"when it got the {'/'.join(sorted(set(dropped.values())))} and worked with the default"))
        return result


def _probe_dated(fn, data, comments, constants_key="constants"):
    """Call ``fn`` with the probe inputs (see :func:`_probe_once`), checking date-time guesses.

    Arguments taken as date-times only by their name are checked: the call is also made with the
    number 0.0 there. Works only as a date-time: confirmed; only as a number: an ordinary input
    instead; both or neither: left as a guess.

    :param fn: the function or bound step method.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param constants_key: the constants table passed with the call.
    :type constants_key: str
    :return: what ``fn`` returned.
    :raises Exception: what the call raised.
    """
    guessed = [arg for arg in comments.get(DATETIME_BY_NAME, []) if isinstance(data.get("time", {}).get(arg), dict)]
    if not guessed:
        return _probe_once(fn, data, comments, constants_key)
    try:
        result, error = _probe_once(fn, data, comments, constants_key), None
    except Exception as e:
        result, error = None, e
    for arg in guessed:
        source = data["time"].pop(arg)
        data.setdefault("inputs", {})[arg] = {"start": 0.0}
        try:
            number_result = _probe_once(fn, data, comments, constants_key)
            number_ok = True
        except Exception:
            number_ok = False
        if error is not None and number_ok:   # not a date-time: keep it an input
            comments.pop(("time", arg), None)
            comments[("inputs", arg)] = (f"its name {arg!r} suggests a date-time, but the probe failed with one "
                                         "and worked with a number")
            result, error = number_result, None
            continue
        del data["inputs"][arg]
        if not data["inputs"]:
            del data["inputs"]
        data["time"][arg] = source
        if error is None and not number_ok:
            comments[("time", arg)] = comments[("time", arg)].replace(
                f"taken as a date-time from its name {arg!r} (not annotated): check it, or replace this with an "
                "input; ", "a date-time argument (the probe worked with one, not with a number): ")
    if not data.get("time"):
        data.pop("time", None)
    if error is not None:
        raise error
    return result


def _call_with_probe_inputs(fn, data, constants_key):
    """Call ``fn`` once with the inferred inputs, time arguments and constants.

    :param fn: the function or bound step method.
    :param data: the config being inferred.
    :type data: dict
    :param constants_key: the constants table passed with the call.
    :type constants_key: str
    :return: what ``fn`` returned.
    """
    args, kwargs = _probe_args(data, ("inputs",), fn, constants_key)
    kwargs.update(_time_kwargs(data.get("time", {})))
    kwargs.update(_probe_constants(data, constants_key, obj=_owner(fn, constants_key)))
    from fmugen.templates.fmugen_runtime import results_of
    return results_of(fn(*args, **kwargs))


#: A made-up series going up and down, for a step that returns nothing from a constant one.
MADE_UP_VARIED = [0.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0, 1.5]


def _empty(result):
    """Whether a step's result is an empty list, tuple or array (nothing to read outputs from)."""
    if isinstance(result, (list, tuple)):
        return not result
    return getattr(result, "size", None) == 0 and hasattr(result, "shape")


def _retry_varied(call, data, comments):
    """Call again with the made-up number inputs as :data:`MADE_UP_VARIED` (numpy arrays), when the step returned
    nothing for made-up ones. The result when it's not empty, else :data:`_FAILED` (the inputs are put back)."""
    made = {name: info for name, info in data.get("inputs", {}).items()
            if (comments.get(("inputs", name)) == MADE_UP or str(comments.get(("inputs", name), "")).startswith(
                "passed as numpy.atleast_1d")) and isinstance(info.get("start"), (int, float))
            and not isinstance(info.get("start"), bool) and "dimensions" not in info}
    if not made or importlib.util.find_spec("numpy") is None:
        return _FAILED
    saved = {name: dict(info) for name, info in made.items()}
    for info in made.values():
        info.pop("convert", None)
        info.update(dimensions=[len(MADE_UP_VARIED)], start=list(MADE_UP_VARIED), numpy=True)
    try:
        result = call()
    except Exception:
        result = None
    if result is None or _empty(result):
        for name, info in made.items():
            info.clear()
            info.update(saved[name])
        return _FAILED
    for name in made:
        comments[("inputs", name)] = ("no default in the code: made up as 8 values going up and down (the step "
                                      "returned nothing for a constant one); check it")
    return result


def _probe_once(fn, data, comments, constants_key="constants"):
    """:func:`_probe_attempt`; when it returns nothing (an empty list or array), again with the made-up inputs
    varied (:func:`_retry_varied`): rainflow's ``count_cycles(series)`` of one value has no cycles."""
    result = _probe_attempt(fn, data, comments, constants_key)
    if _empty(result):
        varied = _retry_varied(lambda: _call_with_probe_inputs(fn, data, constants_key), data, comments)
        return result if varied is _FAILED else varied
    return result


def _probe_attempt(fn, data, comments, constants_key="constants"):
    """Call ``fn`` with the probe inputs, retrying in forms the model may need.

    If it fails: ``**kwargs`` inputs by position (:func:`_keywords_to_positions`); naive date-time
    epochs in UTC; array inputs that are plain lists as numpy arrays, then torch tensors (when
    installed). Without such arrays, in turn: numbers as one-element numpy arrays, dated pandas
    Series and pint quantities; other made-up values (:func:`_retry_made_up`); optional arguments
    (:func:`_retry_options`); the arguments the error names (:func:`_retry_named`); text; positional
    numbers for a callable without a signature; integers a C++ prototype asks for. The first form
    that works is kept in ``data`` with a comment.

    :param fn: the function or bound step method.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param constants_key: the constants table passed with the call.
    :type constants_key: str
    :return: what ``fn`` returned.
    :raises Exception: the original error, when no retry works.
    """
    def call():
        """One probe call with the current inputs.

        :return: what ``fn`` returned.
        """
        return _call_with_probe_inputs(fn, data, constants_key)

    try:
        return call()
    except Exception as error:
        moved = _keywords_to_positions(data, comments, error)
        if moved:   # the model takes them by position only (neuromancer's forward(*inputs)): again, with every retry
            try:
                return _probe_once(fn, data, comments, constants_key)
            except Exception:
                for name, info in moved.items():
                    info.pop("to", None)
                    comments[("inputs", name)] = KWARGS_COMMENT
                raise error from None
        # A date-time epoch is naive (some models refuse time zones); others need one: retry in UTC
        naive = [source for source in data.get("time", {}).values()
                 if isinstance(source, dict) and datetime.datetime.fromisoformat(source["epoch"]).tzinfo is None]
        if naive:
            for source in naive:
                source["epoch"] += "+00:00"
            try:
                result = call()
            except Exception:
                for source in naive:
                    source["epoch"] = source["epoch"].removesuffix("+00:00")
            else:
                for arg, source in data["time"].items():
                    if any(source is s for s in naive):
                        comments[("time", arg)] += "; epoch in UTC: the probe failed without a time zone"
                return result
        plain = [info for info in data.get("inputs", {}).values()
                 if "dimensions" in info and not info.get("numpy") and not info.get("convert")]
        if not plain:
            result = _retry_as_arrays(call, data, comments)
            if result is _FAILED:
                result = _retry_as_series(call, data, comments)
            if result is _FAILED:
                result = _retry_as_pint(call, data, comments)
            if result is _FAILED:
                result = _retry_made_up(call, data, comments, "inputs", getattr(fn, "__self__", None), error)
            if result is _FAILED:
                result = _retry_options(call, fn, data, comments)
            if result is _FAILED:
                result = _retry_named(call, fn, data, comments, "inputs", error)
            if result is _FAILED:
                result = _retry_as_text(call, data, comments, error)
            if result is _FAILED:
                result = _retry_positional(call, fn, data, comments, error)
            if result is _FAILED:
                result = _retry_prototype_integers(call, fn, data, comments, error)
            if result is _FAILED:
                raise
            return result
        for converter in ARRAY_RETRIES:
            module = "numpy" if converter == "numpy" else converter.partition(":")[0]
            if importlib.util.find_spec(module) is None:
                continue
            for info in plain:
                info.update({"numpy": True} if converter == "numpy" else {"convert": converter})
            try:
                result = call()
            except Exception:
                for info in plain:
                    info.pop("numpy", None), info.pop("convert", None)
                continue
            names = [n for n, info in data["inputs"].items() if any(info is p for p in plain)]
            shown = "numpy arrays" if converter == "numpy" else converter.replace(":", ".") + "(...)"
            for name in names:
                comments[("inputs", name)] = f"passed as {shown}: the probe failed with lists"
            return result
        raise


#: An error asking for text: sentence-transformers' "Expected one of: str, dict, ...", transformers' "text input
#: must be of type `str`".
TEXT_ASKED = re.compile(r"\bstr\b|\bstring\b|\btext\b", re.I)


#: The text a made-up input that must be text is tried with.
MADE_UP_TEXT = "The quick brown fox jumps over the lazy dog."


def _retry_as_text(call, data, comments, error):
    """Call again with the made-up inputs as text, when the error asks for a string: every one, then each alone.

    :param call: makes the probe call.
    :type call: callable
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param error: the probe call's error.
    :type error: Exception
    :return: the result, or :data:`_FAILED`.
    """
    if not TEXT_ASKED.search(str(error)):
        return _FAILED
    made = {n: info for n, info in data.get("inputs", {}).items()
            if comments.get(("inputs", n)) == MADE_UP and "dimensions" not in info and not info.get("convert")}
    if not made:
        return _FAILED
    saved = {n: dict(info) for n, info in made.items()}
    for chosen in [list(made)] + ([[n] for n in made] if len(made) > 1 else []):
        for n in chosen:
            made[n].clear()
            made[n].update({k: v for k, v in saved[n].items() if k not in ("start", "type")}, start=MADE_UP_TEXT)
        try:
            result = call()
        except Exception:
            for n in chosen:
                made[n].clear()
                made[n].update(saved[n])
            continue
        for n in chosen:
            comments[("inputs", n)] = "no default in the code: made up as text, since the probe asked for a string; check it"
        return result
    return _FAILED


#: How many positional arguments an error says a callable without a signature takes: "takes exactly 2
#: arguments (0 given)" (sgp4's C method), "expected 3 arguments".
ARGUMENT_COUNT = re.compile(r"takes (?:exactly )?(\d+) (?:positional )?arguments? \(0 given\)|expected (\d+) arguments?")


#: A C++ prototype in a SWIG error ("Possible C/C++ prototypes are: OT::Normal::computePDF(OT::Scalar const) const").
PROTOTYPE = re.compile(r"[\w:]+\(([^()]*)\)")


#: A C++ parameter that is a number.
NUMBER_PARAMETER = re.compile(r"^(?:const )?(?:\w+::)*(?:Scalar|double|float|int|long|UnsignedInteger|Real|Size|Natural|"
                              r"Integer)(?: const)?(?: ?&)?(?: \w+)?$")


def _retry_positional(call, fn, data, comments, error):
    """Call a callable without a readable signature again with as many made-up positional numbers as the error
    says it takes: the count of "takes exactly 2 arguments (0 given)", else the fewest numbers of a SWIG
    prototype that takes only numbers (openturns' ``computePDF(OT::Scalar const)``). They become inputs
    ``x0``, ``x1``, ... passed by position.

    :param call: makes the probe call.
    :type call: callable
    :param fn: the function or step method.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param error: the probe call's error.
    :type error: Exception
    :return: the result, or :data:`_FAILED`.
    """
    from fmugen.arguments import _only_varargs
    if data.get("inputs") or not _only_varargs(fn, skip_self=False):
        return _FAILED
    text = str(error)
    counts = []
    match = ARGUMENT_COUNT.search(text)
    if match:
        counts.append(int(match.group(1) or match.group(2)))
    for prototype in PROTOTYPE.findall(text):
        parts = [p.strip() for p in prototype.split(",") if p.strip()]
        if parts and all(NUMBER_PARAMETER.match(p) for p in parts):
            counts.append(len(parts))
    for count in sorted(set(c for c in counts if 0 < c <= 8)):
        inputs = data.setdefault("inputs", {})
        for i in range(count):
            inputs[f"x{i}"] = {"start": 1.0, "to": f"pos:{i}"}
        try:
            result = call()
        except Exception:
            data.pop("inputs")
            continue
        for i in range(count):
            comments[("inputs", f"x{i}")] = (f"no default in the code: made up, by position: the probe's error says it "
                                             f"takes {count} numbers; name it and check it")
        return result
    return _FAILED


#: A C++ parameter that is an integer or an enum: "Option::Type", "Size", "Natural", "int".
INTEGER_PARAMETER = re.compile(r"^(?:const )?(?:\w+::)*(?:Type|int|long|Size|Natural|Integer|UnsignedInteger|bool)"
                               r"(?: const)?(?: ?&)?(?: \w+)?$")


def _retry_prototype_integers(call, fn, data, comments, error):
    """Call again with the made-up inputs a SWIG prototype the error lists takes as integers or enums as the
    Integer 1 (QuantLib's ``blackFormula(Option::Type,Real,Real,Real,Real,Real)``: optionType), by their position
    in ``fn``'s signature. The first prototype that works is kept, with a comment.

    :return: the result, or :data:`_FAILED`.
    """
    try:
        names = [p.name for p in inspect.signature(fn).parameters.values() if p.name != "self"]
    except (TypeError, ValueError):
        return _FAILED
    inputs = data.get("inputs", {})
    for prototype in PROTOTYPE.findall(str(error)):
        parts = [p.strip() for p in prototype.split(",") if p.strip()]
        chosen = [names[i] for i, part in enumerate(parts) if i < len(names) and INTEGER_PARAMETER.match(part)
                  and comments.get(("inputs", names[i])) == MADE_UP and isinstance(inputs.get(names[i], {}).get("start"), float)]
        if not chosen:
            continue
        for name in chosen:
            inputs[name].update(start=1, type="Integer")
        try:
            result = call()
        except Exception:
            for name in chosen:
                inputs[name].pop("type")
                inputs[name]["start"] = 1.0
            continue
        for name in chosen:
            comments[("inputs", name)] = "no default in the code: made up as 1, an integer the C++ prototype asks for; check it"
        return result
    return _FAILED


#: How many sets of optional arguments :func:`_retry_options` tries, at most.
MAX_OPTION_SETS = 40


def _retry_options(call, fn, data, comments, section="inputs"):
    """Call again giving some numeric arguments whose default is None, made up as 1.0 (1 for a count or size,
    :data:`fmugen.arguments.COUNT_NAME`): one alone, then two, then a count alone as each of
    :data:`OTHER_NUMBERS`.

    For models that need some of several optional arguments (fluids' ``Reynolds(V, D, rho=None, mu=None,
    nu=None)``: "Either density and viscosity, or kinematic viscosity is needed"; pyet's pressure or
    elevation; reservoirpy's ``Reservoir(units=None)`` when building it). The smallest set that works
    becomes variables of ``section``, with a comment.

    :param call: makes the probe call.
    :type call: callable
    :param fn: the function or step method.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param section: ``"inputs"`` (the step's arguments) or ``"parameters"`` (the constructor's).
    :type section: str
    :return: the result, or :data:`_FAILED`.
    """
    from fmugen.arguments import COUNT_NAME
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return _FAILED
    taken = {*data.get("inputs", {}), *data.get("parameters", {}), *data.get("time", {})}
    options = [p.name for p in params if p.default is None and p.name not in taken
               and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
               and static.unwrap_optional(p.annotation) in (p.empty, float, int)]
    if not options:
        return _FAILED
    sets = [[n] for n in options] + [list(pair) for pair in itertools.combinations(options, 2)]
    # a size alone also larger: reservoirpy's Reservoir(units) steps with more than a few units
    sets = [(chosen, 1) for chosen in sets[:MAX_OPTION_SETS]] + [
        ([n], v) for n in options if COUNT_NAME.search(n) for v in OTHER_NUMBERS]
    inputs = data.setdefault(section, {})
    for chosen, value in sets:
        for name in chosen:
            inputs[name] = {"start": value if COUNT_NAME.search(name) else float(value)}
        try:
            result = call()
        except Exception as failed:
            # with them, the inputs may have to be quantities (fluids.units' Reynolds: rho and mu, in units)
            if (result := _converted_retry(call, data, comments, failed)) is _FAILED:
                for name in chosen:
                    inputs.pop(name)
                continue
        for name in chosen:
            for table in ("constants", "call_constants"):
                comments.pop(("model", f"{table}.{name}"), None)
            comments[(section, name)] = (f"default None, made up as {inputs[name]['start']}: the probe failed "
                                         f"without {' and '.join(chosen)}; check it")
        return result
    if not inputs:
        data.pop(section)
    return _FAILED


#: An error asking for a quantity with units: fluids.units' "1.0 has no quantity", "dimensionless".
QUANTITY_ASKED = re.compile(r"quantit|dimensionless|\bunits?\b", re.I)


def _converted_retry(call, data, comments, error):
    """After giving more arguments, the conversion the error asks for, which the retries before tried without them:
    the inputs as dated series ("'float' object has no attribute 'time'": pyet) or pint quantities
    (:data:`QUANTITY_ASKED`: fluids.units). The result, or :data:`_FAILED`."""
    if "object has no attribute" in str(error):
        return _retry_as_series(call, data, comments)
    if QUANTITY_ASKED.search(str(error)):
        return _retry_as_pint(call, data, comments)
    return _FAILED


#: A name an error message quotes: "Provide `context_length` and `prediction_length`".
ERROR_NAME = re.compile(r"[`'\"](\w+)[`'\"]")


#: Names of lengths, made up as an Integer like counts: ``context_length``, ``max_len``, ``horizon``.
LENGTH_NAME = re.compile(r"(^|_)(length|len|horizon|steps|window)$")


#: Values tried for the arguments an error names: a length often has to be one of a model's sizes (512).
NAMED_VALUES = (1, 4, 16, 100, 512)


def _retry_named(call, fn, data, comments, section, error):
    """Call again giving the arguments the error message quotes (transformers-style "Provide `context_length` and
    `prediction_length`"), that ``fn`` declares with a None default or takes through ``**kwargs``: all of them
    made up as each of :data:`NAMED_VALUES` in turn (Integer for counts and lengths). The first that works is kept,
    with a comment.

    :return: the result, or :data:`_FAILED`.
    """
    from fmugen.arguments import COUNT_NAME
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return _FAILED
    keywords = any(p.kind == p.VAR_KEYWORD for p in params.values())
    taken = {*data.get("inputs", {}), *data.get("parameters", {}), *data.get("time", {}),
             *(n for table in ("constants", "call_constants") for n in data["model"].get(table, {}))}
    def named(error):
        # quoted, or a word naming an argument with a None default (pyet's "provide either pressure or the elevation")
        words = ERROR_NAME.findall(str(error)) + [w for w in re.findall(r"\b[a-z]\w{2,}\b", str(error)) if w in params]
        return [n for n in dict.fromkeys(words) if n not in taken and not n[0].isdigit()
                and (n in params and params[n].default is None and params[n].kind in (
                     inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                     or n not in params and keywords and "_" in n)][:4]

    names = named(error)
    if not names:
        return _FAILED
    variables = data.setdefault(section, {})
    for _ in range(4):   # given those, the error may name others (pyet: then the humidity)
        failures = []
        for value in NAMED_VALUES:
            for name in names:
                counted = COUNT_NAME.search(name) or LENGTH_NAME.search(name)
                variables[name] = {"start": value if counted else float(value)}
            try:
                result = call()
            except Exception as failed:
                failures.append(failed)
                # with them, the inputs may have to be dated series (pyet's rs: "'float' object has no attribute
                # 'time'"), which the retries before this one tried without them
                if (result := _converted_retry(call, data, comments, failed)) is _FAILED:
                    continue
            for name in names:
                comments[(section, name)] = (f"made up as {variables[name]['start']}: the probe failed without it "
                                             f"({type(error).__name__}: it asks for {', '.join(names)}); check it")
            return result
        more = [n for failed in failures for n in named(failed) if n not in names]
        if not more and any("'NoneType' object" in str(failed) for failed in failures):
            # an argument left None the error doesn't name (pyet's rs, then lat): each in turn, as 16
            more = _none_needed(call, params, names, taken, variables, failures)
        if not more:
            break
        names += more[:1]   # one: "either `rh` or `tmin`" asks for one of them
    for name in names:
        variables.pop(name)
    if not variables:
        data.pop(section)
    return _FAILED


def _none_needed(call, params, names, taken, variables, failures):
    """The first argument with a None default (not ``names``, not ``taken``) that, made up as 16.0 with ``names``
    as 16, gets the call past a "'NoneType' object" error: ``[name]``, or []."""
    from fmugen.arguments import COUNT_NAME
    candidates = [n for n, p in params.items() if p.default is None and n not in names and n not in taken
                  and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)][:12]
    for candidate in candidates:
        for name in [*names, candidate]:
            counted = COUNT_NAME.search(name) or LENGTH_NAME.search(name)
            variables[name] = {"start": 16 if counted else 16.0}
        try:
            call()
        except Exception as failed:
            plain = re.sub(r"'\w+' object", "object", str(failed))   # 'float' for 'NoneType': not past it
            if any(plain == re.sub(r"'\w+' object", "object", str(f)) for f in failures):
                continue
        variables.pop(candidate)
        return [candidate]
    for candidate in candidates:
        variables.pop(candidate, None)
    return []


#: Sentinel: a retry (:func:`_retry_as_arrays`, :func:`_retry_made_up`, ...) found no call that works.
_FAILED = object()


#: The comment of an input passed through ``**kwargs`` (set by :func:`fmugen.arguments._arguments`).
KWARGS_COMMENT = "start from --start (passed through **kwargs)"


def _keywords_to_positions(data, comments, error):
    """Pass the ``--start`` inputs that went through ``**kwargs`` by position instead, after the others.

    For a step method declaring ``(*args, **kwargs)`` that accepts only positional data
    (neuromancer's ``Block.forward(*inputs)``): the probe call failed with "unexpected keyword
    argument". Positions follow those already taken, in ``--start`` order.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param error: what the probe call raised.
    :type error: Exception
    :return: ``{name: info}`` of the inputs moved, empty when it doesn't apply.
    :rtype: dict
    """
    if not isinstance(error, TypeError) or "unexpected keyword argument" not in str(error):
        return {}
    inputs = data.get("inputs", {})
    moved = {n: info for n, info in inputs.items() if comments.get(("inputs", n)) == KWARGS_COMMENT and "to" not in info}
    if not moved:
        return {}
    taken = [int(str(info["to"]).split(":")[1]) for info in inputs.values() if str(info.get("to", "")).startswith("pos:")]
    taken += [int(k) for k in data["model"].get("call_constants", {}) if str(k).isdigit()]
    position = max(taken, default=-1) + 1
    for name, info in moved.items():
        info["to"] = f"pos:{position}"
        comments[("inputs", name)] = ("start from --start, passed by position: the model takes no keyword "
                                      "arguments there (check the order)")
        position += 1
    return moved


def _construct(build, data, comments):
    """Build the model object for the probe, retrying list parameters as numpy arrays, then torch tensors.

    As the step's inputs are (:func:`_probe_once`): GPy's ``GPRegression(X, Y)`` takes only arrays.
    Without list parameters, other made-up values (:func:`_retry_made_up`), optional arguments
    (:func:`_retry_options`) and the arguments the error names (:func:`_retry_named`) are tried. The
    first form that works is kept in ``data`` with a comment.

    :param build: the class, ``--create`` classmethod or factory.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :return: the object.
    :raises Exception: the original error, when no retry works.
    """
    def make():
        args, kwargs = _probe_args(data, ("parameters",), build, "constants")
        return build(*args, **_probe_constants(data, "constants"), **kwargs)

    try:
        return make()
    except ConstantError:
        raise
    except Exception as error:
        plain = {n: info for n, info in data.get("parameters", {}).items()
                 if "dimensions" in info and not info.get("numpy") and not info.get("convert")}
        if not plain:
            obj = _retry_made_up(make, data, comments, "parameters")
            if obj is _FAILED:
                obj = _retry_options(make, build, data, comments, "parameters")
            if obj is _FAILED:
                obj = _retry_named(make, build, data, comments, "parameters", error)
            if obj is _FAILED:
                raise
            return obj
        for converter in ARRAY_RETRIES:
            module = "numpy" if converter == "numpy" else converter.partition(":")[0]
            if importlib.util.find_spec(module) is None:
                continue
            for info in plain.values():
                info.update({"numpy": True} if converter == "numpy" else {"convert": converter})
            try:
                obj = make()
            except Exception:
                for info in plain.values():
                    info.pop("numpy", None), info.pop("convert", None)
                continue
            shown = "numpy arrays" if converter == "numpy" else converter.replace(":", ".") + "(...)"
            for name in plain:
                comments[("parameters", name)] = f"passed as {shown}: building the model failed with lists"
            return obj
        raise


def _retry_as_arrays(call, data, comments):
    """Call again with each number input as a one-element numpy array (``convert = "numpy:atleast_1d"``).

    Some libraries take only arrays (thermofeel's calculate_heat_index_simplified calls
    numpy.nonzero on its inputs).

    :param call: makes the probe call.
    :type call: callable
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :return: the result, or :data:`_FAILED`.
    """
    numbers = {name: info for name, info in data.get("inputs", {}).items()
               if isinstance(info.get("start"), (int, float)) and not isinstance(info.get("start"), bool)
               and "dimensions" not in info and not info.get("convert") and not info.get("numpy")}
    if not numbers or importlib.util.find_spec("numpy") is None:
        return _FAILED
    for info in numbers.values():
        info["convert"] = "numpy:atleast_1d"
    try:
        result = call()
    except Exception:
        for info in numbers.values():
            del info["convert"]
        return _FAILED
    for name in numbers:
        comments[("inputs", name)] = "passed as numpy.atleast_1d(...): the probe failed with plain numbers"
    return result


#: Shapes tried for a made-up input that fails as a number: a vector of 3, 1×1, 1×3 and 3×3 matrices, and a
#: vector of 4 (commpy's modulate takes a whole number of 2- or 4-bit symbols).
MADE_UP_SHAPES = ([3], [1, 1], [1, 3], [3, 3], [4])


#: An error asking for an integer: "'float' object cannot be interpreted as an integer", "must be int".
INTEGER_ASKED = re.compile(r"\binteger\b|\bmust be int\b|expected int\b|\bint64\b", re.I)


#: Channels-first batches on a grid, tried last for a torch module: neuraloperator's FNO(n_modes=[1, 1]) takes
#: (batch, channels, height, width).
MADE_UP_GRIDS = ([1, 1, 8], [1, 1, 8, 8])


#: Names of a list of sizes, one per layer: deepxde's ``layer_sizes``, ``hidden_sizes``, ``widths``.
PLURAL_SIZES = re.compile(r"(^|_)(sizes|widths|hsizes|layers|hidden_dims|channels_list)$")


#: Numbers tried last for made-up values that fail as 1.0, 0.0, arrays and lists.
OTHER_NUMBERS = (4, 16, 100)


#: Attributes in which a fitted model keeps how many features it takes: scikit-learn's (catboost,
#: xgboost, lightgbm), tabnet's, and common others.
FEATURE_COUNTS = ("n_features_in_", "input_dim", "n_features", "num_features", "input_size")


def _rebuild_with_options(build, method_name, data, comments):
    """When the step fails, build the object again giving numeric options whose default is None
    (:func:`_retry_options`), and keep that if the step works then: reservoirpy's ``Reservoir()`` builds with
    no ``units``, but its step needs them.

    :return: a new object, built as the step starts from, or None.
    """
    def build_and_step():
        _probe_call(_step_method(_build_set_up(build, data), method_name, data), data, comments,
                    constants_key="call_constants")
        return _build_set_up(build, data)

    constructor = build.__init__ if inspect.isclass(build) else build
    obj = _retry_options(build_and_step, constructor, data, comments, "parameters")
    return None if obj is _FAILED else obj


def _rebuild_made_up_as_zero(build, method_name, data, comments):
    """When the step fails, build the object again with its made-up parameters (no default, no
    ``--start``) as 0.0, and keep that if the step works then: a made-up 1.0 builds, but the step
    may fail on it (quantecon's ``LinearStateSpace(A=1.0, ...)``: ``I - A`` is singular).

    :param build: the class, ``--create`` classmethod or factory.
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :return: a new object, built as the step starts from (the trial one was stepped), or None.
    """
    made = {name: info for name, info in data.get("parameters", {}).items()
            if (comments.get(("parameters", name)) == MADE_UP or str(comments.get(("parameters", name), "")).startswith(
                "default None, made up as")) and isinstance(info.get("start"), (int, float))
            and not isinstance(info.get("start"), bool) and info["start"] != 0}
    if not made:
        return None
    saved = {name: info["start"] for name, info in made.items()}
    # then larger: reservoirpy's Reservoir(units) builds with 1 unit, but its step needs more
    for value in (0, *OTHER_NUMBERS):
        for info in made.values():
            info["start"] = type(saved[next(n for n in made if made[n] is info)])(value)
        try:   # not _construct: its retries would change the parameters again (pedon's n = 0.0 fails)
            _probe_call(_step_method(_build_set_up(build, data), method_name, data), data, comments,
                        constants_key="call_constants")
            obj = _build_set_up(build, data)
        except Exception:
            continue
        for name in made:
            comments[("parameters", name)] = (f"no default in the code: made up, as {made[name]['start']} "
                                              "(the step failed with 1.0); check it")
        return obj
    for name, info in made.items():
        info["start"] = saved[name]
    return None


#: A shape an error message names: "cannot reshape array of size 1 into shape (7)", "expected shape (1, 4)".
ERROR_SHAPE = re.compile(r"shape \(?\s*(\d+(?:\s*,\s*\d+)*)\s*,?\s*\)?")


def _error_shapes(error):
    """The shapes an error message names (:data:`ERROR_SHAPE`), with no dimension of 1 alone, at most 3."""
    shapes = []
    for match in ERROR_SHAPE.finditer(str(error or "")):
        shape = [int(x) for x in re.split(r"\s*,\s*", match.group(1))]
        if shape not in shapes and shape != [1] and 0 < math.prod(shape) <= 10_000:
            shapes.append(shape)
    return shapes[:3]


def _retry_made_up(call, data, comments, section, obj=None, error=None):
    """Call again with other values for the made-up arguments of ``section`` (no default, no ``--start``).

    The made-up start is 1.0 (not 0.0, which models refuse when they check it's positive or divide
    by it: blackscholes' "Asset price (S) needs to be larger than 0", pedon's division by zero); some
    models need an array (ahrs' quaternion, GPy's ``X``, scikit-learn's ``predict(X)``). Tried in turn:
    those whose annotation or docstring says they are arrays as arrays of ones in that shape
    (:func:`fmugen.arguments._shape_key`), the others as made up; a value of the object's gymnasium
    space; the Integer 1 when the error asks for an integer; lists of sizes for names like
    ``layer_sizes``; every one as 0.0 (quantecon's matrices, singular when all ones); annotated tensors
    as torch tensors; every one, then each one alone, as an array of ones in the shapes the error names,
    the fitted model's feature count gives and :data:`MADE_UP_SHAPES`; for a torch module, as torch
    tensors (norse, snntorch); every one, then each one alone, as a plain list of two ones, of ints for
    sizes (FNO's ``n_modes``, haversine's points); integer arrays; :data:`OTHER_NUMBERS`. The first
    that works is kept, with a comment.

    :param call: makes the probe call (or builds the object).
    :type call: callable
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param section: ``"inputs"`` or ``"parameters"``.
    :type section: str
    :param obj: the model object the step is called on, if any: a fitted scikit-learn-style model
        gives its number of features (:data:`FEATURE_COUNTS`), so ``(1, n)`` and ``(n,)`` are tried first.
    :param error: the error of the failed call: the shapes it names are tried first, and an Integer when it
        asks for one.
    :return: the result, or :data:`_FAILED`.
    """
    made = {name: info for name, info in data.get(section, {}).items()
            if comments.get((section, name)) == MADE_UP and "dimensions" not in info and not info.get("convert")
            and isinstance(info.get("start"), (int, float)) and not isinstance(info.get("start"), bool)}
    if not made:
        return _FAILED
    saved = {name: dict(info) for name, info in made.items()}

    def scalar(name, value=1):
        made[name].clear()
        made[name].update(saved[name], start=float(value) if isinstance(saved[name]["start"], float) else value)

    def array(name, shape, tensor=False, ints=False):
        made[name].clear()
        made[name].update({k: v for k, v in saved[name].items() if k not in ("start", "type")},
                          dimensions=list(shape), start=[1 if ints else 1.0] * math.prod(shape),
                          **({"convert": "torch:tensor"} if tensor else {"numpy": True}),
                          **({"type": "Int64"} if ints else {}))

    def listed(name, size):   # a plain list, not an array: FNO's n_modes (sizes), haversine's points
        made[name].clear()
        from fmugen.arguments import COUNT_NAME
        counts = isinstance(saved[name]["start"], int) or bool(COUNT_NAME.search(name))   # num_channels: list[int]
        made[name].update({k: v for k, v in saved[name].items() if k not in ("start", "type", "numpy")},
                          dimensions=[size], start=[1 if counts else 1.0] * size, **({"type": "Int64"} if counts else {}))

    def valued(name, value):   # a value from the object's own space
        made[name].clear()
        if isinstance(value, list):
            made[name].update({k: v for k, v in saved[name].items() if k not in ("start", "type")},
                              dimensions=[len(value)], start=value, numpy=True)
        else:
            made[name].update(saved[name], start=value)

    tries = [("as 0.0", {n: 0 for n in made})]
    sized = [n for n in made if PLURAL_SIZES.search(n)]
    if sized:   # deepxde's FNN(layer_sizes): a list of sizes, not an array (a (1, 1) array builds no layers)
        tries.insert(0, ("as lists of two ones", {n: ("list", 2) if n in sized else None for n in made}))
    if INTEGER_ASKED.search(str(error or "")):   # nengo's run_steps(steps): "cannot be interpreted as an integer"
        tries.insert(0, ("as the Integer 1", {n: ("integer", 1) for n in made}))
    spaced = {n: _space_value(obj, n) for n in made}
    if any(v is not None for v in spaced.values()):   # gymnasium's step(action): a value of action_space
        tries.insert(0, ("from the model's action or observation space",
                         {n: ("value", v) if v is not None else None for n, v in spaced.items()}))
    documented = {n: comments[_shape_key(section, n)] for n in made if _shape_key(section, n) in comments}
    if documented and importlib.util.find_spec("numpy") is not None:
        tries.insert(0, ("the arrays as arrays of ones in their documented shape, the others as made up",
                         {n: documented.get(n) for n in made}))
    tensors = {n: comments[_shape_key(section, n, tensor=True)] for n in made
               if _shape_key(section, n, tensor=True) in comments}
    if tensors and importlib.util.find_spec("torch") is not None:   # annotated torch.Tensor: chronos' predict(inputs)
        tries[1:1] = [(f"the tensors as torch tensors of ones, shape {tuple(s)}, the others as made up",
                       {n: ("tensor", s) if n in tensors else None for n in made})
                      for s in [list(next(iter(tensors.values())))] + list(MADE_UP_SHAPES) + [[1, 64]]]
    if importlib.util.find_spec("numpy") is not None:
        features = None
        for attribute in FEATURE_COUNTS:   # a fitted model knows how many features predict(X) takes
            try:
                features = getattr(obj, attribute, None) if obj is not None else None
            except Exception:
                continue
            if isinstance(features, int) and not isinstance(features, bool):
                break
        shapes = ([[1, features], [features]] if isinstance(features, int) and features > 0 else []) + list(MADE_UP_SHAPES)
        shapes = _error_shapes(error) + [s for s in shapes if s not in _error_shapes(error)]   # roboticstoolbox's (7)
        tries += [(f"as arrays of ones, shape {tuple(s)}", {n: s for n in made}) for s in shapes]
        if len(made) > 1:
            tries += [(f"{n} as an array of ones, shape {tuple(s)}, the others as made up",
                       {m: s if m == n else None for m in made}) for n in made for s in shapes]
    if _torch_module(obj):   # a torch module's forward() takes tensors (norse, snntorch)
        tries += [(f"as torch tensors of ones, shape {tuple(s)}", {n: ("tensor", s) for n in made})
                  for s in list(MADE_UP_SHAPES) + _object_shapes(obj) + list(MADE_UP_GRIDS)]
    if "torch" in sys.modules:
        # a float32 batch as wide as made-up training data (skorch's predict(X) on a torch network trained on it)
        tries += [(f"as torch tensors of ones, shape {(1, MADE_UP_TRAINING[1])}",
                   {n: ("tensor", [1, MADE_UP_TRAINING[1]]) for n in made})]
    tries += [("as lists of two ones", {n: ("list", 2) for n in made})]
    if len(made) > 1:
        tries += [(f"{n} as a list of two ones, the others as made up", {m: ("list", 2) if m == n else None for m in made})
                  for n in made]
    if importlib.util.find_spec("numpy") is not None:   # indices or bits: commpy's modulate(input_bits)
        tries += [(f"as integer arrays of ones, shape {tuple(s)}", {n: ("ints", s) for n in made}) for s in ([3], [4])]
    # a larger number: commpy's QAMModem(m) takes a power of 4, reservoirpy's Reservoir(units) more than a few
    tries += [(f"as {v}", {n: ("number", v) for n in made}) for v in OTHER_NUMBERS]
    if len(made) > 1:
        tries += [(f"{n} as {v}, the others as made up", {m: ("number", v) if m == n else None for m in made})
                  for n in made for v in OTHER_NUMBERS]
    for how, plan in tries:
        for name, shape in plan.items():
            if isinstance(shape, tuple) and shape[0] == "list":
                listed(name, shape[1])
            elif isinstance(shape, tuple) and shape[0] == "tensor":
                array(name, shape[1], tensor=True)
            elif isinstance(shape, tuple) and shape[0] == "value":
                valued(name, shape[1])
            elif isinstance(shape, tuple) and shape[0] == "ints":
                array(name, shape[1], ints=True)
            elif isinstance(shape, tuple) and shape[0] == "integer":
                made[name].clear()
                made[name].update({k: v for k, v in saved[name].items() if k != "type"}, start=shape[1], type="Int64")
            elif isinstance(shape, tuple) and shape[0] == "number":
                scalar(name, shape[1])
            else:
                scalar(name) if shape is None else scalar(name, 0) if shape == 0 else array(name, shape)
        try:
            result = call()
        except Exception:
            continue
        for name in made:
            comments[(section, name)] = (f"no default in the code: made up, {how} (the probe failed with 1.0); "
                                         "check it")
        return result
    for name, info in made.items():
        info.clear()
        info.update(saved[name])
    return _FAILED


def _space_value(obj, name):
    """A value of the gymnasium-style space an input takes: ``observation_space`` for an argument named like an
    observation (stable-baselines3's ``predict(observation)``), else ``action_space`` (an environment's
    ``step(action)``). The start of a ``Discrete`` space; the middle of a ``Box`` (0 where unbounded), as a list.

    :param obj: the model object, or None.
    :param name: the argument.
    :type name: str
    :return: an int, a list of floats, or None when there's no such space.
    """
    space = getattr(obj, "observation_space" if "obs" in name.lower() else "action_space", None) \
            if obj is not None else None
    if space is None:
        return None
    try:
        if hasattr(space, "n") and not hasattr(space, "low"):   # Discrete
            return int(getattr(space, "start", 0))
        import numpy
        low, high = numpy.asarray(space.low, float), numpy.asarray(space.high, float)
        middle = numpy.where(numpy.isfinite(low) & numpy.isfinite(high), (low + high) / 2, 0.0)
        return [float(x) for x in middle.ravel()]
    except Exception:
        return None


def _object_shapes(obj):
    """Batch shapes from the lengths a model object (or its ``config``) keeps: ``(1, length, channels)`` and
    ``(1, length)`` for granite's TTM (``config.context_length``, ``config.num_input_channels``).

    :return: up to 4 shapes.
    :rtype: list[list[int]]
    """
    lengths, channels = [], []
    for holder in (obj, getattr(obj, "config", None)):
        try:
            items = vars(holder).items() if holder is not None else ()
        except TypeError:
            continue
        for name, value in items:
            if isinstance(value, int) and not isinstance(value, bool) and 1 < value <= 4096:
                if LENGTH_NAME.search(name) and value not in lengths:
                    lengths.append(value)
                elif "channel" in name and value not in channels:
                    channels.append(value)
    shapes = []
    for length in lengths[:2]:
        shapes += [[1, length, (channels or [1])[0]], [1, length]]
    return shapes


def _torch_module(obj):
    """Whether ``obj`` is a torch module (an ``nn.Module`` instance), without importing torch."""
    torch = sys.modules.get("torch")
    module = getattr(getattr(torch, "nn", None), "Module", None)
    return module is not None and isinstance(obj, module)


def _retry_as_series(call, data, comments):
    """Call again with each number input as a one-value pandas Series indexed by a date (``convert = "series"``).

    Some libraries read the date from their inputs' index (pyet's ``pm_fao56`` takes the day of
    the year from ``rs.index``). The epoch is today at 00:00, as for date-time arguments.

    :param call: makes the probe call.
    :type call: callable
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :return: the result, or :data:`_FAILED`.
    """
    numbers = {name: info for name, info in data.get("inputs", {}).items()
               if isinstance(info.get("start"), (int, float)) and not isinstance(info.get("start"), bool)
               and "dimensions" not in info and not info.get("convert") and not info.get("numpy")}
    if not numbers or importlib.util.find_spec("pandas") is None:
        return _FAILED
    epoch = datetime.datetime.combine(datetime.date.today(), datetime.time()).isoformat()
    for info in numbers.values():
        info.update({"convert": "series", "epoch": epoch})
    try:
        result = call()
    except Exception:
        for info in numbers.values():
            del info["convert"], info["epoch"]
        return _FAILED
    for name in numbers:
        comments[("inputs", name)] = ("passed as a pandas Series dated epoch + t: the probe failed with plain numbers "
                                      "and arrays; set the epoch you need")
    return result


#: pint base dimension -> its SI unit, for an argument whose error names only its dimension (metpy).
PINT_BASE_UNITS = {"[length]": "m", "[mass]": "kg", "[time]": "s", "[temperature]": "K", "[current]": "A",
                   "[substance]": "mol", "[luminosity]": "cd", "dimensionless": "dimensionless"}


def _pint_unit(message, numbers):
    """The input a pint unit error is about, and the unit it asks for; or None.

    Reads fluids.units' ``Converting 2.5 dimensionless to units of m/s`` (the input is found by
    its value) and metpy's ``temperature requires "[temperature]"`` (the name in backquotes) (a dimension, given its SI
    unit: K).

    :param message: the error message.
    :type message: str
    :param numbers: ``{name: info}`` of the inputs still dimensionless.
    :type numbers: dict
    :rtype: tuple[str, str] or None
    """
    import re
    match = re.search(r"Converting (\S+) dimensionless to units of (\S+)", message)
    if match:
        try:
            value = float(match[1])
        except ValueError:
            return None
        names = [n for n, info in numbers.items() if float(info["start"]) == value]
        return (names[0], match[2].rstrip(".,")) if len(names) == 1 else None
    match = re.search(r"`(\w+)` requires \"([^\"]+)\"", message)
    if match and match[1] in numbers:
        unit = match[2]
        for dimension, base in PINT_BASE_UNITS.items():
            unit = unit.replace(dimension, base)
        return (match[1], unit) if "[" not in unit else None
    return None


def _retry_as_pint(call, data, comments):
    """Call again with each number input as a pint quantity (``convert = "pint"``), learning units from the errors.

    Libraries that check units (fluids.units, metpy) refuse plain numbers. Every number input
    starts dimensionless; each error that names an input's unit sets it, until the call works.
    The start values are taken as being in those units.

    :param call: makes the probe call.
    :type call: callable
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :return: the result, or :data:`_FAILED`.
    """
    numbers = {name: info for name, info in data.get("inputs", {}).items()
               if isinstance(info.get("start"), (int, float)) and not isinstance(info.get("start"), bool)
               and "dimensions" not in info and not info.get("convert") and not info.get("numpy")}
    if not numbers or importlib.util.find_spec("pint") is None:
        return _FAILED
    for info in numbers.values():
        info.update({"convert": "pint", "unit": "dimensionless"})
    pending = dict(numbers)
    for _ in range(len(numbers) + 1):
        try:
            result = call()
        except Exception as e:
            found = _pint_unit(str(e), pending)
            if found is None:
                break
            name, unit = found
            pending.pop(name)["unit"] = unit
            continue
        for name in numbers:
            comments[("inputs", name)] = ("passed as a pint quantity: the probe failed with plain numbers; the unit is "
                                          "the one the model asked for, so check the start is in it")
        return result
    for info in numbers.values():
        del info["convert"], info["unit"]
    return _FAILED
