"""Probing, output detection and state detection for interface inference."""
import ast
import dataclasses
import datetime
import enum
import importlib
import inspect
import itertools
import math
import pickle
import re
import sys
import textwrap
import types
import typing
from collections.abc import Mapping

from fmugen import static
from fmugen.config import InterfaceError
from fmugen.toml_output import _key, _value
from fmugen.arguments import (
    MADE_UP, _shape_key, SCALARS, PROBED, DATETIME_BY_NAME, NO_SIGNATURE,
    _fmi_value, _array_info, _scalar_of, _number_like, _typed, _type_name, _same,
    _valid_name, _dotted_name, _misfit, _unfit_value, _unfit, _guessed,
    _apply_converts, _arguments, _only_varargs,
    _start_error, _state_base,
)

#: Step size (and end time) passed to a probe call's time arguments.
PROBE_STEP_SIZE = 1.0

def _run_setup(data, obj=None):
    """Run ``[model] setup`` steps for the probe: functions before construction, methods after.

    The same order as :meth:`Engine._run_setup`, so the probe sees the model as the FMU will.

    :param data: the config being inferred.
    :type data: dict
    :param obj: the constructed object (runs the method steps), or None (runs the function steps).
    """
    from fmugen.config import _setup
    from fmugen.templates.fmugen_runtime import (
        get_path,
        resolve_constant,
        resolve_reference,
    )
    for step in _setup(data["model"].get("setup", []), is_class=True):
        is_method = ":" not in step["call"]
        if is_method != (obj is not None):
            continue
        target = get_path(obj, step["call"]) if is_method else resolve_reference(step["call"])
        target(*[resolve_constant(a) for a in step["args"]],
               **{k: resolve_constant(v) for k, v in step["kwargs"].items()})

def _probe_class(cls, method_name, data, comments, arrays, call_starts, create, factory, converts):
    """``--probe`` for a class: construct the object, call the step method once, and read what changed.

    Outputs come from the return value, from properties, from GEKKO results and from numeric
    attributes the step set (new: outputs) or changed (existing: locals). Also decides
    ``init_call`` (a step that changes nothing can run at initialization) and ``save_state``
    (:func:`_check_picklable`). A failing construction or call is noted and falls back to reading
    the code.

    :param cls: the model class (None for a factory function).
    :type cls: type or None
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param call_starts: ``--start`` values for the step method's arguments.
    :type call_starts: dict
    :param create: a classmethod that builds the object (``--create``), or None.
    :type create: str or None
    :param factory: a factory function that builds the object, or None.
    :type factory: callable or None
    :param converts: ``--convert`` options.
    :type converts: dict
    :return: ``(object or None, returned keys, attribute names seen)``.
    :rtype: tuple
    :raises ConstantError: a configured constant can't be computed.
    :raises InterfaceError: the factory's object has no such method.
    """
    try:
        _run_setup(data)
        build = factory or (getattr(cls, create) if create else cls)
        obj = _construct(build, data, comments)
        _run_setup(data, obj)
    except ConstantError:
        raise
    except Exception as e:
        comments[("outputs", None)] = f"probe construction failed ({e!r}); outputs below are read from the code"
        if cls is not None and ":" not in method_name:
            _arguments(_forwarded(getattr(cls, method_name), method_name, cls), "inputs", data, comments,
                       skip_self=not data["model"].get("call_on_class"),
                       constants_key="call_constants", arrays=arrays, starts=call_starts, probe=True)
            _apply_converts(data, converts)
        return None, set(), set()
    object_step = ":" in method_name   # "module:function(self, ...)": its arguments are in the call text
    if object_step:
        method = None
    elif factory:
        method = getattr(obj, method_name, None)   # bound: no self argument
        if method_name == "__call__" and isinstance(obj, (types.FunctionType, types.MethodType)):
            method = obj   # a function made at runtime (sympy's lambdify): its own signature, not (*args, **kwargs)
        if not callable(method):
            raise InterfaceError(f"the object {factory.__name__}() returns has no method {method_name!r}")
    else:
        method = getattr(cls, method_name)

    # Parameters kept as writable attributes *may* be tunable, but only if the step method
    # reads the attribute (not a value derived from it in __init__), so only suggest it.
    for name in data.get("parameters", {}):
        if _writable_attribute(obj, name) and _same(getattr(obj, name), data["parameters"][name].get("start")):
            comments.setdefault(("parameters", name), "kept as an attribute: add variability = \"tunable\" if the step reads it")

    if object_step:
        if call_starts:
            raise InterfaceError(f"--start {', '.join(call_starts)}: {method_name} takes its arguments in the call "
                                 "text; write them there, or bind inputs by hand (to = \"pos:N\" after those)")
    else:
        _arguments(_forwarded(method, method_name, obj if factory else cls), "inputs", data, comments,
                   skip_self=not factory and not data["model"].get("call_on_class"),
                   constants_key="call_constants",
                   arrays=arrays, starts=call_starts, probe=True)
    _apply_converts(data, converts)
    before, state_before = _numeric_attributes(obj, arrays), _state_bytes(obj)
    dynamic_before = _dynamic_values(obj, arrays)
    descriptors_before = _descriptor_values(obj, arrays)
    # what reading the properties and C attributes raises before the first step (pydmd before fit(),
    # ephem before compute()): those outputs need the model called at initialization
    unread_before = _unreadable(obj)
    try:
        result = _probe_call(_step_method(obj, method_name, data), data, comments, constants_key="call_constants")
    except ConstantError:
        raise
    except Exception as e:
        rebuilt = _rebuild_made_up_as_zero(build, method_name, data, comments)
        if rebuilt is None:   # options left None build, but the step needs them (reservoirpy's units)
            rebuilt = _rebuild_with_options(build, method_name, data, comments)
        if rebuilt is not None:   # the made-up parameters as 0.0 (quantecon's A = 1.0: a singular matrix)
            obj = rebuilt
        elif _reset_first(obj, method_name, data, comments):
            pass
        else:
            set_up = _setup_first(build, obj, method_name, e, data, comments)
            if not isinstance(set_up, str):
                obj = set_up
            else:
                comments[("outputs", None)] = (f"probe call failed ({e!r}); {set_up + '; ' if set_up else ''}"
                                               "outputs below are read from the code")
                return obj, set(), set(getattr(obj, "__dict__", {}))
        # reset() or a set-up worked: the object is the one the step starts from, so its "before" is now
        before, state_before = _numeric_attributes(obj, arrays), _state_bytes(obj)
        dynamic_before = _dynamic_values(obj, arrays)
        descriptors_before = _descriptor_values(obj, arrays)
        unread_before = _unreadable(obj)
        result = _probe_call(_step_method(obj, method_name, data), data, comments, constants_key="call_constants")
    comments[PROBED] = True
    if result is obj:   # a step returning its object for chaining (imufusion's update): it returns nothing new
        result = None
    after = _numeric_attributes(obj, arrays)
    state_after = _state_bytes(obj)
    dynamic_after = _dynamic_values(obj, arrays)
    dynamic_changed = {k: v for k, v in dynamic_after.items() if k not in dynamic_before
                       or not _same(dynamic_before[k], v)}
    if result is not None and _same(before, after) and (state_before is None or state_before == state_after) \
            and not dynamic_changed:
        # the step changed nothing: running it at initialization too gives the outputs at the start time
        data["model"]["init_call"] = True
        comments[("model", "init_call")] = (f"{method_name}() changed nothing in the probe, so it also runs at "
                                            "initialization: outputs have values at the start time")
    _check_picklable(obj, data, comments)

    returned = set()
    if result is not None and result is not obj:
        returned = _outputs_from_return(result, data, comments, default_name="y", is_class=True, arrays=arrays)
    taken = {*data.get("parameters", {}), *data.get("inputs", {}), *data.get("outputs", {})}
    for attr, value in _property_values(obj, arrays).items():   # read-only views such as T1 = sensor reading
        if attr not in taken:
            data.setdefault("outputs", {})[attr] = _typed({}, value, arrays)
            comments[("outputs", attr)] = "a property of the object"
    if result is None:   # the step returns nothing: its results are on the object
        taken |= set(data.get("outputs", {}))
        for attr, value in _descriptor_values(obj, arrays).items():   # C attributes set by the step (ephem's ra)
            if attr in taken:
                continue
            if attr in unread_before:
                comments[("outputs", attr)] = f"set by {method_name}(): reading it failed before"
            elif attr in descriptors_before and not _same(descriptors_before[attr], value):   # imufusion's quaternion
                comments[("outputs", attr)] = f"changed by {method_name}()"
            else:
                continue
            data.setdefault("outputs", {})[attr] = _typed({}, value, arrays)
            taken.add(attr)
        _getter_outputs(obj, data, comments, arrays, taken)
        taken |= set(data.get("outputs", {}))
    needs_call = sorted(set(unread_before) & set(data.get("outputs", {})))
    if needs_call and not data["model"].get("init_call"):
        data["model"]["init_call"] = True
        comments[("model", "init_call")] = (f"reading {', '.join(needs_call)} failed before {method_name}() had run "
                                            f"({unread_before[needs_call[0]]!r}), so it also runs at initialization")
    unfit = _unfit_properties(obj, data, comments, arrays, taken)
    taken |= set(data.get("outputs", {})) | unfit
    taken |= _gekko_outputs(obj, data, comments, taken)
    taken |= static.foreign_attributes(type(obj))   # a library base's own state (GEKKO's csv_status)
    for attr, value in getattr(obj, "__dict__", {}).items():   # complex attributes (None ones are too common)
        if not attr.startswith("_") and attr not in taken and value is not None and _misfit(value):
            _unfit_value(comments, "outputs", attr, value, f"attr:{attr}", arrays)
            unfit.add(attr)
    for attr, value in after.items():
        if attr in taken:
            continue
        if attr not in before:
            data.setdefault("outputs", {})[attr] = _typed({}, value, arrays)
            comments[("outputs", attr)] = f"set by {method_name}()"
        elif not _same(before[attr], value):
            data.setdefault("locals", {})[attr] = _typed({}, value, arrays)
            comments[("locals", attr)] = f"changed by {method_name}()"
    for attr, value in dynamic_changed.items():   # attributes the object makes up (tellurium's species S1, S2)
        if attr in taken or attr in data.get("outputs", {}) or attr in data.get("locals", {}):
            continue
        columns = _table_children(value)
        if columns is not None:   # a result table (pandapipes' net.res_junction): its columns
            for column, part in columns:
                name = _valid_name(f"{attr}.{column}")
                if _fmi_value(part, arrays) and name not in data.get("outputs", {}):
                    data.setdefault("outputs", {})[name] = _typed({"from": f"attr:{attr}.{column}"}, part, arrays)
                    comments[("outputs", name)] = f"a column of {attr}, changed by {method_name}()"
            continue
        data.setdefault("outputs", {})[attr] = _typed({}, value, arrays)
        comments[("outputs", attr)] = f"changed by {method_name}(); an attribute the object provides itself"
    if "outputs" not in data and data.get("locals"):
        # nothing else to read: what the step changes is its result (river's intercept)
        data["outputs"] = data.pop("locals")
        for attr in data["outputs"]:
            comments[("outputs", attr)] = f"changed by {method_name}(); an output, since the probe found no other"
            comments.pop(("locals", attr), None)
    _check_sizes(obj, method_name, data, comments)
    _check_tunable(obj, method_name, data, comments, arrays)
    return obj, returned, set(getattr(obj, "__dict__", {})) | set(_property_values(obj, arrays)) | unfit


def _forwarded(method, method_name, owner):
    """The callable whose arguments ``method`` takes: ``forward`` for a ``__call__`` that only declares
    ``(*args, **kwargs)`` and passes them on to it (torch modules: norse's ``LIFCell()(input_tensor)``).

    :param method: the step method.
    :param method_name: its name.
    :type method_name: str
    :param owner: the class (unbound methods) or the object (bound ones).
    :return: ``owner.forward``, or ``method``.
    """
    from fmugen.arguments import _only_varargs
    forward = getattr(owner, "forward", None)
    if method_name == "__call__" and callable(forward) \
            and _only_varargs(method, skip_self=not inspect.ismethod(method)) \
            and not _only_varargs(forward, skip_self=not inspect.ismethod(forward)):
        return forward
    return method


def _reset_first(obj, method_name, data, comments):
    """After a failed probe step: if the object has a ``reset()`` taking no required argument, and the step
    works after it, run it in ``[model] setup`` (gymnasium's environments refuse ``step()`` before ``reset()``).

    :param obj: the model object.
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :return: whether ``reset`` was added (the object is then reset; the step itself isn't run here).
    :rtype: bool
    """
    reset = getattr(obj, "reset", None)
    if "reset" in data["model"].get("setup", []) or method_name == "reset" or not callable(reset):
        return False
    try:
        required = [p for p in inspect.signature(reset).parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    except (TypeError, ValueError):
        return False
    if required:
        return False
    try:
        reset()
        # with the made-up inputs' retries: CartPole's step() takes an action from its action space, not 1.0
        _probe_call(_step_method(obj, method_name, data), data, comments, constants_key="call_constants")
        reset()   # back to the state the FMU starts from, for the probe's own step
    except Exception:
        return False
    data["model"]["setup"] = [*data["model"].get("setup", []), "reset"]
    comments[("model", "setup")] = f"reset() before the first step: {method_name}() failed without it in the probe"
    return True


#: An error message asking for a call first: "call compile() first", "Use fit() to train model. Then",
#: "not fitted yet. Call 'fit'", "not initialized yet. Call 'initialize'".
SET_UP_ASKED = re.compile(r"\bfirst\b|\bbefore\b|\bthen\b|not (?:yet )?(?:been )?(?:fit|fitted|initiali[sz]ed|compiled|set ?up)\b"
                          r"|\bhas not been\b|\bmust (?:first )?(?:call|run)\b"
                          r"|\bnot having (?:been )?(?:fit|fitted|trained)\b|\bnot (?:yet )?(?:been )?trained\b", re.I)


#: Method names (or prefixes, ending in _) that train the model on the user's data: a set-up the user
#: writes (--setup), never made up (lifelines' fit_interval_censoring, river's learn_many).
TRAINING = ("fit", "train", "learn", "partial_fit", "fit_", "train_", "learn_")


def _setup_first(build, obj, method_name, error, data, comments):
    """After a failed probe step: find a method that sets the model up, run it in ``[model] setup``
    and keep it if the step works then (osqp's ``solve()`` before ``setup()``: its solver is None).

    Candidates, in turn: public methods the error message names ("call setup() first"); then
    methods that assign an attribute the step reads (``self._solver``) which is None or missing on
    the object (:func:`_unset_attributes`). Each is called with no arguments, or with its required
    ones made up as 1.0; then the step. A training method (:data:`TRAINING`) is the user's part: the probe
    trains a new object on made-up data (:func:`_fit_made_up`) only to find the step's inputs and outputs,
    and the set-up is written commented out (``# setup = ["fit(...)"]``), for the user's training data.

    :param build: the class, ``--create`` classmethod or factory (a new object is built for the probe).
    :param obj: the model object the step failed on.
    :param method_name: the step method.
    :type method_name: str
    :param error: what the step raised.
    :type error: Exception
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :return: a new object, built and set up as the FMU will, when a set-up worked; else a hint
        naming the set-up the step seems to need (empty when none is found).
    :rtype: object or str
    """
    cls = type(obj)
    public = [m for m in dir(cls) if not m.startswith("_") and m != method_name and callable(getattr(cls, m, None))
              and not isinstance(inspect.getattr_static(cls, m, None), (property, type))]
    message = str(error)
    # a method the message tells to call first ("call compile() first", "Call 'fit' ... before"), not one it
    # names otherwise ("forward() missing 1 required argument", "'float' object is not iterable")
    asks = SET_UP_ASKED.search(message) is not None
    named = [m for m in public if asks and re.search(rf"\b{re.escape(m)}\b(?!' object)", message)]
    unset = _unset_attributes(obj, method_name)
    assigning = [m for m in public if unset & _assigned(cls, m)]
    # the model's training, when the step fails before it on an attribute the object doesn't have yet or
    # the message asks for a set-up (tabnet's predict(): no network; fit() sets it through a helper)
    unready = asks or f"'{cls.__name__}' object has no attribute" in message
    training = [m for m in ("fit", "train", "learn") if unready and m in public]
    candidates = list(dict.fromkeys(named + assigning + training))
    if "reset" in candidates and "reset" in data["model"].get("setup", []):
        candidates.remove("reset")
    unhelpful = set()   # set up without an error, and the step still failed: not what it needs
    for name in candidates:
        if name in TRAINING or name.startswith(tuple(p for p in TRAINING if p.endswith("_"))):
            fitted = _fit_made_up(build, name, method_name, data, comments)
            if fitted is not None:
                return fitted
            continue
        try:
            params = [p for p in inspect.signature(getattr(obj, name)).parameters.values()
                      if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            continue
        worked = False
        for made in _made_up_variants(getattr(obj, name), params):
            step = f"{name}({', '.join(made)})"
            setup = [*data["model"].get("setup", []), step]
            trial = dict(data, model=dict(data["model"], setup=setup))
            try:
                fresh = _build_set_up(build, trial)
            except Exception:
                continue
            saved = {n: dict(i) for n, i in data.get("inputs", {}).items()}
            saved_comments = dict(comments)
            try:
                # with the made-up inputs' retries: timesfm's forecast() takes a list of arrays, not 1.0
                _probe_call(_step_method(fresh, method_name, data), data, comments, constants_key="call_constants")
                fresh = _build_set_up(build, trial)   # not stepped: the object the FMU starts from
            except Exception:
                if "inputs" in data:
                    data["inputs"] = saved
                comments.clear()
                comments.update(saved_comments)
                continue
            worked = True
            break
        if not worked:
            unhelpful.add(name)
            continue
        data["model"]["setup"] = setup
        made_up = f", its arguments made up ({', '.join(made)}): check them" if params else ""
        comments[("model", "setup")] = (f"{name}() before the first step: {method_name}() failed without it in the "
                                        f"probe{made_up}")
        return fresh
    candidates = [m for m in candidates if m not in unhelpful]
    if not candidates:
        return ""
    name = candidates[0]
    why = f" (it sets {', '.join(sorted(unset & _assigned(cls, name)))}, which {method_name}() uses)" \
            if unset & _assigned(cls, name) else ""
    return f"{method_name}() seems to need {name}() first{why}: add --setup \"{name}(...)\""


#: The made-up training data :func:`_fit_made_up` tries: samples, features.
MADE_UP_TRAINING = (64, 4)   # more samples than a batch (pyod trains in batches of 32)


def _fit_made_up(build, name, method_name, data, comments):
    """Train a new object on made-up data with ``name`` (``fit``), only to probe the step after it.

    The data are deterministic random numbers (:func:`_made_up_training`): arrays ``X, y`` first, then a
    one-column table, a dated table (prophet's ``ds``, ``y``) and a darts ``TimeSeries``, when those
    libraries are loaded. The first that trains and lets the step work (with the made-up inputs' retries,
    which size ``X`` from the fitted model's feature count) is kept: the object is returned for the probe,
    and ``[model] setup`` gets the call only as a commented-out line, for the user's training data. The
    FMU isn't trained on made-up data.

    :param build: the class, ``--create`` classmethod or factory.
    :param name: the training method.
    :type name: str
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :return: the trained object, or None.
    """
    for args, what in _made_up_training():
        try:
            fresh = _build_set_up(build, data)
            getattr(fresh, name)(*args)
            saved = {n: dict(i) for n, i in data.get("inputs", {}).items()}
            saved_comments = dict(comments)
            try:
                _probe_call(_step_method(fresh, method_name, data), data, comments, constants_key="call_constants")
            except Exception:
                if saved or "inputs" in data:
                    data["inputs"] = saved
                    if not saved:
                        data.pop("inputs")
                comments.clear()
                comments.update(saved_comments)
                continue
            fresh = _build_set_up(build, data)   # trained again, for the probe's own step, not stepped
            getattr(fresh, name)(*args)
        except Exception:
            continue
        comments[("model", "#setup")] = (
            f'setup = ["{name}(...)"]  # {method_name}() needs {name}() first: give it your training data and '
            f"uncomment this; the probe trained on made-up data ({what}) only to find the inputs and outputs")
        return fresh
    return None


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


def _made_up_variants(fn, params):
    """The set-up's made-up arguments as call texts (:func:`_made_up_argument`), then with configs built with no
    arguments given sizes (:func:`_sized_instance`), then those made up as 1.0 as numpy
    arrays (a single capital letter or a ``matrix`` name as a 1 x 1 matrix, others a vector of one), then those
    matrices as scipy sparse ones (osqp's ``setup(P, q, A, l, u)``: "P is required to be a sparse matrix")."""
    made = [_made_up_argument(fn, p) for p in params]
    yield made
    # a config built with no arguments, its integer fields that default to 0 as 16 (timesfm's
    # compile(ForecastConfig()): "Horizon must be less than the max horizon. 1 > 0")
    sized = [_sized_instance(text) for text in made]
    if any(sized):
        yield [new or text for new, text in zip(sized, made)]
    numbers = [i for i, text in enumerate(made) if text == "1.0"]
    if not numbers or importlib.util.find_spec("numpy") is None:
        return
    matrices = {i for i in numbers if re.fullmatch(r"[A-Z]|\w*matrix\w*", params[i].name)}
    yield [("numpy.ones((1, 1))" if i in matrices else "numpy.ones(1)") if i in numbers else text
           for i, text in enumerate(made)]
    if matrices and importlib.util.find_spec("scipy") is not None:
        yield [("scipy.sparse.csc_matrix(numpy.ones((1, 1)))" if i in matrices else "numpy.ones(1)")
               if i in numbers else text for i, text in enumerate(made)]


def _sized_instance(text):
    """``module.Class(field=16, ...)`` for call text ``module.Class()`` naming a dataclass with integer fields that
    default to 0, else None."""
    if not text.endswith("()"):
        return None
    module, _, name = text[:-2].rpartition(".")
    try:
        cls = getattr(importlib.import_module(module), name)
    except Exception:
        return None
    zeros = [f.name for f in getattr(cls, "__dataclass_fields__", {}).values()
             if f.default == 0 and type(f.default) is int]
    return f"{text[:-2]}({', '.join(f'{z}=16' for z in zeros)})" if zeros else None


def _made_up_argument(fn, p):
    """A set-up argument with no default, as call text: an instance built with no arguments of the class it's
    annotated with (timesfm's ``compile(forecast_config: ForecastConfig)``), else ``1.0``.

    :param fn: the set-up method.
    :param p: the parameter.
    :type p: inspect.Parameter
    :rtype: str
    """
    try:
        hint = typing.get_type_hints(fn).get(p.name, p.annotation)
    except Exception:
        hint = p.annotation
    from fmugen import static
    from fmugen.arguments import NOT_BUILT
    hint = static.unwrap_optional(hint)
    if isinstance(hint, type) and hint is not p.empty and not issubclass(hint, NOT_BUILT) \
            and hint.__module__ not in ("builtins", "typing", "inspect") \
            and "<" not in hint.__qualname__:
        try:
            hint()
        except Exception:
            return "1.0"
        return f"{hint.__module__}.{hint.__qualname__}()"
    return "1.0"


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


def _method_tree(cls, name):
    """The parsed source of method ``name`` of ``cls``, or None (no Python source: C, SWIG)."""
    try:
        return ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, name))))
    except (OSError, TypeError, SyntaxError, AttributeError):
        return None


def _self_attributes(tree, store):
    """``self.X`` names in a method's tree: assigned (``store``) or read."""
    if tree is None:
        return set()
    fn = tree.body[0] if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)) else None
    args = fn.args.posonlyargs + fn.args.args if fn else []
    me = args[0].arg if args else "self"
    want = ast.Store if store else ast.Load
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == me and isinstance(n.ctx, want)}


def _assigned(cls, name):
    """Attributes method ``name`` of ``cls`` assigns (``self.X = ...``)."""
    return _self_attributes(_method_tree(cls, name), store=True)


def _unset_attributes(obj, method_name):
    """Attributes the step method reads (``self.X``) that are None or missing on ``obj``: something must set them."""
    unset = set()
    for attr in _self_attributes(_method_tree(type(obj), method_name), store=False):
        try:
            value = inspect.getattr_static(obj, attr)
        except AttributeError:
            unset.add(attr)
            continue
        if value is None:
            unset.add(attr)
    return unset


#: The comment on a parameter kept as a writable attribute, before the probe checks whether the step reads it.
KEPT = "kept as an attribute: add variability = \"tunable\" if the step reads it"


def _check_tunable(obj, method_name, data, comments, arrays):
    """Mark parameters kept as attributes tunable when the step reads them: changing one changes the results.

    On copies of the object after the probe's step: two unchanged copies step once (when they
    disagree, the model is random and nothing is decided); a third, with the attribute changed,
    steps too, and its results are compared. When they don't change, the parameter may still be
    read (a PID with zero error), so it only keeps the suggestion, with a note. Objects that can't be
    copied keep the suggestion.

    :param obj: the model object.
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    """
    import copy
    names = [n for n, info in data.get("parameters", {}).items()   # kept as a writable attribute, as given
             if "variability" not in info and isinstance(getattr(obj, n, None), (bool, int, float))
             and _writable_attribute(obj, n) and _same(getattr(obj, n), info.get("start"))]
    if not names:
        return

    def results(o):
        out = _call_with_probe_inputs(_step_method(o, method_name, data), data, "call_constants")
        return _snapshot(out), _numeric_attributes(o, arrays)

    def same(a, b, ignore=None):   # the attribute changed by hand isn't a result
        return _same(a[0], b[0]) and _same({k: v for k, v in a[1].items() if k != ignore},
                                           {k: v for k, v in b[1].items() if k != ignore})

    try:
        if not same(results(copy.deepcopy(obj)), base := results(copy.deepcopy(obj))):
            return   # random by design: changing a value can't be told from chance
    except Exception:
        return
    for name in names:
        value = getattr(obj, name)
        changed = (not value) if isinstance(value, bool) else value + 1 if isinstance(value, int) else value * 1.5 + 1.0
        try:
            other = copy.deepcopy(obj)
            setattr(other, name, changed)
            moved = not same(results(other), base, ignore=name)
        except Exception:
            continue
        before = comments.get(("parameters", name))
        if moved:
            before = None if before == KEPT else before   # decided
            info = data["parameters"][name]
            info["variability"] = "tunable"
            if str(info.get("to", "")).startswith("pos:"):   # a new value reaches the model through its attribute
                info["attr"] = name
            note = "tunable: the step reads it (changing it in the probe changed the results)"
        else:   # with these start values it may just not show (a PID with no error): only a hint
            before = before or KEPT
            note = "changing it in the probe changed nothing, with these start values"
        comments[("parameters", name)] = f"{before}; {note}" if before else note


#: GEKKO classes that hold solved results: Var/SV/CV, Intermediate, FV/MV (not Param, Const).
GEKKO_RESULTS = ("GKVariable", "GK_Intermediate", "GK_FV")


def _gekko_result(value):
    """The value of a solved GEKKO variable, intermediate, FV or MV at the end of its time horizon.

    GEKKO keeps results in these objects, as ``x.value`` after ``m.solve()``; recognised by module
    and class name, without importing gekko.

    :param value: any attribute value.
    :return: the last value of ``value.value``, or None.
    """
    if not type(value).__module__.startswith("gekko.") \
            or not any(k.__name__ in GEKKO_RESULTS for k in type(value).__mro__):
        return None
    try:
        last = value.value[-1]
    except Exception:   # not solved yet: a single value
        return None
    return last if _fmi_value(last) else None


def _gekko_outputs(obj, data, comments, taken):
    """Outputs for the object's attributes that hold GEKKO results.

    ``self.X = m.Var(...)`` and dicts of them (``self.W = {"HOU1": m.Var(...)}``), read as their
    last value (``from = "attr:X.value.-1"``).

    :param obj: the model object.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param taken: names already used.
    :type taken: set[str]
    :return: the names added.
    :rtype: set[str]
    """
    found = {}
    for attr, value in getattr(obj, "__dict__", {}).items():
        if attr.startswith("_"):
            continue
        if _gekko_result(value) is not None:
            found[attr] = (f"{attr}.value.-1", value)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str) and "." not in key and key and _gekko_result(item) is not None:
                    found[_valid_name(f"{attr}.{key}")] = (f"{attr}.{key}.value.-1", item)
    names = set()
    for name, (path, value) in found.items():
        if name in taken:
            continue
        data.setdefault("outputs", {})[name] = _typed({"from": f"attr:{path}"}, _gekko_result(value))
        comments[("outputs", name)] = (f"a GEKKO {type(value).__name__}: its value at the end of the time "
                                       f"horizon (m.time); the whole trajectory is {path[:-3]}")
        names.add(name)
    return names

def _unfit_properties(obj, data, comments, arrays, taken):
    """Commented-out entries for properties annotated as a number, bool or str whose probe value isn't one.

    The code alone would make them outputs, but in the probe they were e.g. None, or raised.

    :param obj: the model object.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param taken: names already used.
    :type taken: set[str]
    :return: their names.
    :rtype: set[str]
    """
    annotated = {name for name, kind in static.own_properties(type(obj)).items() if kind}
    unfit = set()
    for attr, value in _property_values(obj, arrays, everything=True).items():
        if attr in taken or attr not in annotated or (not isinstance(value, Exception) and _fmi_value(value, arrays)):
            continue
        if not _unfit_value(comments, "outputs", attr, value, f"attr:{attr}", arrays):
            why = (f"reading it raised {value!r} in the probe" if isinstance(value, Exception)
                   else f"a {type(value).__name__} in the probe, not an FMI value")
            _unfit(comments, "outputs", attr, why, [f'{_key(attr)} = {{ from = "attr:{attr}" }}'])
        unfit.add(attr)
    return unfit


def _check_sizes(obj, method_name, data, comments):
    """Call the step method a second time and handle arrays whose size changed.

    An array whose size changes can't be an FMI array, which has a fixed size: one that grows by
    entries each call (a history) is read as its last entry; any other is commented out.

    :param obj: the model object.
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    """
    sized = [(section, name, info) for section in ("outputs", "locals")
             for name, info in data.get(section, {}).items() if "dimensions" in info]
    if not sized:
        return
    from fmugen.templates.fmugen_runtime import get_path, pick
    try:
        result = _call_with_probe_inputs(_step_method(obj, method_name, data), data, "call_constants")
    except Exception:
        return
    for section, name, info in sized:
        source = info.get("from", f"attr:{name}")
        kind, _, path = source.partition(":")
        try:
            value = pick(result, path or None) if kind == "return" else get_path(obj, path)
            if kind == "get":
                value = value()
        except Exception as e:
            size = f"unreadable ({e!r})"
        else:
            new = _array_info(_scalar_of(value))
            size = new["dimensions"] if new else "not an array"
        if size == info["dimensions"]:
            continue
        if isinstance(size, list) and len(size) == len(info["dimensions"]) and size[1:] == info["dimensions"][1:] \
                and size[0] > info["dimensions"][0]:
            # a history that grows by one entry per step (seirsplus' numS, pybamm's t): its last entry
            # has a fixed size, and is the value at the end of the step
            sep = "." if path else ":"
            data[section][name] = _typed({"from": f"{source}{sep}-1"}, value[-1], arrays=True)
            comments[(section, name)] = (f"its last entry: the whole {source} grows each step ({info['dimensions']} "
                                         f"then {size}), and an FMI array has a fixed size")
            continue
        del data[section][name]
        comments.pop((section, name), None)
        _unfit(comments, section, name, f"its size changed from {info['dimensions']} to {size} on a second probe "
               "call, and an FMI array has a fixed size", [f'{_key(name)} = {{ ' + (f'from = "{source}", ' if section == "outputs"
                                                                       else "") + "dimensions = [...] }"])
        if not data[section]:
            del data[section]

def _taken(data):
    """Names already used by variables of any section.

    :param data: the config being inferred.
    :type data: dict
    :rtype: set[str]
    """
    return {n for section in ("parameters", "structural_parameters", "inputs", "states", "outputs", "locals")
            for n in data.get(section, {})}


def _add_returned(read, data, comments, is_class, arrays, probe, seen):
    """Outputs from what the code returns (:func:`static.returned`).

    With ``--probe``, only ones the probe didn't see; results of unknown type are added as
    guesses, and ones that may be None are commented out.

    :param read: what the code returns.
    :type read: static.Returned
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param is_class: whether the model is a class (``from`` is then always written).
    :type is_class: bool
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param probe: whether ``--probe`` is in effect.
    :type probe: bool
    :param seen: names the probe already handled.
    :type seen: set[str]
    """
    if read.unknown:
        if not probe:
            comments[("outputs", None)] = (f"outputs could not be read from the code ({read.unknown}); "
                                           "add them by hand, or run init --probe")
        return
    if read.array:
        if arrays and not probe:   # the size can't be read from the code
            comments[("outputs", "#y")] = ('y = { from = "return", dimensions = [...] }  # an array: set its '
                                           'dimensions and uncomment, or run init --probe')
        return
    taken = _taken(data) | set(seen)
    # "y4" and "y4.rewards" when the probe found "y4.rewards.speed"
    parents = {".".join(n.split(".")[:i]) for n in taken for i in range(1, n.count(".") + 1)}
    for name in read.names:
        if name in taken or name in parents:
            continue
        if probe and read.sources[name] == "return" and data.get("outputs"):
            continue   # the probe already split the whole return value (a table, an object) into outputs
        kind = read.types.get(name)
        source = read.sources[name]
        copied = None
        if kind and kind.startswith("arg:"):   # an argument returned as it is: the same type as its variable
            arg = kind[len("arg:"):]
            copied = next((info for sec in ("parameters", "inputs", "states")
                           for n, info in data.get(sec, {}).items()
                           if n == arg or str(info.get("to", "")).endswith(f":{arg}")), None)
            kind = "array" if copied and "dimensions" in copied else None
        if kind == "array":
            if arrays and not probe:
                comments[("outputs", f"#{name}")] = (f'{name} = {{ from = "{source}", dimensions = [...] }}  '
                                                     "# an array: set its dimensions and uncomment, or run init --probe")
            continue
        if kind in ("other", "none"):
            continue
        if comments.get(PROBED):   # the probe's call worked, and didn't return it
            _unfit(comments, "outputs", name, "not in what the probe's call returned; reading it fails the step "
                   "whenever the model doesn't return it", [f'{_key(name)} = {{ from = "{source}" }}'])
            continue
        info = {} if (source == f"return:{name}" and not is_class) else {"from": source}
        if copied:
            info.update({k: v for k, v in copied.items() if k in ("type", "enum", "items")})
        elif kind in ("Boolean", "String", "Integer"):
            info["type"] = kind
        elif kind is None:
            _guessed(comments, name)
        if name in read.nullable and not probe:   # e.g. a database column: None fails the step
            comments[("outputs", f"#{name}")] = f"{_key(name)} = {_value(info)}"
            comments.setdefault(("outputs", None), "results that can be None (nullable database columns) are "
                                                   "commented out: uncomment the ones this model always has, "
                                                   "or run init --probe")
            continue
        data.setdefault("outputs", {})[name] = info
        if probe:
            comments[("outputs", name)] = "from the code; not seen in the probe"
        elif read.may_be_none and source == "return":
            comments[("outputs", name)] = "the code can also return None, which fails the step: check it, or run init --probe"


def _add_attributes(cls, method_name, data, comments, arrays, probe, seen):
    """Outputs and locals from the attributes the step method assigns, and the object's own properties.

    Attributes also set in ``__init__`` become locals (they exist before the first step), the
    others outputs. Unannotated properties are only suggested: reading them would run code.

    :param cls: the model class.
    :type cls: type
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param probe: whether ``--probe`` is in effect.
    :type probe: bool
    :param seen: names the probe already handled.
    :type seen: set[str]
    """
    in_step, in_init = static.class_attributes(cls, method_name)
    taken = _taken(data) | set(seen)
    for attr, kind in in_step.items():
        if attr.startswith("_") or attr in taken:
            continue
        kind = static._merge([kind, in_init.get(attr)])
        section = "locals" if attr in in_init else "outputs"
        if kind == "array":
            if arrays and not probe:
                comments[(section, f"#{attr}")] = (f"{attr} = {{ dimensions = [...] }}  # an array set by "
                                                   f"{method_name}(): set its dimensions and uncomment, or run init --probe")
            continue
        if kind == "other":
            continue
        if comments.get(PROBED) and section == "outputs":   # the probe's step worked, and didn't set it
            _unfit(comments, "outputs", attr, f"not set by {method_name}() in the probe; reading it fails the step "
                   "whenever the model hasn't set it", [f'{_key(attr)} = ' + (f'{{ type = "{kind}" }}' if kind in
                                                        ("Boolean", "String") else "{}")])
            continue
        info = {"type": kind} if kind in ("Boolean", "String") else {}
        if kind is None and section == "outputs":
            _guessed(comments, attr)
        data.setdefault(section, {})[attr] = info
        comments[(section, attr)] = ("from the code; not seen in the probe" if probe else
                                     f"{'changed' if section == 'locals' else 'set'} by {method_name}()")
    for name, kind in static.own_properties(cls).items():
        if name in _taken(data) or name in seen:
            continue
        if kind:   # annotated as a number, bool or str
            data.setdefault("outputs", {})[name] = {"type": kind} if kind != "Real" else {}
            comments[("outputs", name)] = "from the code; not seen in the probe" if probe else "a property of the object"
        elif not probe:   # without an annotation its value could be anything: reading it would run code
            comments[("outputs", f"#{name}")] = (f"{name} = {{}}  # a property of the object (type unknown): "
                                                 "uncomment if it is a number, or run init --probe")

#: Converters tried in order when a probe call fails with list inputs.
ARRAY_RETRIES = ("numpy", "torch:tensor")


def _probe_constants(data, table, positional=False):
    """Constants set in the config (e.g. computed by a call), resolved as the FMU would.

    :param data: the config being inferred.
    :type data: dict
    :param table: ``"constants"`` or ``"call_constants"``.
    :type table: str
    :param positional: return ``{position: value}`` of those named ``"0"``, ``"1"``, ... (passed to
        ``*args``) instead of the keyword ones.
    :type positional: bool
    :return: name (or position) -> value.
    :rtype: dict
    :raises ConstantError: a constant fails to compute.
    """
    from fmugen.templates.fmugen_runtime import resolve_constant
    values = {}
    for k, v in data["model"].get(table, {}).items():
        if k.isdigit() != positional:
            continue
        try:
            values[int(k) if positional else k] = resolve_constant(v)
        except Exception as e:
            what = f"call:{v['call']}" if isinstance(v, dict) and "call" in v else _value(v)
            raise ConstantError(f"[model] {table}.{k} = {what} fails: {type(e).__name__}: {e}") from e
    return values


class ConstantError(InterfaceError):
    """A constant (e.g. ``--const NAME=call:...``) that can't be computed: the FMU would fail the same way.

    Re-raised instead of being turned into a comment, because no config could fix it.
    """


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
    kwargs.update(_probe_constants(data, constants_key))
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


def _state_bytes(obj):
    """The object's attributes pickled, to tell whether a call changed them.

    :param obj: the model object.
    :return: the pickle, or None if they can't be pickled.
    :rtype: bytes or None
    """
    try:
        return pickle.dumps(vars(obj))
    except Exception:
        return None


def _check_picklable(obj, data, comments):
    """Pick ``[model] save_state`` by trying to pickle the object as the runtime would.

    FMU state save/restore pickles the model object (pickle, cloudpickle or dill). If that fails,
    save only the attributes that can be pickled; if none can, declare save/restore unsupported.

    :param obj: the model object.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    """
    from fmugen.templates.fmugen_runtime import dump_state
    try:
        dump_state(obj)
        return
    except Exception as e:
        error = f"{type(e).__name__}: {e}"[:120]
    try:
        attrs = vars(obj)
    except TypeError:
        attrs = {}
    saved, skipped = [], []
    for name, value in attrs.items():
        try:
            dump_state(value)
            saved.append(name)
        except Exception:
            skipped.append(name)
    if saved and skipped:
        data["model"]["save_state"] = saved
        comments[("model", "save_state")] = (
            f"the whole object can't be pickled ({error}); saved: these attributes only. Not saved, "
            f"kept as they are on restore: {', '.join(skipped)}. Check those don't change during a simulation, "
            "else set save_state = false")
    else:
        data["model"]["save_state"] = False
        comments[("model", "save_state")] = f"the model object can't be pickled ({error}): " \
                                            "the FMU can't save and restore its state"

def _step_method(obj, method_name, data):
    """The step method to call: bound to ``obj``, or taken from its class with ``call_on_class``.

    :param obj: the model object.
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred.
    :type data: dict
    :rtype: callable
    """
    if ":" in method_name:   # "module:function(self, ...)": a function taking the object (pandapipes' pipeflow)
        from fmugen.templates.fmugen_runtime import object_call
        return object_call(method_name, obj)
    return getattr(type(obj) if data["model"].get("call_on_class") else obj, method_name)


def _varargs_by_keyword(constructor, init_starts, data, probe):
    """Whether a ``(*args, **kwargs)`` constructor needs its ``--start`` values as keywords.

    The signature can't tell: ``StateSpace(A, B, C, D)`` takes them by position,
    ``Quaternion(axis=..., angle=...)`` by keyword. Without ``--probe`` they go by position.
    With it, by keyword only when the positional call fails and the keyword one works.

    :param constructor: the class (or ``create`` classmethod).
    :param init_starts: the values given to the constructor.
    :type init_starts: dict
    :param data: the config being inferred (for its setup calls).
    :type data: dict
    :param probe: whether ``--probe`` is in effect.
    :type probe: bool
    :rtype: bool
    """
    try:
        params = inspect.signature(constructor).parameters.values()
    except (TypeError, ValueError):
        return False
    if not probe or not init_starts or not any(p.kind is p.VAR_KEYWORD for p in params) \
            or not any(p.kind is p.VAR_POSITIONAL for p in params):
        return False
    from fmugen.templates.fmugen_runtime import resolve_constant
    try:
        _run_setup(data)
        values = {n: resolve_constant(v) for n, v in init_starts.items()}
    except Exception:
        return False
    try:
        constructor(*values.values())
        return False
    except Exception:
        pass
    try:
        constructor(**values)
    except Exception:
        return False
    return True


def _split_varargs(cls, create, method_name, init_starts, call_starts, data, comments, probe):
    """Split ``--start`` values between a constructor and a step method that both take only ``*args``.

    SWIG classes such as ``openturns.Normal(mu, sigma).computePDF(x)``, and C built-ins without a
    signature (``vqf.VQF(gyrTs).update(gyr, acc)``, sgp4's ``Satrec.twoline2rv(l1, l2).sgp4(jd, fr)``):
    the signatures can't tell which values go where, so _assign_starts gives them all to the
    constructor. With ``--probe``, the first values (in ``--start`` order) go to the constructor, the
    rest to the method, trying the method with as many as possible first, and the first split where
    both calls work is kept: a value the step can take is an input, which can change each step
    (ephem's ``Sun().compute(date)``, not ``Sun(date)``, which computes once).

    :param cls: the model class.
    :type cls: type
    :param create: ``--create``: the classmethod that builds the object, or None.
    :type create: str or None
    :param method_name: the step method.
    :type method_name: str
    :param init_starts: the values given to the constructor so far.
    :type init_starts: dict
    :param call_starts: the values given to the method so far.
    :type call_starts: dict
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param probe: whether ``--probe`` is in effect.
    :type probe: bool
    :return: ``(constructor starts, method starts)``.
    :rtype: tuple[dict, dict]
    """
    method = getattr(cls, method_name, None)
    build = getattr(cls, create, None) if create else cls
    constructor = build if create else cls.__init__
    if method is None or build is None or not _only_varargs(constructor, skip_self=not create) \
            or not _only_varargs(method) or len(init_starts) < 1:
        return init_starts, call_starts
    names = list(init_starts)
    built = f"{cls.__name__}.{create}" if create else cls.__name__
    if not probe:
        comments[("inputs", None)] = (f"{built}() and {method_name}() both take *args (or have no signature): every "
                                      f"--start value was given to the constructor; run init --probe to split them")
        return init_starts, call_starts
    from fmugen.templates.fmugen_runtime import resolve_constant
    try:
        _run_setup(data)
        values = [resolve_constant(init_starts[n]) for n in names]
    except Exception:
        return init_starts, call_starts
    for k in range(len(names) + 1):
        try:
            getattr(build(*values[:k]), method_name)(*values[k:], **{n: resolve_constant(v) for n, v in call_starts.items()})
        except Exception:
            continue
        if k < len(names):
            comments[("inputs", None)] = (f"{built}() and {method_name}() both take *args (or have no signature): the "
                                          f"probe worked with {', '.join(names[:k]) or 'nothing'} for the constructor "
                                          f"and {', '.join(names[k:])} for {method_name}()")
        return ({n: init_starts[n] for n in names[:k]},
                {**{n: init_starts[n] for n in names[k:]}, **call_starts})
    return init_starts, call_starts

def _probe_args(data, sections, fn=None, constants_key=None):
    """Positional and keyword arguments for a probe call from the inferred start values.

    Includes the positional constants of ``constants_key``; item and field bindings
    (``arg:pair[0]``, ``arg:u.T``) are assembled as the runtime does (:func:`assemble_items`).

    :param data: the config being inferred.
    :type data: dict
    :param sections: the variable sections passed (``("inputs",)`` or ``("parameters",)``).
    :type sections: tuple[str, ...]
    :param fn: the function called (to build object arguments).
    :type fn: callable or None
    :param constants_key: the constants table for positional constants, or None.
    :type constants_key: str or None
    :return: ``(args, kwargs)``.
    :rtype: tuple[list, dict]
    """
    values = _probe_kwargs(data, sections)
    positional = [
        (int(info["to"].split(":")[1]), name)
        for section in sections for name, info in data.get(section, {}).items()
        if str(info.get("to", "")).startswith("pos:")
    ]
    by_position = {i: values.pop(name) for i, name in positional}
    if constants_key:
        by_position.update(_probe_constants(data, constants_key, positional=True))
    args = [by_position[i] for i in sorted(by_position)]
    from fmugen.templates.fmugen_runtime import assemble_items
    targets = {name: str(info["to"]).partition(":")[2] for section in sections
               for name, info in data.get(section, {}).items()
               if any(c in str(info.get("to", "")).partition(":")[2] for c in "[.")
               and str(info["to"]).startswith(("arg:", "init:"))}
    for name, target in targets.items():
        values[target] = values.pop(name)
    return args, assemble_items(values, fn)


def _probe_kwargs(data, sections):
    """Python values of the start values of these sections, as the runtime would pass them.

    Enum members, reshaped (numpy) arrays, Binary bytes and ``convert`` functions are applied.

    :param data: the config being inferred.
    :type data: dict
    :param sections: the variable sections.
    :type sections: tuple[str, ...]
    :return: variable name -> value.
    :rtype: dict
    """
    from fmugen.templates.fmugen_runtime import reshape, resolve_reference
    kwargs = {}
    for section in sections:
        for name, info in data.get(section, {}).items():
            if "enum" in info:
                kwargs[name] = getattr(resolve_reference(info["enum"]), info["start"])
            elif "dimensions" in info:
                value = reshape(info["start"], info["dimensions"])
                if info.get("numpy"):
                    import numpy
                    value = numpy.array(value)
                kwargs[name] = value
            elif info.get("type") == "Binary":
                kwargs[name] = bytes.fromhex(info["start"])
            else:
                kwargs[name] = info.get("start", 0.0)
            if info.get("convert") in ("series", "pint"):   # dated at the epoch, as at time 0; in the unit
                from fmugen.templates.fmugen_runtime import converter
                kwargs[name] = converter(info)(kwargs[name])
            elif info.get("convert"):
                kwargs[name] = resolve_reference(info["convert"])(kwargs[name])
    return kwargs


def _time_kwargs(time_args):
    """Values of the ``[time]`` arguments for a probe call.

    Date-times get their epoch; ``step_size`` and ``end_time`` get :data:`PROBE_STEP_SIZE`;
    everything else 0.0.

    :param time_args: the ``[time]`` table.
    :type time_args: dict
    :rtype: dict
    """
    kwargs = {}
    for arg, source in time_args.items():
        if isinstance(source, dict):   # { source = ..., epoch = ... }: a date-time
            kwargs[arg] = datetime.datetime.fromisoformat(source["epoch"])
        else:
            kwargs[arg] = {"step_size": PROBE_STEP_SIZE, "end_time": PROBE_STEP_SIZE}.get(source, 0.0)
    return kwargs

def _outputs_from_return(result, data, comments, default_name, is_class, arrays=False):
    """Outputs from a probe result.

    Dicts, pandas/xarray tables, NamedTuples, tuples/lists, single values and objects (their
    public attributes and properties) are split into outputs; nested ones get dotted names.

    :param result: what the probe call returned.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param default_name: name of a single result, and prefix of tuple positions (``y0``, ``y1``).
    :type default_name: str
    :param is_class: whether the model is a class.
    :type is_class: bool
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :return: the names of the parts it looked at (dict keys, tuple positions y0, y1, ...), also
        those left out because they aren't FMI values.
    :rtype: set[str]
    """
    outputs = data.setdefault("outputs", {})
    wrapped = any(info.get("convert") == "numpy:atleast_1d" for info in data.get("inputs", {}).values())
    if wrapped and getattr(result, "size", None) == 1 and hasattr(result, "reshape"):
        result = result.reshape(())[()]   # numbers passed as one-element arrays: a number comes back as one
    parts = set()   # properties of a returned object, which may share a name with an input (ufloat's nominal_value)
    if isinstance(result, Mapping):
        items = [(str(k), v, f"return:{k}") for k, v in result.items()]
    elif _table_children(result) == [] and getattr(result, "size", None) == 1:
        items = [(default_name, result.item(), "return")]   # one value labelled by a date (pyet): read as a number
    elif _table_children(result):   # pandas Series / one-row DataFrame, xarray Dataset, labelled by names
        items = [(k, v, f"return:{k}") for k, v in _table_children(result)]
    elif _table_children(result) is not None and not (arrays and _array_info(result)):
        items = []   # a table labelled by dates or numbers (pastas' simulate) is read below as an array
    elif arrays and _array_info(result):
        items = [(default_name, result, "return")]
    elif isinstance(result, tuple) and hasattr(result, "_fields"):   # NamedTuple: use the field names
        items = [(field, getattr(result, field), f"return:{field}") for field in result._fields]
    elif isinstance(result, (tuple, list)):
        items = [leaf for i, v in enumerate(result) for leaf in _positions(f"{default_name}{i}", v, f"return:{i}", arrays)]
    elif _fmi_value(result, arrays and isinstance(result, bytes)) or (result is not None and _misfit(result)):
        # bytes (an image, rembg's remove()) are a Binary output in FMI 3
        items = [(default_name, result, "return")]
    elif result is not None and (hasattr(result, "__dict__") or _public_properties(result, arrays=True)):
        # an object: its public attributes and properties, e.g. uncertainties' nominal_value, std_dev
        public = {k: v for k, v in getattr(result, "__dict__", {}).items() if not k.startswith("_")}
        public.update({k: v for k, v in _public_properties(result, arrays=True).items() if k not in public})
        items = [_single(k, v, f"return:{k}") for k, v in public.items()]
        parts = set(public) - set(getattr(result, "__dict__", {}))   # computed, not a stored copy of an input
    else:
        items = []
    looked_at = {name for name, _, _ in items}
    items = [leaf for name, value, source in items for leaf in _nested(name, value, source, arrays)]
    items = [part for item in items for part in _complex_parts(*item)]
    # keys that aren't names (radioactivedecay's "N-14"): a valid name, the real key kept in `from`
    items = [(_valid_name(name), value, source) for name, value, source in items]
    taken = {n for section in ("parameters", "inputs", "states") for n in data.get(section, {})}
    for name, value, source in items:
        if name in taken and name in parts and not is_class:
            # a property of the object a function returns, named like an input: y_nominal_value
            # (a class's step that echoes its input, as pyfluids' with_state, still gives none)
            name = f"{default_name}_{name}"
        if name in taken and source == "return":   # the whole result, named like an input (lambdify's f(x, y))
            name = next(n for n in ("result", "result_y", f"{default_name}_result") if n not in taken)
        if name in taken:  # e.g. an object that echoes its inputs as attributes
            given = data.get("inputs", {}).get(name, {})
            if source == f"return:{name}" and "start" in given and _fmi_value(value, arrays) \
                    and not _same(_scalar_of(value), given["start"]):
                # returned under its own name with a new value: the next value of a state (x -> {"x": ...})
                comments.setdefault(RETURNED_INPUTS, set()).add(name)
            continue
        if _dotted_name(name) and _unfit_value(comments, "outputs", name, value, source, arrays):
            continue
        if not _dotted_name(name) or not _fmi_value(value, arrays):
            comments[("outputs", None)] = f"skipped {name!r}: not an FMI value or not a valid name"
            continue
        # from = "return:<name>" is the default for functions
        info = {} if source == f"return:{name}" and not is_class else {"from": source}
        outputs[name] = _typed(info, value, arrays)
        scalar = _scalar_of(value)
        if isinstance(scalar, int) and not isinstance(scalar, bool) and "type" not in outputs[name]:
            comments[("outputs", name)] = 'an int in the probe; Real, since results often vary: type = "Integer" if it is a count'
    if not outputs:
        data.pop("outputs")
    return looked_at


def _complex_parts(name, value, source):
    """A complex result as two outputs, its real and imaginary parts (FMI has no complex type).

    ``y`` read from ``return`` gives ``y_real`` (``return:real``) and ``y_imag``; anything else
    is returned as it is.

    :param name: the output name.
    :type name: str
    :param value: its probe value.
    :param source: its ``from`` binding.
    :type source: str
    :rtype: list[tuple]
    """
    scalar = _scalar_of(value)
    if not (isinstance(scalar, complex) or "complex" in str(getattr(value, "dtype", ""))):
        return [(name, value, source)]
    sep = "." if ":" in source else ":"
    return [(f"{name}_{part}", getattr(value, part), f"{source}{sep}{part}") for part in ("real", "imag")]


#: comments key (not rendered): inputs the probe saw returned under their own name with a new value.
RETURNED_INPUTS = ("init", "returned inputs")

#: How deep nested results are split into dotted outputs.
MAX_NESTING = 4


def _nested(name, value, source, arrays, depth=0):
    """``(name, value, source)`` for each value inside nested dicts, NamedTuples, dataclasses and namespaces.

    Named with dots: ``{"zone": {"T": 21}}`` gives ``("zone.T", 21, "return:zone.T")``.

    :param name: the outer name.
    :type name: str
    :param value: the value.
    :param source: its ``from`` binding.
    :type source: str
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param depth: the current nesting level.
    :type depth: int
    :rtype: list[tuple]
    """
    if depth < MAX_NESTING and not _fmi_value(value, arrays):
        children = _children(value)
        if children:
            sep = ":" if source == "return" else "."
            return [leaf for key, child in children
                    for leaf in _nested(f"{name}.{key}", child, f"{source}{sep}{key}", arrays, depth + 1)]
    return [(name, value, source)]


def _positions(name, value, source, arrays, depth=0):
    """``(name, value, source)`` for each item of tuples nested in a tuple result, which would be skipped whole.

    nashpy's ``support_enumeration`` gives ``((sigma_r, sigma_c), ...)``: ``y0`` becomes ``y0_0``
    (``return:0.0``) and ``y0_1``. Only plain tuples are split, and only when they aren't an FMI
    value themselves (a tuple of numbers stays one array).

    :param name: the variable name of ``value``.
    :type name: str
    :param value: an item of the result.
    :param source: its ``from`` binding.
    :type source: str
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param depth: the current nesting level.
    :type depth: int
    :rtype: list[tuple]
    """
    if depth < MAX_NESTING and type(value) is tuple and value and not _fmi_value(value, arrays):
        return [leaf for i, v in enumerate(value)
                for leaf in _positions(f"{name}_{i}", v, f"{source}.{i}", arrays, depth + 1)]
    return [(name, value, source)]


def _children(value):
    """The named parts of a container value (tables, mappings, NamedTuples, dataclasses, namespaces, objects).

    An object's parts are its public attributes and properties (simdkalman's ``result.observations.mean``,
    ultralytics' ``results[0].boxes.xyxy``); a list or tuple of such objects gives its items by position.

    :param value: the value.
    :return: ``[(name, part)]``, empty for anything else.
    :rtype: list[tuple]
    """
    table = _table_children(value)
    if table is not None:
        return table
    if isinstance(value, Mapping):
        return [(k, v) for k, v in value.items() if isinstance(k, str)]
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return [(f, getattr(value, f)) for f in value._fields]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]
    if isinstance(value, types.SimpleNamespace):
        return list(vars(value).items())
    if type(value) in (list, tuple) and 0 < len(value) <= MAX_ITEMS and all(_plain_object(v) for v in value):
        return [(str(i), v) for i, v in enumerate(value)]
    if _plain_object(value):
        public = {k: v for k, v in vars(value).items() if not k.startswith("_") and v is not None}
        public.update({k: v for k, v in _public_properties(value, arrays=True).items() if k not in public})
        # inputs kept on the object (ultralytics' orig_img) aren't results: too large to be one
        return [(k, v) for k, v in public.items() if not _size(v) > MAX_NESTED_SIZE]
    return []


#: How many items of a list of objects are split into outputs (ultralytics' one Results per image).
MAX_ITEMS = 8

#: Arrays inside a returned object larger than this aren't made outputs (an input image kept on it).
MAX_NESTED_SIZE = 10_000


def _size(value):
    """The number of elements of an array-like value (numpy, torch), else 0.

    :param value: the value.
    :rtype: int
    """
    size = getattr(value, "size", 0)
    size = size() if callable(size) else size   # torch: a method returning a Size
    try:
        return int(math.prod(size)) if isinstance(size, tuple) else int(size)
    except (TypeError, ValueError):
        return 0


def _plain_object(value):
    """Whether ``value`` is an instance of a library or user class with public state to read (not a module,
    class, function, number, string, container or array).

    :param value: the value.
    :rtype: bool
    """
    if isinstance(value, (type, types.ModuleType, types.FunctionType, types.MethodType, str, bytes, Mapping,
                          list, tuple, set, frozenset, enum.Enum)) or callable(value) or _fmi_value(value, True):
        return False
    return type(value).__module__ not in ("builtins",) and hasattr(value, "__dict__")


def _table_children(value):
    """``[(label, value)]`` for a pandas Series, a DataFrame or an xarray Dataset of single values.

    A one-row DataFrame gives each column's value; a longer one its numeric columns as arrays
    (neuralforecast's forecast column, pandapipes' result tables), each read by its label. Recognised
    by module and class name, so pandas/xarray aren't imported; only string labels.

    :param value: the value.
    :return: the pairs, or None for anything else.
    :rtype: list[tuple] or None
    """
    kind = type(value).__name__
    library = type(value).__module__.split(".")[0]
    try:
        if library == "pandas" and kind == "Series":
            pairs = list(value.items())
        elif library == "pandas" and kind == "DataFrame" and len(value) == 1:
            pairs = [(c, value[c].iloc[0]) for c in value.columns]
        elif library == "pandas" and kind == "DataFrame" and len(value) > 1:
            pairs = [(c, value[c].to_numpy()) for c in value.columns if value[c].dtype.kind in "biuf"]
        elif library == "xarray" and kind == "Dataset":
            pairs = [(k, v.item()) for k, v in value.data_vars.items() if v.size == 1]
        else:
            return None
    except Exception:
        return None
    return [(k, v) for k, v in pairs if isinstance(k, str)]


def _single(name, value, source):
    """A one-element array read as its element: ambiance's properties are ``array([288.15])``.

    :param name: the output name.
    :type name: str
    :param value: its probe value.
    :param source: its ``from`` binding.
    :type source: str
    :return: ``(name, value, source)``, with ``value[0]`` and ``source.0`` for a one-element array.
    :rtype: tuple
    """
    if getattr(value, "ndim", 0) == 1 and getattr(value, "size", 0) == 1:
        return name, value[0], f"{source}.0"
    return name, value, source


#: Attributes a C wrapper adds to every object, not the model's: SWIG's ownership flag (openturns).
WRAPPER_FLAGS = ("thisown",)


def _public_properties(obj, arrays=False):
    """``{name: value}`` of the public properties of obj's class whose value is an FMI value.

    Of properties that are the same object under several names (uncertainties' n and
    nominal_value), only the longest name. The class is searched with
    :func:`inspect.getattr_static`: getattr on the class runs descriptors, which can fail.

    :param obj: the object.
    :param arrays: also numeric arrays.
    :type arrays: bool
    :rtype: dict
    """
    by_property = {}
    for name in dir(type(obj)):
        # static lookup: getattr on the class runs descriptors, which can fail (SQLAlchemy hybrid properties)
        prop = inspect.getattr_static(type(obj), name, None)
        if isinstance(prop, property) and not name.startswith("_") and name not in WRAPPER_FLAGS \
                and len(name) > len(by_property.get(id(prop), "")):
            by_property[id(prop)] = name
    values = {}
    for name in sorted(by_property.values()):
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if _fmi_value(value, arrays):
            values[name] = value
    return values

def _detect_states(data, comments, returned, attrs):
    """Turn inputs into states: those the probe saw returned under their own name with a new value,
    and those named like a previous value.

    An input named like a previous value (x_prev, x_old, last_x, ...) whose next value is
    returned or stored as an attribute as x_next, next_x, x_new, new_x or x is a state, with
    ``next`` bound to that value.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param returned: keys of the return value.
    :type returned: set[str]
    :param attrs: attribute names of the object.
    :type attrs: set[str]
    """
    inputs = data.get("inputs", {})
    for name in list(inputs):
        if name in comments.get(RETURNED_INPUTS, ()):   # the probe saw its next value returned under its own name
            info = inputs.pop(name)
            data.setdefault("states", {})[name] = {**info, "next": f"return:{name}"}
            comments[("states", name)] = (f"fed back from return:{name} after each step: the model returns a new "
                                          f"{name} (seen in the probe); make it an input if that's not its next value")
            comments.pop(("inputs", name), None)
            continue
        base = _state_base(name)
        if base is None:
            continue
        for candidate in (f"{base}_next", f"next_{base}", f"{base}_new", f"new_{base}", base):
            if candidate in returned:
                source = f"return:{candidate}"
            elif candidate in attrs:
                source = f"attr:{candidate}"
            else:
                continue
            info = inputs.pop(name)
            data.setdefault("states", {})[name] = {**info, "next": source}
            comments[("states", name)] = f"fed back from {source} after each step"
            break
    if not inputs:
        data.pop("inputs", None)


def _numeric_attributes(obj, arrays=False):
    """Snapshots of the object's public attributes that hold FMI values.

    :param obj: the model object.
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :rtype: dict
    """
    try:
        attrs = vars(obj)
    except TypeError:
        return {}
    return {k: _snapshot(v) for k, v in attrs.items()
            if not k.startswith("_") and k not in WRAPPER_FLAGS and _fmi_value(v, arrays)}


def _dynamic_values(obj, arrays=False):
    """Snapshots of public attributes the object provides itself (``__getattr__`` listed by ``__dir__``),
    not its class or ``__dict__``: tellurium's RoadRunner gives each species (``rr.S1``) this way.

    :param obj: the model object.
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :rtype: dict
    """
    try:
        names = set(dir(obj)) - set(dir(type(obj))) - set(getattr(obj, "__dict__", {}))
    except Exception:
        return {}
    values = {}
    for name in sorted(names):
        if name.startswith("_") or not name.isidentifier():
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if _fmi_value(value, arrays) or _table_children(value):   # tables: pandapipes' net.res_junction
            values[name] = _snapshot(value)
    return values


def _snapshot(value):
    """A copy that later in-place changes (e.g. numpy ``+=``) don't affect.

    :param value: the value.
    """
    return value.copy() if hasattr(value, "copy") and not isinstance(value, (str, bytes)) else value


def _property_values(obj, arrays=False, everything=False):
    """Public properties of the object whose current value is an FMI value.

    Only properties defined in the model's own package count: base classes from frameworks
    (torch.nn.Module, transformers' PreTrainedModel, ...) add properties that aren't model results.

    :param obj: the model object.
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param everything: include every property, with the exception in place of the value when
        reading it raised.
    :type everything: bool
    :rtype: dict
    """
    package = type(obj).__module__.split(".")[0]
    own = {name for klass in type(obj).__mro__ if klass.__module__.split(".")[0] == package
           for name, member in vars(klass).items() if isinstance(member, property)}
    values = {}
    for name, prop in inspect.getmembers(type(obj), lambda m: isinstance(m, property)):
        if name.startswith("_") or prop.fget is None or name not in own or name in WRAPPER_FLAGS:
            continue
        try:
            value = getattr(obj, name)
        except Exception as e:
            if everything:   # the exception, in place of the value
                values[name] = e
            continue
        if everything or _fmi_value(value, arrays):
            values[name] = _snapshot(value)
    return values


def _own_descriptors(obj):
    """Names of the public C attributes (getset and member descriptors) of obj's class from the model's package.

    Extension types such as ephem's bodies keep their results in these, not in ``__dict__``.

    :param obj: the model object.
    :rtype: list[str]
    """
    package = type(obj).__module__.split(".")[0]
    names = []
    for name in dir(type(obj)):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(type(obj), name, None)
        if (inspect.isgetsetdescriptor(member) or inspect.ismemberdescriptor(member)) \
                and getattr(member, "__objclass__", type(obj)).__module__.split(".")[0] == package:
            names.append(name)
    return names


def _descriptor_values(obj, arrays=False):
    """``{name: value}`` of the object's own C attributes (:func:`_own_descriptors`) that hold FMI values.

    :param obj: the model object.
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :rtype: dict
    """
    values = {}
    for name in _own_descriptors(obj):
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if _fmi_value(value, arrays):
            values[name] = _snapshot(value)
    return values


def _unreadable(obj):
    """Properties and C attributes of the object whose reading raises an error an FMU would fail on.

    Missing attributes, keys and indexes (AttributeError, KeyError, IndexError) don't count: the FMU
    skips those outputs at initialization.

    :param obj: the model object.
    :return: ``{name: the exception}``.
    :rtype: dict
    """
    errors = {}
    values = _property_values(obj, everything=True)
    for name in _own_descriptors(obj):
        try:
            getattr(obj, name)
        except Exception as e:
            values[name] = e
    for name, value in values.items():
        if isinstance(value, Exception) and not isinstance(value, (AttributeError, KeyError, IndexError)):
            errors[name] = value
    return errors


#: Getter method names: ``get_quaternion``, ``getQuat6D`` (not ``getattr``-like lower-case words).
GETTER = re.compile(r"get(_[a-z0-9]\w*|[A-Z0-9]\w*)")


def _getter_outputs(obj, data, comments, arrays, taken):
    """Outputs read through getter methods called without arguments (``get:``), for a step that returns nothing.

    Methods named like ``get_farm_power`` or ``getQuat6D``, from the model's own package, that take
    no required argument (or whose signature can't be read, as for C built-ins) are called once;
    those returning an FMI value become outputs named without the ``get`` prefix.

    :param obj: the model object, after the probe's step.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param taken: names already used; updated in place.
    :type taken: set[str]
    """
    package = type(obj).__module__.split(".")[0]
    for name in dir(type(obj)):
        if not GETTER.fullmatch(name):
            continue
        member = inspect.getattr_static(type(obj), name, None)
        owner = getattr(member, "__objclass__", None) or next(
            (k for k in type(obj).__mro__ if name in vars(k)), type(obj))
        if owner.__module__.split(".")[0] != package:
            continue
        method = getattr(obj, name, None)
        if not callable(method):
            continue
        try:
            required = [p for p in inspect.signature(method).parameters.values()
                        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        except (TypeError, ValueError):   # a C built-in: try calling it
            required = []
        if required:
            continue
        try:
            value = method()
        except Exception:
            continue
        if not _fmi_value(value, arrays):
            continue
        output = name[4:] if name.startswith("get_") else name[3].lower() + name[4:]
        if output in taken or not output.isidentifier():
            output = name
        if output in taken:
            continue
        data.setdefault("outputs", {})[output] = _typed({"from": f"get:{name}"}, value, arrays)
        comments[("outputs", output)] = f"returned by {name}() after the step"
        taken.add(output)


def _writable_attribute(obj, name):
    """Whether ``name`` can be set on the object: an instance attribute, or a property with a setter.

    :param obj: the model object.
    :param name: the attribute.
    :type name: str
    :rtype: bool
    """
    if name not in getattr(obj, "__dict__", {}):
        prop = inspect.getattr_static(type(obj), name, None)
        return isinstance(prop, property) and prop.fset is not None
    return True
