"""Checks after the probe's step: set-up order, made-up feeds, sizes, tunable attributes, results."""
import ast
import importlib
import inspect
import re
import textwrap
import typing
from collections.abc import Mapping

from fmugen import static
from fmugen.arguments import (
    _array_info, _fmi_value, _same, _scalar_of, _typed, _unfit, _unfit_value, _valid_name,
)
from fmugen.probe._arguments import _step_method
from fmugen.probe._calling import _build_set_up, _call_with_probe_inputs, _made_up_training, _probe_call
from fmugen.probe._reading import _numeric_attributes, _property_values, _snapshot, _writable_attribute
from fmugen.toml_output import _key


#: ONNX element types (``tensor(float)``) as numpy dtypes.
ONNX_TYPES = {"float": "float32", "double": "float64", "float16": "float16", "int64": "int64", "int32": "int32",
              "int16": "int16", "int8": "int8", "uint8": "uint8", "uint16": "uint16", "bool": "bool"}


def _made_up_feed(obj, method_name, call_starts, comments):
    """An ONNX session's ``run(output_names, input_feed)``: ``input_feed`` not given is made up from the
    session's own inputs (``get_inputs()``: name, shape, element type) as zeros, a symbolic dimension as 1;
    ``output_names`` not given is None (all outputs). Updates ``call_starts`` in place.
    """
    if method_name != "run" or "input_feed" in call_starts or importlib.util.find_spec("numpy") is None:
        return
    try:
        params = inspect.signature(getattr(obj, "run")).parameters
        specs = list(obj.get_inputs())
        feed = {}
        for spec in specs:
            kind = re.fullmatch(r"tensor\((\w+)\)", spec.type)
            if kind is None or kind.group(1) not in ONNX_TYPES or not isinstance(spec.name, str):
                return
            shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in spec.shape)
            feed[spec.name] = f"numpy.zeros({shape!r}, dtype={ONNX_TYPES[kind.group(1)]!r})"
    except Exception:
        return
    if "input_feed" not in params or not feed:
        return
    call_starts["input_feed"] = {"call": "builtins:dict({" + ", ".join(f"{k!r}: {v}" for k, v in feed.items()) + "})"}
    comments[("constant", "call_constants.input_feed")] = (
        f"made up from the session's inputs ({', '.join(feed)}), as zeros: give your data with --const")
    if "output_names" in params and "output_names" not in call_starts:
        call_starts["output_names"] = None


def _made_up_actions(obj, method_name, call_starts, comments):
    """A multi-agent environment's ``step(actions)`` (PettingZoo's parallel API): ``actions`` not given is made
    up as a dict of one action per agent (``possible_agents``), each the first of its ``action_space(agent)``:
    a discrete space's ``start``, a box's zeros clipped to its bounds. A dict start, so each agent's action
    is an FMU input (``actions[agent_0]``). Updates ``call_starts`` in place."""
    if method_name != "step" or "actions" in call_starts:
        return
    try:
        if "actions" not in inspect.signature(getattr(obj, "step")).parameters:
            return
        agents = list(getattr(obj, "possible_agents", None) or getattr(obj, "agents", None) or [])
        actions = {}
        for agent in agents:
            space = obj.action_space(agent)
            if not isinstance(agent, str) or not agent.isidentifier():
                return
            if type(space).__name__ == "Discrete":
                actions[agent] = int(space.start)
            elif type(space).__name__ == "Box" and len(space.shape) <= 1:
                import numpy
                actions[agent] = [float(v) for v in numpy.atleast_1d(numpy.clip(0.0, space.low, space.high))]
            else:
                return
    except Exception:
        return
    if actions:
        call_starts["actions"] = actions
        comments[("inputs", "actions")] = ("made up: one action per agent (possible_agents), the first of each "
                                          "action_space(agent); give yours with --start")


def _environment(obj):
    """Whether ``obj`` is a Gym-style environment: ``reset`` and ``step`` methods, an action and an observation
    space."""
    try:
        return all(getattr(obj, n, None) is not None for n in ("action_space", "observation_space"))             and callable(getattr(obj, "reset", None)) and callable(getattr(obj, "step", None))
    except Exception:
        return False


def _reset_first(obj, method_name, data, comments, why=None):
    """After a failed probe step: if the object has a ``reset()`` taking no required argument, and the step
    works after it, run it in ``[model] setup`` (gymnasium's environments refuse ``step()`` before ``reset()``).

    :param obj: the model object.
    :param method_name: the step method.
    :type method_name: str
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param why: the reason in the comment, when the step didn't fail.
    :type why: str or None
    :return: whether ``reset`` was added (the object is then reset; the step itself isn't run here).
    :rtype: bool
    """
    reset = getattr(obj, "reset", None)
    if any(s == "reset" or s.startswith("reset(") for s in data["model"].get("setup", [])) \
            or method_name == "reset" or not callable(reset):
        return False
    try:
        params = inspect.signature(reset).parameters
    except (TypeError, ValueError):
        return False
    if any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
           for p in params.values()):
        return False
    # a seed it takes is given, so the FMU gives the same results every run (gymnasium's reset(seed=None))
    tries = [({"seed": 0}, "reset(seed=0)")] if "seed" in params and params["seed"].default is None else []
    for kwargs, call in [*tries, ({}, "reset")]:
        saved = {n: dict(i) for n, i in data.get("inputs", {}).items()}
        try:
            reset(**kwargs)
            # with the made-up inputs' retries: CartPole's step() takes an action from its action space, not 1.0
            _probe_call(_step_method(obj, method_name, data), data, comments, constants_key="call_constants")
            reset(**kwargs)   # back to the state the FMU starts from, for the probe's own step
        except Exception:
            if saved:
                data["inputs"] = saved
            continue
        data["model"]["setup"] = [*data["model"].get("setup", []), call]
        seed = "; seed=0 so each run gives the same results" if kwargs else ""
        comments[("model", "setup")] = (f"{call if kwargs else 'reset()'} before the first step: "
                                        f"{why or f'{method_name}() failed without it in the probe'}{seed}")
        return True
    return False


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
            signature = inspect.signature(getattr(obj, name)).parameters
            params = [p for p in signature.values()
                      if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            continue
        # a seed it takes is given, so the FMU gives the same results every run (gymnasium's reset(seed=None))
        seeded = "seed" in signature and signature["seed"].default is None
        worked = False
        for made in (m for made in _made_up_variants(getattr(obj, name), params)
                     for m in ([*made, "seed=0"], made)[0 if seeded else 1:]):
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
        seed = "; seed=0 so each run gives the same results" if "seed=0" in made else ""
        comments[("model", "setup")] = (f"{name}() before the first step: {method_name}() failed without it in the "
                                        f"probe{made_up}{seed}")
        return fresh
    candidates = [m for m in candidates if m not in unhelpful]
    if not candidates:
        return ""
    name = candidates[0]
    why = f" (it sets {', '.join(sorted(unset & _assigned(cls, name)))}, which {method_name}() uses)" \
            if unset & _assigned(cls, name) else ""
    return f"{method_name}() seems to need {name}() first{why}: add --setup \"{name}(...)\""


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
