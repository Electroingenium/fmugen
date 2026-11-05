"""``--probe`` for a class: construct the object, call the step once, read what changed."""
import inspect
import types

from fmugen import static
from fmugen.arguments import (
    PROBED, _apply_converts, _arguments, _fmi_value, _misfit, _only_varargs, _same, _typed, _unfit_value,
    _valid_name,
)
from fmugen.config import InterfaceError
from fmugen.probe._arguments import _check_picklable, _state_bytes, _step_method
from fmugen.probe._base import ConstantError, _requested_outputs, _run_setup
from fmugen.probe._calling import (
    _build_set_up, _call_with_probe_inputs, _construct, _probe_call, _rebuild_made_up_as_zero, _rebuild_with_options,
)
from fmugen.probe._checks import (
    _check_sizes, _check_tunable, _environment, _gekko_outputs, _made_up_actions, _made_up_feed,
    _reset_first, _setup_first, _unfit_properties,
)
from fmugen.probe._reading import (
    _descriptor_values, _dynamic_values, _getter_outputs, _numeric_attributes, _outputs_from_return,
    _property_values, _snapshot, _table_children, _unreadable, _writable_attribute,
)


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

    if not object_step:
        _made_up_feed(obj, method_name, call_starts, comments)
        _made_up_actions(obj, method_name, call_starts, comments)
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
    if method_name == "step" and _environment(obj):
        # the Gym API resets an environment before its first step; some step without it, from a state that
        # isn't the episode's start (a gym distillation column: its outputs are left unset)
        if _reset_first(obj, method_name, data, comments,
                        why="it's an environment (an action and an observation space), which starts with reset()"):
            try:   # a new one: the probe's trial step may have set attributes that reset() leaves
                obj = _build_set_up(build, data)
            except Exception:
                pass
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
        try:   # a new object: the failed calls may have left this one broken (gym-pybullet-drones' step([[...]]))
            fresh = _build_set_up(build, data) if rebuilt is None else None
        except Exception:
            fresh = None
        if rebuilt is not None:   # the made-up parameters as 0.0 (quantecon's A = 1.0: a singular matrix)
            obj = rebuilt
        elif fresh is not None and _reset_first(fresh, method_name, data, comments):
            obj = fresh
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
    # the object couldn't be pickled to compare, so "nothing changed" rests on its numbers only
    # (pybamm's Simulation keeps its state in a solution object): a second step confirms it
    unconfirmed = data["model"].get("init_call") and (state_before is None or state_after is None)
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
    elif isinstance(result, bool) or type(result).__name__ == "bool_" or type(result) is int and result in (0, 1):
        # the step returns only a status: its results come from getters (vosk's AcceptWaveform() says whether an
        # utterance ended, as 0 or 1; FinalResult() gives the text)
        _getter_outputs(obj, data, comments, arrays, taken, results=True)
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
    _requested_outputs(data, comments, arrays, obj=obj, result=result)
    if unconfirmed:
        _confirm_unchanged(obj, method_name, result, data, comments)
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


def _output_values(obj, result, data):
    """``{output: value}`` of the outputs found so far, read as the FMU reads them; unreadable ones left out.

    :param obj: the model object.
    :param result: what the step returned.
    :param data: the config being inferred.
    :type data: dict
    :rtype: dict
    """
    from fmugen.templates.fmugen_runtime import get_path, pick
    values = {}
    for name, info in data.get("outputs", {}).items():
        kind, _, path = info.get("from", f"attr:{name}").partition(":")
        try:
            if kind == "return":
                value = pick(result, path or None)
            else:
                value = get_path(obj, path)
                if kind == "get":
                    value = value()
            values[name] = _snapshot(value)
        except Exception:
            continue
    return values


def _confirm_unchanged(obj, method_name, result, data, comments):
    """Undo ``init_call`` when a second step with the same inputs gives other outputs.

    Running the step at initialization is right only for a step that changes nothing; one that
    advances a state hidden from the probe (pybamm's ``step()``: the solution's time) would put every
    output one step ahead.

    :param obj: the model object, after the probe's step.
    :param method_name: the step method.
    :type method_name: str
    :param result: what that step returned.
    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    """
    first = _output_values(obj, result, data)
    if not first:
        return
    try:
        again = _call_with_probe_inputs(_step_method(obj, method_name, data), data, "call_constants")
    except Exception:
        return
    second = _output_values(obj, None if again is obj else again, data)
    changed = sorted(n for n in first if n in second and not _same(first[n], second[n]))
    if changed:
        del data["model"]["init_call"]
        comments.pop(("model", "init_call"), None)
        shown = ", ".join(changed[:3]) + (f" and {len(changed) - 3} more" if len(changed) > 3 else "")
        comments[("model", "#init_call")] = (f"no init_call: a second {method_name}() with the same inputs changed "
                                              f"{shown}, so running it at initialization would put the outputs a step ahead")
