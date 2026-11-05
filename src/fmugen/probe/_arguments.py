"""The arguments a probe call passes: start values, constants, ``*args`` and the step method."""
import inspect
import pickle

from fmugen.arguments import _only_varargs
from fmugen.probe._base import ConstantError, _run_setup
from fmugen.toml_output import _value


def _probe_constants(data, table, positional=False, obj=None):
    """Constants set in the config (e.g. computed by a call), resolved as the FMU would.

    :param data: the config being inferred.
    :type data: dict
    :param table: ``"constants"`` or ``"call_constants"``.
    :type table: str
    :param positional: return ``{position: value}`` of those named ``"0"``, ``"1"``, ... (passed to
        ``*args``) instead of the keyword ones.
    :type positional: bool
    :param obj: the model object, for ``call_constants`` computed from it (``self.…``).
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
            values[int(k) if positional else k] = resolve_constant(v, obj)
        except Exception as e:
            what = f"call:{v['call']}" if isinstance(v, dict) and "call" in v else _value(v)
            raise ConstantError(f"[model] {table}.{k} = {what} fails: {type(e).__name__}: {e}") from e
    return values


def _owner(fn, constants_key):
    """The model object a step method is bound to, for ``call_constants`` computed from it, or None.

    :param fn: the step (a bound method) or function.
    :param constants_key: the constants table passed with the call.
    :type constants_key: str or None
    """
    return getattr(fn, "__self__", None) if constants_key == "call_constants" else None


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
        by_position.update(_probe_constants(data, constants_key, positional=True, obj=_owner(fn, constants_key)))
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
