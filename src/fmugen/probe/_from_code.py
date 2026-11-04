"""Outputs and locals read from the model's code (what it returns, the attributes it assigns)."""

from fmugen import static
from fmugen.arguments import PROBED, _guessed, _unfit
from fmugen.toml_output import _key, _value


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


def _add_attributes(cls, method_name, data, comments, arrays, probe, seen, obj=None):
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
    :param obj: the probed object, or None.
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
        if obj is not None and not _scalar_or_unset(obj, attr):
            # not among the attributes the probe reads, but not a single value either (botorch's train_targets,
            # a tensor kept as a torch buffer): typed as one, it would fail the FMU
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


def _scalar_or_unset(obj, attr):
    """Whether ``obj.attr`` is a single FMI value, None or missing (not an array or another object)."""
    try:
        value = getattr(obj, attr, None)
    except Exception:
        return True
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    return type(value).__module__ == "numpy" and getattr(value, "ndim", None) == 0
