"""Standalone utility functions used by Engine, CompositeEngine and the build-time tools."""
from fmugen.templates.fmugen_runtime._constants import (
    FLOAT_TYPES, INT_RANGES, TRUE_STRINGS, FALSE_STRINGS, SPEC_VERSION, DEFAULT_VALUES, Status,
)
import ast
import builtins
import contextlib
import copy
import dataclasses
import datetime
import importlib
import inspect
import json
import logging
import math
import os
import pickle
import random
import sys
import tempfile
import traceback
import types
import typing
import warnings
from collections.abc import Iterable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path


# ================= helpers =================


def _clock_of(var):
    """The clock a variable belongs to, or None for an unclocked variable.

    :param var: an entry of ``interface["variables"]``.
    :type var: dict
    :return: the first name in its ``clocks``, or None.
    :rtype: str or None
    """
    clocks = var.get("clocks")
    return clocks[0] if clocks else None


def setup_sys_path(interface, resources_dir):
    """Prepend the interface's ``sys_path`` folders (relative to ``resources/``) to ``sys.path``.

    So the user's modules, copied into ``resources/fmugen_model/``, import by their own names.
    Folders already on the path are not added twice; the order of ``sys_path`` is kept.

    :param interface: a parsed ``interface.json``.
    :type interface: dict
    :param resources_dir: the folder the paths are relative to.
    :type resources_dir: str or pathlib.Path
    """
    for path in reversed(interface.get("sys_path", [])):
        path = str((Path(resources_dir) / path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def load_entry(interface, resources_dir):
    """Import the user's module and get its entry class or function.

    Uses :func:`importlib.import_module` on ``entry.module`` after :func:`setup_sys_path`, so the
    module is imported exactly as the user's own code would import it.

    :param interface: a parsed ``interface.json``.
    :type interface: dict
    :param resources_dir: the FMU's ``resources/`` folder.
    :type resources_dir: str or pathlib.Path
    :return: ``(module, entry object)``.
    :rtype: tuple
    :raises ImportError: the module can't be imported.
    :raises AttributeError: the module has no ``entry.name``.
    """
    setup_sys_path(interface, resources_dir)
    entry = interface["entry"]
    module = importlib.import_module(entry["module"])
    return module, getattr(module, entry["name"])


def rng_states():
    """States of the global random number generators the model may draw from.

    Saved with the FMU state, so a rolled-back step draws the same numbers again. numpy and torch
    are only looked up in :data:`sys.modules` (never imported here): if the model didn't import
    them, it can't have used them.

    :return: ``{"random": ..., "numpy": ..., "torch": ..., "torch.cuda": ...}`` for those loaded.
    :rtype: dict
    """
    states = {"random": random.getstate()}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        states["numpy"] = numpy.random.get_state()
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "get_rng_state"):
        states["torch"] = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            states["torch.cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states):
    """Restore random generator states saved by :func:`rng_states`.

    Generators whose module isn't loaded (or whose state wasn't saved) are left alone.

    :param states: as returned by :func:`rng_states`.
    :type states: dict
    """
    if "random" in states:
        random.setstate(states["random"])
    if "numpy" in states and "numpy" in sys.modules:
        sys.modules["numpy"].random.set_state(states["numpy"])
    torch = sys.modules.get("torch")
    if torch is not None:
        if "torch" in states:
            torch.set_rng_state(states["torch"])
        if "torch.cuda" in states:
            torch.cuda.set_rng_state_all(states["torch.cuda"])


def split_global(ref):
    """Resolve a ``[model] globals`` entry into its module and name.

    :param ref: ``"package.module:NAME"``.
    :type ref: str
    :return: ``(module, "NAME")``.
    :rtype: tuple
    :raises AttributeError: the module has no such global.
    """
    module_name, _, name = ref.partition(":")
    module = importlib.import_module(module_name)
    if not hasattr(module, name):
        raise AttributeError(f"module {module_name} has no global {name!r}")
    return module, name


def resolve_reference(ref):
    """Import the object a reference names.

    Accepts ``"package.module:attr.path"`` or a dotted ``"package.module.attr"``. In the dotted
    form the longest importable module prefix is imported and the rest is followed as attributes.
    With a colon, if the module part isn't a real submodule (``pybamm.lithium_ion:SPM``, where
    ``lithium_ion`` is an attribute set lazily by ``pybamm``), the dotted form is tried instead.

    :param ref: the reference.
    :type ref: str
    :return: the object.
    :raises ImportError: nothing in the reference can be imported.
    :raises AttributeError: an attribute on the path doesn't exist.
    """
    if ":" in ref:
        module_name, _, attr = ref.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # 'pybamm.lithium_ion:SPM': lithium_ion is an attribute of pybamm (set lazily), not a submodule
            if e.name is None or not (module_name + ".").startswith(e.name + ".") or "." not in module_name:
                raise
            try:
                return resolve_reference(f"{module_name}.{attr}")
            except (ImportError, AttributeError):
                raise e from None
        return get_path(module, attr)
    parts = ref.split(".")
    for i in range(len(parts), 0, -1):  # longest importable module prefix
        try:
            module = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        return get_path(module, ".".join(parts[i:])) if i < len(parts) else module
    raise ImportError(f"cannot import {ref!r}")


def dump_state(state):
    """Pickle an FMU state with the first library that can.

    pickle, else cloudpickle (lambdas, local functions), else dill (when installed); the first
    byte of the result says which (``P``, ``C``, ``D``), for :func:`load_state`.

    :param state: the state dict built by :meth:`Engine.serialize`.
    :type state: dict
    :return: the tagged bytes.
    :rtype: bytes
    :raises Exception: the last pickling error when none of them can pickle the state.
    """
    try:
        return b"P" + pickle.dumps(state)
    except Exception as e:
        error = e
    for prefix, name in ((b"C", "cloudpickle"), (b"D", "dill")):
        try:
            return prefix + importlib.import_module(name).dumps(state)
        except ImportError:
            continue
        except Exception as e:
            error = e
    raise error


def load_state(data):
    """Unpickle bytes made by :func:`dump_state`.

    cloudpickle pickles load with plain pickle once cloudpickle is importable, so it is only
    imported; dill's are loaded by dill.

    :param data: tagged bytes.
    :type data: bytes
    :return: the state dict.
    :rtype: dict
    :raises ValueError: the first byte isn't a known tag.
    """
    kind, payload = data[:1], data[1:]
    if kind == b"C":
        import cloudpickle  # noqa: F401  (its pickles load with pickle once it is importable)
    elif kind == b"D":
        import dill
        return dill.loads(payload)
    elif kind != b"P":
        raise ValueError("not an FMU state saved by this FMU")
    return pickle.loads(payload)


def split_item(name):
    """Split a bound argument name into its base argument and item key or field.

    ``"pair[0]"`` -> ``("pair", 0)``; ``"params[key]"`` -> ``("params", "key")``;
    ``"u.temp"`` -> ``("u", Field("temp"))``; ``"x"`` -> ``("x", None)``.

    :param name: the name after ``arg:`` / ``init:`` in ``to``.
    :type name: str
    :return: ``(base, key)``.
    :rtype: tuple
    """
    if "." in name and "[" not in name:
        base, _, field = name.partition(".")
        return base, Field(field)
    if not name.endswith("]") or "[" not in name:
        return name, None
    base, _, key = name[:-1].partition("[")
    key = key.strip().strip("'\"")
    return base, int(key) if key.isdigit() else key


class Field(str):
    """The field name in ``to = "arg:u.temp"``: ``u`` is an object built from its fields.

    A ``str`` subclass, so :func:`assemble_items` can tell fields (dataclass, pydantic, attrs
    objects) from dict keys.
    """


def assemble_items(bound, fn=None):
    """Combine bound item and field arguments into whole arguments.

    ``{"pair[0]": a, "pair[1]": b, "u.T": c, "x": d}`` -> ``{"pair": (a, b), "u": U(T=c), "x": d}``.
    Integer keys make a tuple (numbered 0, 1, ... without gaps), other keys a dict; fields make
    an object from ``fn``'s signature (:func:`build_object`).

    :param bound: bound name -> value.
    :type bound: dict
    :param fn: the function whose arguments these are (needed only for fields).
    :type fn: callable or None
    :return: argument name -> value.
    :rtype: dict
    :raises ValueError: fields mixed with items, or tuple items with gaps.
    """
    result, items = {}, {}
    for name, value in bound.items():
        base, key = split_item(name)
        if key is None:
            result[name] = value
        else:
            items.setdefault(base, {})[key] = value
    for base, parts in items.items():
        if all(isinstance(k, Field) for k in parts):
            result[base] = build_object(fn, base, {str(k): v for k, v in parts.items()})
        elif any(isinstance(k, Field) for k in parts):
            raise ValueError(f"{base}: use either fields (arg:{base}.name) or items (arg:{base}[key]), not both")
        elif all(isinstance(k, int) for k in parts):
            if sorted(parts) != list(range(len(parts))):
                raise ValueError(f"{base}: tuple items must be numbered 0, 1, 2, ... without gaps")
            result[base] = tuple(parts[i] for i in range(len(parts)))
        else:
            result[base] = parts
    return result


#: Cache of :func:`object_argument`: (function, argument name) -> (default, class).
_OBJECT_ARGS = {}


def object_argument(fn, name):
    """The default instance and the annotated class of ``fn``'s argument ``name``.

    Uses :func:`inspect.signature` for the default and :func:`typing.get_type_hints` for the
    class (which resolves string and ``from __future__ import annotations`` hints);
    ``Optional[X]`` / ``X | None`` is unwrapped to ``X`` with :func:`typing.get_origin` /
    :func:`typing.get_args`. For a class, the hints are those of its ``__init__``. Cached per
    (function, name), because it runs on every call.

    :param fn: the function, method or class.
    :type fn: callable
    :param name: the argument name.
    :type name: str
    :return: ``(default instance or None, class or None)``.
    :rtype: tuple
    """
    key = (getattr(fn, "__func__", fn), name)
    if key not in _OBJECT_ARGS:
        target = fn.__init__ if isinstance(fn, type) else fn
        param = inspect.signature(fn).parameters.get(name)
        default = None if param is None or param.default is inspect.Parameter.empty else param.default
        try:   # get_type_hints, not __annotations__: it resolves string (postponed) annotations
            cls = typing.get_type_hints(target).get(name)
        except Exception:
            cls = None
        if typing.get_origin(cls) in (typing.Union, types.UnionType):   # Optional[Inputs]
            rest = [a for a in typing.get_args(cls) if a is not type(None)]
            cls = rest[0] if len(rest) == 1 else None
        _OBJECT_ARGS[key] = default, cls if isinstance(cls, type) else None
    return _OBJECT_ARGS[key]


def build_object(fn, name, fields):
    """Build an object argument from bound fields.

    With a default instance, a copy of it with the fields replaced (dataclasses.replace,
    pydantic ``model_copy``, attrs.evolve, namedtuple ``_replace``, else copy + setattr), so
    fields that aren't FMU variables keep their default; otherwise the annotated class called
    with the fields.

    :param fn: the function whose argument it is.
    :type fn: callable or None
    :param name: the argument name.
    :type name: str
    :param fields: field name -> value.
    :type fields: dict
    :return: the object.
    :raises ValueError: neither a default nor a class annotation.
    """
    if fn is None:
        raise ValueError(f"{name}: cannot tell which object to build from {sorted(fields)}")
    default, cls = object_argument(fn, name)
    if default is not None:
        if dataclasses.is_dataclass(default):
            return dataclasses.replace(default, **fields)
        if hasattr(default, "model_copy"):          # pydantic 2
            return default.model_copy(update=fields)
        if hasattr(type(default), "__attrs_attrs__"):
            import attrs
            return attrs.evolve(default, **fields)
        if isinstance(default, tuple) and hasattr(default, "_replace"):   # a namedtuple
            return default._replace(**fields)
        obj = copy.copy(default)
        for field, value in fields.items():
            setattr(obj, field, value)
        return obj
    if cls is None:
        raise ValueError(f"{name}: no default and no class annotation to build it from {sorted(fields)}")
    return cls(**fields)


def keyword_constants(constants):
    """The constants passed by keyword: all but those named ``"0"``, ``"1"``, ... (``*args`` positions).

    :param constants: resolved constants.
    :type constants: dict
    :return: the keyword ones.
    :rtype: dict
    """
    return {k: v for k, v in constants.items() if not k.isdigit()}


def is_literal_constant(value):
    """Whether a constant is a value, not a ``{ call = ... }`` or ``{ ref = ... }``.

    Literal constants are resolved again on reset and saved with the FMU state; calls and
    references are objects shared by the whole run.

    :param value: the constant as written in ``interface.json``.
    :return: True for a literal.
    :rtype: bool
    """
    return not (isinstance(value, dict) and len(value) == 1 and ("call" in value or "ref" in value))


def resolve_constant(value):
    """The Python value of a constant from ``interface.json``.

    ``{ python = "..." }`` is evaluated with :func:`ast.literal_eval` (literals only, never
    arbitrary code); ``{ ref = ... }`` is imported (:func:`resolve_reference`);
    ``{ call = ... }`` is called (:func:`call_reference`); anything else is the value itself.

    :param value: the constant as written in ``interface.json``.
    :return: its value.
    """
    if isinstance(value, dict) and len(value) == 1:
        if "python" in value:
            return ast.literal_eval(value["python"])
        if "ref" in value:
            return resolve_reference(value["ref"])
        if "call" in value:
            return call_reference(value["call"])
    return value


def parse_call_text(text):
    """Parse ``"module:function(arg, key=arg, ...)"`` without calling anything.

    The argument list is parsed with :func:`ast.parse` as the call ``_(...)``, so Python's own
    grammar handles nesting and quoting; the nodes are evaluated later by :func:`call_argument`.

    :param text: the call text.
    :type text: str
    :return: ``(target, [argument nodes], {keyword: node})``.
    :rtype: tuple
    :raises ValueError: a missing ``)`` or ``*`` / ``**`` arguments.
    :raises SyntaxError: arguments that aren't valid Python.
    """
    target, paren, rest = text.partition("(")
    if not paren:
        return target.strip(), [], {}
    if not rest.rstrip().endswith(")"):
        raise ValueError(f"{text!r}: missing ')'")
    node = ast.parse(f"_({rest}", mode="eval").body
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_"):
        # "pathlib:Path('x').read_bytes()" would otherwise call Path() and drop the rest
        raise ValueError(f"{text!r}: only one call is supported, nothing after its ')' (no .method(), "
                         "[index] or operators)")
    if any(k.arg is None for k in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args):
        raise ValueError(f"{text!r}: * and ** arguments are not supported")
    return target.strip(), list(node.args), {k.arg: k.value for k in node.keywords}


def call_reference(text):
    """Call ``"module:function(args)"`` and return the result.

    E.g. ``load_from_hub(repo_id="sb3/x", filename="m.zip")``. Without parentheses,
    ``"module:attr.path"`` is the object itself, not called (a function to pass, an enum member).
    Arguments: see :func:`call_argument`.

    :param text: the call text.
    :type text: str
    :return: what the call returned (or the object).
    """
    target, paren, _ = text.partition("(")
    if not paren:
        return resolve_reference(target.strip())
    target, args, kwargs = parse_call_text(text)
    return resolve_reference(target)(*[call_argument(a) for a in args],
                                     **{k: call_argument(v) for k, v in kwargs.items()})


def call_argument(node):
    """Evaluate one argument node of a call constant.

    Allowed: a Python literal; a dotted name of an importable object or a builtin
    (``numpy.sin``, ``float``); a call of one (``numpy.linspace(0, 1, 5)``, ``float("inf")``);
    a list, tuple, set or dict of these; a negated value. Evaluating the AST ourselves (instead
    of ``eval``) keeps this to imports and calls the user wrote, nothing else.

    :param node: an expression node from :func:`parse_call_text`.
    :type node: ast.expr
    :return: the value.
    :raises ValueError: any other kind of expression.
    """
    with contextlib.suppress(ValueError, SyntaxError, TypeError):
        return ast.literal_eval(node)
    if isinstance(node, ast.Name) and hasattr(builtins, node.id):
        return getattr(builtins, node.id)
    if isinstance(node, (ast.Name, ast.Attribute)):
        return resolve_reference(ast.unparse(node))
    if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
        if any(k.arg is None for k in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args):
            raise ValueError(f"{ast.unparse(node)!r}: * and ** arguments are not supported")
        return call_argument(node.func)(*[call_argument(a) for a in node.args],
                                        **{k.arg: call_argument(k.value) for k in node.keywords})
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [call_argument(e) for e in node.elts]
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict) and None not in node.keys:
        return {call_argument(k): call_argument(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = call_argument(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError(f"{ast.unparse(node)!r}: arguments must be literals, names of importable objects, "
                     "calls of those, or lists/tuples/dicts of these")


def converter(var, now=lambda: 0.0):
    """The function applied to a value before the model gets it (``convert``).

    ``"pint"`` makes a quantity in the variable's ``unit`` from pint's application registry;
    ``"series"`` a one-value pandas Series indexed by the date ``epoch`` + ``now()`` seconds (for
    models that read the date from their inputs' index, such as pyet); anything else is a
    reference to a function.

    :param var: an entry of ``interface["variables"]`` with ``convert``.
    :type var: dict
    :param now: the FMU time, for ``"series"``.
    :type now: callable
    :return: the converter.
    :rtype: callable
    """
    if var["convert"] == "pint":   # a quantity in the variable's unit, from pint's application registry
        import pint
        registry, unit = pint.get_application_registry(), var.get("unit", "")
        return lambda value: registry.Quantity(value, unit)
    if var["convert"] == "series":
        import pandas
        epoch = datetime.datetime.fromisoformat(var["epoch"])
        return lambda value: pandas.Series(
            [value], index=pandas.DatetimeIndex([epoch + datetime.timedelta(seconds=now())]))
    return resolve_reference(var["convert"])


def _labelled(obj):
    """Whether ``obj`` is a pandas or xarray object indexed by labels (has ``keys``).

    Checked by module name, so pandas and xarray are never imported here.

    :param obj: any object.
    :return: True for a labelled pandas/xarray object.
    :rtype: bool
    """
    return type(obj).__module__.split(".")[0] in ("pandas", "xarray") and hasattr(obj, "keys")


def results_of(result):
    """What a model call returned, with a generator read into a tuple.

    A generator (nashpy's ``support_enumeration``) can only be read once; as a tuple, several
    outputs can read it and it can be saved with the FMU state. Anything else is returned
    unchanged.

    :param result: what the model returned.
    :return: ``tuple(result)`` for a generator, else ``result``.
    """
    return tuple(result) if isinstance(result, types.GeneratorType) else result


def plain_number(value, unit=None):
    """A quantity or one-value pandas/xarray object as a plain number.

    A pint quantity (anything with ``.magnitude`` and ``.units``) or an astropy one becomes its
    magnitude, in ``unit`` when given; a one-value pandas/xarray column becomes its value.
    Anything else is returned unchanged.

    :param value: what the model returned.
    :param unit: the variable's unit, to convert to.
    :type unit: str or None
    :return: the number (or ``value``).
    """
    if type(value).__module__.split(".")[0] in ("pandas", "xarray") and getattr(value, "size", None) == 1:
        value = value.item()
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return value.m_as(unit) if unit and hasattr(value, "m_as") else value.magnitude
    if hasattr(value, "unit") and hasattr(value, "to_value"):   # an astropy Quantity
        try:
            return value.to_value(unit) if unit else value.value
        except Exception:   # a unit astropy doesn't know: the value in its own unit
            return value.value
    return value


def _indexable(obj):
    """Whether a path part like ``"0"`` or ``"-1"`` indexes ``obj`` by position.

    Lists, tuples and arrays (numpy, torch: anything with ``ndim`` of 1 or more), but not pandas or
    xarray objects, whose ``[0]`` is a label.

    :param obj: any object.
    :rtype: bool
    """
    if isinstance(obj, (tuple, list)):
        return True
    module = type(obj).__module__.split(".")[0]
    return module not in ("pandas", "xarray") and (getattr(obj, "ndim", 0) or 0) >= 1 \
        and hasattr(obj, "__getitem__")


_MISSING = object()   # _nth / _item: the fallback doesn't apply


def _nth(obj, part):
    """Element ``part`` (``"0"``, ``"-1"``) of an iterable that can't be indexed, such as a ``dict_keys`` view.

    Only tried where an attribute lookup has failed. Strings, mappings, pandas/xarray objects and
    iterators (which reading would consume) are left out.

    :param obj: any object.
    :param part: the path part.
    :type part: str
    :return: the element, or ``_MISSING`` when this doesn't apply.
    :raises IndexError: ``obj`` has fewer elements.
    """
    if not (part.lstrip("-").isdigit() and part.count("-") <= 1) or isinstance(obj, (str, bytes, Mapping)) \
            or type(obj).__module__.split(".")[0] in ("pandas", "xarray") or not isinstance(obj, Iterable) \
            or isinstance(obj, Iterator):
        return _MISSING
    return list(obj)[int(part)]


def _item(obj, part):
    """``obj[part]`` for an object that isn't a mapping but takes keys, such as pybamm's ``Solution``.

    Only tried where an attribute lookup has failed. Strings and mappings are left out (a
    mapping's part is already a key).

    :param obj: any object.
    :param part: the path part, used as the key.
    :type part: str
    :return: the item, or ``_MISSING`` when ``obj`` has no such item.
    """
    if isinstance(obj, (str, bytes, Mapping)) or not hasattr(type(obj), "__getitem__"):
        return _MISSING
    try:
        return obj[part]
    except (KeyError, IndexError, TypeError, ValueError):
        return _MISSING


def get_path(obj, path):
    """Follow a dotted path through attributes, mapping keys and sequence indexes.

    ``"state.T"``, ``"rewards.speed"``, ``"4.x"``, ``"X.value.-1"`` (the last element). On a
    mapping a part is a key, unless it's only an attribute (``"options.update"``: the method).
    A number on any other iterable without indexing (``"bodies.0"`` on a ``dict_keys`` view)
    picks that element, and on an object that takes keys (``obj["Voltage [V]"]``) a part is
    then tried as a key, but both only where there's no such attribute.

    :param obj: the start object (usually the model object or a call result).
    :param path: the dotted path.
    :type path: str
    :return: the value at the end of the path.
    :raises AttributeError: a missing attribute (or KeyError / IndexError).
    """
    for part in path.split("."):
        if isinstance(obj, Mapping) and (part in obj or not hasattr(obj, part)) or _labelled(obj) and part in obj:   # "options.update": the method
            obj = obj[part]
        elif part.lstrip("-").isdigit() and part.count("-") <= 1 and _indexable(obj):
            try:
                obj = obj[int(part)]
            except TypeError as e:   # a list-like wrapping a single value (a GEKKO variable before its first solve)
                raise IndexError(f"{part}: {e}") from e
        else:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                element = _nth(obj, part)
                if element is _MISSING:
                    element = _item(obj, part)
                if element is _MISSING:
                    raise
                obj = element
    return obj


def set_path(obj, path, value):
    """Set the attribute (or item) at the end of a dotted path.

    The parts before the last are followed as attributes; only where an attribute lookup fails is
    the part read as :func:`get_path` reads it (a key, an index). The last part is set as an item
    only where there's no such attribute and it can only be an item: a key already in a mapping or
    a labelled pandas/xarray object (``"loads.p_set.l"`` sets ``obj.loads.p_set["l"]``), or an
    index of a list or array (``"qpos.0"``). Everything else is set as an attribute, as before.

    :param obj: the start object (the model object).
    :param path: the dotted path.
    :type path: str
    :param value: the new value.
    """
    *parents, last = path.split(".")
    table = column = None
    for part in parents:
        table, column = obj, part
        try:
            obj = getattr(obj, part)
        except AttributeError:   # a key or an index: "loads.p_set.l", "data.qpos.0"
            obj = get_path(obj, part)
    try:
        attribute = hasattr(obj, last)
        if attribute and _labelled(obj) and not hasattr(type(obj), last) and last in obj:
            attribute = False   # pandas reads labels as attributes (series.l): it's an item
    except Exception:   # a property that fails to read: set it as before
        attribute = True
    if attribute:
        setattr(obj, last, value)
    elif (isinstance(obj, Mapping) or _labelled(obj)) and last in obj:
        if _labelled(table) and hasattr(table, "loc") and column in getattr(table, "columns", ()):
            # a column of a table is a copy under pandas' copy-on-write: set the table's cell
            table.loc[last, column] = value
        else:
            obj[last] = value
    elif last.lstrip("-").isdigit() and last.count("-") <= 1 and _indexable(obj):
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def _self_path(node):
    """The path after ``self`` of an argument like ``self`` or ``self.model.data``, or None.

    :param node: an argument node from :func:`parse_call_text`.
    :type node: ast.expr
    :return: ``""`` for ``self``, ``"model.data"`` for ``self.model.data``, None for anything else.
    :rtype: str or None
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "self":
        return ".".join(reversed(parts))
    return None


def object_call(text, obj):
    """The step call ``"module:function(self, self.attr, ...)"`` of ``[model] call``, bound to ``obj``.

    For libraries whose step is a function taking the model object (``pandapipes:pipeflow(self)``,
    ``mujoco:mj_step(self.model, self.data)``). ``self`` is the model object and ``self.<path>``
    follows a path on it (:func:`get_path`), read on every call; other arguments are evaluated like
    those of a call constant (:func:`call_argument`). Arguments bound to variables (``arg:``,
    ``pos:``) come after these.

    :param text: the call text.
    :type text: str
    :param obj: the model object.
    :return: a callable taking the remaining arguments.
    :rtype: callable
    """
    target, args, kwargs = parse_call_text(text)
    fn = resolve_reference(target)

    def value(node):
        path = _self_path(node)
        if path is None:
            return call_argument(node)
        return get_path(obj, path) if path else obj

    def call(*more, **more_kwargs):
        return fn(*[value(a) for a in args], *more, **{k: value(v) for k, v in kwargs.items()}, **more_kwargs)
    return call


def pick(result, key):
    """Pick a value out of a call result.

    :param result: what the model returned.
    :param key: None for the whole result; a key (possibly containing dots) of a mapping;
        else a path for :func:`get_path`.
    :type key: str or None
    :return: the value.
    """
    if key is None:
        return result
    if isinstance(result, Mapping) and key in result:   # a key that itself contains dots
        return result[key]
    return get_path(result, key)


def describe(source):
    """Describe a ``from`` / ``next`` source for error messages.

    :param source: ``{"kind": "return" | "attr" | "get", "name": ...}``.
    :type source: dict
    :return: e.g. ``"the return value ['y']"``, ``"attribute 'state.T'"`` or ``"method 'getQuat6D'()"``.
    :rtype: str
    """
    if source["kind"] == "return":
        return "the return value" + (f" [{source['name']!r}]" if source.get("name") else "")
    if source["kind"] == "get":
        return f"method {source['name']!r}()"
    return f"attribute {source['name']!r}"


def flatten(value):
    """Flatten nested lists and tuples into one list.

    :param value: a scalar or nested lists.
    :return: the elements in row-major order.
    :rtype: list
    """
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in flatten(item)]
    return [value]


def reshape(flat, dims):
    """Turn a flat list into nested lists of the given dimensions (row-major).

    :param flat: the elements.
    :type flat: list
    :param dims: the dimensions.
    :type dims: list[int]
    :return: nested lists.
    :rtype: list
    """
    if len(dims) <= 1:
        return list(flat)
    step = len(flat) // dims[0]
    return [reshape(flat[i * step:(i + 1) * step], dims[1:]) for i in range(dims[0])]


def _is_number(text):
    """Whether a string can be read as a number by :func:`float`.

    :param text: the string.
    :type text: str
    :rtype: bool
    """
    try:
        float(text)
    except ValueError:
        return False
    return True


def coerce(fmi_type, value):
    """Convert a value to an FMI type, checking it fits.

    :param fmi_type: an FMI 2 or FMI 3 type name (``Real``, ``Int16``, ``Boolean``, ...).
    :type fmi_type: str
    :param value: the value to convert. Booleans accept ``TRUE_STRINGS`` / ``FALSE_STRINGS``;
        Binary accepts hex strings.
    :return: the converted value.
    :raises ValueError: a non-integer for an integer type, a value out of range, or a string
        that isn't a Boolean.
    """
    if fmi_type in FLOAT_TYPES:
        return float(value)
    if fmi_type in INT_RANGES or fmi_type == "Enumeration":
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{value!r} is not an integer")
        value = int(value)
        low, high = INT_RANGES.get(fmi_type, (-2**63, 2**63 - 1))
        if not low <= value <= high:
            raise ValueError(f"{value} is out of range for {fmi_type}")
        return value
    if fmi_type == "Boolean":
        if isinstance(value, str):
            text = value.strip().lower()
            if text in TRUE_STRINGS:
                return True
            if text in FALSE_STRINGS:
                return False
            raise ValueError(f"{value!r} is not a Boolean")
        return bool(value)
    if fmi_type == "Binary":
        if isinstance(value, str):
            return bytes.fromhex(value)
        return bytes(value)
    return str(value)


def _quiet_debug_logging():
    """Inside UniFMU's backend, raise the root logger from DEBUG to INFO.

    The backend's main.py calls logging.basicConfig(level=logging.DEBUG), so every library logs
    its debug records to the console: numba writes about 3 MB while compiling antropy's functions.
    UniFMU 0.14 crashes (its dispatcher panics) when the backend writes that much, and the importer
    then waits forever. INFO and above still reach the console and the importer's log.
    """
    root = logging.getLogger()
    if "UNIFMU_DISPATCHER_ENDPOINT" in os.environ and root.level < logging.INFO:
        root.setLevel(logging.INFO)


def _capture_output(enabled):
    """Inside UniFMU's backend, send stdout and stderr (file descriptors 1 and 2) to a file.

    Only with ``fmugen build --capture-output``. UniFMU 0.14 crashes when its backend writes more
    than about 4 KB to the console, which models that print warnings or progress bars do. The
    output then goes to the importer's log instead (:meth:`Engine._forward_output`). Outside
    UniFMU (build probe, tests), nothing is redirected. ``os.dup2`` is used rather than replacing
    ``sys.stdout`` so output from C extensions is captured too.

    :param enabled: the interface's ``capture_output``.
    :type enabled: bool
    :return: the temporary file the output goes to, or None.
    :rtype: typing.BinaryIO or None
    """
    if not enabled or "UNIFMU_DISPATCHER_ENDPOINT" not in os.environ:
        return None
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    sink = tempfile.TemporaryFile()
    for fd in (1, 2):
        with contextlib.suppress(OSError):
            _CONSOLE.setdefault(fd, os.dup(fd))
            os.dup2(sink.fileno(), fd)
    return sink


#: The backend's own stdout and stderr while :func:`_capture_output` redirects them.
_CONSOLE = {}


def _release_output(sink, first=""):
    """Undo :func:`_capture_output` and write what it captured to the console.

    For an error that ends the backend (the model fails to import at instantiation): UniFMU shows
    the backend's console output, which would otherwise be in a file nobody reads.

    :param sink: what :func:`_capture_output` returned (None: nothing captured).
    :type sink: typing.BinaryIO or None
    :param first: a line written before the captured output: UniFMU cuts the output short, and
        warnings printed while importing (torch_geometric's) would push it out.
    :type first: str
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    for fd, console in _CONSOLE.items():
        with contextlib.suppress(OSError):
            os.dup2(console, fd)
    with contextlib.suppress(OSError):
        os.write(2, first.encode(errors="replace"))
    if sink is None:
        return
    with contextlib.suppress(OSError, ValueError):
        sink.seek(0)
        os.write(2, sink.read()[-20000:])


#: Largest FMI 3 uint64: the limit of a ``*Fraction`` resolution.
UINT64_MAX = 2**64 - 1


def fraction(value):
    """A float as ``(counter, resolution)`` for FMI 3 ``*Fraction`` functions.

    Both are uint64, so a value that needs a larger resolution (1e-20) is rounded to the nearest
    fraction that fits. Uses the float's decimal text, so 0.1 gives 1/10, not its binary value.

    :param value: an interval or shift in seconds.
    :type value: float
    :return: ``(counter, resolution)``.
    :rtype: tuple[int, int]
    """
    f = Fraction(str(value))
    if f.denominator > UINT64_MAX:
        f = f.limit_denominator(UINT64_MAX)
    return f.numerator, f.denominator


def outside_bundle(path):
    """The folder of the FMU's ``resources/`` that ``path`` stands for, when the backend is compiled.

    A compiled backend (``fmugen build --compile``) reads its code and ``interface.json`` from the
    bundle: ``resources/dist/main/_internal`` (PyInstaller's ``sys._MEIPASS``) or
    ``resources/dist/main`` (Nuitka). The data the build keeps outside it (``[model] cwd`` folders,
    ``hf_cache``) stays at the same place under ``resources/``. Uncompiled, ``path`` is returned as is.

    :param path: a folder inside the bundle, e.g. the resources folder the engine was given.
    :type path: str or pathlib.Path
    :rtype: pathlib.Path
    """
    path = Path(path)
    if not getattr(sys, "frozen", False) and "__compiled__" not in globals():
        return path
    exe_dir = Path(sys.executable).resolve().parent   # resources/dist/main
    bundle = Path(getattr(sys, "_MEIPASS", exe_dir)).resolve()
    try:
        return exe_dir.parent.parent / path.resolve().relative_to(bundle)
    except ValueError:   # not inside the bundle
        return path
