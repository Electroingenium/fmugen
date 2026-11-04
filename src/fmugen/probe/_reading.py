"""Reading results out of what the probe's call returned and the object it changed."""
import dataclasses
import enum
import inspect
import math
import re
import types
from collections.abc import Mapping

from fmugen.arguments import (
    _array_info, _dotted_name, _fmi_value, _misfit, _same, _scalar_of, _state_base, _typed, _unfit_value,
    _valid_name,
)


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
    if isinstance(result, Mapping) and _table_children(result) is None:   # an xarray Dataset is a Mapping too
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
    elif result is not None and not hasattr(result, "__dict__") and (serialized := _serialized(result)) is not None:
        # an extension type (spaCy's Doc): its properties are internals (array_head, tensor); to_json() is its result
        method, mapping = serialized
        items = [(k, v, f"return:{method}().{k}") for k, v in mapping.items() if isinstance(k, str)]
    elif result is not None and (hasattr(result, "__dict__") or _public_properties(result, arrays=True)):
        # an object: its public attributes and properties, e.g. uncertainties' nominal_value, std_dev
        public = {k: v for k, v in getattr(result, "__dict__", {}).items() if not k.startswith("_")}
        public.update({k: v for k, v in _public_properties(result, arrays=True).items() if k not in public})
        items = [_single(k, v, f"return:{k}") for k, v in public.items()]
        parts = set(public) - set(getattr(result, "__dict__", {}))   # computed, not a stored copy of an input
    else:
        items = []
    if not items and result is not None and (serialized := _serialized(result)) is not None:
        # an object with nothing readable: what its own to_dict()/to_json() gives
        method, mapping = serialized
        items = [(k, v, f"return:{method}().{k}") for k, v in mapping.items() if isinstance(k, str)]
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
    if type(value) in (list, tuple) and 0 < len(value) <= MAX_ITEMS \
            and all(_plain_object(v) or isinstance(v, Mapping) for v in value):
        return [(str(i), v) for i, v in enumerate(value)]   # also dicts: spaCy's entities, {start, end, label}
    if _plain_object(value):
        public = {k: v for k, v in vars(value).items() if not k.startswith("_") and v is not None}
        public.update({k: v for k, v in _public_properties(value, arrays=True).items() if k not in public})
        # inputs kept on the object (ultralytics' orig_img) aren't results: too large to be one
        return [(k, v) for k, v in public.items() if not _size(v) > MAX_NESTED_SIZE]
    return []


#: How many items of a list of objects are split into outputs (ultralytics' one Results per image).
MAX_ITEMS = 8


#: Methods an object serializes itself with, tried in order when nothing else of it can be read: to_json first,
#: as spaCy's Doc.to_dict() is its binary form (array_head, tensor), to_json() its result (a string elsewhere: skipped).
SERIALIZERS = ("to_json", "to_dict", "as_dict", "model_dump")


def _serialized(value):
    """``(method, mapping)`` of an object read through its own serializer (:data:`SERIALIZERS`), or None.

    For results whose state isn't readable as attributes or properties: spaCy's ``Doc`` (a Cython
    type) gives ``to_json()``, ``{"text", "ents": [{"start", "end", "label"}], "tokens": [...]}``. The
    first of them that takes no arguments and returns a non-empty mapping.

    :param value: the result.
    :rtype: tuple or None
    """
    if isinstance(value, (type, str, bytes, Mapping, list, tuple)) or _table_children(value) is not None:
        return None
    for name in SERIALIZERS:
        method = getattr(value, name, None)
        if not callable(method):
            continue
        try:
            mapping = method()
        except Exception:   # it needs arguments, or fails here
            continue
        if isinstance(mapping, Mapping) and mapping:
            return name, mapping
    return None


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
    """``[(label, value)]`` for a pandas Series, a DataFrame or an xarray Dataset.

    A one-row DataFrame gives each column's value; a longer one its numeric columns as arrays
    (neuralforecast's forecast column, pandapipes' result tables), each read by its label. A
    Dataset gives its single values, and its other numeric variables as arrays (py_wake's
    ``WS_eff`` by turbine, wind direction and speed). Recognised by module and class name,
    also of a base class (py_wake's ``SimulationResult`` is a Dataset), so pandas/xarray aren't
    imported; only string labels.

    :param value: the value.
    :return: the pairs, or None for anything else.
    :rtype: list[tuple] or None
    """
    from fmugen.templates.fmugen_runtime import table_library
    library = table_library(value)
    kind = next((k.__name__ for k in type(value).__mro__ if k.__module__.split(".")[0] == library), None)
    try:
        if library == "pandas" and kind == "Series":
            pairs = list(value.items())
        elif library == "pandas" and kind == "DataFrame" and len(value) == 1:
            pairs = [(c, value[c].iloc[0]) for c in value.columns]
        elif library == "pandas" and kind == "DataFrame" and len(value) > 1:
            pairs = [(c, value[c].to_numpy()) for c in value.columns if value[c].dtype.kind in "biuf"]
        elif library == "xarray" and kind == "Dataset":
            pairs = [(k, v.item() if v.size == 1 else v.to_numpy()) for k, v in value.data_vars.items()
                     if v.size == 1 or v.dtype.kind in "biuf"]
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
        names = set(object_names(obj)) - set(dir(type(obj))) - set(getattr(obj, "__dict__", {}))
        # properties made at run time for this model, by a function (closures): roadrunner adds one per
        # species and parameter to its class when it loads a model (RoadRunner._makeProperties)
        names |= {n for n in dir(type(obj))
                  if isinstance(prop := inspect.getattr_static(type(obj), n, None), property)
                  and "<locals>" in getattr(prop.fget, "__qualname__", "")}
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


def object_names(obj):
    """``dir(obj)``, or, when the object's ``__dir__`` fails, the names of its class and ``__dict__``.

    scikit-learn's ``__dir__`` reads every attribute, so one property that raises (scikit-survival's
    ``feature_importances_``: NotImplementedError) makes ``dir`` raise.

    :param obj: the object.
    :rtype: list[str]
    """
    try:
        return dir(obj)
    except Exception:
        return sorted(set(dir(type(obj))) | set(getattr(obj, "__dict__", None) or {}))


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


#: Method names that give a result, read after a step that returns only a status: vosk's ``FinalResult``,
#: ``result``, ``final_result``, ``partial_results``.
RESULT_GETTER = re.compile(r"(?:[Ff]inal|[Pp]artial|[Ll]ast)?_?[Rr]esults?")


def _getter_outputs(obj, data, comments, arrays, taken, results=False):
    """Outputs read through getter methods called without arguments (``get:``), for a step that returns nothing.

    Methods named like ``get_farm_power`` or ``getQuat6D``, from the model's own package, that take
    no required argument (or whose signature can't be read, as for C built-ins) are called once;
    those returning an FMI value become outputs named without the ``get`` prefix. With ``results``, for a step
    that returns only a status, also those named like a result (:data:`RESULT_GETTER`: vosk's ``FinalResult``).

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
        if not GETTER.fullmatch(name) and not (results and RESULT_GETTER.fullmatch(name)):
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
        output = name if not GETTER.fullmatch(name) else name[4:] if name.startswith("get_") \
            else name[3].lower() + name[4:]
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
