"""Reading a model's source code to find its results, without running it.

`fmugen init` (without --probe) uses this instead of calling the model: it parses the
source of the function or step method with `ast` and looks at what it returns and which
attributes it assigns, and at type annotations. Nothing here calls into the model's code.
What can't be known this way (array sizes, results built at runtime, code without Python
source) is reported back so the config can say so.
"""
import ast
import dataclasses
import inspect
import textwrap
import types
import typing

#: Python type -> FMI type, for annotations and values that are plain scalars.
PRIMITIVE_TYPES = {float: "Real", int: "Integer", bool: "Boolean", str: "String"}
#: Substrings of an annotation's repr that mean an array (size unknown statically).
ARRAY_ANNOTATIONS = ("ndarray", "Tensor", "jax.Array", "tf.Tensor", "list[", "List[", "Sequence[", "numpy")
#: How far to follow self.method() calls when collecting assigned attributes.
MAX_DEPTH = 6


@dataclasses.dataclass
class Returned:
    """What a function or method returns, as far as the code tells.

    :ivar names: output names, in order.
    :ivar sources: name -> ``"return"`` / ``"return:<key>"`` (the ``from`` binding).
    :ivar types: name -> FMI type, when known.
    :ivar array: the whole result is an array (size unknown).
    :ivar unknown: why the result couldn't be read, if it couldn't.
    :ivar may_be_none: some return statements return None (simpy's Environment.run).
    :ivar nullable: names whose value may be None (database columns).
    """
    names: list = dataclasses.field(default_factory=list)   # output names, in order
    sources: dict = dataclasses.field(default_factory=dict)  # name -> "return" / "return:<key>"
    types: dict = dataclasses.field(default_factory=dict)    # name -> FMI type (when known)
    array: bool = False      # the whole result is an array (size unknown)
    unknown: str = None      # why the result couldn't be read, if it couldn't
    may_be_none: bool = False   # some return statements return None (simpy's Environment.run)
    nullable: list = dataclasses.field(default_factory=list)   # names whose value may be None (database columns)


def function_node(fn):
    """The ``ast.FunctionDef`` of ``fn``, parsed from its source.

    :func:`inspect.unwrap` removes decorators (functools.wraps) so the real function's source
    is read; :func:`inspect.getsource` gives the text, dedented so a method parses on its own.

    :param fn: a function, method, classmethod or staticmethod.
    :return: the function node, or None when there's no Python source (C extensions, builtins).
    :rtype: ast.FunctionDef or ast.AsyncFunctionDef or None
    """
    fn = inspect.unwrap(getattr(fn, "__func__", fn))
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    node = tree.body[0] if tree.body else None
    return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def _own_nodes(node):
    """All nodes inside a function body, in source order, not entering nested functions, classes or lambdas.

    Unlike :func:`ast.walk` (breadth-first, enters everything), this keeps a nested function's
    ``return`` from being taken as the outer function's.

    :param node: a function node.
    :type node: ast.FunctionDef
    :return: the nodes.
    :rtype: collections.abc.Iterator[ast.AST]
    """
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    def walk(items):   # in source order
        """Depth-first walk that skips nested scopes.

        :param items: the nodes to walk.
        :type items: collections.abc.Iterable[ast.AST]
        """
        for item in items:
            if not isinstance(item, nested):
                yield item
                yield from walk(ast.iter_child_nodes(item))

    yield from walk(node.body)


#: Function/method names whose call obviously builds an array or sequence.
ARRAY_CALLS = {"array", "asarray", "zeros", "ones", "empty", "full", "arange", "linspace", "eye", "tensor",
               "zeros_like", "ones_like", "empty_like", "full_like", "roll", "stack", "concatenate", "list", "tuple"}
#: Function names whose call obviously gives a number.
NUMBER_CALLS = {"float", "int", "abs", "round", "sqrt", "exp", "log", "log10", "log2", "sin", "cos", "tan",
                "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "pow", "hypot", "floor", "ceil", "radians",
                "degrees", "isclose"}
#: Function/class names whose call gives something that is not an FMI value.
OTHER_CALLS = {"dict", "set", "frozenset", "deque", "defaultdict", "OrderedDict", "Counter", "open", "Lock",
               "RLock", "Thread", "Queue", "Event", "object"}


def _literal_type(node, namespace=None):
    """What an assigned or returned expression obviously is.

    Only the expression's form is looked at (literals, arithmetic, comparisons, calls of
    well-known names); nothing is evaluated.

    :param node: the expression.
    :type node: ast.expr
    :param namespace: the function's globals, to recognise calls of classes (objects).
    :type namespace: dict or None
    :return: ``"Boolean"``, ``"String"``, ``"Real"``, ``"array"``, ``"other"`` (not an FMI value) or None (not obvious); ``"none"`` for a None literal.
    :rtype: str or None
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "Boolean"
        if isinstance(node.value, str):
            return "String"
        if node.value is None:
            return "none"
        return "Real" if isinstance(node.value, (int, float)) else None
    if isinstance(node, ast.BinOp):   # a * b, x ** 2; "a" + b is a string, [a] * n a list
        kinds = {_literal_type(node.left, namespace), _literal_type(node.right, namespace)}
        if kinds & {"String", "array", "other"}:
            return None
        return "Real"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return "Real"
    if isinstance(node, (ast.Compare, ast.BoolOp)) or isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "Boolean"   # a > b, a and b, not a (BoolOp of non-bools is rare in model code)
    if isinstance(node, (ast.List, ast.ListComp, ast.Tuple)):
        return "array"
    if isinstance(node, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp, ast.Lambda, ast.GeneratorExp)):
        return "other"
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name in ARRAY_CALLS:
            return "array"
        if name in NUMBER_CALLS:
            return "Real"
        if name in OTHER_CALLS:
            return "other"
        target = _resolve(node.func, namespace or {})
        if isinstance(target, type) and target not in (int, float, bool, str, complex):
            return "other"   # an instance of a class: an object, not a number
    return None


def _hint_type(hint):
    """The FMI type of a scalar annotation (``Optional`` unwrapped), or None.

    :param hint: an annotation.
    :rtype: str or None
    """
    return PRIMITIVE_TYPES.get(unwrap_optional(hint))


#: Type names in string annotations (``"Optional[float]"``) -> the type.
OPTIONAL_NAMES = {"float": float, "int": int, "bool": bool, "str": str}


def written_globals(module):
    """Names of module-level variables that a function of ``module`` rebinds (``global X; X = ...``).

    State kept outside the model object, e.g. pythermalcomfort's PRE_SHIV, which must be saved
    with the FMU state (``[model] globals``). Found by parsing the module's source.

    :param module: the model's module.
    :type module: types.ModuleType
    :return: the names, in order of first appearance.
    :rtype: list[str]
    """
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError, UnicodeDecodeError):
        return []
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        declared = {n for node in ast.walk(fn) if isinstance(node, ast.Global) for n in node.names}
        assigned = {t.id for node in ast.walk(fn) if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                    for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                    if isinstance(t, ast.Name)}
        found += [n for n in sorted(declared & assigned) if n not in found]
    return found


def unwrap_optional(hint):
    """X for ``Optional[X]`` / ``X | None`` / ``Union[X, None]`` (also written as strings), else the hint unchanged.

    :func:`typing.get_origin` is ``Union`` for ``Optional[X]`` and ``types.UnionType`` for
    ``X | None``, so both are checked. String annotations (unresolved forward references) are
    matched as text.

    :param hint: an annotation.
    :return: the unwrapped annotation.
    """
    if isinstance(hint, str):
        text = hint.replace(" ", "").replace("typing.", "")
        if text.startswith("Optional[") and text.endswith("]"):
            text = text[len("Optional["):-1]
        parts = [t for t in text.split("|") if t != "None"]
        return OPTIONAL_NAMES.get(parts[0], hint) if len(parts) == 1 else hint
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        rest = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(rest) == 1:
            return rest[0]
    return hint


def _is_array_hint(hint):
    """Whether an annotation names an array type (numpy, torch, lists, ...), by its repr.

    :param hint: an annotation, or None.
    :rtype: bool
    """
    return hint is not None and any(a in repr(hint) for a in ARRAY_ANNOTATIONS)


def _fields_of(hint):
    """Field names and types of a NamedTuple, TypedDict or dataclass type.

    :param hint: a return annotation.
    :return: ``(names, {name: FMI type})``, or None for other types.
    :rtype: tuple or None
    """
    if not isinstance(hint, type):
        return None
    if issubclass(hint, tuple) and hasattr(hint, "_fields"):
        names = list(hint._fields)
    elif typing.is_typeddict(hint) or dataclasses.is_dataclass(hint):
        names = list(getattr(hint, "__annotations__", {}))
    else:
        return None
    try:
        hints = typing.get_type_hints(hint)
    except Exception:
        hints = getattr(hint, "__annotations__", {})
    return names, {n: _hint_type(hints.get(n)) for n in names if _hint_type(hints.get(n))}


def _table_columns(hint):
    """The typed columns of a SQLAlchemy model class (mendeleev's Element).

    Read from the class's ``__table__.columns``; columns without a Python type are skipped.

    :param hint: a return annotation.
    :return: ``(names, {name: type}, nullable names)``, or None.
    :rtype: tuple or None
    """
    columns = getattr(getattr(hint, "__table__", None), "columns", None) if isinstance(hint, type) else None
    if columns is None:
        return None
    names, types, nullable = [], {}, []
    for column in columns:
        try:
            kind = PRIMITIVE_TYPES.get(column.type.python_type)
        except Exception:   # NotImplementedError for types without one
            kind = None
        name = getattr(column, "key", None) or column.name
        if kind is None or not str(name).isidentifier():
            continue
        names.append(name)
        types[name] = kind
        if getattr(column, "nullable", True):
            nullable.append(name)
    return (names, types, nullable) if names else None


def returned(fn, namespace=None, attributes=None):
    """What ``fn`` returns: dict keys, tuple positions, NamedTuple/dataclass fields, or a single value.

    First the return annotation (:func:`typing.get_type_hints`, which resolves string
    annotations); without a usable one, the function's source is parsed and its ``return``
    statements read: a dict literal (or a local dict filled key by key) gives its keys, a tuple
    its positions, a call of a NamedTuple/dataclass its fields. Local variables assigned in the
    body give the kinds of returned names.

    :param fn: the function or step method.
    :param namespace: globals to resolve names in (default: the function's ``__globals__``).
    :type namespace: dict or None
    :param attributes: ``{attr: kind}`` for a method, giving the kind of a returned ``self.attr``.
    :type attributes: dict or None
    :return: what was found.
    :rtype: Returned
    """
    try:
        hint = typing.get_type_hints(inspect.unwrap(getattr(fn, "__func__", fn))).get("return")
    except Exception:
        hint = None
    fields = _fields_of(hint)
    if fields:
        names, types = fields
        return Returned(names, {n: f"return:{n}" for n in names}, types)
    table = _table_columns(unwrap_optional(hint))
    if table:
        names, types, nullable = table
        return Returned(names, {n: f"return:{n}" for n in names}, types, nullable=nullable)
    if hint is type(None):
        return Returned()
    if _hint_type(hint):
        return Returned(["y"], {"y": "return"}, {"y": _hint_type(hint)})
    if _is_array_hint(hint):
        return Returned(["y"], {"y": "return"}, array=True)

    node = function_node(fn)
    if node is None:
        return Returned(unknown="no Python source to read (a C extension or a compiled model)")
    namespace = namespace or getattr(inspect.unwrap(getattr(fn, "__func__", fn)), "__globals__", {})
    self_name = node.args.args[0].arg if node.args.args else None
    dict_vars = _local_dicts(node)
    arg_names = {a.arg for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]} - {self_name}

    found = {}   # local variables: x = a * b
    for item in _own_nodes(node):
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            value = item.value
            if isinstance(value, ast.Name):   # y = x: x's kind (an argument, or an earlier local)
                value_kind = f"arg:{value.id}" if value.id in arg_names else _merge(found.get(value.id, []))
            else:
                value_kind = _literal_type(value)
            found.setdefault(item.targets[0].id, []).append(value_kind)
        elif isinstance(item, ast.AugAssign) and isinstance(item.target, ast.Name):
            found.setdefault(item.target.id, []).append(None)   # x += ...: its kind stays that of x = ...
    local_kinds = {name: _merge(kinds) for name, kinds in found.items()}

    def kind(value):   # an argument returned as it is takes that argument's type ("arg:<name>")
        """The kind of a returned expression; an argument returned as it is gives ``"arg:<name>"``.

        :param value: the expression.
        :type value: ast.expr
        :return: ``"Boolean"``, ``"String"``, ``"Real"``, ``"array"``, ``"other"`` (not an FMI value) or None (not obvious), or ``"arg:<name>"``.
        :rtype: str or None
        """
        if isinstance(value, ast.Name) and value.id in arg_names:
            return f"arg:{value.id}"
        if isinstance(value, ast.Name) and value.id in local_kinds:
            return local_kinds[value.id]
        if attributes is not None and isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
                and value.value.id == self_name:
            return attributes.get(value.attr)
        return _literal_type(value)

    shapes, none = [], False
    for item in _own_nodes(node):
        if not isinstance(item, ast.Return):
            continue
        value = item.value
        if value is None or isinstance(value, ast.Constant) and value.value is None:
            none = True
            continue
        if isinstance(value, ast.Name) and value.id == self_name:
            continue
        if isinstance(value, ast.Dict) and _str_keys(value):
            leaves = _dict_leaves(value, kind)
            shapes.append(("keys", list(leaves), leaves))
        elif isinstance(value, ast.Name) and value.id in dict_vars:
            shapes.append(("keys", dict_vars[value.id], {}))
        elif isinstance(value, ast.Tuple):
            shapes.append(("tuple", [f"y{i}" for i in range(len(value.elts))], {}))
        elif isinstance(value, ast.Call) and _fields_of(_resolve(value.func, namespace)):
            names, types = _fields_of(_resolve(value.func, namespace))
            shapes.append(("fields", names, types))
        else:
            shapes.append(("single", ["y"], {"y": kind(value)}))
    if not shapes:
        return Returned()
    kinds = {kind for kind, _, _ in shapes}
    if kinds == {"keys"} or kinds == {"fields"}:
        names, types = [], {}
        for _, keys, key_types in shapes:   # union over several returns, in order
            names += [k for k in keys if k not in names]
            types.update({k: t for k, t in key_types.items() if t})
        return Returned(names, {n: f"return:{n}" for n in names}, types)
    if len(kinds) == 1 and kinds <= {"tuple", "single"} and len({len(n) for _, n, _ in shapes}) == 1:
        kind, names, types = shapes[0]
        sources = {n: f"return:{i}" for i, n in enumerate(names)} if kind == "tuple" else {"y": "return"}
        return Returned(names, sources, {k: t for k, t in types.items() if t}, may_be_none=none)
    return Returned(unknown="its return statements return different shapes")


def _dict_leaves(node, kind, prefix="", depth=0):
    """``{name: type}`` for a dict literal, with dict literals inside it named with dots.

    ``{"zone": {"T": t}}`` -> ``zone.T`` (up to three levels).

    :param node: the dict literal.
    :type node: ast.Dict
    :param kind: gives the kind of a value expression.
    :type kind: callable
    :param prefix: dotted prefix of the enclosing keys.
    :type prefix: str
    :param depth: nesting level.
    :type depth: int
    :rtype: dict
    """
    leaves = {}
    for key, value in zip(_str_keys(node), node.values):
        if isinstance(value, ast.Dict) and _str_keys(value) and depth < 3:
            leaves.update(_dict_leaves(value, kind, f"{prefix}{key}.", depth + 1))
        else:
            leaves[prefix + key] = kind(value)
    return leaves


def _str_keys(node):
    """The keys of a dict literal, if all are identifier strings.

    :param node: the dict literal.
    :type node: ast.Dict
    :return: the keys, or ``[]``.
    :rtype: list[str]
    """
    keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    return keys if len(keys) == len(node.keys) and all(k.isidentifier() for k in keys) else []


def _local_dicts(node):
    """Local dicts and their keys: ``v = {...}`` / ``v = dict()`` plus ``v["k"] = ...`` in the function.

    :param node: the function node.
    :type node: ast.FunctionDef
    :return: ``{variable: [keys]}`` (only those with keys).
    :rtype: dict
    """
    dicts = {}
    for item in _own_nodes(node):
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            value = item.value
            if isinstance(value, ast.Dict) and (_str_keys(value) or not value.keys):
                dicts[item.targets[0].id] = list(_str_keys(value))
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict" \
                    and not value.args:
                dicts[item.targets[0].id] = [k.arg for k in value.keywords if k.arg]
    for item in _own_nodes(node):
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) \
                        and target.value.id in dicts and isinstance(target.slice, ast.Constant) \
                        and isinstance(target.slice.value, str) and target.slice.value.isidentifier() \
                        and target.slice.value not in dicts[target.value.id]:
                    dicts[target.value.id].append(target.slice.value)
    return {k: v for k, v in dicts.items() if v}


def _resolve(node, namespace):
    """The object a Name / dotted Attribute refers to in a namespace, or None.

    Uses :func:`inspect.getattr_static`, so no property, ``__getattr__`` or import runs: reading
    the code must never execute it.

    :param node: the expression.
    :type node: ast.expr
    :param namespace: the globals.
    :type namespace: dict
    :return: the object, or None.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.insert(0, node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in namespace:
        return None
    obj = namespace[node.id]
    for part in parts:
        obj = inspect.getattr_static(obj, part, None)
        if obj is None:
            return None
    return obj


def _installed_packages():
    """Top-level import names of the installed distributions.

    :rtype: set[str]
    """
    return set(packages_distributions())


#: (folders holding distributions, their modification times) -> packages_distributions().
_DISTRIBUTIONS = {}


def packages_distributions():
    """:func:`importlib.metadata.packages_distributions`, cached.

    It reads the file list of every installed distribution (about 1 s with a few hundred
    packages). The key is the sys.path folders that hold distributions (``*.dist-info``,
    ``*.egg-info``) and their modification times: installing or removing a package changes its
    folder's time. Folders of model code (which init adds to sys.path) don't count.

    :return: top-level module name -> distribution names.
    :rtype: dict[str, list[str]]
    """
    import importlib.metadata
    import os
    import sys

    def holding(path):
        """The modification time of a sys.path folder that holds distributions, else None.

        :param path: a sys.path entry (``""`` is the current folder).
        :type path: str
        :rtype: int or None
        """
        try:
            with os.scandir(path or ".") as entries:
                if any(e.name.endswith((".dist-info", ".egg-info")) for e in entries):
                    return os.stat(path or ".").st_mtime_ns
        except OSError:
            pass
        return None
    key = tuple((path, t) for path in sys.path for t in [holding(path)] if t is not None)
    if key not in _DISTRIBUTIONS:
        _DISTRIBUTIONS.clear()
        _DISTRIBUTIONS[key] = importlib.metadata.packages_distributions()
    return _DISTRIBUTIONS[key]


def foreign_class(klass, cls):
    """Whether ``klass``, a base of the model class ``cls``, comes from another installed library.

    GEKKO for ``class Plant(GEKKO)``: what its code sets on self is the library's state, not the
    model's results. A base in the user's own files is not foreign, and for a model class from
    an installed library (a library built on another), no base is.

    :param klass: a class in ``cls.__mro__``.
    :type klass: type
    :param cls: the model class.
    :type cls: type
    :rtype: bool
    """
    package, own = klass.__module__.split(".")[0], cls.__module__.split(".")[0]
    installed = _installed_packages()
    return package != own and package in installed and own not in installed


def foreign_attributes(cls):
    """Attributes that the code of ``cls``'s foreign bases (see :func:`foreign_class`) sets on self.

    :param cls: the model class.
    :type cls: type
    :rtype: set[str]
    """
    names = set()
    for klass in cls.__mro__[1:]:
        if klass is object or not foreign_class(klass, cls):
            continue
        for name, member in vars(klass).items():
            if inspect.isfunction(member):
                names |= set(assigned_attributes(klass, name, depth=0, raw=True))
    return names


def assigned_attributes(cls, method_name, depth=MAX_DEPTH, raw=False):
    """The attributes ``method_name`` sets on self, with their kinds, following self.other() calls.

    Methods of a foreign base (see :func:`foreign_class`) aren't followed: what they set is the
    library's. Methods are found with :func:`inspect.getattr_static` so no descriptor runs.

    :param cls: the model class.
    :type cls: type
    :param method_name: the method to start from.
    :type method_name: str
    :param depth: how many levels of self.method() calls to follow.
    :type depth: int
    :param raw: return every assignment's kind (including ``("ref", attr)`` for copies of other
        attributes) instead of one merged kind per attribute.
    :type raw: bool
    :return: ``{attribute: kind}``; kinds are ``"Boolean"``, ``"String"``, ``"Real"``, ``"array"``, ``"other"`` (not an FMI value) or None (not obvious).
    :rtype: dict
    """
    kinds, seen = {}, set()

    def defined_in(name):
        """The class in ``cls.__mro__`` that defines ``name``.

        :param name: a method name.
        :type name: str
        :rtype: type or None
        """
        return next((klass for klass in cls.__mro__ if name in vars(klass)), None)

    def visit(name, level):
        """Collect the self-attribute assignments of one method, then visit the methods it calls on self.

        :param name: the method name.
        :type name: str
        :param level: the current call depth.
        :type level: int
        """
        if name in seen or level > depth:
            return
        seen.add(name)
        owner = defined_in(name)
        if owner is not None and owner is not cls and foreign_class(owner, cls):
            return
        method = inspect.getattr_static(cls, name, None)
        node = function_node(method) if method is not None else None
        if node is None or not node.args.args:
            return
        self_name = node.args.args[0].arg
        namespace = getattr(inspect.unwrap(getattr(method, "__func__", method)), "__globals__", {})
        for item in _own_nodes(node):
            targets = item.targets if isinstance(item, ast.Assign) else \
                [item.target] if isinstance(item, (ast.AugAssign, ast.AnnAssign)) else []
            for target in targets:
                for t in (target.elts if isinstance(target, ast.Tuple) else [target]):
                    base = t.value if isinstance(t, ast.Subscript) else t
                    if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) \
                            and base.value.id == self_name:
                        value = item.value if isinstance(item, ast.Assign) and t is target else None
                        if isinstance(t, ast.Subscript):
                            value = None   # self.x[i] = ...: x is a container, its kind comes from elsewhere
                        ref = _attribute_ref(value, self_name) if value is not None else None
                        if isinstance(item, ast.AugAssign) and t is base:   # self.x += dx: x has dx's kind
                            found = _literal_type(item.value, namespace)
                            found = "Real" if not isinstance(item.op, ast.Add) or found == "Real" else None
                        else:
                            found = ("ref", ref) if ref else _literal_type(value, namespace) if value is not None else None
                        kinds.setdefault(base.attr, []).append(found)
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) \
                    and isinstance(item.func.value, ast.Name) and item.func.value.id == self_name:
                visit(item.func.attr, level + 1)

    visit(method_name, 0)
    if raw:
        return kinds
    return {attr: _merge([k for k in found if not isinstance(k, tuple)]) for attr, found in kinds.items()}


#: Methods that return a copy of an attribute (self.x.copy()): the copy has the attribute's kind.
COPY_METHODS = {"copy", "clone", "detach", "astype", "reshape", "flatten", "ravel", "tolist", "cpu", "numpy"}


def _attribute_ref(node, self_name):
    """The attribute name if ``node`` is self.y, self.y.copy() (and similar) or np.copy(self.y).

    :param node: an assigned value.
    :type node: ast.expr
    :param self_name: the method's first parameter name.
    :type self_name: str
    :rtype: str or None
    """
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in COPY_METHODS:
            node = node.func.value
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "copy" and len(node.args) == 1:
            node = node.args[0]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == self_name:
        return node.attr
    return None


def class_attributes(cls, method_name):
    """The attributes set by the step method and by ``__init__``, with their kinds.

    Assignments copied from other attributes (``self.x_prior = self.x.copy()``) take that
    attribute's kind.

    :param cls: the model class.
    :type cls: type
    :param method_name: the step method.
    :type method_name: str
    :return: ``({attr: kind} of the step method, {attr: kind} of __init__)``.
    :rtype: tuple[dict, dict]
    """
    step, init = assigned_attributes(cls, method_name, raw=True), assigned_attributes(cls, "__init__", raw=True)

    def kind(attr, seen=()):
        """The merged kind of an attribute, following copies of other attributes.

        :param attr: the attribute.
        :type attr: str
        :param seen: attributes already followed (stops cycles).
        :type seen: tuple
        :rtype: str or None
        """
        found = []
        for k in step.get(attr, []) + init.get(attr, []):
            if isinstance(k, tuple):
                found.append(kind(k[1], seen + (attr,)) if k[1] not in seen and k[1] != attr else None)
            else:
                found.append(k)
        return _merge(found)

    return {a: kind(a) for a in step}, {a: kind(a) for a in init}


def _merge(found):
    """One kind from all assignments of a name: not-FMI and array win; None placeholders are ignored.

    :param found: the kinds.
    :type found: list
    :rtype: str or None
    """
    for kind in ("other", "array"):
        if kind in found:
            return kind
    known = [k for k in found if k not in (None, "none")]
    return known[0] if known else None


def init_copies(cls):
    """Constructor arguments the constructor stores unchanged as attributes (``self.x = x``).

    Those parameters can be tuned through the attribute of the same name.

    :param cls: the model class.
    :type cls: type
    :rtype: set[str]
    """
    node = function_node(inspect.getattr_static(cls, "__init__", None))
    if node is None or not node.args.args:
        return set()
    self_name = node.args.args[0].arg
    copies = set()
    for item in _own_nodes(node):
        if isinstance(item, ast.Assign) and isinstance(item.value, ast.Name):
            for target in item.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                        and target.value.id == self_name and target.attr == item.value.id:
                    copies.add(target.attr)
    return copies


def own_properties(cls):
    """Public properties defined in the model's own package, with their FMI types (not read).

    :param cls: the model class.
    :type cls: type
    :return: ``{name: FMI type or None}``; array-typed properties are left out.
    :rtype: dict
    """
    package = cls.__module__.split(".")[0]
    found = {}
    for klass in cls.__mro__:
        if klass.__module__.split(".")[0] != package:
            continue
        for name, member in vars(klass).items():
            if isinstance(member, property) and member.fget is not None and not name.startswith("_"):
                try:
                    hint = typing.get_type_hints(member.fget).get("return")
                except Exception:
                    hint = None
                if not _is_array_hint(hint):
                    found.setdefault(name, _hint_type(hint))
    return found


def factory_class(factory, depth=0):
    """The class a factory function returns, if the code tells.

    Its return annotation (``-> torchani.arch.ANI``), else what all its return statements call,
    followed through other functions up to three levels (tellurium's loada returns
    loadAntimonyModel(...), which returns roadrunner.RoadRunner(...)).

    :param factory: the factory function.
    :param depth: the current level.
    :type depth: int
    :return: the class, or None (builtins don't count).
    :rtype: type or None
    """
    try:
        hint = typing.get_type_hints(factory).get("return")
    except Exception:
        hint = None
    if isinstance(hint, type):
        return hint if hint.__module__ != "builtins" else None
    node = function_node(factory)
    if node is None or depth >= 3:
        return None
    namespace = getattr(inspect.unwrap(factory), "__globals__", {})
    found = set()
    for item in _own_nodes(node):
        if not isinstance(item, ast.Return) or item.value is None:
            continue
        if not isinstance(item.value, ast.Call):
            return None
        target = _resolve(item.value.func, namespace)
        if isinstance(target, type):
            found.add(target)
        elif inspect.isfunction(target):
            found.add(factory_class(target, depth + 1))
        else:
            return None
    if len(found) != 1 or None in found:
        return None
    cls = found.pop()
    return cls if cls.__module__ != "builtins" else None
