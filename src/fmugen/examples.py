"""Read how a model is used from its package's own examples, for ``fmugen init``.

Doctests in the docstrings, code blocks in the README (the package metadata's description, or a
README next to a model file) and scripts in an ``examples`` folder often show a complete use: the
object built (and by which classmethod), the calls that set it up, the step method and its
arguments. :func:`find_usage` reads them with :mod:`ast` (nothing in them is run) and returns what
``init`` would otherwise need ``--call``, ``--create``, ``--setup``, ``--start`` and ``--const`` for.
"""

import ast
import doctest
import importlib.metadata
import importlib.util
import inspect
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

#: Calls on the object that aren't a step: they show, save, copy or close it.
NOT_A_STEP = ("plot", "show", "print", "save", "summary", "describe", "copy", "close", "to_", "plot_", "save_",
              "print_", "get_params", "set_params", "__repr__", "__str__")
#: How many example files are read, at most, from an ``examples`` folder.
MAX_FILES = 40
#: Example files larger than this (bytes) are skipped.
MAX_SIZE = 200_000

#: A Markdown fence line: ```` ``` ```` or ```` ~~~ ````, with the block's language.
_FENCE = re.compile(r"^[ \t]*(```+|~~~+)[ \t]*([\w+-]*)")
#: The languages of a fenced block read as Python (none: often Python in a Python package's README).
PYTHON_FENCES = ("", "python", "py", "python3", "pycon", "ipython")
_RST = re.compile(r"^\.\. (?:code-block|code|sourcecode)::[ \t]*(?:python|py|python3|pycon)?[ \t]*\n"
                  r"(?:[ \t]+:\w[\w-]*:.*\n)*\n?((?:[ \t]+.*\n|[ \t]*\n)+)", re.M)


@dataclass
class Usage:
    """A use of the model found in an example.

    :ivar source: where it was found (``"the README of numpy"``, ``"examples/run.py"``).
    :ivar create: the classmethod that built the object, or None.
    :ivar call: the step method, or None (a function model: the function itself).
    :ivar setup: the calls on the object before the step, as ``[model] setup`` strings.
    :ivar init: the constructor's (or classmethod's, or function's) arguments that have no default in the
        code: name -> a start value (a literal), ``{"call": "module:expr"}`` / ``{"python": repr}`` for a
        value that isn't an FMU variable, or ``{"expr": text}`` for arithmetic on imports, computed by init.
    :ivar step: the step method's arguments, likewise.
    :ivar dropped: calls before the step left out of ``setup``: their arguments aren't literals or imports.
    :ivar sources: argument name -> where its value was found, when not in ``source`` (:func:`find_values`).
    """
    source: str
    create: str | None = None
    call: str | None = None
    setup: list = field(default_factory=list)
    init: dict = field(default_factory=dict)
    step: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)


def find_usage(entry, module, model_path=None):
    """The first use of ``entry`` in its package's examples that builds it (a class) or calls it (a function).

    :param entry: the model class or function.
    :param module: the module it was imported from.
    :param model_path: the model file, for a model given as a file (its README and ``examples`` folder).
    :type model_path: pathlib.Path or None
    :return: the best use found (one that calls a step first), or None.
    :rtype: Usage or None
    """
    found = []
    for source, code in _sources(entry, module, model_path):
        try:
            usage = _read(code, source, entry, module)
        except Exception:   # an odd snippet: none from it
            usage = None
        if usage is None:
            continue
        if usage.call and source == f"the docstring of {entry.__qualname__}.{usage.call}" and _a_variant(entry, usage.call):
            # a variant showing itself (timezonefinder's timezones_at, of timezone_at): not that it's the step;
            # pyproj's transform and radioactivedecay's decay show themselves, and they are
            usage.call, usage.step, usage.setup = None, {}, []
        if usage.call or not inspect.isclass(entry):
            return usage
        found.append(usage)
    return found[0] if found else None


def find_values(entry, module, model_path=None):
    """The values the package's examples pass to arguments of anything of the package they call (not only the
    model): ``{argument name: (value, source)}``, the first one seen for each name.

    For a model's argument no use of it shows: openap's README builds ``FuelFlow("A320")``, not ``Thrust``, and
    both take the aircraft as ``ac``. Only arguments with no default where they are passed (as :func:`_bound`).

    :param entry: the model class or function.
    :param module: the module it was imported from.
    :param model_path: the model file, for a model given as a file.
    :rtype: dict
    """
    values = {}
    for source, code in _sources(entry, module, model_path):
        try:
            _collect_values(code, source, entry, module, values)
        except Exception:   # an odd snippet: none from it
            continue
    return values


def _collect_values(code, source, entry, module, values):
    """Add the values of one snippet's calls to ``values`` (:func:`find_values`)."""
    tree = _parse(code)
    snippet = _Snippet(entry, module)
    for node, _ in _statements(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _add_import(snippet, node)
            continue
        for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
            dotted = _dotted(call.func)
            full = snippet.resolve(dotted) if dotted else None
            if not full and dotted and source.startswith("the docstring") \
                    and hasattr(module, dotted.partition(".")[0]):
                full = f"{module.__name__}.{dotted}"   # a doctest runs in its module: thermo's Chemical('Nitrogen')
            if not full or full.partition(".")[0] != snippet.top:   # the package's own: numpy's x isn't the model's
                continue
            fn = _imported(full)
            if fn is None or not callable(fn):
                continue
            for name, arg in (_bound(fn, call.args, call.keywords, skip_self=False) or {}).items():
                if name not in values:
                    value = snippet.value(arg)
                    if value is not None:
                        values[name] = (value, source)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            snippet.values[node.targets[0].id] = _Inline(snippet).visit(_copy(node.value))


def _imported(dotted):
    """The object a full dotted name names (a module's attribute, through classes), or None."""
    if not dotted:
        return None
    module = _module_prefix(dotted) if "." in dotted else None
    if module is None:
        return None
    try:
        obj = importlib.import_module(module)
        for part in dotted[len(module) + 1:].split("."):
            obj = getattr(obj, part)
    except Exception:
        return None
    return obj


def _add_import(snippet, node):
    """Bind the names an import statement brings in."""
    if isinstance(node, ast.Import):
        for a in node.names:
            snippet.aliases[a.asname or a.name.partition(".")[0]] = a.name if a.asname else a.name.partition(".")[0]
    elif node.module and not node.level:
        for a in node.names:
            if a.name != "*":
                snippet.aliases[a.asname or a.name] = f"{node.module}.{a.name}"


def _a_variant(cls, name):
    """Whether method ``name`` of ``cls`` is a variant of another of its public methods (:func:`_variant_of`), or
    the same words in the plural (``timezones_at`` of ``timezone_at``)."""
    from fmugen.interface import _variant_of

    def singular(method):
        return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in method.split("_")]

    others = [m for m in dir(cls) if not m.startswith("_") and m != name and callable(getattr(cls, m, None))]
    return any(_variant_of(name, m) or singular(name) == singular(m) for m in others)


# ---------------- where examples are ----------------

def _sources(entry, module, model_path):
    """``(source, code)`` of each example snippet: docstrings first, then READMEs, then example files.

    :rtype: iterator
    """
    objects = [entry]
    if inspect.isclass(entry):
        objects += [getattr(entry, "__init__", None)] + [
            getattr(entry, n, None) for n, v in vars(entry).items() if not n.startswith("_") and callable(v)]
    objects.append(module)
    for obj in objects:
        doc = inspect.getdoc(obj) if obj is not None else None
        if doc and ">>>" in doc:
            yield f"the docstring of {getattr(obj, '__qualname__', getattr(obj, '__name__', 'the module'))}", \
                _doctest_code(doc)
    for source, text in _readmes(module, model_path):
        for code in _code_blocks(text):
            yield source, code
    for path in _example_files(module, model_path):
        try:
            yield path.as_posix(), path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue


def _readmes(module, model_path):
    """``(source, text)`` of the README of the model's distribution (its metadata), or next to the model file."""
    if model_path is not None:
        for folder in (model_path.parent, model_path.parent.parent):
            for name in ("README.md", "README.rst", "README.txt", "README"):
                path = folder / name
                if path.is_file() and path.stat().st_size < MAX_SIZE:
                    yield path.name, path.read_text(encoding="utf-8", errors="replace")
        return
    top = (getattr(module, "__name__", "") or "").partition(".")[0]
    try:
        dists = importlib.metadata.packages_distributions().get(top, [])
    except Exception:
        dists = []
    for dist in dict.fromkeys(dists):
        try:
            meta = importlib.metadata.metadata(dist)
        except Exception:
            continue
        text = meta.get_payload() or meta.get("Description") or ""
        if text:
            yield f"the README of {dist}", text


def _example_files(module, model_path):
    """``.py`` files in an ``examples`` (or ``example``) folder of the package or next to the model file."""
    roots = []
    if model_path is not None:
        roots += [model_path.parent, model_path.parent.parent]
    else:
        file = getattr(module, "__file__", None)
        if file:
            package = Path(file).parent
            top = (module.__name__ or "").partition(".")[0]
            while package.name != top and package.parent != package and (package.parent / "__init__.py").exists():
                package = package.parent
            roots.append(package)
    files = []
    for root in roots:
        for name in ("examples", "example"):
            folder = root / name
            if folder.is_dir():
                files += sorted(p for p in folder.rglob("*.py") if p.stat().st_size < MAX_SIZE)
    return files[:MAX_FILES]


def _doctest_code(text):
    """The code of the ``>>>`` examples in ``text``, one statement per example."""
    try:
        return "\n".join(e.source.rstrip() for e in doctest.DocTestParser().get_examples(text))
    except ValueError:
        return ""


def _fenced_blocks(text):
    """The Python blocks of a Markdown document, its fences paired in order: a ```` ```bash ```` block's closing
    fence doesn't open one (nashpy's README)."""
    blocks, lines, fence, language = [], [], None, ""
    for line in text.splitlines():
        match = _FENCE.match(line)
        if fence is None:
            if match:
                fence, language, lines = match.group(1), match.group(2).lower(), []
        elif match and match.group(1).startswith(fence[:3]) and not match.group(2):
            if language in PYTHON_FENCES:
                blocks.append(textwrap.dedent("\n".join(lines)))
            fence = None
        else:
            lines.append(line)
    return blocks


def _code_blocks(text):
    """The Python code blocks of a Markdown or reStructuredText document (and its ``>>>`` examples)."""
    blocks = _fenced_blocks(text) + [textwrap.dedent(m.group(1)) for m in _RST.finditer(text)]
    for block in blocks:
        yield _doctest_code(block) if ">>>" in block else block
    if ">>>" in text and not blocks:
        yield _doctest_code(text)


# ---------------- reading a snippet ----------------

def _parse(code):
    """The module of ``code``, dropping shell and notebook lines (``$ pip ...``, ``!pip``, ``%matplotlib``)."""
    lines = [line for line in code.splitlines() if not line.lstrip().startswith(("$", "!", "%"))]
    return ast.parse(textwrap.dedent("\n".join(lines)))


def _statements(body, loop=0):
    """The statements of ``body`` in order, with how many loops each is in, through if/with/try/for blocks and
    the ``main()`` style functions an example script defines."""
    for node in body:
        if isinstance(node, (ast.For, ast.AsyncFor)):
            # what it iterates is used, in a loop: nashpy's "for eq in game.support_enumeration():"
            header = ast.Assign(targets=[ast.Name("_")], value=node.iter, lineno=node.lineno, col_offset=0)
            yield header, loop + 1
            yield from _statements(node.body, loop + 1)
        elif isinstance(node, ast.While):
            yield from _statements(node.body, loop + 1)
        elif isinstance(node, (ast.If, ast.With, ast.AsyncWith)):
            yield from _statements(node.body, loop)
            yield from _statements(getattr(node, "orelse", []), loop)
        elif isinstance(node, ast.Try):
            yield from _statements(node.body, loop)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.args.args:
            yield from _statements(node.body, loop)
        else:
            yield node, loop


def _dotted(node):
    """``a.b.c`` of a Name/Attribute chain, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


class _Snippet:
    """What a snippet's statements bind, read in order: imports, literal and computed values, the model object."""

    def __init__(self, entry, module):
        self.entry, self.module = entry, module
        self.top = (getattr(module, "__name__", "") or "").partition(".")[0]
        self.aliases = {}   # name -> full dotted name, from imports
        self.values = {}    # name -> expression (names in it already resolved)

    def resolve(self, dotted):
        """The full dotted name of ``dotted`` through the imports, or None when its root isn't imported."""
        root, _, rest = dotted.partition(".")
        if root not in self.aliases:
            return None
        return self.aliases[root] + (f".{rest}" if rest else "")

    def is_entry(self, func):
        """Whether ``func`` (a call's callee) names the model: ``(True, None)``, or ``(True, classmethod)``."""
        dotted = _dotted(func)
        if dotted is None:
            return False, None
        name = self.entry.__name__
        full = self.resolve(dotted) or dotted
        parts = full.split(".")
        ours = parts[0] in (self.top, name) or dotted.partition(".")[0] not in self.aliases
        if parts[-1] == name and ours:
            return True, None
        if inspect.isclass(self.entry) and len(parts) >= 2 and parts[-2] == name and ours \
                and isinstance(inspect.getattr_static(self.entry, parts[-1], None), (classmethod, staticmethod)):
            return True, parts[-1]
        return False, None

    def value(self, node):
        """``node`` as a start value (a literal), ``{"python": repr}``, ``{"call": "module:expr"}`` or
        ``{"expr": text}`` (arithmetic on imports), or None when it uses names the snippet doesn't import or
        bind to such a value."""
        node = _Inline(self).visit(_copy(node))
        if node is None:
            return None
        try:
            literal = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            literal = _FAIL
        if literal is not _FAIL:
            return _start(literal, node)
        if any(isinstance(n, (ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Await,
                              ast.NamedExpr, ast.Starred)) for n in ast.walk(node)):
            return None
        names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
        if not names or any(not _importable(n) for n in set(names)):
            return None
        if not isinstance(node, ast.Call):   # arithmetic on imports (pysindy's 3 * numpy.exp(...)): computed by init
            return {"expr": ast.unparse(node)} if all(_module_prefix(d) or _importable(d) for d in _roots(node)) \
                else None
        callee = node.func
        while isinstance(callee, (ast.Call, ast.Subscript)):
            callee = callee.func if isinstance(callee, ast.Call) else callee.value
        dotted = _dotted(callee)
        if dotted is None:
            return None
        module = _module_prefix(dotted)
        if module is None:
            return None
        text = ast.unparse(node)
        if not text.startswith(module + "."):
            return {"expr": text} if all(_module_prefix(d) or _importable(d) for d in _roots(node)) else None
        return {"call": f"{module}:{text[len(module) + 1:]}"}


_FAIL = object()


def _roots(node):
    """The dotted names an expression uses (``numpy.exp``), at their longest."""
    names = []
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and (dotted := _dotted(n)):
            names.append(dotted)
        elif isinstance(n, ast.Name):
            names.append(n.id)
    return [d for d in names if not any(o != d and o.startswith(d + ".") for o in names)]


def _copy(node):
    """A deep copy of an AST node."""
    return ast.parse(ast.unparse(node), mode="eval").body


class _Inline(ast.NodeTransformer):
    """Replace the snippet's names by what they stand for: an import's full dotted name, a bound value."""

    def __init__(self, snippet):
        self.snippet = snippet

    def visit_Name(self, node):
        if node.id in self.snippet.values:
            return _copy(self.snippet.values[node.id])
        if node.id in self.snippet.aliases:
            return ast.parse(self.snippet.aliases[node.id], mode="eval").body
        if node.id in ("True", "False", "None"):
            return node
        return node   # unknown: left as it is, and value() rejects it unless it's an importable module


def _importable(name):
    """Whether ``name`` is a top-level module that can be imported (``numpy``, ``torch``)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _module_prefix(dotted):
    """The longest prefix of ``dotted`` that is an importable module (``numpy.random`` of ``numpy.random.rand``)."""
    parts = dotted.split(".")
    for n in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:n])
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError, AttributeError):
            continue
    return None


def _start(literal, node):
    """A literal as a start value: a number, bool, string or a (nested) list of numbers; else ``{"python": repr}``
    for a dict, list or tuple of other things, or None."""
    if isinstance(literal, (bool, int, float, str)):
        return literal

    def numbers(v):
        return isinstance(v, (bool, int, float)) or (isinstance(v, (list, tuple)) and v != () and v != []
                                                       and all(numbers(x) for x in v))

    if isinstance(literal, (list, tuple)) and numbers(literal):
        return _as_lists(literal)
    if isinstance(literal, (dict, list, tuple)):
        return {"python": repr(literal)}
    return None


def _as_lists(value):
    """Tuples as lists, nested."""
    return [_as_lists(v) for v in value] if isinstance(value, (list, tuple)) else value


def _bound(fn, args, kwargs, skip_self):
    """``{argument name: node}`` of a call's arguments, positional ones named by ``fn``'s signature."""
    try:
        params = [p for p in inspect.signature(fn).parameters.values()][1 if skip_self else 0:]
    except (TypeError, ValueError):
        return None
    named = {}
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(args) > len(positional) or any(isinstance(a, ast.Starred) for a in args):
        return None
    for p, node in zip(positional, args):
        if p.default is p.empty:   # the code's own default is kept: the example only fills what it lacks
            named[p.name] = node
    declared = {p.name for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    for k in kwargs:
        if k.arg is None:   # **options
            return None
        if k.arg in declared and next(p for p in params if p.name == k.arg).default is inspect.Parameter.empty:
            named[k.arg] = k.value   # not **kwargs: options the model has defaults for
    return named


def _method_calls(node, obj):
    """The calls ``obj.method(...)`` in ``node``."""
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == obj]


def _read(code, source, entry, module):
    """The use of ``entry`` in one snippet, or None.

    :rtype: Usage or None
    """
    tree = _parse(code)
    snippet = _Snippet(entry, module)
    is_class = inspect.isclass(entry)
    obj, built, calls = None, None, []   # calls: (name, call node, loops, result used)
    for node, loops in _statements(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _add_import(snippet, node)
            continue
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) else None
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        target = targets[0].id if len(targets) == 1 and isinstance(targets[0], ast.Name) else None
        if isinstance(value, ast.Call):
            ours, create = snippet.is_entry(value.func)
            if ours and is_class and obj is None and target:
                obj, built = target, (create, value)
                continue
            if ours and not is_class and built is None:
                built = (None, value)
                calls.append((None, value, loops, target is not None))
                continue
        if obj is not None:
            for call in _method_calls(node, obj):
                used = not (isinstance(node, ast.Expr) and node.value is call)
                calls.append((call.func.attr, call, loops, used))
            if target == obj:   # rebound: later calls are on something else
                break
        if target and value is not None and target != obj:
            snippet.values[target] = _Inline(snippet).visit(_copy(value))
    if built is None:
        return None
    usage = Usage(source)
    create, ctor_call = built
    if is_class:
        usage.create = create
        ctor = getattr(entry, create) if create else entry.__init__
        _take(usage.init, snippet, _bound(ctor, ctor_call.args, ctor_call.keywords, skip_self=not create))
        steps = [c for c in calls if c[0] and not c[0].startswith("_") and not _named(c[0], NOT_A_STEP)
                 and callable(getattr(entry, c[0], None))]
        if not steps:
            return usage
        from fmugen.interface import CALL_NAMES, STEP_LIKE

        # a run-like name, then a call in a loop, then one whose result is used, then the first: a README showing
        # several queries in turn leads with the main one (blackscholes' price(), then delta(), ... charm())
        # (among run-like names, by their order there: simpy's run() before its process())
        runlike = CALL_NAMES + STEP_LIKE

        def rank(c):
            return (c[0] not in runlike, runlike.index(c[0]) if c[0] in runlike else 0, not c[2], not c[3],
                    calls.index(c))

        step = min(steps, key=rank)
        usage.call = step[0]
        for name, call, _, _ in calls[:calls.index(step)]:
            if name == step[0] or name is None or _named(name, NOT_A_STEP) or name.startswith("_"):
                continue
            text = _setup_text(snippet, name, call)
            if text is None:
                usage.dropped.append(name)
            elif text not in usage.setup:
                usage.setup.append(text)
        method = getattr(entry, step[0])
        static = isinstance(inspect.getattr_static(entry, step[0], None), (staticmethod, classmethod))
        _take(usage.step, snippet, _bound(method, step[1].args, step[1].keywords, skip_self=not static))
    else:
        _take(usage.init, snippet, _bound(entry, ctor_call.args, ctor_call.keywords, skip_self=False))
    return usage


def _named(name, names):
    """Whether ``name`` is one of ``names`` (or starts with one ending in ``_``)."""
    return any(name == n or (n.endswith("_") and name.startswith(n)) for n in names)


def _take(args, snippet, bound):
    """Add the arguments of a call that have a start or constant value to ``args``."""
    for name, node in (bound or {}).items():
        value = snippet.value(node)
        if value is not None:
            args.setdefault(name, value)


def _setup_text(snippet, name, call):
    """``name(args)`` for ``[model] setup``, its arguments resolved, or None when one can't be."""
    parts = []
    for a in call.args:
        text = _argument_text(snippet, a)
        if text is None:
            return None
        parts.append(text)
    for k in call.keywords:
        if k.arg is None:
            return None
        text = _argument_text(snippet, k.value)
        if text is None:
            return None
        parts.append(f"{k.arg}={text}")
    return f"{name}({', '.join(parts)})"


def _argument_text(snippet, node):
    """A setup argument as source text: a literal, or a call of an importable object with full dotted names."""
    value = snippet.value(node)
    if value is None:
        return None
    if isinstance(value, dict):
        if "python" in value:
            return value["python"]
        if "expr" in value:   # arithmetic, which a setup call's arguments can't hold
            return None
        module, _, rest = value["call"].partition(":")
        return f"{module}.{rest}"
    return repr(value)
