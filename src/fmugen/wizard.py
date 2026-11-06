"""The questions ``fmugen init --wizard`` asks.

They fill in the same arguments as the command-line options, so an option given is never asked, and
what the wizard answers is exactly what those options would do. Prompts go to stderr, like the warnings,
so ``-o -`` can still be redirected. questionary is imported only when a question is asked: the FMU's
runtime never gets here.
"""
import ast
import contextlib
import io
import re
import shlex
import sys
from pathlib import Path

from fmugen.config import InterfaceError

#: Set while the wizard reads the model: the inference asks nothing then (see :func:`interactive`).
_assume_yes = False

#: Python files in the folder that are not models, left out of the first question.
_NOT_MODELS = {"setup.py", "conftest.py", "noxfile.py"}
_OTHER = "Something else (a path, or package.module:Name)"
_OWN_ENTRY = "Write my own"
_OWN_METHOD = "Write my own"
_USE_RETURNED = "None: read what the function returns on every step"
_BUILD_EACH_STEP = "None: build the object on every step and read its results"
_DONE = "Nothing else: write the config"
_CALL_VALUE = re.compile(r"call:(self\.|[\w.]+:)\w.*")


def assume_yes(value=True):
    """Never ask: the inference takes its own pick.

    :param value: whether to assume yes.
    :type value: bool
    """
    global _assume_yes
    _assume_yes = value


def interactive():
    """Whether questions can be asked: a terminal on stdin and stderr.

    :rtype: bool
    """
    return not _assume_yes and bool(sys.stdin and sys.stdin.isatty() and sys.stderr.isatty())


def _output():
    """Where prompts are drawn: stderr (tests replace this).

    :rtype: prompt_toolkit.output.Output
    """
    from prompt_toolkit.output import create_output
    return create_output(stdout=sys.stderr)


def _ask(kind, message, **options):
    """Ask one question.

    :param kind: the questionary prompt: ``select``, ``confirm``, ``text`` or ``path``.
    :type kind: str
    :param message: the question.
    :type message: str
    :param options: the prompt's options.
    :return: the answer.
    :raises KeyboardInterrupt: Ctrl-C.
    """
    import questionary
    return getattr(questionary, kind)(message, output=_output(), **options).unsafe_ask()


def select_method(cls_name, ranked):
    """Ask which method runs a step.

    :param cls_name: the class's name.
    :type cls_name: str
    :param ranked: the candidates, best first.
    :type ranked: list[str]
    :rtype: str
    :raises KeyboardInterrupt: Ctrl-C.
    """
    return _ask("select", f"Which method of {cls_name} runs a step?", choices=list(ranked))


def _note(text):
    """Say something between the questions, on stderr with them.

    :param text: the text.
    :type text: str
    """
    print(text, file=sys.stderr, flush=True)


def _select(message, choices, **options):
    """Ask to pick one, filtering by typing when the list is long.

    :param message: the question.
    :type message: str
    :param choices: the choices.
    :type choices: list
    :param options: the prompt's options.
    :return: the answer.
    """
    if len(choices) > 15:
        message += " (type to filter)"
        options.update(use_search_filter=True, use_jk_keys=False)
    return _ask("select", message, choices=choices, **options)


def _unquoted(text):
    """An answer without the spaces and the quotes a terminal adds to a path dragged into it.

    :param text: the answer.
    :type text: str
    :rtype: str
    """
    return text.strip().strip("\"'").strip()


def _split_target(target):
    """Split a model target, also a module without a name, for which the wizard lists the names.

    :param target: the target.
    :type target: str
    :return: ``(path or None, module name or None, Name or None)``.
    :rtype: tuple
    :raises InterfaceError: neither a file nor a module.
    """
    from fmugen.interface import parse_target
    try:
        return parse_target(target)
    except InterfaceError:
        if all(part.isidentifier() for part in target.split(".")):
            return None, target, None
        raise


def _module_found(name):
    """Whether a module's top package can be imported, not importing it.

    :param name: the module.
    :type name: str
    :rtype: bool
    """
    import importlib.util
    top = name.partition(".")[0]
    cwd = Path.cwd()
    try:
        return (cwd / top).is_dir() or (cwd / f"{top}.py").is_file() or importlib.util.find_spec(top) is not None
    except (ImportError, ValueError):
        return False


def _check_target(text):
    """Validate a model target for a prompt.

    :param text: the answer.
    :type text: str
    :return: True, or what's wrong with it.
    :rtype: bool or str
    """
    try:
        path, _module, _name = _split_target(_unquoted(text))
    except InterfaceError:
        return "Give a .py file, or package.module[:Name] for an installed module"
    if path and not path.is_file():
        return f"No file {path}"
    if not path and not _module_found(_module):
        return f"No module {_module} here or in this environment"
    return True


def _ask_model():
    """Ask for the model: the Python files in this folder, or anything else.

    :rtype: str
    """
    files = sorted(p.name for p in Path.cwd().glob("*.py") if not p.name.startswith("_") and p.name not in _NOT_MODELS)
    if files:
        answer = _select("Which model?", [*files, _OTHER])
        if answer != _OTHER:
            return answer
    return _unquoted(_ask("path", "Model (model.py[:Name] or package.module[:Name]):", validate=_check_target))


def entries_in_file(path):
    """The public functions and classes a model file defines, read without importing it.

    Importing runs the file's top-level code, and ``init`` does that once it knows what to do. Enums and
    exceptions don't count, by the name of their bases, as for a target without a name.

    :param path: the file.
    :type path: pathlib.Path
    :return: the names; empty if the file can't be parsed (``init`` reports that).
    :rtype: list[str]
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return []
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [getattr(b, "attr", getattr(b, "id", "")) for b in node.bases]
            if any(b.endswith(("Enum", "Flag", "Error", "Exception", "Warning")) for b in bases):
                continue
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_"):
            names.append(node.name)
    return names


def _entries_in_module(name):
    """The public functions and classes of an installed module.

    :param name: the module.
    :type name: str
    :rtype: list[str]
    :raises InterfaceError: it can't be imported, or defines none.
    """
    import importlib

    from fmugen.builder import isolated_imports
    from fmugen.interface import public_entries
    if str(Path.cwd()) not in sys.path:
        sys.path.append(str(Path.cwd()))   # a local package, as infer_config finds it
    try:
        with isolated_imports(), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            names = [e.__name__ for e in public_entries(importlib.import_module(name))]
    except Exception as e:
        raise InterfaceError(f"cannot import {name}: {e}") from e
    if not names:
        raise InterfaceError(f"{name} defines no public function or class; give package.module:Name")
    return names


def _ask_entry(model):
    """Add the entry to a target that doesn't name one: asked for a file or module that has several.

    :param model: the target.
    :type model: str
    :rtype: str
    :raises InterfaceError: a module that can't be read.
    """
    path, module, name = _split_target(model)
    if name:
        return model
    names = entries_in_file(path) if path else _entries_in_module(module)
    if path and len(names) < 2:
        return model
    if len(names) == 1:
        return f"{model}:{names[0]}"
    where = path.name if path else module
    answer = _select(f"{where} defines several: which one is the model?", [*names, _OWN_ENTRY])
    if answer == _OWN_ENTRY:   # imported names aren't listed
        answer = _ask("text", "Function or class name:", validate=_check_name).strip()
    return f"{model}:{answer}"


def _check_name(text):
    """Validate a function, class or method name for a prompt.

    :param text: the answer.
    :type text: str
    :return: True, or what's wrong with it.
    :rtype: bool or str
    """
    return text.strip().isidentifier() or "Give a name, without arguments"


def _output_problem(output, model):
    """Why the config can't be written there.

    :param output: the config's path.
    :type output: str
    :param model: the model target.
    :type model: str
    :return: what's wrong, or None.
    :rtype: str or None
    """
    path = _split_target(model)[0]
    if output == "-" or not path or path.resolve().is_relative_to(Path(output).parent.resolve()):
        return None
    return f"The config must be in the folder of {path.name} or one above it, since it names the model by a relative path"


def _ask_output(args):
    """Settle where the config goes, and whether to replace one that exists: asked unless ``-o`` works as given.

    :param args: the options; ``output`` and ``force`` are set.
    :type args: argparse.Namespace
    """
    path = _split_target(args.model)[0]
    default = (path.parent if path else Path()) / "fmugen.toml"

    def in_folder(text):
        return str(Path(text) / "fmugen.toml") if text != "-" and Path(text).is_dir() else text

    def check(text):
        return _output_problem(in_folder(_unquoted(text) or str(default)), args.model) or True

    output = args.output
    while True:
        if output is None:
            output = _unquoted(_ask("path", "Write the config to (- prints it):", default=str(default),
                                    validate=check)) or str(default)
        output = in_folder(output)
        problem = _output_problem(output, args.model)
        if problem:
            _note(problem)
        elif output == "-" or args.force or not Path(output).exists():
            break
        elif _ask("confirm", f"{output} exists. Overwrite it?", default=False):
            args.force = True
            break
        output = None
    args.output = output


def guessed_starts(data, comments):
    """The variables whose start value ``init`` chose: made up, or from a docstring, the code's checks or an example.

    :param data: the inferred config.
    :type data: dict
    :param comments: its comments.
    :type comments: dict
    :return: ``{name: (start, where it came from)}``.
    :rtype: dict
    """
    found = {}
    for section in ("structural_parameters", "parameters", "inputs", "states"):
        for name, info in data.get(section, {}).items():
            why = comments.get((section, name))
            # one with "to" is bound under another name (an item of an argument): --start can't name it
            if isinstance(why, str) and "start" in info and "to" not in info and any(
                    mark in why for mark in ("no default", "start from the example", "made up")):
                found[name] = (info["start"], " ".join(why.splitlines()))
    return found


def _check_value(text, start=None):
    """Validate a value for a prompt: of the kind of the value it replaces, or computed.

    :param text: the answer.
    :type text: str
    :param start: the value it replaces, if any: a typo in a number mustn't become a text, typing it String.
    :return: True, or what's wrong with it.
    :rtype: bool or str
    """
    from fmugen.__main__ import _literal
    text = text.strip()
    if not text:
        return "Give a value"
    if text.startswith("call:"):
        return bool(_CALL_VALUE.fullmatch(text)) or "Give call:module:function(...)"
    if isinstance(start, (int, float, list, tuple)) and isinstance(_literal(text), str):
        return f"That isn't a {'list' if isinstance(start, (list, tuple)) else 'number'}; a text goes in quotes"
    return True


def _draft(args, parsed_values):
    """Read the model as ``init`` will, also calling it if the probe was chosen, asking nothing and printing no warning.

    :param args: the options so far.
    :type args: argparse.Namespace
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :return: ``(config, comments, a class's candidate step methods or None)``.
    :rtype: tuple[dict, dict, list[str] or None]
    """
    from fmugen.builder import isolated_imports
    from fmugen.interface import (
        create_candidates,
        infer_config,
        parse_target,
        step_candidates,
    )
    path = parse_target(args.model)[0]
    config_dir = Path(args.output).parent if args.output not in (None, "-") else path.parent if path else None
    converts, reads = _parsed(args)
    assume_yes(True)
    try:
        with isolated_imports(), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            data, comments = infer_config(args.model, call=args.call, config_dir=config_dir, fmi_version=args.fmi,
                                          starts=parsed_values(), setup=args.setup, kind=args.kind, create=args.create,
                                          probe=args.probe, converts=converts, reads=reads)
            methods, comments[("init", "factory")] = (None, False) if args.kind else step_candidates(
                args.model, parsed_values())
            comments[("init", "builders")] = [] if args.kind else create_candidates(args.model)
            return data, comments, methods
    finally:
        assume_yes(False)


def _read(args, parsed_values):
    """Read the model with the options so far, saying so, as it can take a while.

    :param args: the options so far.
    :type args: argparse.Namespace
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :return: what :func:`_draft` gives, or None when that failed, which is said.
    :rtype: tuple or None
    """
    _note("Calling the model..." if args.probe else "Reading the model...")
    try:
        return _draft(args, parsed_values)
    except InterfaceError as e:
        _note(f"That doesn't work: {e}")
    except (Exception, SystemExit) as e:
        _note(f"That doesn't work: {type(e).__name__}: {e}")
    return None


def _ask_method(picked, candidates, cls_name, none_label=None, can_build=False):
    """Ask which method runs a step: the one ``init`` picked first, then the others, or one typed.

    :param picked: the method ``init`` picked, or None when it builds the object on every step instead.
    :type picked: str or None
    :param candidates: the other candidates.
    :type candidates: list[str]
    :param cls_name: the class's name.
    :type cls_name: str
    :param none_label: the choice of no method, shown first when ``picked`` is None (default: building the object).
    :type none_label: str or None
    :param can_build: also offer no method when ``picked`` is one: ``--kind function``.
    :type can_build: bool
    :return: the method, or None for no method.
    :rtype: str or None
    """
    none_label = none_label or _BUILD_EACH_STEP
    choices = [picked or none_label, *(m for m in candidates if m != picked), _OWN_METHOD]
    if picked and can_build:
        choices.append(none_label)
    answer = _select(f"Which method of {cls_name} runs a step?", choices)
    if answer == none_label:
        return None
    return _ask("text", "Method name:", validate=_check_name).strip() if answer == _OWN_METHOD else answer


def _choose_create(args, state, parsed_values):
    """Ask which classmethod builds the object, for a class that has some.

    :param args: the options; ``create`` is set.
    :type args: argparse.Namespace
    :param state: what :func:`_draft` gave.
    :type state: tuple
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :return: what :func:`_draft` gives for the answer.
    :rtype: tuple
    """
    while not args.create and not args.kind:
        data, comments, _ = state
        builders = comments.get(("init", "builders"))
        if not builders:
            break
        picked = data["model"].get("create")
        name = data["model"]["entry"].rpartition(":")[2]
        constructor = f"The constructor: {name}(...)"
        # no option says the constructor: it's offered only when it's what init does anyway
        first = [picked] if picked else [constructor]
        answer = _select(f"How is {name} built?", [*first, *(b for b in builders if b != picked)])
        if answer == constructor:
            break
        args.create = answer
        if answer == picked:
            break
        new = _read(args, parsed_values)
        if new:
            return new
        args.create = None
    return state


def _choose_method(args, state, parsed_values):
    """Ask which method runs a step, for a class or a factory with more than one way to choose.

    :param args: the options; ``call`` and ``kind`` are set.
    :type args: argparse.Namespace
    :param state: what :func:`_draft` gave.
    :type state: tuple
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :return: what :func:`_draft` gives for the answer.
    :rtype: tuple
    """
    while not args.call and not args.kind:
        data, comments, candidates = state
        if candidates is None:
            break
        # the probe can settle on building the object every step: it's the first choice then, and the methods follow
        factory = comments.get(("init", "factory"))
        default = None if factory or data["model"].get("kind") == "function" else "__call__"
        picked = data["model"].get("call", default)
        if not [m for m in candidates if m != picked]:   # nothing to choose from
            break
        name = data["model"]["entry"].rpartition(":")[2]
        call = _ask_method(picked, candidates, name, _USE_RETURNED if factory else None, can_build=not factory)
        args.call = call
        args.kind = "function" if call is None and picked else None
        if call == picked:
            break
        new = _read(args, parsed_values)   # another method has other arguments
        if new:
            return new
        args.call = args.kind = None
    return state


def data_arguments(comments):
    """The arguments ``init`` found to look like data or files, and how each could be given.

    :param comments: the comments of a draft.
    :type comments: dict
    :return: ``{name: example}``, the example as in the warning's ``--const`` or ``--start``, without its option.
    :rtype: dict[str, str]
    """
    found = {}
    for line in comments.get(("init", "data_arguments"), []):
        name = line.partition(":")[0]
        found[name] = shlex.split(line.rpartition("e.g. ")[2])[-1].partition("=")[2]
    return found


def _ask_data(args, comments, asked):
    """Ask for the new arguments that look like files or data. Enter leaves one out, and ``init`` warns about it.

    :param args: the options; ``start``, ``const`` and ``value_order`` are added to.
    :type args: argparse.Namespace
    :param comments: the comments of a draft.
    :type comments: dict
    :param asked: the names already asked; added to.
    :type asked: set[str]
    :return: whether a value was given.
    :rtype: bool
    """
    given = False
    for name, example in data_arguments(comments).items():
        if name in asked:
            continue
        asked.add(name)
        answer = _ask("text", f"{name} looks like a file or data the model runs on. Its value, e.g. {example} "
                              "(Enter to skip):",
                      validate=lambda text: not text.strip() or _check_value(text)).strip()
        if answer:
            # a computed value, or a file read into one, is fixed; a literal or a path can be a variable
            option = "const" if answer.startswith("call:") else "start"
            _add_value(args, option, f"{name}={answer}")
            given = True
    return given


def _probe_failure(comments):
    """Why the probe failed, so the questions after it can be answered with that in mind.

    :param comments: the comments of a draft.
    :type comments: dict
    :return: the reason, or None.
    :rtype: str or None
    """
    note = str(comments.get(("outputs", None), ""))
    return note if note.startswith(("probe call failed", "probe construction failed")) else None


def _add_value(args, option, item):
    """Add a ``--start`` or ``--const`` answer.

    :param args: the options.
    :type args: argparse.Namespace
    :param option: ``start`` or ``const``.
    :type option: str
    :param item: ``NAME=VALUE``.
    :type item: str
    """
    getattr(args, option).append(item)
    args.value_order.append(item.partition("=")[0].strip())


def _ask_starts(args, data, comments, asked):
    """Ask for the start values ``init`` chose, pre-filled; Enter keeps one.

    :param args: the options; ``start``, ``const`` and ``value_order`` are added to.
    :type args: argparse.Namespace
    :param data: the config of a draft.
    :type data: dict
    :param comments: its comments.
    :type comments: dict
    :param asked: the names already asked; added to.
    :type asked: set[str]
    :return: whether a value was changed.
    :rtype: bool
    """
    from fmugen.__main__ import _literal
    changed = False
    for name, (start, why) in guessed_starts(data, comments).items():
        if name in asked:
            continue
        asked.add(name)
        answer = _ask("text", f"Start value of {name} ({why}):", default=repr(start),
                      validate=lambda text, start=start: _check_value(text, start)).strip()
        if answer.startswith("call:"):   # a computed value is fixed, not a variable
            _add_value(args, "const", f"{name}={answer}")
            changed = True
            continue
        value = _literal(answer)
        if repr(value) == repr(start):
            continue
        if type(start) is float and type(value) is int:   # 5 for 1.0 mustn't quietly make a Real variable an Integer
            kind = _ask("select", f"{name}: is {value} a real number or a count?",
                        choices=[f"Real ({float(value)})", f"Integer ({value})"])
            answer = repr(float(value)) if kind.startswith("Real") else answer
        _add_value(args, "start", f"{name}={answer}")
        changed = True
    return changed


def _ask_values(args, state, parsed_values, asked):
    """Ask for data and start values until a new reading of the model asks for nothing new.

    With the probe, a value given can get it further, and its next failure can name another argument.

    :param args: the options.
    :type args: argparse.Namespace
    :param state: what :func:`_draft` gave.
    :type state: tuple
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :param asked: the names already asked; added to.
    :type asked: set[str]
    :return: what :func:`_draft` gives for the answers.
    :rtype: tuple
    """
    shown = None
    while True:
        data, comments, _ = state
        failure = _probe_failure(comments)
        if failure and failure != shown:
            _note(failure)
        shown = failure
        before = list(args.start), list(args.const), list(args.value_order)
        given = _ask_data(args, comments, asked)
        if not _ask_starts(args, data, comments, asked) and not given:
            return state
        new = _read(args, parsed_values)
        if not new:   # init would stop on these too
            args.start, args.const, args.value_order = before
            _note("Those values were left out.")
            return state
        state = new


def _ask_needs(name, needs):
    """Ask what an output left commented out needs to become a real one.

    :param name: the output.
    :type name: str
    :param needs: ``"type"`` and / or ``"dimensions"``.
    :type needs: set[str]
    :return: what follows its name in ``--keep``: ``=Real``, ``=[3]``, ``=Real[3]`` or nothing.
    :rtype: str
    """
    kind = _ask("select", f"Type of {name}?", choices=["Real", "Integer", "Boolean", "String"]) if "type" in needs else ""
    size = ""
    if "dimensions" in needs:
        answer = _ask("text", f"Size of {name} (e.g. 3, or 3,2):", validate=_check_size)
        size = f"[{answer.replace(' ', '')}]"
    return f"={kind}{size}" if kind or size else ""


def _check_size(text):
    """Validate an array size for a prompt.

    :param text: the answer.
    :type text: str
    :return: True, or what's wrong with it.
    :rtype: bool or str
    """
    return all(part.strip().isdigit() for part in text.split(",")) or "Give numbers separated by commas"


def _ask_outputs(args, data, comments, probed):
    """Ask which outputs to keep, those left commented out among them, and which of the probe's ints are counts.

    Without the probe, only when some are commented out: that's the choice they were left for.

    :param args: the options; ``keep`` and ``integer`` are added to.
    :type args: argparse.Namespace
    :param data: the config of a draft.
    :type data: dict
    :param comments: its comments.
    :type comments: dict
    :param probed: whether the probe was chosen.
    :type probed: bool or float
    """
    import questionary

    from fmugen.builder import commented_outputs
    outputs = list(data.get("outputs", {}))
    commented = {n: c for n, c in commented_outputs(comments).items() if n not in outputs}
    kept = [item.partition("=")[0] for item in args.keep] or outputs
    if (probed or commented) and len(outputs) + len(commented) > 1 and not args.keep:
        # what the object holds inside it, rather than what it computes, unless that's all there is
        inner = {n for n in outputs if "." in n and str(data["outputs"][n].get("from", "attr:")).startswith("attr:")}
        inner = inner if len(inner) < len(outputs) else set()
        choices = [questionary.Choice(n, checked=n not in inner) for n in outputs]
        choices += [questionary.Choice(f"{n}  (left commented out: {c['why'].partition(';')[0] or 'unchecked'})", value=n)
                    for n, c in commented.items()]
        kept = _ask("checkbox", "Which outputs to keep?", choices=choices,
                    validate=lambda chosen: bool(chosen) or "Keep at least one")
        if len(kept) < len(outputs) or commented:   # those not kept are decided on: gone from the config
            for name in kept:
                item = name + _ask_needs(name, commented[name]["needs"]) if name in commented else name
                args.keep.append(item)
    ints = [n for n in kept if "an int in the probe; Real" in str(comments.get(("outputs", n), ""))
            and n not in args.integer]
    if ints:
        for name in _ask("checkbox", "Ints in the probe, typed Real. Which are counts, to type Integer?",
                              choices=ints):
            args.integer.append(name)


def _parsed(args):
    """The ``--convert`` and ``--read`` options as ``infer_config`` takes them.

    :param args: the options.
    :type args: argparse.Namespace
    :rtype: tuple[dict, dict]
    """
    from fmugen.__main__ import _parse_converts, _parse_reads
    return _parse_converts(args.convert), _parse_reads(args.read)


def _ask_extra(args, comments):
    """Ask for one more option, of those only the user can know. Nothing is added unless chosen.

    :param args: the options; added to.
    :type args: argparse.Namespace
    :param comments: the comments of a draft.
    :type comments: dict
    :return: whether one was added.
    :rtype: bool
    """
    from fmugen.__main__ import _parse_converts, _parse_reads, _parse_starts

    def parses(parse, *more):
        def check(text):
            try:
                parse([text.strip()], *more)
            except InterfaceError as e:
                return str(e)
            return True
        return check

    def check_const(text):
        ok = parses(_parse_starts, "--const")(text)
        return _check_value(text.partition("=")[2]) if ok is True else ok

    def check_start(text):
        ok = parses(_parse_starts)(text)
        return _check_value(text.partition("=")[2]) if ok is True else ok

    defaults = [str(k[1]).partition(".")[2] for k in comments if k[0] == "model"
                and str(k[1]).startswith(("constants.", "call_constants."))]
    hint = f" (left at their default: {', '.join(defaults[:6])})" if defaults else ""
    start, const, setup, read, convert = ("A start value (--start)", "A fixed value, not an FMU variable (--const)",
                                          "A call before the model is used (--setup)",
                                          "An output at a path (--read)", "How an argument is passed (--convert)")
    answer = _ask("select", "Anything else to set?", choices=[_DONE, start, const, setup, read, convert])
    if answer == start:
        _add_value(args, "start", _ask("text", "NAME=VALUE:", validate=check_start).strip())
    elif answer == const:
        _add_value(args, "const", _ask("text", f"NAME=VALUE or NAME=call:module:function(...){hint}:",
                                       validate=check_const).strip())
    elif answer == setup:
        args.setup.append(_ask("text", "module:function(args), or a method such as reset:",
                               validate=lambda text: bool(text.strip()) or "Give a call").strip())
    elif answer == read:
        args.read.append(_ask("text", "NAME=attr:PATH or NAME=return[:PATH]:", validate=parses(_parse_reads)).strip())
    elif answer == convert:
        args.convert.append(_ask("text", "NAME=module:function or NAME=numpy:", validate=parses(_parse_converts)).strip())
    return answer != _DONE


def _ask_extras(args, state, parsed_values, asked):
    """Offer the options no question covers, each checked on the model, until there's nothing else.

    :param args: the options.
    :type args: argparse.Namespace
    :param state: what :func:`_draft` gave.
    :type state: tuple
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :param asked: the names already asked; added to.
    :type asked: set[str]
    :return: what :func:`_draft` gives for the answers.
    :rtype: tuple
    """
    names = "start", "const", "value_order", "setup", "read", "convert"
    while True:
        before = [list(getattr(args, n)) for n in names]
        if not _ask_extra(args, state[1]):
            return state
        new = _read(args, parsed_values)
        if not new:
            for name, value in zip(names, before):
                setattr(args, name, value)
            _note("It was left out.")
            continue
        state = _ask_values(args, new, parsed_values, asked)


def _settle_model(args, parsed_values):
    """Ask what decides how the model is read: the model, the FMI version, the config's path and the probe.

    A model that can't be read is asked again, unless other options given may be why: ``init`` stops on those.

    :param args: the options; updated in place.
    :type args: argparse.Namespace
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :return: what :func:`_draft` gives.
    :rtype: tuple
    :raises InterfaceError: options given on the command line that don't fit.
    """
    options_given = any((args.call, args.create, args.kind, args.start, args.const, args.setup, args.read, args.convert))
    output_given, force_given = args.output, args.force
    probe_asked = False
    while True:
        try:
            if args.model is None:
                args.model = _ask_model()
            args.model = _ask_entry(args.model)
        except InterfaceError as e:
            if options_given:
                raise
            _note(f"That doesn't work: {e}")
            args.model = None
            continue
        if args.fmi is None:
            args.fmi = _ask("select", "Which FMI version?", choices=_fmi_choices())
        _ask_output(args)
        if not args.probe and not probe_asked:
            probe_asked = True
            _note("The model can be called once. Don't, if it needs a device, a network or a licence.")
            if _ask("confirm", "Call it, to find array sizes, exact types and run-time results?", default=False):
                args.probe = args.probe_timeout or True
        state = _read(args, parsed_values)
        if state:
            return state
        if options_given:
            raise InterfaceError("the model can't be read with the options given")
        args.model, args.output, args.force = None, output_given, force_given


def _command(args):
    """The options that give the same config with no questions.

    :param args: the options, answered.
    :type args: argparse.Namespace
    :rtype: list[str]
    """
    order = list(dict.fromkeys(args.value_order))
    values = [("--start", item) for item in args.start] + [("--const", item) for item in args.const]
    values.sort(key=lambda v: order.index(v[1].partition("=")[0].strip()))
    command = [args.model, *(part for value in values for part in value)]
    if args.probe:
        command.append("--probe")
        if args.probe_timeout:
            command += ["--probe-timeout", f"{args.probe_timeout:g}"]
    for option, value in (("--create", args.create), ("--kind", args.kind), ("--call", args.call)):
        if value:
            command += [option, value]
    for option, items in (("--setup", args.setup), ("--read", args.read), ("--convert", args.convert),
                          ("--keep", args.keep), ("--integer", args.integer)):
        for item in items:
            command += [option, item]
    command += ["--fmi", str(args.fmi)]
    path = _split_target(args.model)[0]
    if args.output != str((path.parent if path else Path()) / "fmugen.toml"):
        command += ["-o", args.output]
    return [*command, "--force"] if args.force else command


def fill_in(args, parsed_values):
    """Ask what the options left open, and set it on ``args``.

    Prints the command that gives the same config without questions.

    :param args: the parsed ``init`` options; updated in place.
    :type args: argparse.Namespace
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :raises KeyboardInterrupt: Ctrl-C.
    :raises InterfaceError: options given on the command line that don't fit the model.
    """
    from fmugen.builder import commented_outputs
    state = _settle_model(args, parsed_values)
    state = _choose_create(args, state, parsed_values)
    state = _choose_method(args, state, parsed_values)
    asked = set()
    state = _ask_values(args, state, parsed_values, asked)
    state = _ask_extras(args, state, parsed_values, asked)
    if (not args.probe and not state[0].get("outputs") and not commented_outputs(state[1])
            and _ask("confirm", "No outputs found in the code. Call the model once to look for them?", default=True)):
        args.probe = args.probe_timeout or True
        state = _read(args, parsed_values) or state
    _ask_outputs(args, state[0], state[1], args.probe)
    _note("To repeat this without questions: " + shlex.join(["fmugen", "init", *_command(args)]))


def _fmi_choices():
    """The FMI versions, as prompt choices.

    :rtype: list[questionary.Choice]
    """
    import questionary
    return [questionary.Choice("FMI 2: supported by more tools", value=2),
            questionary.Choice("FMI 3: real arrays, Binary, Float32, clocks and events", value=3)]
