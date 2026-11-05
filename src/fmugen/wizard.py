"""The questions ``fmugen init`` asks in a terminal.

They fill in the same arguments as the command-line options, so an option given is never asked, and
what the wizard answers is exactly what those options would do. Prompts go to stderr, like the warnings,
so ``-o -`` can still be redirected. questionary is imported only when a question is asked: the FMU's
runtime never gets here.
"""
import ast
import contextlib
import io
import shlex
import sys
from pathlib import Path

from fmugen.config import InterfaceError

#: Set by ``--yes``: nothing is asked, also by the inference (see :func:`interactive`).
_assume_yes = False

#: Python files in the folder that are not models, left out of the first question.
_NOT_MODELS = {"setup.py", "conftest.py", "noxfile.py"}
_OTHER = "Something else (a path, or package.module:Name)"
_OWN_ENTRY = "Write my own"
_OWN_METHOD = "Write my own"
_USE_RETURNED = "None: read what the function returns on every step"
_BUILD_EACH_STEP = "None: build the object on every step and read its results"


def assume_yes(value=True):
    """Never ask: every question takes its default (``--yes``).

    :param value: whether to assume yes.
    :type value: bool
    """
    global _assume_yes
    _assume_yes = value


def interactive():
    """Whether questions can be asked: a terminal on stdin and stderr, without ``--yes``.

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


def _check_target(text):
    """Validate a model target for a prompt.

    :param text: the answer.
    :type text: str
    :return: True, or what's wrong with it.
    :rtype: bool or str
    """
    from fmugen.interface import parse_target
    try:
        path, _module, _name = parse_target(text.strip())
    except InterfaceError:
        return "Give a .py file, or package.module:Name for an installed module"
    if path and not path.is_file():
        return f"No file {path}"
    return True


def _ask_model():
    """Ask for the model: the Python files in this folder, or anything else.

    :rtype: str
    """
    files = sorted(p.name for p in Path.cwd().glob("*.py") if not p.name.startswith("_") and p.name not in _NOT_MODELS)
    if files:
        answer = _ask("select", "Which model?", choices=[*files, _OTHER])
        if answer != _OTHER:
            return answer
    return _ask("path", "Model (model.py[:Name] or package.module:Name):", validate=_check_target).strip()


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


def _ask_entry(model):
    """Add the entry to a model file target that defines several.

    :param model: the target.
    :type model: str
    :rtype: str
    """
    from fmugen.interface import parse_target
    path, _module, name = parse_target(model)
    if not path or name:
        return model
    names = entries_in_file(path)
    if len(names) < 2:
        return model
    answer = _ask("select", f"{path.name} defines several: which one is the model?", choices=[*names, _OWN_ENTRY])
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


def _ask_output(default, force):
    """Ask where the config goes, and whether to replace one that exists.

    :param default: the suggested path.
    :type default: pathlib.Path
    :param force: ``--force`` was given.
    :type force: bool
    :return: ``(path, whether to overwrite it)``.
    :rtype: tuple[str, bool]
    """
    while True:
        output = _ask("path", "Write the config to:", default=str(default)).strip() or str(default)
        if force or not Path(output).exists():
            return output, force
        if _ask("confirm", f"{output} exists. Overwrite it?", default=False):
            return output, True


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


def _check_value(text):
    """Validate a start value for a prompt.

    :param text: the answer.
    :type text: str
    :return: True, or what's wrong with it.
    :rtype: bool or str
    """
    return bool(text.strip()) or "Give a value"


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
    from fmugen.interface import infer_config, parse_target, step_candidates
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
            return data, comments, methods
    finally:
        assume_yes(False)


def _ask_method(picked, candidates, cls_name, none_label=None):
    """Ask which method runs a step: the one ``init`` picked first, then the others, or one typed.

    :param picked: the method ``init`` picked, or None when it builds the object on every step instead.
    :type picked: str or None
    :param candidates: the other candidates.
    :type candidates: list[str]
    :param cls_name: the class's name.
    :type cls_name: str
    :param none_label: the choice of no method, shown first when ``picked`` is None (default: building the object).
    :type none_label: str or None
    :return: the method, or None for no method.
    :rtype: str or None
    """
    none_label = none_label or _BUILD_EACH_STEP
    choices = [picked or none_label, *(m for m in candidates if m != picked), _OWN_METHOD]
    long_list = {"use_search_filter": True, "use_jk_keys": False} if len(choices) > 15 else {}
    answer = _ask("select", f"Which method of {cls_name} runs a step?"
                            + (" (type to filter)" if long_list else ""), choices=choices, **long_list)
    if answer == none_label:
        return None
    return _ask("text", "Method name:", validate=_check_name).strip() if answer == _OWN_METHOD else answer


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


def _ask_data(args, comments, command, asked):
    """Ask for the new arguments that look like files or data. Enter leaves one out, and ``init`` warns about it.

    :param args: the options; ``start``, ``const`` and ``value_order`` are added to.
    :type args: argparse.Namespace
    :param comments: the comments of a draft.
    :type comments: dict
    :param command: the equivalent command; added to.
    :type command: list[str]
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
                              "(Enter to skip):").strip()
        if answer:
            # a computed value, or a file read into one, is fixed; a literal or a path can be a variable
            option = "const" if answer.startswith("call:") else "start"
            _add_value(args, command, option, f"{name}={answer}")
            given = True
    return given


def _show_probe_failure(comments):
    """Say why the probe failed, so the questions after it can be answered with that in mind.

    :param comments: the comments of a draft.
    :type comments: dict
    """
    note = str(comments.get(("outputs", None), ""))
    if note.startswith(("probe call failed", "probe construction failed")):
        print(note, file=sys.stderr)


def _add_value(args, command, option, item):
    """Add a ``--start`` or ``--const`` answer.

    :param args: the options.
    :type args: argparse.Namespace
    :param command: the equivalent command; added to.
    :type command: list[str]
    :param option: ``start`` or ``const``.
    :type option: str
    :param item: ``NAME=VALUE``.
    :type item: str
    """
    getattr(args, option).append(item)
    args.value_order.append(item.partition("=")[0])
    command += [f"--{option}", item]


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


def _ask_outputs(args, data, comments, command, probed):
    """Ask which outputs to keep, those left commented out among them, and which of the probe's ints are counts.

    Without the probe, only when some are commented out: that's the choice they were left for.

    :param args: the options; ``keep`` and ``integer`` are added to.
    :type args: argparse.Namespace
    :param data: the config of a draft.
    :type data: dict
    :param comments: its comments.
    :type comments: dict
    :param command: the equivalent command; added to.
    :type command: list[str]
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
        kept = _ask("checkbox", "Which outputs to keep? (space toggles, a toggles all)", choices=choices,
                    validate=lambda chosen: bool(chosen) or "Keep at least one")
        if len(kept) < len(outputs) or commented:   # those not kept are decided on: gone from the config
            for name in kept:
                item = name + _ask_needs(name, commented[name]["needs"]) if name in commented else name
                args.keep.append(item)
                command += ["--keep", item]
    ints = [n for n in kept if "an int in the probe; Real" in str(comments.get(("outputs", n), ""))
            and n not in args.integer]
    if ints:
        for name in _ask("checkbox", "These were ints in the probe, and are typed Real. Which are counts, to type "
                                     "Integer? (space toggles)", choices=ints):
            args.integer.append(name)
            command += ["--integer", name]


def _parsed(args):
    """The ``--convert`` and ``--read`` options as ``infer_config`` takes them.

    :param args: the options.
    :type args: argparse.Namespace
    :rtype: tuple[dict, dict]
    """
    from fmugen.__main__ import _parse_converts, _parse_reads
    return _parse_converts(args.convert), _parse_reads(args.read)


def fill_in(args, argv, parsed_values):
    """Ask what the options left open, and set it on ``args``.

    Prints the command that gives the same config without questions.

    :param args: the parsed ``init`` options; updated in place.
    :type args: argparse.Namespace
    :param argv: the arguments given after ``init``, for that command.
    :type argv: list[str]
    :param parsed_values: gives the ``--start`` and ``--const`` values of ``args``.
    :type parsed_values: collections.abc.Callable
    :raises KeyboardInterrupt: Ctrl-C.
    """
    from fmugen.interface import parse_target
    given = args.model
    args.model = _ask_model() if given is None else given
    added = []
    if not args.probe and _ask("confirm", "Call the model once, to find array sizes, exact types and results built "
                                          "at run time? (not for models that need a device, network or licence)",
                               default=False):
        args.probe = args.probe_timeout or True
        added.append("--probe")
    args.model = _ask_entry(args.model)
    command = [args.model if a == given else a for a in argv]
    if given is None:
        command.insert(0, args.model)
    command += added

    data, comments, candidates = _draft(args, parsed_values)
    if candidates is not None and not args.call:
        # the probe can settle on building the object every step: it's the first choice then, and the methods follow
        name = data["model"]["entry"].rpartition(":")[2]
        factory = comments.get(("init", "factory"))
        default = None if factory or data["model"].get("kind") == "function" else "__call__"
        picked = data["model"].get("call", default)
        args.call = _ask_method(picked, candidates, name, _USE_RETURNED if factory else None)
        if args.call:
            command += ["--call", args.call]
        if args.call != picked:   # another method has other arguments
            data, comments, _ = _draft(args, parsed_values)

    asked = set()
    # with the probe, a value given can get it further, and its next failure can name another argument
    _show_probe_failure(comments)
    while _ask_data(args, comments, command, asked) and args.probe:
        data, comments, _ = _draft(args, parsed_values)
        _show_probe_failure(comments)
    changed = False
    for name, (start, why) in guessed_starts(data, comments).items():
        if name not in asked:
            answer = _ask("text", f"Start value of {name} ({why}):", default=repr(start),
                          validate=_check_value).strip()
            if answer != repr(start):
                # a computed value is fixed, not a variable
                _add_value(args, command, "const" if answer.startswith("call:") else "start", f"{name}={answer}")
                changed = True
    if args.probe and changed:   # other values can give other outputs
        data, comments, _ = _draft(args, parsed_values)
    _ask_outputs(args, data, comments, command, args.probe)

    if args.fmi is None:
        args.fmi = _ask("select", "Which FMI version?", choices=_fmi_choices())
        command += ["--fmi", str(args.fmi)]
    if args.output is None:
        path = parse_target(args.model)[0]
        default = (path.parent if path else Path()) / "fmugen.toml"
        args.output, overwrite = _ask_output(default, args.force)
        if str(default) != args.output:
            command += ["-o", args.output]
        if overwrite and not args.force:
            args.force = True
            command.append("--force")
    print("To repeat this without questions: " + shlex.join(["fmugen", "init", *command, "--yes"]), file=sys.stderr)


def _fmi_choices():
    """The FMI versions, as prompt choices.

    :rtype: list[questionary.Choice]
    """
    import questionary
    return [questionary.Choice("FMI 2: supported by more tools", value=2),
            questionary.Choice("FMI 3: real arrays, Binary, Float32, clocks and events", value=3)]
