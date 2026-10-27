"""fmugen CLI.

    fmugen init path/to/model.py[:Name]                      # write path/to/fmugen.toml for review
    fmugen build path/to/fmugen.toml -o out/model.fmu        # packaged FMU (default)
    fmugen build path/to/model.py -o out/model.fmu           # same, with the config inferred in memory
    fmugen build path/to/fmugen.toml -o out/model --format folder
    fmugen build path/to/fmugen.toml -o out/model.fmu --fmi 3        # FMI 3.0 instead of 2.0
    fmugen build path/to/fmugen.toml -o out/model.fmu --vendor       # runs on other machines with Python
    fmugen build path/to/fmugen.toml -o out/model.fmu --compile pyinstaller   # ... without Python, no sources
    fmugen build path/to/fmugen.toml -o out/model.fmu --distributed  # proxy FMU + out/model_private/
"""
import argparse
import sys

from fmugen.builder import FORMATS, build, init
from fmugen.config import InterfaceError
from fmugen.distribute import COMPILERS


def _parse_converts(items):
    """Parse ``--convert`` options.

    :param items: ``["NAME=module:function" | "NAME=numpy", ...]``.
    :type items: list[str]
    :return: ``{NAME: converter}``.
    :rtype: dict[str, str]
    :raises InterfaceError: an item of another form.
    """
    converts = {}
    for item in items:
        name, sep, converter = item.partition("=")
        name, converter = name.strip(), converter.strip()
        if not sep or not name.isidentifier() or not (converter == "numpy" or ":" in converter):
            raise InterfaceError(f"--convert {item!r}: expected NAME=module:function (e.g. x=torch:tensor) or NAME=numpy")
        converts[name] = converter
    return converts


def _literal(text):
    """A ``--start`` / ``--const`` VALUE: a Python literal (:func:`ast.literal_eval`, never executed),
    ``true`` / ``false`` in any case as Booleans, anything else a string.

    ``inf`` and ``nan`` are floats, also inside lists (tmm's layer thicknesses ``[inf, 100.0, inf]``),
    as in TOML.

    :param text: the VALUE.
    :type text: str
    """
    import ast
    import math
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return text
    names = {"inf": math.inf, "nan": math.nan}

    def value(node):
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, (ast.List, ast.Tuple)):
            items = [value(e) for e in node.elts]
            return items if isinstance(node, ast.List) else tuple(items)
        return ast.literal_eval(node)

    try:
        return value(tree.body)
    except (ValueError, TypeError, SyntaxError):
        return text


def _parse_starts(items, option="--start"):
    """Parse ``--start NAME=VALUE`` options: start values of the model's arguments, as FMU variables.

    :param items: the options.
    :type items: list[str]
    :param option: the option name, for messages.
    :type option: str
    :return: ``{NAME: value}``; NAME may be dotted (``ARG.FIELD``).
    :rtype: dict
    :raises InterfaceError: an item without ``=``, with an invalid name, or a ``call:`` value.
    """
    starts = {}
    for item in items:
        name, sep, text = item.partition("=")
        if not sep or not all(part.isidentifier() for part in name.strip().split(".")):
            raise InterfaceError(f"{option} {item!r}: expected NAME=VALUE")
        text = text.strip()
        if option == "--start" and text.startswith("call:"):
            raise InterfaceError(f"--start {item!r}: a call: value is computed once, not a start value: "
                                 f"give it with --const {item}")
        starts[name.strip()] = text if option == "--const" else _literal(text)
    return starts


def _parse_consts(items):
    """Parse ``--const NAME=VALUE`` options: fixed values passed to the model, not FMU variables.

    For what only the model's user knows: a file, a model name, a mode, a computed object.
    ``call:module:function(...)`` is computed when the FMU initializes; anything else is a literal
    as for ``--start``.

    :param items: the options.
    :type items: list[str]
    :return: ``{NAME: {"call": ...} | {"python": repr}}``.
    :rtype: dict
    :raises InterfaceError: an item without ``=`` or with an invalid name.
    """
    consts = {}
    for name, text in _parse_starts(items, "--const").items():
        consts[name] = {"call": text[len("call:"):]} if text.startswith("call:") else {"python": repr(_literal(text))}
    return consts


def _starts_and_consts(starts, consts, order=None):
    """The ``--start`` and ``--const`` values together, as :func:`fmugen.builder.init` takes them.

    In the order they were given on the command line: for callables without a signature (pinocchio's
    ``rnea(model, data, q, v, a)``) that order is the order of the positional arguments.

    :param starts: parsed ``--start`` values.
    :type starts: dict
    :param consts: parsed ``--const`` values.
    :type consts: dict
    :param order: the names in command-line order (default: the starts, then the consts).
    :type order: list[str] or None
    :rtype: dict
    :raises InterfaceError: a name given to both.
    """
    both = sorted(set(starts) & set(consts))
    if both:
        raise InterfaceError(f"{', '.join(both)}: given with both --start and --const; an argument is either an "
                             "FMU variable (--start) or a fixed value (--const)")
    merged = {**starts, **consts}
    return {name: merged[name] for name in (order or merged)} if order else merged


class _InOrder(argparse.Action):
    """``append`` that also records the option's NAME in one list shared by ``--start`` and ``--const``."""

    def __call__(self, parser, namespace, values, option_string=None):
        getattr(namespace, self.dest).append(values)
        namespace.value_order.append(values.partition("=")[0].strip())


def main(argv=None):
    """The ``fmugen`` command: parse arguments and run ``init`` or ``build``.

    Config and interface errors print a one-line message and exit with code 2; other errors exit
    with code 1 (with the traceback when ``--debug`` is given). A successful build prints a
    summary of the FMU's variables.

    :param argv: the arguments (default: ``sys.argv[1:]``).
    :type argv: list[str] or None
    """
    parser = argparse.ArgumentParser(prog="fmugen", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    i = sub.add_parser("init", help="write a fmugen.toml by inspecting a model",
                       description="Import the model, read its signatures and source code, and write a commented "
                                   "fmugen.toml to review. The model is not called unless --probe is given. "
                                   "See docs/models.md.")
    i.add_argument("model", help="model.py, model.py:Name, or package.module:Name")
    i.add_argument("-o", "--output", help="config path (default: fmugen.toml next to the model; - prints it)")
    i.add_argument("--call", help="method run on each step, for classes (default: step/do_step/update/__call__)")
    i.add_argument("--force", action="store_true", help="overwrite an existing config")
    i.add_argument("--fmi", type=int, choices=(2, 3), help="target FMI version; 3 also infers Binary, Float32 and resizable arrays")
    i.set_defaults(value_order=[])
    i.add_argument("--start", action=_InOrder, default=[], metavar="NAME=VALUE",
                   help="start (and probe) value of an argument that becomes an FMU variable (a Python literal: "
                        "number, string, bool, list), e.g. for arguments without a default, or ARG.FIELD=VALUE "
                        "for a field of an object argument; repeatable")
    i.add_argument("--const", action=_InOrder, default=[], metavar="NAME=VALUE",
                   help="fixed value of an argument, not an FMU variable: a file, a model name, a mode, or "
                        "call:module:function(...) for an object computed when the FMU initializes; repeatable")
    i.add_argument("--setup", action="append", default=[], metavar="CALL",
                   help="run before the model is used, e.g. 'psychrolib:SetUnitSystem(psychrolib.SI)' "
                        "or 'reset' (a method, after construction); repeatable")
    i.add_argument("--create", metavar="CLASSMETHOD",
                   help="classes built by a factory: the classmethod that creates the object, "
                        "e.g. from_pretrained; its arguments become parameters")
    i.add_argument("--kind", choices=("function",),
                   help="function: treat a class whose constructor does the work as a function called every step")
    i.add_argument("--probe", action="store_true",
                   help="also call the model once (setup, construction, one step) with the start values, to find "
                        "what the code doesn't show: array sizes, exact types, results built at runtime, whether "
                        "its state can be saved. Without it, the model is never called")
    i.add_argument("--convert", action="append", default=[], metavar="NAME=module:function",
                   help="how an argument is passed to the model, e.g. x=torch:tensor or x=numpy; repeatable")

    b = sub.add_parser("build", help="build a UniFMU from a fmugen.toml or a model file",
                       description="Build an FMI 2.0 or 3.0 Co-Simulation FMU. MODEL is a fmugen.toml, a directory "
                                   "containing one, or a model file whose config is inferred in memory. "
                                   "See docs/config.md and docs/packaging.md.")
    b.add_argument("model", help="fmugen.toml, a directory containing one, or model.py[:Name]")
    b.add_argument("-o", "--output", required=True, help="output path, used exactly as given")
    b.add_argument("--format", choices=FORMATS, default="fmu",
                   help="fmu: zipped .fmu archive (default); folder: unzipped UniFMU folder")
    b.add_argument("--name", help="modelName in modelDescription.xml (default: [model] name, else the entry name)")
    b.add_argument("--author", default=None, help="author in modelDescription.xml (default: [model] author)")
    b.add_argument("--call", help="method run on each step when MODEL is a .py file with a class")
    b.add_argument("--probe", action="store_true",
                   help="when MODEL is a .py file: infer its config like init --probe (calls the model once)")
    b.add_argument("--vendor", action="store_true",
                   help="put wheels of every requirement into the FMU; it installs them offline on its first "
                        "run on a machine (the target needs Python, not this environment)")
    b.add_argument("--platform", action="append", default=[], metavar="TAG",
                   help="with --vendor: also vendor wheels for this platform, e.g. win_amd64, "
                        "manylinux2014_x86_64, macosx_11_0_arm64; repeatable (default: this machine)")
    b.add_argument("--python-version", action="append", default=[], metavar="X.Y",
                   help="with --vendor: vendor wheels for this Python version; repeatable (default: this one)")
    b.add_argument("--compile", choices=sorted(COMPILERS),
                   help="freeze the model, its packages and Python into an executable: the FMU contains no "
                        "source code and needs no Python, but only runs on this OS")
    b.add_argument("--fmi", type=int, choices=(2, 3), help="FMI version (default: [model] fmi_version, else 2)")
    b.add_argument("--hf-weights", action=argparse.BooleanOptionalAction, default=None,
                   help="put the Hugging Face models the model loads (string parameters such as "
                        "\"amazon/chronos-bolt-tiny\") into the FMU, which then loads them offline "
                        "(default: on with --vendor and --compile)")
    b.add_argument("--capture-output", action="store_true",
                   help="send what the model prints (stdout/stderr) to the importer's log instead of the "
                        "console; use it when the model prints more than a few KB, which hangs UniFMU 0.14")
    b.add_argument("--distributed", action="store_true",
                   help="make the FMU a proxy and write the model to a separate folder (--private-output) "
                        "that runs on another host or container and connects to the proxy over TCP")
    b.add_argument("--endpoint", metavar="IP",
                   help="with --distributed: IP address of the host running the proxy FMU (default: 127.0.0.1)")
    b.add_argument("--private-output", metavar="PATH",
                   help="with --distributed: the private folder (default: <output stem>_private next to -o)")

    for p in (i, b):
        p.add_argument("--probe-timeout", type=float, metavar="SECONDS",
                       help="with --probe: exit if the model's probe call takes longer (default: no limit)")
        p.add_argument("--debug", action="store_true",
                       help="show the full traceback of an unexpected error (default: a one-line message)")
    args = parser.parse_args(argv)
    if args.probe and args.probe_timeout:
        args.probe = args.probe_timeout
    try:
        if args.command == "init":
            output, _data = init(args.model, args.output, args.call, args.force, args.fmi,
                                _starts_and_consts(_parse_starts(args.start), _parse_consts(args.const),
                                                   list(dict.fromkeys(args.value_order))),
                                args.setup, args.kind, args.create, args.probe, _parse_converts(args.convert))
            if output:
                print(f"Wrote {output.resolve()}; review it, then run: fmugen build {output}")
            return
        output, interface = build(args.model, args.output, args.name, args.author,
                                  args.format, args.call, args.fmi, args.vendor, args.platform,
                                  args.python_version, args.compile, args.capture_output, args.probe,
                                  args.hf_weights, args.distributed, args.endpoint, args.private_output)
    except (FileExistsError, InterfaceError) as e:
        parser.exit(2, f"fmugen: error: {e}\n")
    except Exception as e:
        if args.debug:
            raise
        parser.exit(1, f"fmugen: error: {type(e).__name__}: {e}\n(run with --debug for the full traceback)\n")

    counts = {}
    for v in interface["variables"]:
        if v["causality"] != "independent":
            counts[v["causality"]] = counts.get(v["causality"], 0) + 1
    summary = ", ".join(f"{n} {c}s" for c, n in counts.items())
    if interface.get("clocks"):
        summary += f", {len(interface['clocks'])} clocks"
    print(f"Built {output.resolve()} (FMI {interface['fmi_version']}.0; {summary})")
    if interface.get("private_output"):
        print(f"Wrote its private side to {interface['private_output'].resolve()}; when the simulation logs "
              f"the proxy's port, run there: {interface['private_start']}")


if __name__ == "__main__":
    main()
