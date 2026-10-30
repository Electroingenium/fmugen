"""Composite FMUs: several fmugen models in one FMU, connected output to input.

A composite fmugen.toml lists other fmugen configs (the parts) and connections:

    [composite]
    name = "pv_system"
    parts = { sky = "irradiance/fmugen.toml", cell = "cell/fmugen.toml" }   # run in this order
    connections = ["sky.poa_global -> cell.poa_global"]

Each part is built as usual into resources/parts/<name>/ and run by its own runtime Engine
(templates/fmugen_runtime/, CompositeEngine). This module validates the composite config and
combines the parts' interfaces into the FMU's interface.
"""
import keyword
import tomllib
from pathlib import Path

from fmugen.config import FMI_VERSIONS, INT_TYPES, InterfaceError, element_count, load_config
from fmugen.templates.fmugen_runtime import SPEC_VERSION

#: Keys allowed in ``[composite]``.
COMPOSITE_KEYS = {"name", "description", "author", "parts", "connections", "fmi_version"}
#: Tables allowed in a composite config.
TOP_KEYS = {"composite", "experiment"}
#: Folder under ``resources/`` holding one subfolder per part.
PARTS_DIR = "parts"


class Composite:
    """A validated composite config: several fmugen configs (the parts) and their connections.

    :ivar path: the composite fmugen.toml.
    :ivar data: the ``[composite]`` table.
    :ivar experiment: the ``[experiment]`` table.
    :ivar fmi_version: the FMI version of the FMU.
    :ivar parts: part name -> its :class:`fmugen.config.Config`, in run order.
    :ivar connections: ``((src part, src variable), (dst part, dst variable))`` pairs.
    """
    def __init__(self, path):
        """Read and validate a composite config and load each part's config.

        All parts run in one Python process, which imports a module name only once, so two parts
        with modules of the same name are refused (one would run the other's code).

        :param path: the composite fmugen.toml.
        :type path: str or pathlib.Path
        :raises InterfaceError: unknown keys, fewer than two parts, an invalid part name or path,
            a malformed connection, or clashing module names.
        """
        self.path = Path(path)
        data = tomllib.loads(self.path.read_text(encoding="utf-8"))
        unknown = set(data) - TOP_KEYS
        if unknown:
            raise InterfaceError(f"{self.path}: a composite config has only [composite] and [experiment], "
                                 f"not {sorted(unknown)}")
        table = data["composite"]
        unknown = set(table) - COMPOSITE_KEYS
        if unknown:
            raise InterfaceError(f"[composite]: unknown keys {sorted(unknown)}")
        self.data = table
        self.experiment = dict(data.get("experiment", {}))
        self.fmi_version = table.get("fmi_version", 2)
        if self.fmi_version not in FMI_VERSIONS:
            raise InterfaceError(f"[composite] fmi_version must be one of {FMI_VERSIONS}")
        parts = table.get("parts")
        if not isinstance(parts, dict) or len(parts) < 2:
            raise InterfaceError('[composite] parts must name two or more configs: { a = "a/fmugen.toml", ... }')
        self.parts = {}
        for name, rel in parts.items():
            if not name.isidentifier() or keyword.iskeyword(name):
                raise InterfaceError(f"[composite] parts: {name!r} must be a Python identifier")
            part_path = (self.path.parent / rel).resolve()
            if part_path.is_dir():
                part_path = part_path / "fmugen.toml"
            if not part_path.is_file():
                raise InterfaceError(f"[composite] parts.{name}: {rel!r} is not a fmugen.toml")
            self.parts[name] = load_config(part_path)
        self.connections = [_connection(text) for text in table.get("connections", [])]
        owners = {}
        for name, config in self.parts.items():
            for module in config.local_modules():
                owners.setdefault(module, []).append(name)
        clashes = {module: names for module, names in sorted(owners.items()) if len(names) > 1}
        if clashes:
            shown = "; ".join(f"{m} (in {', '.join(names)})" for m, names in clashes.items())
            raise InterfaceError(f"[composite] parts have modules with the same name: {shown}. All parts run in "
                                 "one Python process, which imports a module name once, so one part would run "
                                 "the other's code: rename one of them")


def is_composite(target):
    """Whether a build target is a composite config.

    :param target: a fmugen.toml, a folder containing one, or anything else.
    :type target: str or pathlib.Path
    :return: the config path when it has a ``[composite]`` table, else None.
    :rtype: pathlib.Path or None
    :raises InterfaceError: the file isn't valid TOML.
    """
    path = Path(target)
    if path.is_dir():
        path = path / "fmugen.toml"
    if path.suffix != ".toml" or not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise InterfaceError(f"{path}: {e}") from e
    return path if "composite" in data else None


def _connection(text):
    """Parse a connection ``"part.output -> part.input"``.

    :param text: the connection.
    :type text: str
    :return: ``((source part, variable), (target part, variable))``.
    :rtype: tuple
    :raises InterfaceError: a malformed connection.
    """
    source, arrow, target = (str(text)).partition("->")
    ends = []
    for end in (source.strip(), target.strip()):
        part, dot, variable = end.partition(".")
        if not arrow or not dot or not part or not variable:
            raise InterfaceError(f"[composite] connection {text!r} must look like 'part.output -> part.input'")
        ends.append((part, variable))
    return tuple(ends)


#: FMI floating-point types (FMI 2 and FMI 3 names).
FLOAT_TYPES = ("Real", "Float32", "Float64")


def _type_mismatch(source, target, source_interface, target_interface):
    """Why a value of ``source`` can't be passed to ``target``, or None.

    Same types, integers and Booleans into floats, and integers into integers (the range is
    checked when it is set) work. Enumerations must have the same items.

    :param source: the source part's variable.
    :type source: dict
    :param target: the target part's variable.
    :type target: dict
    :param source_interface: the source part's interface (for its type definitions).
    :type source_interface: dict
    :param target_interface: the target part's interface.
    :type target_interface: dict
    :return: the reason, or None.
    :rtype: str or None
    """
    a, b = source["type"], target["type"]
    if a == "Enumeration" or b == "Enumeration":
        if a != b:
            return f"{a} to {b}"
        items = [i.get("type_definitions", {}).get(v.get("declared_type"), {}).get("items")
                 for i, v in ((source_interface, source), (target_interface, target))]
        return None if items[0] == items[1] else f"the enumerations have different items {items[0]} and {items[1]}"
    if a == b or b in FLOAT_TYPES and (a in INT_TYPES or a in FLOAT_TYPES or a == "Boolean") \
            or a in INT_TYPES and b in INT_TYPES:
        return None
    return f"a {a} can't be passed to a {b}"


def combine(composite, interfaces, version, model_name=None, author=None):
    """Build the composite FMU's interface from its parts' interfaces.

    Each connection is checked (any variable but an input into an input, fed once, unclocked, same
    size, compatible types) and stored as the parts' value references. The FMU's variables are the
    parts' variables not fed by a connection, named ``<part>.<name>`` and carrying ``part`` and
    ``local`` (the reference inside the part), which :class:`CompositeEngine` uses to route
    calls. Clocks follow the variables. Parts with a fixed step must agree on it.

    :param composite: the validated composite config.
    :type composite: Composite
    :param interfaces: part name -> its interface spec, in run order.
    :type interfaces: dict
    :param version: the FMI version.
    :type version: int
    :param model_name: ``--name``, overriding ``[composite] name``.
    :type model_name: str or None
    :param author: ``--author``, overriding ``[composite] author``.
    :type author: str or None
    :return: the composite interface spec (``composite = true``).
    :rtype: dict
    :raises InterfaceError: an invalid connection or conflicting fixed step sizes.
    """
    by_name = {name: {v["name"]: v for v in interface["variables"]} for name, interface in interfaces.items()}
    connections, fed = [], set()
    for (src, src_var), (dst, dst_var) in composite.connections:
        for part, var in ((src, src_var), (dst, dst_var)):
            if part not in by_name:
                raise InterfaceError(f"[composite] connection: there is no part {part!r}")
            if var not in by_name[part]:
                raise InterfaceError(f"[composite] connection: part {part!r} has no variable {var!r}")
        source, target = by_name[src][src_var], by_name[dst][dst_var]
        if target["causality"] != "input":
            raise InterfaceError(f"[composite] connection: {dst}.{dst_var} is not an input")
        if source["causality"] in ("input", "independent"):
            raise InterfaceError(f"[composite] connection: {src}.{src_var} is an input, not a result")
        if (dst, dst_var) in fed:
            raise InterfaceError(f"[composite] connection: {dst}.{dst_var} is fed twice")
        for end, var in ((f"{src}.{src_var}", source), (f"{dst}.{dst_var}", target)):
            if var.get("clocks"):
                raise InterfaceError(f"[composite] connection: {end} is a clocked variable; only variables "
                                     "that have a value at every step can be connected")
            if any(isinstance(d, str) for d in var.get("dimensions", ())):
                raise InterfaceError(f"[composite] connection: {end} is an array sized by a structural "
                                     "parameter; only fixed-size arrays can be connected")
        if element_count(source) != element_count(target):
            raise InterfaceError(f"[composite] connection: {src}.{src_var} and {dst}.{dst_var} have different sizes")
        why = _type_mismatch(source, target, interfaces[src], interfaces[dst])
        if why:
            raise InterfaceError(f"[composite] connection: {src}.{src_var} -> {dst}.{dst_var}: {why}")
        fed.add((dst, dst_var))
        connections.append({"from": [src, source["valueReference"]], "to": [dst, target["valueReference"]]})

    variables, type_definitions = [], {}
    for part, interface in interfaces.items():
        for td_name, definition in interface.get("type_definitions", {}).items():
            type_definitions[f"{part}.{td_name}"] = definition
        for var in interface["variables"]:
            if var["causality"] == "independent" or (part, var["name"]) in fed:
                continue
            new = dict(var, name=f"{part}.{var['name']}", part=part, local=var["valueReference"])
            if var.get("declared_type"):
                new["declared_type"] = f"{part}.{var['declared_type']}"
            if "depends_on" in var:
                new["depends_on"] = [f"{part}.{d}" for d in var["depends_on"] if (part, d) not in fed]
            if var.get("clocks"):
                new["clocks"] = [f"{part}.{c}" for c in var["clocks"]]
            if var.get("dimensions"):   # a structural parameter sizing it is one of the FMU's variables too
                new["dimensions"] = [f"{part}.{d}" if isinstance(d, str) else d for d in var["dimensions"]]
            variables.append(new)
    if version == 3:
        variables.append({"name": "time", "causality": "independent", "variability": "continuous",
                          "type": "Float64", "description": "Simulation time"})
    vr = 0
    for var in variables:
        var["valueReference"] = vr
        vr += element_count(var) if version == 2 else 1
    # the parts' clocks, named <part>.<clock>, after the variables (FMI 3 only: a part's config refuses
    # clocks for FMI 2)
    clocks = [dict(clock, name=f"{part}.{clock['name']}", part=part, local=clock["valueReference"])
              for part, interface in interfaces.items() for clock in interface.get("clocks", [])]
    for vr, clock in enumerate(clocks, start=vr):
        clock["valueReference"] = vr

    experiment = dict(composite.experiment)
    fixed = {i.get("experiment", {}).get("step_size") for i in interfaces.values()
             if i.get("experiment", {}).get("fixed_step")}
    if len(fixed) > 1:
        raise InterfaceError(f"[composite] parts need different fixed step sizes: {sorted(fixed)}")
    if fixed and "step_size" not in experiment:
        experiment.update(step_size=fixed.pop(), fixed_step=True)
    return {
        "spec_version": SPEC_VERSION,
        "composite": True,
        "fmi_version": version,
        "model_name": model_name or composite.data.get("name") or composite.path.parent.name,
        "description": composite.data.get("description", "fmugen composite of " + ", ".join(interfaces)),
        "author": author if author is not None else composite.data.get("author", ""),
        "parts": [{"name": part, "dir": f"{PARTS_DIR}/{part}"} for part in interfaces],
        "connections": connections,
        "variables": variables,
        "type_definitions": type_definitions,
        "experiment": experiment,
        "clocks": clocks,
        "events": {},
        "has_event_mode": any(i.get("has_event_mode") for i in interfaces.values()),
        "can_get_and_set_state": all(i.get("can_get_and_set_state", True) for i in interfaces.values()),
        "capture_output": any(i.get("capture_output") for i in interfaces.values()),
    }
