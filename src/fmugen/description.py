"""modelDescription.xml (FMI 2.0 or 3.0 Co-Simulation) from a fmugen interface spec."""
import hashlib
import itertools
import math
import os
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from fmugen.units import base_unit, display_unit

#: FMI 2 log categories: (name, description or None).
LOG_CATEGORIES = [
    ("logStatusWarning", None),
    ("logStatusDiscard", None),
    ("logStatusError", None),
    ("logStatusFatal", None),
    ("logStatusPending", None),
    ("logAll", None),
    ("logUnifmuMessages",
     ("Messages related to internal UniFMU functionality. "
     "Enabling this category is required for distributed UniFMUs.")),
]

#: Replaced after the FMU's files are written by a token derived from their content (stamp_guid),
#: so an unchanged model rebuilds to the same GUID / instantiationToken.
GUID_PLACEHOLDER = "{00000000-0000-0000-0000-000000000000}"
#: Namespace of the name-based UUID made from the FMU's content hash.
GUID_NAMESPACE = uuid.UUID("6f3c1d52-8a4e-4b7a-9d0e-2f5c8b1a7e43")

#: FMI floating-point types (FMI 2 and FMI 3 names).
FLOAT_TYPES = ("Real", "Float32", "Float64")

#: ``[experiment]`` key -> <DefaultExperiment> attribute.
EXPERIMENT_ATTRIBUTES = {
    "start_time": "startTime", "stop_time": "stopTime",
    "tolerance": "tolerance", "step_size": "stepSize",
}


def build_time():
    """The build time: ``SOURCE_DATE_EPOCH`` when set (reproducible builds), else now.

    :rtype: datetime.datetime
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    return datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)


def _format_value(fmi_type, value):
    """A value as XML attribute text: ``true`` / ``false``, ``NaN`` / ``INF`` / ``-INF``, full-precision floats.

    :param fmi_type: the variable's FMI type.
    :type fmi_type: str
    :param value: the value.
    :rtype: str
    """
    if fmi_type == "Boolean":
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "INF" if value > 0 else "-INF"
        return repr(value)
    return str(value)


def build_model_description(interface, model_name=None, author=None):
    """Return the FMI 2.0 Co-Simulation modelDescription.xml ElementTree for an interface spec.

    Built with :mod:`xml.etree.ElementTree`. FMI 2 has no arrays, so each array element is its
    own <ScalarVariable> (``x[1,2]``, see :func:`_scalars`) with the array's value reference plus
    its index. ModelStructure lists outputs (with ``depends_on`` dependencies) and initial
    unknowns by 1-based ScalarVariable index. The GUID is a placeholder until :func:`stamp_guid`.

    :param interface: the interface spec (see ``docs/interface.md``).
    :type interface: dict
    :param model_name: overrides ``model_name``.
    :type model_name: str or None
    :param author: overrides ``author``.
    :type author: str or None
    :rtype: xml.etree.ElementTree.ElementTree
    """
    variables = interface["variables"]
    experiment = interface.get("experiment", {})
    can_state = "true" if interface.get("can_get_and_set_state", True) else "false"

    root = ET.Element("fmiModelDescription", {
        "fmiVersion": "2.0",
        "modelName": model_name or interface["model_name"],
        "guid": GUID_PLACEHOLDER,
        "description": interface.get("description", ""),
        "author": interface.get("author", "") if author is None else author,
        "generationTool": "fmugen + unifmu",
        "generationDateAndTime": build_time().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variableNamingConvention": "structured" if any(v.get("dimensions") for v in variables) else "flat",
    })

    ET.SubElement(root, "CoSimulation", {
        "modelIdentifier": "unifmu",
        "needsExecutionTool": "true",
        "canNotUseMemoryManagementFunctions": "true",
        "canHandleVariableCommunicationStepSize": "false" if experiment.get("fixed_step") else "true",
        "canGetAndSetFMUstate": can_state,
        "canSerializeFMUstate": can_state,
    })

    _unit_definitions(root, [v for v in variables if v["type"] == "Real"])

    type_definitions = interface.get("type_definitions", {})
    if type_definitions:
        type_defs = ET.SubElement(root, "TypeDefinitions")
        for name, definition in type_definitions.items():
            simple_type = ET.SubElement(type_defs, "SimpleType", {"name": name})
            enumeration = ET.SubElement(simple_type, "Enumeration")
            for value, item in enumerate(definition["items"], start=1):
                ET.SubElement(enumeration, "Item", {"name": item, "value": str(value)})

    log_categories = ET.SubElement(root, "LogCategories")
    for name, description in LOG_CATEGORIES:
        attrs = {"name": name}
        if description:
            attrs["description"] = description
        ET.SubElement(log_categories, "Category", attrs)

    experiment_attrs = {
        xml_name: _format_value("Real", float(experiment[key]))
        for key, xml_name in EXPERIMENT_ATTRIBUTES.items() if key in experiment
    }
    if experiment_attrs:
        ET.SubElement(root, "DefaultExperiment", experiment_attrs)
    _annotations(root, "VendorAnnotations", interface)

    model_variables = ET.SubElement(root, "ModelVariables")
    index_of = {}   # name -> indexes of its ScalarVariables (several for an array)
    for index, (var, element, name) in enumerate(_scalars(variables), start=1):
        index_of.setdefault(var["name"], []).append(index)
        attrs = {
            "name": name,
            "valueReference": str(var["valueReference"] + element),
            "causality": var["causality"],
            "variability": var["variability"],
        }
        if var.get("description"):
            attrs["description"] = var["description"]
        if var.get("initial") and var["causality"] != "parameter":
            attrs["initial"] = var["initial"]
        extra = var.get("extra", {})
        attrs.update((k, v) for k, v in extra.items() if k not in FMI2_TYPE_ATTRIBUTES)
        model_variables.append(ET.Comment(f'Index of variable = "{index}"'))
        scalar = ET.SubElement(model_variables, "ScalarVariable", attrs)

        type_attrs = {}
        if var.get("declared_type"):
            type_attrs["declaredType"] = var["declared_type"]
        if var.get("quantity"):
            type_attrs["quantity"] = var["quantity"]
        if var.get("unit") and var["type"] == "Real":
            type_attrs["unit"] = var["unit"]
        for key in ("min", "max", "nominal"):
            if key in var:
                type_attrs[key] = _format_value(var["type"], var[key])
        if "start" in var:
            start = var["start"][element] if isinstance(var["start"], list) else var["start"]
            type_attrs["start"] = _format_value(var["type"], start)
        type_attrs.update((k, v) for k, v in extra.items() if k in FMI2_TYPE_ATTRIBUTES)
        ET.SubElement(scalar, var["type"], type_attrs)

    # Omitting "dependencies" means an output may depend on all inputs.
    structure = ET.SubElement(root, "ModelStructure")
    outputs = [v for v in variables if v["causality"] == "output"]
    if outputs:
        outputs_el = ET.SubElement(structure, "Outputs")
        for var in outputs:
            for index in index_of[var["name"]]:
                ET.SubElement(outputs_el, "Unknown", _unknown(var, index, index_of))
    initial_unknowns = [
        v for v in variables
        if (v["causality"] == "output" and v.get("initial") != "exact")
        or v["causality"] == "calculatedParameter"
    ]
    if initial_unknowns:
        initial_el = ET.SubElement(structure, "InitialUnknowns")
        for index in sorted(i for v in initial_unknowns for i in index_of[v["name"]]):
            ET.SubElement(initial_el, "Unknown", {"index": str(index)})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


#: FMI 2 attributes of the <Real>/<Integer>/... element; the others go on <ScalarVariable>.
FMI2_TYPE_ATTRIBUTES = {"displayUnit", "relativeQuantity", "unbounded", "reinit", "derivative"}


def _unit_definitions(root, variables):
    """Add <UnitDefinitions>: each unit by name, with its SI <BaseUnit> when the unit is recognised.

    Also the <DisplayUnit>s variables ask for (``extra = { displayUnit = ... }``), with factor and
    offset when both units are known.

    :param root: the <fmiModelDescription> element.
    :type root: xml.etree.ElementTree.Element
    :param variables: the floating-point variables.
    :type variables: list[dict]
    """
    displays = {}
    for v in variables:
        if v.get("unit"):
            displays.setdefault(v["unit"], set())
            if v.get("extra", {}).get("displayUnit"):
                displays[v["unit"]].add(v["extra"]["displayUnit"])
    if not displays:
        return
    unit_defs = ET.SubElement(root, "UnitDefinitions")
    for unit in sorted(displays):
        element = ET.SubElement(unit_defs, "Unit", {"name": unit})
        base = base_unit(unit)
        if base is not None:
            ET.SubElement(element, "BaseUnit", base)
        for name in sorted(displays[unit] - {unit}):
            conversion = display_unit(unit, name)
            attrs = {"name": name}
            if conversion is not None:
                factor, offset = conversion
                if not math.isclose(factor, 1.0, rel_tol=1e-12):
                    attrs["factor"] = repr(factor)
                if offset:
                    attrs["offset"] = repr(offset)
            ET.SubElement(element, "DisplayUnit", attrs)


def _annotations(root, tag, interface):
    """Add ``[model] annotations``: <Tool name="..."> elements.

    Inside <VendorAnnotations> (FMI 2), or as <Annotation type="..."> inside <Annotations> (FMI 3).
    The text was validated by ``config._check_annotations``.

    :param root: the <fmiModelDescription> element.
    :type root: xml.etree.ElementTree.Element
    :param tag: ``"VendorAnnotations"`` or ``"Annotations"``.
    :type tag: str
    :param interface: the interface spec (see ``docs/interface.md``).
    :type interface: dict
    """
    text = interface.get("annotations")
    if text:
        element = ET.fromstring(f"<{tag}>{text}</{tag}>")
        if tag == "Annotations":
            for tool in element:
                tool.tag = "Annotation"
                tool.attrib["type"] = tool.attrib.pop("name")
        root.append(element)


def _unknown(var, index, index_of):
    """The attributes of an FMI 2 <Unknown>: its index and the indexes of the inputs it depends on.

    :param var: the output variable.
    :type var: dict
    :param index: its ScalarVariable index.
    :type index: int
    :param index_of: variable name -> its ScalarVariable indexes.
    :type index_of: dict
    :rtype: dict
    """
    attrs = {"index": str(index)}
    if "depends_on" in var:
        attrs["dependencies"] = " ".join(str(i) for name in var["depends_on"] for i in index_of[name])
    return attrs


def _scalars(variables):
    """``(variable, element, name)`` for each FMI 2 ScalarVariable.

    An array x with dimensions [2, 3] becomes x[1,1], x[1,2], ... x[2,3] (row-major, 1-based,
    FMI 2 "structured" naming).

    :param variables: the interface's variables.
    :type variables: list[dict]
    :rtype: collections.abc.Iterator[tuple]
    """
    for var in variables:
        dims = var.get("dimensions")
        if not dims:
            yield var, 0, var["name"]
            continue
        for element, position in enumerate(itertools.product(*(range(1, d + 1) for d in dims))):
            yield var, element, f"{var['name']}[{','.join(map(str, position))}]"


LOG_CATEGORIES_FMI3 = [
    ("logStatusWarning", None),
    ("logStatusDiscard", None),
    ("logStatusError", None),
    ("logStatusFatal", None),
    ("logEvents", None),
    ("logUnifmuMessages",
     ("Messages related to internal UniFMU functionality. "
     "Enabling this category is required for distributed UniFMUs.")),
]


def build_model_description_fmi3(interface, model_name=None, author=None):
    """Return the FMI 3.0 Co-Simulation modelDescription.xml ElementTree for an interface spec.

    FMI 3 variables are typed elements (<Float64>, <Int32>, ...) with <Dimension> children for
    arrays (a structural parameter's value reference, or a fixed size); String and Binary starts
    are <Start> children. Clocks follow the variables. ModelStructure uses value references;
    clocked variables are never initial unknowns.

    :param interface: the interface spec (see ``docs/interface.md``).
    :type interface: dict
    :param model_name: overrides ``model_name``.
    :type model_name: str or None
    :param author: overrides ``author``.
    :type author: str or None
    :rtype: xml.etree.ElementTree.ElementTree
    """
    variables = interface["variables"]
    clocks = interface.get("clocks", [])
    experiment = interface.get("experiment", {})
    can_state = "true" if interface.get("can_get_and_set_state", True) else "false"
    vr_of = {v["name"]: v["valueReference"] for v in [*variables, *clocks]}

    root = ET.Element("fmiModelDescription", {
        "fmiVersion": "3.0",
        "modelName": model_name or interface["model_name"],
        "instantiationToken": GUID_PLACEHOLDER,
        "description": interface.get("description", ""),
        "author": interface.get("author", "") if author is None else author,
        "generationTool": "fmugen + unifmu",
        "generationDateAndTime": build_time().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variableNamingConvention": "flat",
    })

    ET.SubElement(root, "CoSimulation", {
        "modelIdentifier": "unifmu",
        "needsExecutionTool": "true",
        "canBeInstantiatedOnlyOncePerProcess": "false",
        "canGetAndSetFMUState": can_state,
        "canSerializeFMUState": can_state,
        "canHandleVariableCommunicationStepSize": "false" if experiment.get("fixed_step") else "true",
        "hasEventMode": "true" if interface.get("has_event_mode") else "false",
        "canReturnEarlyAfterIntermediateUpdate": "false",
        "providesIntermediateUpdate": "false",
    })

    _unit_definitions(root, [v for v in variables if v["type"] in FLOAT_TYPES])

    type_definitions = interface.get("type_definitions", {})
    if type_definitions:
        type_defs = ET.SubElement(root, "TypeDefinitions")
        for name, definition in type_definitions.items():
            enumeration = ET.SubElement(type_defs, "EnumerationType", {"name": name})
            for value, item in enumerate(definition["items"], start=1):
                ET.SubElement(enumeration, "Item", {"name": item, "value": str(value)})

    log_categories = ET.SubElement(root, "LogCategories")
    for name, description in LOG_CATEGORIES_FMI3:
        attrs = {"name": name}
        if description:
            attrs["description"] = description
        ET.SubElement(log_categories, "Category", attrs)

    experiment_attrs = {
        xml_name: _format_value("Real", float(experiment[key]))
        for key, xml_name in EXPERIMENT_ATTRIBUTES.items() if key in experiment
    }
    if experiment_attrs:
        ET.SubElement(root, "DefaultExperiment", experiment_attrs)

    model_variables = ET.SubElement(root, "ModelVariables")
    for var in variables:
        attrs = {
            "name": var["name"],
            "valueReference": str(var["valueReference"]),
            "causality": var["causality"],
            "variability": var["variability"],
        }
        for key, xml_name in (("description", "description"), ("declared_type", "declaredType"),
                              ("quantity", "quantity")):
            if var.get(key):
                attrs[xml_name] = var[key]
        if var.get("initial") and var["causality"] not in ("parameter", "structuralParameter"):
            attrs["initial"] = var["initial"]
        if var.get("unit") and var["type"] in FLOAT_TYPES:
            attrs["unit"] = var["unit"]
        for key in ("min", "max", "nominal"):
            if key in var:
                attrs[key] = _format_value(var["type"], var[key])
        if var.get("clocks"):
            attrs["clocks"] = " ".join(str(vr_of[c]) for c in var["clocks"])
        attrs.update(var.get("extra", {}))
        start = var.get("start")
        start_children = []
        if start is not None:
            values = start if isinstance(start, list) else [start]
            if var["type"] in ("String", "Binary"):
                start_children = values
            else:
                attrs["start"] = " ".join(_format_value(var["type"], v) for v in values)
        element = ET.SubElement(model_variables, "Float64" if var["type"] == "Real" else var["type"], attrs)
        for value in start_children:
            ET.SubElement(element, "Start", {"value": str(value)})
        for d in var.get("dimensions", []):
            ET.SubElement(element, "Dimension",
                          {"valueReference": str(vr_of[d])} if isinstance(d, str) else {"start": str(d)})

    for clock in clocks:
        attrs = {
            "name": clock["name"],
            "valueReference": str(clock["valueReference"]),
            "causality": clock["causality"],
            "variability": "discrete",
            "intervalVariability": clock["interval_variability"],
        }
        if clock.get("description"):
            attrs["description"] = clock["description"]
        if "interval" in clock:
            attrs["intervalDecimal"] = _format_value("Real", clock["interval"])
        if clock.get("shift"):
            attrs["shiftDecimal"] = _format_value("Real", clock["shift"])
        ET.SubElement(model_variables, "Clock", attrs)

    structure = ET.SubElement(root, "ModelStructure")
    outputs = [v for v in [*variables, *clocks] if v["causality"] == "output"]
    for var in sorted(outputs, key=lambda v: v["valueReference"]):
        attrs = {"valueReference": str(var["valueReference"])}
        if "depends_on" in var:
            attrs["dependencies"] = " ".join(str(vr_of[name]) for name in var["depends_on"])
        ET.SubElement(structure, "Output", attrs)
    for var in variables:
        if var.get("clocks"):
            continue  # clocked variables only get values when their clock ticks
        if (var["causality"] == "output" and var.get("initial") != "exact") or var["causality"] == "calculatedParameter":
            ET.SubElement(structure, "InitialUnknown", {"valueReference": str(var["valueReference"])})
    _annotations(root, "Annotations", interface)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


def write_model_description(interface, path, **kwargs):
    """Write modelDescription.xml for the interface's FMI version.

    :param interface: the interface spec (see ``docs/interface.md``).
    :type interface: dict
    :param path: the file.
    :type path: str or pathlib.Path
    :param kwargs: ``model_name`` / ``author`` overrides.
    """
    build = build_model_description_fmi3 if interface.get("fmi_version", 2) == 3 else build_model_description
    tree = build(interface, **kwargs)
    tree.write(Path(path), encoding="utf-8", xml_declaration=True)


def stamp_guid(fmu_dir):
    """Replace :data:`GUID_PLACEHOLDER` in the FMU's modelDescription.xml by a token hashed from every file.

    SHA-256 over every file's path and content (the model description without its build time,
    which changes on every build), turned into a :func:`uuid.uuid5`: an unchanged model rebuilds
    to the same GUID / instantiationToken, and any change gives a new one. Files are hashed in
    chunks: a frozen FMU can hold files of hundreds of MB.

    :param fmu_dir: the FMU folder.
    :type fmu_dir: str or pathlib.Path
    :return: the token, ``{...}``.
    :rtype: str
    """
    fmu_dir = Path(fmu_dir)
    digest = hashlib.sha256()
    for file in sorted(f for f in fmu_dir.rglob("*") if f.is_file() and "__pycache__" not in f.parts):
        digest.update(file.relative_to(fmu_dir).as_posix().encode() + b"/")
        if file == fmu_dir / "modelDescription.xml":   # without the build time, which changes on every build
            digest.update(re.sub(rb'generationDateAndTime="[^"]*"', b"", file.read_bytes()))
            continue
        with open(file, "rb") as f:   # in chunks: a frozen FMU can hold files of hundreds of MB
            digest.update(hashlib.file_digest(f, "sha256").digest())
    description = fmu_dir / "modelDescription.xml"
    token = "{" + str(uuid.uuid5(GUID_NAMESPACE, digest.hexdigest())) + "}"
    description.write_bytes(description.read_bytes().replace(GUID_PLACEHOLDER.encode(), token.encode()))
    return token
