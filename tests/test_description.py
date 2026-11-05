"""modelDescription.xml details: SI base units, display units, extra attributes and annotations."""
import xml.etree.ElementTree as ET

import pytest
from fmpy.validation import validate_fmu
from test_runtime import write

from fmugen.__main__ import main
from fmugen.builder import build
from fmugen.config import InterfaceError
from fmugen.units import base_unit, display_unit

CONFIG = '''
    [model]
    entry = "heat.py:heat"
    fmi_version = {fmi}
    annotations = '<Tool name="example.org/tool"><Note text="hello"/></Tool>'
    [inputs]
    T = {{ start = 293.15, unit = "K", extra = {{ displayUnit = "degC", unbounded = true }} }}
    v = {{ start = 1.0, unit = "km/h" }}
    [outputs]
    q = {{ unit = "W/(m2.K)", extra = {{ relativeQuantity = false }} }}
    n = {{ unit = "furlongs" }}
'''
SOURCE = "def heat(T=293.15, v=1.0):\n    return {'q': 0.1 * T * v, 'n': 1.0}\n"


@pytest.mark.parametrize("fmi", [2, 3])
def test_units_extra_attributes_and_annotations(tmp_path, fmi):
    write(tmp_path, "heat.py", SOURCE, CONFIG.format(fmi=fmi))
    fmu = build(tmp_path / "fmugen.toml", tmp_path / "heat.fmu")[0]
    assert validate_fmu(str(fmu)) == []
    out = tmp_path / "folder"
    build(tmp_path / "fmugen.toml", out, output_format="folder")
    root = ET.parse(out / "modelDescription.xml").getroot()

    units = {u.get("name"): u for u in root.find("UnitDefinitions")}
    assert units["K"].find("BaseUnit").attrib == {"K": "1"}
    assert units["K"].find("DisplayUnit").attrib == {"name": "degC", "offset": "-273.15"}
    assert units["km/h"].find("BaseUnit").attrib == {"m": "1", "s": "-1", "factor": "0.277777777777778"}
    assert units["W/(m2.K)"].find("BaseUnit").attrib == {"kg": "1", "s": "-3", "K": "-1"}
    assert units["furlongs"].find("BaseUnit") is None          # not recognised: declared by name only

    if fmi == 2:
        T = next(v for v in root.iter("ScalarVariable") if v.get("name") == "T").find("Real")
        tool = root.find("VendorAnnotations/Tool")
    else:
        T = next(v for v in root.find("ModelVariables") if v.get("name") == "T")
        tool = root.find("Annotations/Annotation")
    assert (T.get("displayUnit"), T.get("unbounded")) == ("degC", "true")
    assert tool.get("name" if fmi == 2 else "type") == "example.org/tool"
    assert tool.find("Note").get("text") == "hello"


def test_unit_parsing():
    assert base_unit("kW.h") == {"kg": "1", "m": "2", "s": "-2", "factor": "3600000.0"}
    assert base_unit("m/s2") == {"m": "1", "s": "-2"}
    assert base_unit("kg.m-3") == {"kg": "1", "m": "-3"}
    assert base_unit("degC") == {"K": "1", "offset": "273.15"}
    assert base_unit("degC/s") is None and base_unit("%") is None
    for ambiguous in ("C", "F", "S", "T"):                   # Celsius or coulomb? Fahrenheit or farad?
        assert base_unit(ambiguous) is None
    assert base_unit("uF") == {"kg": "-1", "m": "-2", "s": "4", "A": "2", "factor": "1e-06"}
    assert display_unit("Pa", "bar") == (1e-05, 0.0)
    assert display_unit("m", "s") is None                       # different quantities


@pytest.mark.parametrize("line, message", [
    ('x = { extra = { start = 1.0 } }', "not in extra"),
    ('x = { extra = { displayUnit = ["a"] } }', "single value"),
])
def test_extra_errors(tmp_path, line, message):
    write(tmp_path, "m.py", "def f(x=1.0):\n    return {'y': x}\n",
          f'[model]\nentry = "m.py:f"\n[inputs]\n{line}\n[outputs]\ny = {{}}\n')
    with pytest.raises(InterfaceError, match=message):
        build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")


def test_annotations_must_be_tool_elements(tmp_path):
    write(tmp_path, "m.py", "def f(x=1.0):\n    return {'y': x}\n",
          '[model]\nentry = "m.py:f"\nannotations = "<Other/>"\n[inputs]\nx = {}\n[outputs]\ny = {}\n')
    with pytest.raises(InterfaceError, match="Tool"):
        build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")


def test_cli_errors_are_one_line_unless_debug(tmp_path, capsys):
    (tmp_path / "broken.py").write_text("raise RuntimeError('cannot open the port')\n\ndef f(x=1.0):\n    return x\n")
    with pytest.raises(SystemExit) as exit_info:
        main(["build", str(tmp_path / "broken.py"), "-o", str(tmp_path / "out.fmu")])
    assert exit_info.value.code != 0
    err = capsys.readouterr().err
    assert "cannot open the port" in err and "Traceback" not in err
