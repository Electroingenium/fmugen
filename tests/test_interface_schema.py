import json
from importlib.resources import files

import jsonschema
import pytest

from fmugen.builder import build
from test_composite import write_composite


SCHEMA = json.loads(files("fmugen").joinpath("interface.schema.json").read_text())


def test_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def validate(fmu_dir):
    """Every interface.json in a built FMU folder (the top-level one and each part's) matches the schema."""
    paths = sorted((fmu_dir / "resources").rglob("interface.json"))
    assert paths
    for path in paths:
        jsonschema.validate(json.loads(path.read_text()), SCHEMA, cls=jsonschema.Draft202012Validator)


@pytest.mark.parametrize("fmi", [2, 3])
def test_composite_matches_the_schema(tmp_path, fmi):
    build(write_composite(tmp_path), tmp_path / "out", output_format="folder", fmi_version=fmi)
    validate(tmp_path / "out")
