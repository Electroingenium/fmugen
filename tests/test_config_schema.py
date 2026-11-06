"""The JSON Schema of fmugen.toml: valid, in step with config.py, and accepting real configs."""
import json
import tomllib
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest

from fmugen.__main__ import main
from fmugen.builder import init
from fmugen.compose import COMPOSITE_KEYS
from fmugen.compose import TOP_KEYS as COMPOSITE_TOP_KEYS
from fmugen.config import (
    CLOCK_KEYS,
    COMMON_VAR_KEYS,
    EVENT_KEYS,
    EXPERIMENT_KEYS,
    FMI_TYPES,
    INTERVAL_VARIABILITIES,
    MODEL_KEYS,
    TIME_SOURCES,
    TOP_KEYS,
    VAR_KEYS,
)

SCHEMA = json.loads(files("fmugen").joinpath("fmugen.schema.json").read_text())
DEFS = SCHEMA["$defs"]
ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted((ROOT / "models" / "r3").glob("*/fmugen.toml"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def errors(config):
    return list(VALIDATOR.iter_errors(config))


def test_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_schema_keys_match_the_config_loader():
    model_top, composite_top = SCHEMA["oneOf"]
    assert set(model_top["properties"]) == TOP_KEYS
    assert set(composite_top["properties"]) == COMPOSITE_TOP_KEYS
    assert set(DEFS["model"]["properties"]) == MODEL_KEYS
    assert set(DEFS["experiment"]["properties"]) == EXPERIMENT_KEYS
    assert set(DEFS["clock"]["properties"]) == CLOCK_KEYS
    assert set(DEFS["events"]["properties"]) == EVENT_KEYS
    assert set(DEFS["composite"]["properties"]) == COMPOSITE_KEYS
    assert set(DEFS["common"]["properties"]) == COMMON_VAR_KEYS
    for section, keys in VAR_KEYS.items():
        variable = DEFS[f"section_{section}"]["additionalProperties"]
        assert set(variable["properties"]) | COMMON_VAR_KEYS == keys
    assert set(DEFS["common"]["properties"]["type"]["enum"]) == set(FMI_TYPES)
    assert set(DEFS["time_source"]["enum"]) == set(TIME_SOURCES)
    assert set(DEFS["clock"]["properties"]["interval_variability"]["enum"]) == set(INTERVAL_VARIABILITIES)


@pytest.mark.parametrize("config", [
    {"model": {}},
    {"model": {"entry": "m.py:f", "strat": 1}},
    {"model": {"entry": "m.py:f"}, "inputs": {"x": {"strat": 1}}},
    {"model": {"entry": "m.py:f"}, "inputs": {"x": {"from": "return"}}},
    {"model": {"entry": "m.py:f"}, "outputs": {"y": {"type": "Double"}}},
    {"model": {"entry": "m.py:f"}, "unknown": {}},
    {"composite": {"parts": {"a": "a"}}},
    {"composite": {"parts": {"a": "a", "b": "b"}}, "inputs": {}},
])
def test_invalid_configs_are_refused(config):
    assert errors(config)


def test_every_published_model_config_validates():
    assert MODELS
    for path in MODELS:
        assert not errors(tomllib.loads(path.read_text(encoding="utf-8"))), path


def test_the_config_init_writes_validates(tmp_path):
    (tmp_path / "m.py").write_text("def f(x=1.0):\n    return {'y': x}\n")
    output, _ = init(tmp_path / "m.py")
    assert not errors(tomllib.loads(output.read_text(encoding="utf-8")))


def test_composite_validates():
    assert not errors({"composite": {"parts": {"a": "a", "b": "b/fmugen.toml"}, "connections": ["a.y -> b.x"]},
                       "experiment": {"step_size": 0.1}})


def test_schema_command(capsys):
    main(["schema"])
    assert json.loads(capsys.readouterr().out) == SCHEMA
