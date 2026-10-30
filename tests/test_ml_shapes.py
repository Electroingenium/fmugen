"""Model shapes found by testing published neural networks (docs/testing/tested-models.md):
factory functions, computed constants, tuple/dict arguments, NamedTuple and 0-d results,
and array inputs that must be numpy arrays without saying so."""
import textwrap

import pytest

from fmugen.builder import init, isolated_imports
from fmugen.config import InterfaceError
from fmugen.__main__ import _parse_consts, _parse_starts, _starts_and_consts
from fmugen.interface import infer_config


def write(tmp_path, source):
    (tmp_path / "lib.py").write_text(textwrap.dedent(source))
    return tmp_path / "lib.py"


def test_start_values_in_toml_spelling_and_computed_constants():
    starts = _parse_starts(["fast=true", "slow=FALSE", "n=3", "name=abc", "d=[inf, 100.0, -inf]", "x=nan"])
    assert {k: v for k, v in starts.items() if k != "x"} == {"fast": True, "slow": False, "n": 3, "name": "abc",
                                                             "d": [float("inf"), 100.0, float("-inf")]}
    assert starts["x"] != starts["x"]   # nan
    with pytest.raises(InterfaceError, match="give it with --const path=call:"):
        _parse_starts(["path=call:os.path:join('a', 'b')"])
    consts = _parse_consts(["path=call:os.path:join('a', 'b')", "model=yolo11n.pt", "size=[640, 480]", "on=TRUE"])
    assert consts == {"path": {"call": "os.path:join('a', 'b')"}, "model": {"python": "'yolo11n.pt'"},
                      "size": {"python": "[640, 480]"}, "on": {"python": "True"}}
    with pytest.raises(InterfaceError, match="both --start and --const"):
        _starts_and_consts({"n": 3}, {"n": {"python": "3"}})


def test_start_and_const_keep_the_command_line_order(tmp_path):
    # like pinocchio's rnea(model, data, q, v, a), a C function with no signature: positions follow the order given
    from fmugen.__main__ import main
    (tmp_path / "lib.py").write_text(textwrap.dedent("""
        def rnea(*args):
            model, data, q = args
            assert model == "M" and data == "D", args
            return {"tau": 2.0 * q}
    """))
    main(["init", f"{tmp_path / 'lib.py'}:rnea", "-o", str(tmp_path / "fmugen.toml"), "--probe",
          "--const", "model=M", "--const", "data=D", "--start", "q=1.5"])
    text = (tmp_path / "fmugen.toml").read_text()
    assert '0 = "M"' in text and '1 = "D"' in text and 'q = { start = 1.5, to = "pos:2" }' in text


def test_factory_function_with_a_computed_constant(tmp_path, make_fmu, adapter):
    # like silero_vad.load_silero_vad / torchani.models.ANI2x: a function returns the model object
    model = write(tmp_path, '''
        class Gain:
            def __init__(self, k, path):
                self.k, self.path = k, path
            def step(self, u):
                return self.k * u

        def make(path, k=2.0, verbose=False):
            return Gain(k, path)

        def weights_file():                     # like huggingface_sb3.load_from_hub(...)
            return "weights.bin"
    ''')
    with isolated_imports():
        init(f"{model}:make", call="step", probe=True, starts=_starts_and_consts(_parse_starts(["u=1.0"]), _parse_consts(["path=call:lib:weights_file()"])))
    text = (tmp_path / "fmugen.toml").read_text()
    assert 'call = "step"' in text and "[model.constants]" in text and 'path = { call = "lib:weights_file()" }' in text
    assert 'k = { start = 2.0, variability = "tunable" }' in text and "verbose = { start = false }" in text

    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.engine.obj.path == "weights.bin"        # computed when the FMU initializes
    assert fmu.set("u", 3.0) == 0 and fmu.fmi2DoStep(0.0, 1.0, False) == 0
    assert fmu.get("y") == 6.0


def test_tuple_argument_namedtuple_and_0d_results_numpy_retry(tmp_path, make_fmu, adapter):
    # like torchani: forward((species, coordinates)) returning SpeciesEnergies(species, energies)
    model = write(tmp_path, '''
        from collections import namedtuple
        import numpy as np

        Result = namedtuple("Result", "total count")

        def combine(pair, scale=1.0):
            offset, values = pair
            return Result(np.array(offset + scale * values.sum()), values.size)   # values must be an ndarray
    ''')
    with isolated_imports():
        data, comments = infer_config(f"{model}:combine", fmi_version=3, probe=True, starts={"pair": (1.0, [1.0, 2.0, 3.0])})
    assert data["inputs"]["pair_0"] == {"start": 1.0, "to": "arg:pair[0]"}
    assert data["inputs"]["pair_1"] == {"dimensions": [3], "start": [1.0, 2.0, 3.0], "to": "arg:pair[1]", "numpy": True}
    assert "the probe failed with lists" in comments[("inputs", "pair_1")]
    assert data["outputs"] == {"total": {}, "count": {}}    # functions: from = "return:<name>" is the default

    with isolated_imports():
        init(f"{model}:combine", fmi_version=3, probe=True, starts={"pair": (1.0, [1.0, 2.0, 3.0])})
    fmu = adapter(make_fmu(tmp_path), fmi=3)
    fmu.initialize()
    assert fmu.get("total") == 7.0
    assert fmu.set("pair_1", [2.0, 2.0, 2.0]) == 0 and fmu.fmi3DoStep(0.0, 1.0, False)[0] == 0
    assert fmu.get("total", "count") == [7.0, 3.0]


def test_dict_argument(tmp_path, make_fmu, adapter):
    # like surfaces' test functions: f({"n_estimators": ..., "max_depth": ...})
    model = write(tmp_path, '''
        def score(params):
            return params["a"] * params["b"]
    ''')
    with isolated_imports():
        init(f"{model}:score", starts={"params": {"a": 2.0, "b": 3.0}})
    text = (tmp_path / "fmugen.toml").read_text()
    assert 'a = { start = 2.0, to = "arg:params[a]" }' in text
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.get("y") == 6.0
    assert fmu.set("a", 4.0) == 0 and fmu.fmi2DoStep(0.0, 1.0, False) == 0
    assert fmu.get("y") == 12.0


def test_only_the_models_own_properties_become_outputs(tmp_path):
    # like transformers models: framework base classes add properties that aren't results
    model = write(tmp_path, '''
        from fractions import Fraction

        class Sensor(Fraction):                  # Fraction (stdlib) has properties numerator/denominator
            @property
            def reading(self):
                return 1.5
            def step(self, u=0.0):
                return None
    ''')
    with isolated_imports():
        data, _ = infer_config(f"{model}:Sensor", call="step", probe=True)
    assert set(data["outputs"]) == {"reading"}
