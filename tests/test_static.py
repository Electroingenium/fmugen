"""`fmugen init` without --probe: everything is read from the signatures and the source code,
and the model is never called."""
import textwrap
import tomllib

import pytest

from fmugen.builder import init, isolated_imports
from fmugen.__main__ import _parse_converts
from fmugen.config import InterfaceError
from fmugen.interface import infer_config
from fmugen.toml_output import render_toml


def infer(target, **kwargs):
    with isolated_imports():
        return infer_config(target, **kwargs)


def write(tmp_path, source, name="m.py"):
    (tmp_path / name).write_text(textwrap.dedent(source))
    return tmp_path / name


def test_nothing_runs_without_probe(tmp_path):
    # like a model that needs hardware: constructing it or stepping it would fail (or do something)
    model = write(tmp_path, '''
        import pathlib
        MARK = pathlib.Path(__file__).with_name("ran")

        class Heater:
            def __init__(self, port="COM3"):
                MARK.write_text("constructed")
                raise RuntimeError(f"no device on {port}")

            def step(self, power=0.0):
                MARK.write_text("stepped")
                self.temperature = 20.0 + power
                return {"T": self.temperature}
    ''')
    data, comments = infer(model)
    assert not (tmp_path / "ran").exists()
    assert data["parameters"] == {"port": {"start": "COM3"}}
    assert data["inputs"] == {"power": {"start": 0.0}}
    assert data["outputs"] == {"T": {"from": "return:T"}, "temperature": {}}
    assert "save_state = false  # not checked without --probe" in render_toml(data, comments)


@pytest.mark.parametrize("body, expected", [
    ("return a, a * 2", {"y0": {"from": "return:0"}, "y1": {"from": "return:1"}}),
    ("return a + 1", {"y": {"from": "return"}}),
    ("r = {}\n    r['total'] = a\n    r['ok'] = a > 0\n    return r", {"total": {}, "ok": {}}),
    ("if a > 0:\n        return {'up': a}\n    return {'down': -a}", {"up": {}, "down": {}}),
])
def test_function_returns(tmp_path, body, expected):
    model = write(tmp_path, f"def f(a=1.0):\n    {body}\n")
    data, _ = infer(model)
    assert data["outputs"] == expected


def test_annotations_namedtuple_and_bool(tmp_path):
    model = write(tmp_path, '''
        from typing import NamedTuple

        class Result(NamedTuple):
            power: float
            count: int
            ok: bool

        def f(a=1.0) -> Result:
            return compute(a)          # not readable: the annotation says what comes back
    ''')
    data, _ = infer(f"{model}:f")
    assert data["outputs"] == {"power": {}, "count": {"type": "Integer"}, "ok": {"type": "Boolean"}}


def test_properties_are_not_read(tmp_path):
    model = write(tmp_path, '''
        class Sensor:
            def step(self, u=0.0):
                self.last = u

            @property
            def reading(self) -> float:          # annotated: an output
                raise RuntimeError("reading the sensor runs code")

            @property
            def status(self):                    # unannotated: could be anything
                raise RuntimeError("reading the sensor runs code")
    ''')
    data, comments = infer(model)
    assert data["outputs"] == {"last": {}, "reading": {}}
    assert "# status = {}  # a property of the object (type unknown)" in render_toml(data, comments)


def test_no_source_says_so(tmp_path):
    data, comments = infer("math:hypot")
    assert "outputs" not in data
    assert "no Python source" in comments[("outputs", None)] and "--probe" in comments[("outputs", None)]


def test_arrays_of_unknown_size_are_left_to_fill_in(tmp_path):
    model = write(tmp_path, '''
        import numpy as np

        class Filter:
            def __init__(self):
                self.history = np.zeros(3)
                self.gain = 0.5

            def step(self, u=0.0) -> np.ndarray:
                self.history = np.roll(self.history, 1)
                self.gain = self.gain * 0.9
                return self.history.copy()
    ''')
    data3, comments3 = infer(model, fmi_version=3)
    text3 = render_toml(data3, comments3)
    assert 'y = { from = "return", dimensions = [...] }' in text3 and "history = { dimensions = [...] }" in text3
    assert data3["locals"] == {"gain": {}}

    data2, comments2 = infer(model)                       # FMI 2 too: one scalar per element once sized
    assert render_toml(data2, comments2).count("dimensions = [...]") == 2 and data2["locals"] == {"gain": {}}


def test_factory_without_a_return_annotation(tmp_path):
    model = write(tmp_path, '''
        class Model:
            def step(self, u=0.0):
                return u

        MODELS = {"weights.bin": Model}

        def build(path):
            return Model()

        def load(path="weights.bin"):   # like tellurium's loada, which returns loadAntimonyModel(...)
            return build(path)

        def lookup(path="weights.bin"):
            return MODELS[path]()
    ''')
    # the class is found through the return statements
    data, comments = infer(f"{model}:load", call="step")
    assert data["parameters"] == {"path": {"start": "weights.bin"}} and data["inputs"] == {"u": {"start": 0.0}}
    data, comments = infer(f"{model}:lookup", call="step")
    assert data["parameters"] == {"path": {"start": "weights.bin"}}
    assert "known only by running it" in comments[("inputs", None)]


def test_database_rows_give_their_typed_columns(tmp_path):
    pytest.importorskip("sqlalchemy")
    model = write(tmp_path, '''
        from sqlalchemy import Column, Float, Integer, String
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class Element(Base):   # like mendeleev's Element
            __tablename__ = "elements"
            atomic_number = Column(Integer, primary_key=True)
            symbol = Column(String, nullable=False)
            density = Column(Float)

        def element(ids: int) -> Element:
            return _get(ids)

        def _get(ids):
            raise RuntimeError("needs the database")
    ''')
    data, comments = infer(f"{model}:element", starts={"ids": 8})
    assert data["outputs"] == {"atomic_number": {"type": "Integer"}, "symbol": {"type": "String"}}
    assert "# density = {}" in render_toml(data, comments)   # nullable: commented out


def test_convert(tmp_path):
    assert _parse_converts(["x=torch:tensor", "y=numpy"]) == {"x": "torch:tensor", "y": "numpy"}
    with pytest.raises(InterfaceError):
        _parse_converts(["x=tensor"])
    model = write(tmp_path, "def f(x=(1.0, 2.0), y=(0.0,)):\n    return {'s': x}\n")
    data, _ = infer(model, fmi_version=3, starts={"x": [1.0, 2.0], "y": [0.0]},
                    converts={"x": "torch:tensor", "y": "numpy"})
    assert data["inputs"]["x"]["convert"] == "torch:tensor" and data["inputs"]["y"]["numpy"] is True
    with pytest.raises(InterfaceError, match="--convert names"):
        infer(model, converts={"z": "numpy"})


def test_probe_adds_what_only_the_code_shows(tmp_path):
    model = write(tmp_path, '''
        class Valve:
            def step(self, u=1.0):
                if u > 10:
                    self.alarm = True                # not reached with the start value
                self.flow = 2 * u
    ''')
    data, comments = infer(model, probe=True)
    # the probe's step worked without setting alarm: reading it would fail the FMU's step
    assert data["outputs"] == {"flow": {}}
    assert comments[("outputs", "#alarm#0")] == 'alarm = { type = "Boolean" }'
    assert "not set by step() in the probe" in comments[("outputs", "#alarm#why")]


def test_probe_comments_out_values_that_cant_be_fmi_variables(tmp_path):
    # like pyproj (a str property that is None); impedance's complex result is split into its real
    # and imaginary parts; a growing history (seirsplus) gives its last entry
    model = write(tmp_path, '''
        import numpy as np

        class Circuit:
            def __init__(self):
                self.history = np.zeros(1)

            @property
            def remarks(self) -> str:
                return None

            def predict(self, f=1.0):
                self.history = np.append(self.history, f)
                return np.array([1 + 2j, 3 - 1j]) * f
    ''')
    data, comments = infer(model, probe=True, fmi_version=3)
    assert data["outputs"] == {"y_real": {"from": "return:real", "dimensions": [2]},
                               "y_imag": {"from": "return:imag", "dimensions": [2]}}
    assert data["locals"] == {"history": {"from": "attr:history.-1"}}
    assert comments[("outputs", "#remarks#0")] == 'remarks = { from = "attr:remarks", type = ... }'
    assert "its last entry" in comments[("locals", "history")] and "([2] then [3])" in comments[("locals", "history")]
    with isolated_imports():
        output, _ = init(str(model), probe=True, fmi_version=3, output=str(tmp_path / "fmugen.toml"))
    text = output.read_text()
    assert "# remarks: None in the probe" in text
    assert "remarks" not in tomllib.loads(text).get("outputs", {})


def test_probe_reads_the_properties_of_a_returned_object(tmp_path):
    # like uncertainties' ufloat (properties named like its arguments) and ambiance's Atmosphere
    # (properties that are one-element arrays)
    model = write(tmp_path, '''
        import numpy as np

        class Value:
            def __init__(self, nominal_value, std_dev):
                self._v, self._s = nominal_value, std_dev

            @property
            def nominal_value(self):
                return self._v

            @property
            def temperature(self):
                return np.array([288.15 - 0.0065 * self._v])

        def measure(nominal_value=1.0, std_dev=0.1):
            return Value(nominal_value, std_dev)
    ''')
    data, _ = infer(f"{model}:measure", probe=True)
    assert data["outputs"] == {"y_nominal_value": {"from": "return:nominal_value"},
                               "temperature": {"from": "return:temperature.0"}}


def test_init_prints_the_assumptions_it_wrote_as_comments(tmp_path, capsys):
    model = write(tmp_path, '''
        class Tank:
            def __init__(self, area, n=3):
                self.area, self.n = area, n

            def step(self, *, inflow, when=None):
                self.level = inflow / self.area
    ''')
    with isolated_imports():
        init(str(model), output=str(tmp_path / "fmugen.toml"))
    err = capsys.readouterr().err
    assert "warning: area, inflow: no default in the code" in err
    assert "warning: n: typed Integer because the default is an int" in err
    assert "warning: when: taken as a date-time from the name only" in err




def test_pint_units_are_read_from_the_errors():
    # fluids.units names the unit and the value, metpy the argument and its dimension
    from fmugen.probe import _pint_unit
    numbers = {"V": {"start": 2.5}, "D": {"start": 0.25}}
    assert _pint_unit("Converting 2.5 dimensionless to units of m/s raised DimensionalityError", numbers) == ("V", "m/s")
    assert _pint_unit("Converting 9.0 dimensionless to units of m raised", numbers) is None
    assert _pint_unit('`D` requires "[length] / [time]" but given "dimensionless"', numbers) == ("D", "m / s")
    assert _pint_unit("something else", numbers) is None


def test_probe_outputs_what_the_step_changes_when_nothing_else(tmp_path):
    # like river's LinearRegression (learns intercept); a history that grows each step gives its last entry
    model = write(tmp_path, '''
        import numpy as np

        class Learner:
            def __init__(self, lr=0.1):
                self.lr = lr
                self.intercept = 0.0
                self.history = np.zeros(1)

            def learn_one(self, y=3.0):
                self.intercept += self.lr * (y - self.intercept)
                self.history = np.append(self.history, self.intercept)
    ''')
    data, _ = infer(f"{model}:Learner", call="learn_one", probe=True)
    assert data["outputs"] == {"intercept": {}, "history": {"from": "attr:history.-1"}}
    assert "locals" not in data
