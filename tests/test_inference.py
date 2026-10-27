"""`fmugen init`: configs inferred from unmodified models."""
import sys
import textwrap
import tomllib

import pytest

from fmugen.builder import init, isolated_imports
from fmugen.config import Config, InterfaceError, load_config, parse_call
from fmugen.interface import infer_config
from fmugen.toml_output import render_toml


def infer(target, **kwargs):
    kwargs.setdefault("probe", True)   # these tests check what the probe call finds; tests/test_static.py: without
    with isolated_imports():
        return infer_config(target, **kwargs)


def test_installed_class_with_time_argument():
    data, comments = infer("simple_pid:PID")
    assert data["time"] == {"dt": "step_size"}
    assert data["inputs"] == {"input_": {"start": 0.0}}
    # components and tunings are 3-tuples: arrays, one scalar per element in FMI 2
    assert data["outputs"] == {"y": {"from": "return"}, "components": {"dimensions": [3]},
                               "tunings": {"dimensions": [3]}}
    assert data["parameters"]["Kp"] == {"start": 1.0}
    assert "tunable" in comments[("parameters", "Kp")]
    assert comments[("model", "constants.output_limits")] == {"python": "(None, None)"}


def test_rendered_config_round_trips(tmp_path):
    (tmp_path / "tank.py").write_text(textwrap.dedent('''
        """A leaky tank."""
        class Tank:
            def __init__(self, area=2.0, leak=0.1, limits=(0, 5)):
                self.area, self.leak, self.limits = area, leak, limits
                self.level = 1.0

            def step(self, inflow: float, dt=1.0):
                self.level += (inflow - self.leak * self.level) / self.area * dt
                self.overflow = self.level > self.limits[1]
                return self.level
    '''))
    output, data = init(tmp_path / "tank.py")
    text = output.read_text()
    assert tomllib.loads(text)["model"] == {"entry": "tank.py:Tank", "call": "step"}
    # integers stay Integer, as an int scalar default does (layer sizes: torch refuses floats)
    assert tomllib.loads(text)["parameters"]["limits"] == {"dimensions": [2], "type": "Integer", "start": [0, 5]}
    assert data["outputs"] == {"y": {"from": "return"}, "overflow": {"type": "Boolean"}}
    assert data["locals"] == {"level": {}}
    assert load_config(output).entry_name == "Tank"

    with pytest.raises(FileExistsError):
        init(tmp_path / "tank.py")


def test_ambiguous_module_needs_a_name(tmp_path):
    (tmp_path / "two.py").write_text("def a(x=1.0):\n    return x\n\ndef b(x=1.0):\n    return x\n")
    with pytest.raises(InterfaceError, match="cannot tell which"):
        infer(tmp_path / "two.py")
    data, _ = infer(f"{tmp_path / 'two.py'}:b")
    assert data["model"]["entry"] == "two.py:b"


def test_model_that_fails_when_imported(tmp_path):
    # a script that solves at module level: its error, and where, not a traceback through fmugen
    (tmp_path / "helper.py").write_text("def solve():\n    raise FileNotFoundError('options.json')\n")
    (tmp_path / "script.py").write_text("import helper\n\ndef f(x=1.0):\n    return x\n\nhelper.solve()\n")
    message = (r"importing .*script\.py failed: FileNotFoundError: options\.json "
               r"\(at script\.py:6, in helper\.py:2\); importing a module runs its top-level code")
    with pytest.raises(InterfaceError, match=message):
        infer(tmp_path / "script.py", probe=False)
    (tmp_path / "fmugen.toml").write_text('[model]\nentry = "script.py:f"\nsources = ["helper.py"]\n\n'
                                         '[outputs]\ny = { from = "return" }\n')
    from fmugen.builder import build
    with pytest.raises(InterfaceError, match=r"importing script\.py:f failed: FileNotFoundError: options\.json"):
        build(tmp_path, tmp_path / "script.fmu")


def test_unknown_config_keys_are_rejected(tmp_path):
    with pytest.raises(InterfaceError, match="unknown key"):
        Config({"model": {"entry": "m.py:f"}, "inputs": {"x": {"strat": 1.0}}}, tmp_path)
    with pytest.raises(InterfaceError, match="entry"):
        Config({"model": {}}, tmp_path)


def test_render_toml_multiline_comment_stays_a_comment():
    # do-mpc: save_state's comment quotes casadi's multi-line pickling error
    import tomllib
    text = render_toml({"model": {"entry": "m.py:f", "save_state": False}},
                       {("model", "save_state"): "can't be pickled (Exception: Use something like:\nwith ctx(): \n  pi)"})
    assert tomllib.loads(text)["model"]["save_state"] is False


def test_render_toml_values():
    text = render_toml({"model": {"entry": "m.py:f"}, "inputs": {"x": {"start": float("-inf"), "unit": "degC"}}})
    assert tomllib.loads(text)["inputs"]["x"]["start"] == float("-inf")


def test_fmi3_infers_arrays_binary_and_float32(tmp_path):
    (tmp_path / "filt.py").write_text(textwrap.dedent('''
        import numpy as np

        class Filter:
            def __init__(self, weights=(0.5, 0.25, 0.25), tag=b"\x01"):
                self.weights = np.array(weights, dtype=np.float32)
                self.history = np.zeros(3)

            def step(self, u=0.0):
                self.history = np.roll(self.history, 1)
                self.history[0] = u
                self.out = np.float32(self.weights @ self.history)
                return self.history.copy()
    '''))
    data, comments = infer(tmp_path / "filt.py", fmi_version=3)
    assert data["model"]["fmi_version"] == 3
    assert data["parameters"]["weights"] == {"dimensions": [3], "start": [0.5, 0.25, 0.25]}
    assert data["parameters"]["tag"] == {"type": "Binary", "start": "01"}
    assert data["outputs"]["y"] == {"from": "return", "dimensions": [3]}
    assert data["outputs"]["out"] == {"type": "Float32"}
    assert "[clocks.sample]" in render_toml(data, comments)

    data2, comments2 = infer(tmp_path / "filt.py")              # FMI 2: arrays, but no Binary or Float32
    assert data2["parameters"]["weights"] == {"dimensions": [3], "start": [0.5, 0.25, 0.25]}
    assert "tag" not in data2["parameters"] and "--fmi 3" in comments2[("parameters", "#tag")]
    assert data2["outputs"]["out"] == {}


def test_start_setup_kind_and_errors(tmp_path):
    (tmp_path / "gh.py").write_text(textwrap.dedent('''
        STARTED = False

        def start():
            global STARTED
            STARTED = True

        class Filter:
            def __init__(self, x, g, h=0.1, **kwargs):
                if not STARTED:
                    raise RuntimeError("call start() first")
                self.x, self.g, self.h = x, g, h

            def update(self, z, g=None):
                self.x += (g or self.g) * (z - self.x)
                return self.x

        class Props:
            def __init__(self, **kwargs):
                self.T = kwargs["T"]
                self.double = 2 * kwargs["T"]
    '''))
    # g has no default in the constructor and one in update(): --start g goes to the constructor
    data, _ = infer(f"{tmp_path / 'gh.py'}:Filter", starts={"x": 0.0, "g": 0.5, "z": 1.0},
                    setup=["gh:start"])
    assert data["model"]["setup"] == ["gh:start"]
    assert data["parameters"]["g"] == {"start": 0.5} and data["inputs"]["z"] == {"start": 1.0}
    assert "g" not in data["inputs"] and data["outputs"] == {"y": {"from": "return"}}

    # a class whose constructor does the work, called like a function; inputs aren't echoed as outputs
    data, _ = infer(f"{tmp_path / 'gh.py'}:Props", kind="function", starts={"T": 300.0})
    assert data["model"]["kind"] == "function"
    assert data["inputs"] == {"T": {"start": 300.0}} and data["outputs"] == {"double": {}}

    with pytest.raises(InterfaceError, match="bytes need --fmi 3"):
        infer(f"{tmp_path / 'gh.py'}:Filter", starts={"x": b"", "g": 0.5})
    with pytest.raises(InterfaceError, match="not arguments"):
        infer(f"{tmp_path / 'gh.py'}:start", starts={"nope": 1.0})


@pytest.mark.parametrize("annotation", ["int | None", "Optional[int]", "typing.Optional[int]"])
def test_optional_annotation_keeps_the_type(tmp_path, annotation):
    (tmp_path / "opt.py").write_text(textwrap.dedent(f"""
        from __future__ import annotations
        import typing
        from typing import Optional
        def f(n: {annotation}, flag: bool | None, x: float | None) -> float:
            return 1.0
    """))
    data, _ = infer(tmp_path / "opt.py", probe=False)
    assert data["inputs"]["n"]["start"] == 0 and type(data["inputs"]["n"]["start"]) is int
    assert data["inputs"]["flag"]["start"] is False
    assert data["inputs"]["x"]["start"] == 0.0


def test_optional_from_typing_objects():
    import typing

    from fmugen.static import unwrap_optional
    assert unwrap_optional(typing.Optional[bool]) is bool
    assert unwrap_optional(int | None) is int
    assert unwrap_optional(int | str) == int | str


def test_number_like_outputs_are_kept(tmp_path):
    (tmp_path / "nums.py").write_text(textwrap.dedent("""
        from decimal import Decimal
        from fractions import Fraction
        import numpy
        def f(x: float = 1.0):
            return {"d": Decimal("1.5"), "q": Fraction(1, 3), "h": numpy.float16(2.0), "c": 1j}
    """))
    data, _ = infer(tmp_path / "nums.py")
    assert list(data["outputs"]) == ["d", "q", "h", "c_real", "c_imag"]   # complex: its two parts


def test_pint_outputs_get_their_unit(tmp_path):
    pytest.importorskip("pint")
    (tmp_path / "q.py").write_text(textwrap.dedent("""
        import pint
        u = pint.get_application_registry()
        def f(x: float = 1.0):
            return {"v": u.Quantity(x, "m/s"), "n": u.Quantity(2.0, "")}
    """))
    data, _ = infer(tmp_path / "q.py")
    assert data["outputs"]["v"]["unit"] == "m/s"
    assert "unit" not in data["outputs"]["n"]


def test_gekko_results_kept_on_the_object_are_outputs(tmp_path):
    pytest.importorskip("gekko")
    (tmp_path / "tank.py").write_text(textwrap.dedent("""
        import numpy as np
        from gekko import GEKKO

        class Tank:
            def __init__(self, k=0.5):
                self.m = GEKKO(remote=False)
                self.m.time = np.linspace(0, 1, 11)
                self.k = self.m.Const(k)
                self.inflow = self.m.Param(1.0)
                self.h = self.m.Var(1.0)
                self.out = self.m.Intermediate(self.k * self.h)
                self.levels = {"main": self.h}
                self.m.Equation(self.h.dt() == self.inflow - self.out)
                self.m.options.IMODE = 4

            def solve(self):
                self.m.solve(disp=False)
    """))
    data, comments = infer(f"{tmp_path / 'tank.py'}:Tank", call="solve")
    # Var and Intermediate are results; Const and Param are not; a dict of them is read by key
    assert data["outputs"] == {"h": {"from": "attr:h.value.-1"}, "out": {"from": "attr:out.value.-1"},
                               "levels.main": {"from": "attr:levels.main.value.-1"}}
    assert "end of the time horizon" in comments[("outputs", "h")]


def test_a_library_base_class_state_is_not_an_output(tmp_path):
    # like `class Plant(GEKKO)`: what the library's own code sets on self (GHFilter.update's x, dx, y)
    # is its state; only what the model's code sets is a result. A base in the user's files still counts.
    pytest.importorskip("filterpy")
    (tmp_path / "base.py").write_text(textwrap.dedent("""
        class Smoother:
            def smooth(self, z):
                self.smoothed = 0.5 * z
    """))
    (tmp_path / "tracker.py").write_text(textwrap.dedent("""
        from filterpy.gh import GHFilter
        from base import Smoother

        class Tracker(GHFilter):
            def track(self, z=1.0):
                self.update(z)
                self.estimate = float(self.x)

        class Local(Smoother):
            def step(self, z=1.0):
                self.smooth(z)
    """))
    starts = {"x": 0.0, "dx": 0.0, "dt": 1.0, "g": 0.5, "h": 0.1}
    for probe in (False, True):
        data, _ = infer(f"{tmp_path / 'tracker.py'}:Tracker", call="track", starts=starts, probe=probe)
        assert list(data["outputs"]) == ["estimate"], probe
        assert not set(data.get("locals", {})) & {"x", "dx", "y", "x_prediction", "dx_prediction"}
        data, _ = infer(f"{tmp_path / 'tracker.py'}:Local", probe=probe)
        assert list(data["outputs"]) == ["smoothed"], probe


NESTED = """
from dataclasses import dataclass
from types import SimpleNamespace

@dataclass
class Zone:
    T: float
    occupied: bool

def f(x: float = 1.0):
    return {"zone": {"T": x, "air": {"rh": 0.5}}, "obj": Zone(2.0, True), "ns": SimpleNamespace(q=3.0)}
"""


def test_nested_outputs_are_named_with_dots(tmp_path):
    (tmp_path / "nested.py").write_text(NESTED)
    data, _ = infer(f"{tmp_path / 'nested.py'}:f")
    assert list(data["outputs"]) == ["zone.T", "zone.air.rh", "obj.T", "obj.occupied", "ns.q"]
    assert data["outputs"]["obj.occupied"]["type"] == "Boolean"
    static, _ = infer(f"{tmp_path / 'nested.py'}:f", probe=False)   # dict literals are read from the code too
    assert {"zone.T", "zone.air.rh"} <= set(static["outputs"])
    assert '"zone.air.rh" = {}' in render_toml(data)


def test_fmi3_time_attribute_is_renamed(tmp_path):
    (tmp_path / "clock.py").write_text(textwrap.dedent("""
        class M:
            def __init__(self):
                self.time = 0.0
            def step(self, dt: float = 1.0):
                self.time += dt
    """))
    # FMI 2 too: FMPy's results call the simulation time `time`, hiding an output of that name (nengo)
    for fmi in (2, 3):
        data, _ = infer(tmp_path / "clock.py", fmi_version=fmi)
        # the step's only result, so an output rather than a local
        assert "time" not in data.get("outputs", {})
        assert data["outputs"]["model_time"] == {"from": "attr:time"}


@pytest.mark.parametrize("kind", ["series", "frame", "xarray"])
def test_table_results_become_named_outputs(tmp_path, kind):
    pytest.importorskip("pandas")
    if kind == "xarray":
        pytest.importorskip("xarray")
    body = {
        "series": "pandas.Series({'p': x, 'q': 2 * x})",
        "frame": "pandas.DataFrame({'p': [x], 'q': [2 * x]})",
        "xarray": "__import__('xarray').Dataset({'p': x, 'q': 2 * x})",
    }[kind]
    (tmp_path / "table.py").write_text(f"import pandas\ndef f(x: float = 1.0):\n    return {body}\n")
    data, _ = infer(tmp_path / "table.py")
    assert data["outputs"] == {"p": {}, "q": {}}


OBJECT_ARGS = """
import dataclasses
from typing import Optional

@dataclasses.dataclass
class Weather:
    T: float
    windy: bool = False
    label: object = None

@dataclasses.dataclass
class Plant:
    gain: float = 2.0
    lag: int = 3
    notes: tuple = ("a",)

def f(w: Weather, plant: Plant = Plant(), opt: Optional[Weather] = None):
    return w.T * plant.gain
"""


def test_object_arguments_become_one_variable_per_field(tmp_path):
    (tmp_path / "objs.py").write_text(OBJECT_ARGS)
    data, comments = infer(f"{tmp_path / 'objs.py'}:f", starts={"w.T": 20.0})
    inputs = data["inputs"]
    assert inputs["w_T"] == {"start": 20.0, "to": "arg:w.T"}
    assert inputs["w_windy"] == {"start": False, "to": "arg:w.windy"}
    assert "w_label" not in inputs and "plant_notes" not in inputs   # not FMI values: defaults kept
    assert inputs["plant_gain"] == {"start": 2.0, "to": "arg:plant.gain"}
    assert inputs["plant_lag"] == {"start": 3, "to": "arg:plant.lag"}
    assert "opt_T" not in inputs   # defaults to None: stays "not given"
    assert data["outputs"] == {"y": {"from": "return"}}   # the probe call built both objects
    _, comments = infer(f"{tmp_path / 'objs.py'}:f", probe=False)
    assert "no default" in comments[("inputs", "w_T")]


@pytest.mark.parametrize("library", ["pydantic", "attrs"])
def test_pydantic_and_attrs_arguments(tmp_path, library):
    pytest.importorskip(library)
    source = {
        "pydantic": "import pydantic\nclass In(pydantic.BaseModel):\n    T: float\n    n: int = 2\n",
        "attrs": "import attrs\n@attrs.define\nclass In:\n    T: float\n    n: int = 2\n",
    }[library]
    (tmp_path / "m.py").write_text(source + "def f(u: In):\n    return u.T * u.n\n")
    data, _ = infer(f"{tmp_path / 'm.py'}:f", starts={"u.T": 1.5})
    assert data["inputs"] == {"u_T": {"start": 1.5, "to": "arg:u.T"}, "u_n": {"start": 2, "to": "arg:u.n"}}
    assert data["outputs"] == {"y": {"from": "return"}}


def test_int_results_are_real_with_a_note(tmp_path):
    (tmp_path / "m.py").write_text(textwrap.dedent("""
        def f(x: float = 1.0):
            return {"count": 3, "y": x}
    """))
    data, comments = infer(tmp_path / "m.py")
    assert data["outputs"]["count"] == {} and "Integer" in comments[("outputs", "count")]
    assert ("outputs", "y") not in comments


def test_setup_call_keeps_keyword_arguments():
    # setup = ["reset(seed=0)"] must seed the reset, not call reset() (Gymnasium envs)
    assert parse_call("reset(seed=0)") == {"call": "reset", "args": [], "kwargs": {"seed": 0}}
    assert parse_call("psychrolib:SetUnitSystem(psychrolib.SI, strict=True)") == {
        "call": "psychrolib:SetUnitSystem", "args": [{"ref": "psychrolib.SI"}], "kwargs": {"strict": True}}
    with pytest.raises(InterfaceError, match=r"\*\*kwargs"):
        parse_call("reset(**opts)")


def test_setup_call_takes_calls_as_arguments():
    # fit(numpy.array(...), ...): init --probe ran it, so build must accept it too (xgboost, skorch)
    assert parse_call("fit(numpy.array([[0.1], [0.2]]), numpy.array([1.0, 2.0]), epochs=2)") == {
        "call": "fit", "args": [{"call": "numpy.array([[0.1], [0.2]])"}, {"call": "numpy.array([1.0, 2.0])"}],
        "kwargs": {"epochs": 2}}
    from fmugen.templates.fmugen_runtime import resolve_constant
    assert resolve_constant({"call": "numpy.array([1.0, 2.0])"}).tolist() == [1.0, 2.0]
    with pytest.raises(InterfaceError, match="calls of names"):
        parse_call("fit(x + 1)")
    with pytest.raises(InterfaceError, match="only one call"):
        parse_call("fit(1).predict(2)")


def test_call_constant_refuses_text_after_the_call():
    # was Path() (the current folder): everything after the first ')' was dropped (rembg, easyocr)
    from fmugen.templates.fmugen_runtime import resolve_constant
    with pytest.raises(ValueError, match="only one call"):
        resolve_constant({"call": "pathlib:Path('x.png').read_bytes()"})
    with pytest.raises(ValueError, match="only one call"):
        resolve_constant({"call": "numpy:zeros(3) + numpy.ones(3)"})


def test_isolated_imports_forgets_namespace_subpackages(tmp_path):
    # like mesa.examples.advanced: a namespace package's __path__ looks its parent up in
    # sys.modules when read, so forgetting the parent first made restoring sys.modules fail
    (tmp_path / "pkg" / "ns").mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "ns" / "mod.py").write_text("x = 1\n")
    with isolated_imports():
        sys.path.insert(0, str(tmp_path))
        import pkg.ns.mod  # noqa: F401
    assert not {"pkg", "pkg.ns", "pkg.ns.mod"} & set(sys.modules)
    assert str(tmp_path) not in sys.path


def test_names_build_would_reject_are_renamed(tmp_path):
    (tmp_path / "names.py").write_text(textwrap.dedent('''
        import dataclasses

        def source(time, alpha=1.0):   # e.g. pygfunction's finite_line_source(time, alpha, ...)
            return alpha * time

        class Inventory:   # e.g. radioactivedecay's Inventory(contents, units).decay(decay_time, units)
            def __init__(self, amount=1.0, units="Bq"):
                self.amount, self.units = amount, units

            def decay(self, decay_time, units="s"):
                return self.amount / (1.0 + decay_time)

        @dataclasses.dataclass(frozen=True)
        class Parameter:   # a descriptor used as a signature default, like astropy.cosmology's
            default: float = 0.0
            doc: str = ""

            def __get__(self, obj, owner=None):
                return self

        class Cosmology:
            def __init__(self, H0, Neff=Parameter(default=3.04, doc="neutrino species")):
                self.H0, self.Neff = H0, Neff

            def age(self, z):
                return 1.0 / (self.H0 * (1.0 + z))
    '''))
    # FMI 3 reserves `time`: the argument becomes model_time, bound to it
    data, _ = infer(f"{tmp_path / 'names.py'}:source", starts={"time": 10.0}, fmi_version=3)
    assert data["inputs"]["model_time"] == {"start": 10.0, "to": "arg:time"} and "time" not in data["inputs"]
    # the step method's units, beside the constructor's
    data, _ = infer(f"{tmp_path / 'names.py'}:Inventory", call="decay", starts={"decay_time": 1.0, "units": "y"})
    assert data["parameters"]["units"] == {"start": "Bq"}
    assert data["inputs"]["decay_units"] == {"start": "y", "to": "arg:units"}
    # both copies: the method's by its variable name, then the plain name is the constructor's
    data, _ = infer(f"{tmp_path / 'names.py'}:Inventory", call="decay",
                    starts={"decay_time": 1.0, "units": "Ci", "decay_units": "y"})
    assert data["parameters"]["units"] == {"start": "Ci"}
    assert data["inputs"]["decay_units"] == {"start": "y", "to": "arg:units"}
    # a descriptor default stands for its `default`; it isn't split into fields
    data, _ = infer(f"{tmp_path / 'names.py'}:Cosmology", call="age", starts={"H0": 70.0, "z": 1.0})
    assert data["parameters"] == {"H0": {"start": 70.0}, "Neff": {"start": 3.04}}
    for target, kwargs in ((f"{tmp_path / 'names.py'}:source", {"starts": {"time": 10.0}, "fmi_version": 3}),
                           (f"{tmp_path / 'names.py'}:Inventory", {"call": "decay",
                                                                   "starts": {"decay_time": 1.0, "units": "y"}})):
        output = tmp_path / "fmugen.toml"
        with isolated_imports():
            init(target, output=output, force=True, probe=True, **kwargs)
        load_config(output)   # and build accepts it:
        from fmugen.builder import build
        build(output, tmp_path / "out", output_format="folder")


def test_entry_keeps_the_name_it_was_given(tmp_path):
    # PyCO2SYS:sys is engine.nd.CO2SYS, while PyCO2SYS:CO2SYS is an older, different function;
    # rdkit's Descriptors.MolWt is a lambda
    (tmp_path / "carbonate_engine.py").write_text("def CO2SYS(par1=1.0):\n    return {'new': par1}\n")
    (tmp_path / "carbonate.py").write_text(textwrap.dedent('''
        from carbonate_engine import CO2SYS as sys

        def CO2SYS(par1=1.0, old_option=0):
            return {"old": par1}

        MolWt = lambda *x, **y: 46.069
    '''))
    data, _ = infer(f"{tmp_path / 'carbonate.py'}:sys")
    assert data["model"]["entry"] == "carbonate.py:sys" and "new" in data["outputs"]
    output = tmp_path / "fmugen.toml"
    with isolated_imports():
        init(f"{tmp_path / 'carbonate.py'}:MolWt", output=output, starts={"mol": 1.0}, probe=True)
    assert load_config(output).model["entry"] == "carbonate.py:MolWt"


def test_factory_annotated_with_a_base_class(tmp_path):
    # tirex's load_model() -> PretrainedModel returns a subclass; only the subclass has forecast()
    (tmp_path / "fac.py").write_text(textwrap.dedent('''
        class Base:
            pass

        class Forecaster(Base):
            def forecast(self, x=1.0):
                return 2.0 * x

        def load(path="w") -> Base:
            return Forecaster()
    '''))
    data, _ = infer(f"{tmp_path / 'fac.py'}:load", call="forecast")
    assert "x" in data["inputs"] and "y" in data["outputs"]
    data, comments = infer(f"{tmp_path / 'fac.py'}:load", call="forecast", probe=False)   # was AttributeError
    assert "known only by running it" in comments[("inputs", None)]


def test_bytes_dated_series_and_object_arrays_are_outputs(tmp_path):
    # rembg returns PNG bytes, pastas a Series indexed by dates, facenet-pytorch object arrays of
    # numbers: each was left out ("no outputs found")
    pd = pytest.importorskip("pandas")
    (tmp_path / "res.py").write_text(textwrap.dedent('''
        import numpy as np, pandas as pd
        def png(n=3):
            return bytes(range(n))
        def series(a=1.0):
            return pd.Series([a, 2 * a, 3 * a], index=pd.date_range("2020-01-01", periods=3))
        def boxes(a=1.0):
            return np.array([[a, 2.0], [3.0, 4.0]], dtype=object), np.array([0.9], dtype=object)
    '''))
    data, _ = infer(f"{tmp_path / 'res.py'}:png", fmi_version=3)
    assert data["outputs"] == {"y": {"from": "return", "type": "Binary"}}
    data, _ = infer(f"{tmp_path / 'res.py'}:series")
    assert data["outputs"] == {"y": {"from": "return", "dimensions": [3]}}
    data, _ = infer(f"{tmp_path / 'res.py'}:boxes")
    assert data["outputs"]["y0"]["dimensions"] == [2, 2] and data["outputs"]["y1"]["dimensions"] == [1]


def test_a_keyword_named_like_the_kwargs_parameter(tmp_path):
    # MOMENTPipeline.from_pretrained(name, **model_kwargs) is called with model_kwargs={...}
    (tmp_path / "kw.py").write_text(textwrap.dedent('''
        def load(path="x", **model_kwargs):
            return {"y": float(len(model_kwargs.get("model_kwargs", {})))}
    '''))
    data, _ = infer(f"{tmp_path / 'kw.py'}:load", starts={"model_kwargs": {"task_name": "r"}})
    assert data["model"]["constants"] == {"model_kwargs": {"python": "{'task_name': 'r'}"}}


def test_fmi3_integer_starts_beyond_int32_get_64_bit_types(tmp_path):
    # numpy's default_rng(seed=12345678901234567890): typed Integer (Int32), which build refused
    (tmp_path / "rng.py").write_text("def draw(seed=0, n=1):\n    return {'y': float(seed % 7)}\n")
    data, _ = infer(f"{tmp_path / 'rng.py'}:draw", fmi_version=3, starts={"seed": 12345678901234567890, "n": -2**40})
    assert data["inputs"]["seed"]["type"] == "UInt64" and data["inputs"]["n"]["type"] == "Int64"


def test_generator_results_are_read_and_nested_tuples_split(tmp_path):
    # like nashpy's Game.support_enumeration: a generator of (row strategy, column strategy) pairs
    (tmp_path / "gen.py").write_text(textwrap.dedent('''
        class Game:
            def __init__(self, a=2.0):
                self.a = a

            def equilibria(self, tol=1e-9):
                yield (0.4, 0.6), (self.a, 1.0, 0.5)

            def pairs(self):
                return (1.0, 2.0), 3.0
    '''))
    data, _ = infer(f"{tmp_path / 'gen.py'}:Game", call="equilibria")
    assert data["outputs"] == {"y0_0": {"from": "return:0.0", "dimensions": [2]},
                               "y0_1": {"from": "return:0.1", "dimensions": [3]}}
    # a tuple of numbers stays one array, as before
    data, _ = infer(f"{tmp_path / 'gen.py'}:Game", call="pairs")
    assert data["outputs"] == {"y0": {"from": "return:0", "dimensions": [2]}, "y1": {"from": "return:1"}}


def test_inputs_that_need_a_date_index_are_passed_as_series(tmp_path):
    pytest.importorskip("pandas")
    # like pyet's pm_fao56: the day of the year comes from an input's index
    (tmp_path / "evap.py").write_text(textwrap.dedent('''
        def pet(tmean=20.0, rs=10.0):
            day = rs.index.dayofyear
            return (tmean * 0.1 + rs * 0.01 + day * 0.0).rename("PET")
    '''))
    data, _ = infer(f"{tmp_path / 'evap.py'}:pet")
    assert {name: info["convert"] for name, info in data["inputs"].items()} == {"tmean": "series", "rs": "series"}
    assert data["outputs"] == {"y": {"from": "return"}}
    output = tmp_path / "fmugen.toml"
    with isolated_imports():
        init(f"{tmp_path / 'evap.py'}:pet", output=output, probe=True)
    config = load_config(output)
    assert "pandas" in " ".join(config.model["requirements"])


def test_results_that_refuse_float_are_read_through_their_properties(tmp_path):
    (tmp_path / "errors.py").write_text(textwrap.dedent('''
        class Affine:   # like uncertainties' AffineScalarFunc: __float__ raises
            __slots__ = ("_value", "_sigma")

            def __init__(self, value, sigma):
                self._value, self._sigma = value, sigma

            def __float__(self):
                raise TypeError("can't convert an affine function to float; use x.nominal_value")

            @property
            def nominal_value(self):
                return self._value

            n = nominal_value

            @property
            def std_dev(self):
                return self._sigma

        def scaled(x=1.0, sigma=0.1):
            return Affine(2 * x, 2 * sigma)
    '''))
    data, _ = infer(f"{tmp_path / 'errors.py'}:scaled")
    assert data["outputs"] == {"nominal_value": {}, "std_dev": {}}   # from = "return:<name>", the default


def test_time_arguments_are_recognized_by_common_names(tmp_path):
    (tmp_path / "clocks.py").write_text(textwrap.dedent('''
        class Environment:   # like simpy's Environment.run(until)
            def __init__(self):
                self.now = 0.0

            def run(self, until=None):
                if until is None:
                    return None
                if until <= self.now:
                    raise ValueError(f"until ({until}) must be greater than the current simulation time")
                self.now = until
                return self.now

        class Solver:   # like tellurium's oneStep(currentTime, stepSize)
            def oneStep(self, currentTime, stepSize):
                return currentTime + stepSize
    '''))
    data, comments = infer(f"{tmp_path / 'clocks.py'}:Environment", call="run")
    assert data["time"] == {"until": "end_time"} and data["outputs"]["y"] == {"from": "return"}
    data, _ = infer(f"{tmp_path / 'clocks.py'}:Environment", call="run", probe=False)
    assert "can also return None" in render_toml(data, _)
    data, _ = infer(f"{tmp_path / 'clocks.py'}:Solver", call="oneStep")
    assert data["time"] == {"currentTime": "time", "stepSize": "step_size"}
    # given by --start, it is a fixed value, with a hint
    data, comments = infer(f"{tmp_path / 'clocks.py'}:Environment", call="run", starts={"until": 5.0})
    assert data["inputs"]["until"] == {"start": 5.0}
    assert '[time] until = "end_time"' in comments[("inputs", "until")]


def test_requirements_are_the_installed_distributions_the_model_uses(tmp_path):
    import importlib.metadata
    (tmp_path / "helper.py").write_text("SCALE = 2.0\n")
    (tmp_path / "uses.py").write_text(textwrap.dedent('''
        import json
        import pytest   # a third-party package, as numpy would be
        from helper import SCALE

        def f(x=1.0):
            return {"y": SCALE * x}
    '''))
    data, comments = infer(tmp_path / "uses.py", probe=False)
    assert data["model"]["requirements"] == [f"pytest=={importlib.metadata.version('pytest')}"]
    assert "requirements = [" in render_toml(data, comments)


def test_requirements_skip_a_venv_next_to_the_model(tmp_path):
    import importlib
    import sys
    from fmugen.interface import _local_sources
    # like cftoc/.venv beside cftoc/CFTOCSolverV2.py: its packages aren't the model's imports,
    # and one of them imports __main__, which has no spec under the fmugen console script
    import importlib.metadata
    folder = tmp_path / "cftoc"
    (folder / ".venv" / "lib" / "site-packages").mkdir(parents=True)
    (folder / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (folder / ".venv" / "lib" / "site-packages" / "tool.py").write_text("import __main__\nimport numpy\n")
    (folder / "solver.py").write_text(textwrap.dedent('''
        import pytest

        def f(x=1.0):
            return {"y": 2 * x}
    '''))
    data, _ = infer(folder / "solver.py", config_dir=tmp_path, probe=False)
    assert data["model"]["sources"] == ["cftoc"]
    assert data["model"]["requirements"] == [f"pytest=={importlib.metadata.version('pytest')}"]
    # with the config beside the model, a module imported from the venv isn't a source
    (folder / ".venv" / "lib" / "venvtool.py").write_text("X = 1\n")
    with isolated_imports():
        sys.path.insert(0, str(folder / ".venv" / "lib"))
        try:
            importlib.import_module("venvtool")
            sources = _local_sources((folder / "solver.py").resolve(), folder.resolve())
        finally:
            sys.path.pop(0)
    assert sources == []


def test_result_keys_that_are_not_names_get_valid_ones(tmp_path):
    (tmp_path / "decay.py").write_text(textwrap.dedent('''
        def decay(t=1.0):   # like radioactivedecay's Inventory.contents
            return {"contents": {"C-14": 1.0 - t / 10, "N-14": t / 10}, "2x": 2 * t, "class": 0.5}
    '''))
    data, _ = infer(tmp_path / "decay.py")
    assert data["outputs"] == {"contents.C_14": {"from": "return:contents.C-14"},
                               "contents.N_14": {"from": "return:contents.N-14"},
                               "_2x": {"from": "return:2x"}, "_class": {"from": "return:class"}}


def test_outputs_are_read_from_the_object_the_step_returns(tmp_path):
    (tmp_path / "fluid.py").write_text(textwrap.dedent('''
        class Fluid:   # like pyfluids' Fluid(...).with_state(...): a new object, read through properties
            def __init__(self, name="Water"):
                self.__name, self.__temperature = name, None

            def with_state(self, temperature=20.0):
                fluid = Fluid(self.__name)
                fluid.__temperature = temperature
                return fluid

            @property
            def temperature(self) -> float:
                if self.__temperature is None:
                    raise ValueError("Invalid or not defined state!")
                return self.__temperature

            @property
            def density(self) -> float:
                return 1000.0 - 0.2 * self.temperature
    '''))
    data, _ = infer(f"{tmp_path / 'fluid.py'}:Fluid", call="with_state", starts={"temperature": 20.0})
    assert data["inputs"] == {"temperature": {"start": 20.0}}   # echoed back: not an output too
    assert data["outputs"] == {"density": {"from": "return:density"}}   # not the original object's raising ones


@pytest.mark.parametrize("previous, following", [("x_old", "x_new"), ("last_x", "next_x"), ("prev_x", "x"),
                                                 ("x_last", "x_next")])
def test_state_naming_conventions(tmp_path, previous, following):
    (tmp_path / "m.py").write_text(f"def step(u: float = 1.0, {previous}: float = 0.0):\n"
                                   f"    return {{'{following}': {previous} + u}}\n")
    data, _ = infer(tmp_path / "m.py", probe=False)
    assert data["states"] == {previous: {"start": 0.0, "next": f"return:{following}"}}


def test_a_file_that_only_re_exports_its_model(tmp_path):
    (tmp_path / "plant_core.py").write_text("def model(x: float = 1.0):\n    return {'y': 2 * x}\n\n"
                                            "def helper():\n    pass\n")
    (tmp_path / "plant.py").write_text("from plant_core import model, helper\n__all__ = ['model']\n")
    data, _ = infer(tmp_path / "plant.py", probe=False)
    assert data["model"]["entry"] == "plant.py:model"
    assert data["model"]["sources"] == ["plant_core.py"]


def test_probe_timeout_stops_the_process(tmp_path):
    import subprocess
    (tmp_path / "slow.py").write_text("import time\n\ndef slow(x: float = 1.0):\n    time.sleep(60)\n"
                                      "    return {'y': x}\n")
    result = subprocess.run([sys.executable, "-m", "fmugen", "init", str(tmp_path / "slow.py"), "--probe",
                             "--probe-timeout", "2"], capture_output=True, text=True, timeout=50)
    assert result.returncode == 2 and "did not finish in 2 s" in result.stderr


def test_model_files_named_like_standard_library_modules_are_refused(tmp_path):
    from fmugen.builder import build, init
    from fmugen.config import InterfaceError
    (tmp_path / "queue.py").write_text("def step(x: float = 1.0):\n    return {'y': x}\n")
    with pytest.raises(InterfaceError, match="queue.*standard library"):
        init(tmp_path / "queue.py")
    # a helper next to the entry file, copied with sources
    (tmp_path / "plant.py").write_text("from queue import step\n\ndef plant(x: float = 1.0):\n    return step(x)\n")
    (tmp_path / "fmugen.toml").write_text('[model]\nentry = "plant.py:plant"\nsources = ["queue.py"]\n'
                                          '[inputs]\nx = { start = 1.0 }\n[outputs]\ny = {}\n')
    with pytest.raises(InterfaceError, match="queue.*standard library"):
        build(tmp_path / "fmugen.toml", tmp_path / "out", output_format="folder")


def test_ufunc_arguments_go_by_position(tmp_path):
    # like numba's DUFunc (psychrolib with numba installed): named parameters, but no keyword arguments
    (tmp_path / "vec.py").write_text(textwrap.dedent("""
        import inspect

        class DUFunc:
            nin = 2

            def __init__(self, fn):
                self.fn, self.__signature__ = fn, inspect.signature(fn)

            def __call__(self, *args, **kwargs):
                if kwargs:
                    raise TypeError("unexpected keyword arguments to ufunc")
                return self.fn(*args)

        DUFunc.__module__ = "numba.np.ufunc.dufunc"

        def _ratio(dry=25.0, rh=0.5):
            return dry * rh

        ratio = DUFunc(_ratio)
    """))
    data, _ = infer(f"{tmp_path / 'vec.py'}:ratio", probe=False)
    assert data["inputs"]["dry"]["to"] == "pos:0" and data["inputs"]["rh"]["to"] == "pos:1"


def test_star_args_kwargs_constructor_gets_keywords_when_the_probe_needs_them(tmp_path):
    # like pyquaternion's Quaternion(*args, **kwargs): Quaternion(axis=..., angle=...), not by position
    (tmp_path / "rot.py").write_text(textwrap.dedent("""
        class Turn:
            def __init__(self, *args, **kwargs):
                if args:
                    raise ValueError("expected 4 values")
                self.angle = kwargs["angle"] * kwargs["scale"]

            def rotate(self, x=1.0):
                return x * self.angle
    """))
    data, _ = infer(f"{tmp_path / 'rot.py'}:Turn", call="rotate", starts={"angle": 0.5, "scale": 2.0, "x": 3.0})
    assert data["parameters"]["angle"] == {"start": 0.5} and data["parameters"]["scale"] == {"start": 2.0}
    assert "x" in data["inputs"]


@pytest.mark.parametrize("probe", [False, True])
def test_method_without_self_is_called_on_the_class(tmp_path, probe):
    # like Model.building(df, ...): a plain function in the class body, called on the class
    (tmp_path / "heat.py").write_text(textwrap.dedent("""
        class Model:
            def building(load=2.0, gain=3.0):
                return load * gain
    """))
    data, comments = infer(f"{tmp_path / 'heat.py'}:Model", call="building", probe=probe)
    assert data["model"]["call_on_class"] is True and "has no self" in comments[("model", "call_on_class")]
    assert set(data["inputs"]) == {"load", "gain"}   # load isn't dropped as if it were self


def test_normal_method_is_not_called_on_the_class(tmp_path):
    (tmp_path / "heat.py").write_text(textwrap.dedent("""
        class Model:
            def building(self, load=2.0):
                return load
    """))
    data, _ = infer(f"{tmp_path / 'heat.py'}:Model", call="building")
    assert "call_on_class" not in data["model"]


@pytest.mark.parametrize("probe", [False, True])
def test_function_attached_to_a_class_is_a_method(tmp_path, probe):
    # like pypsa's Network.pf = pf, written as pf(n, ...): called on the object, not on the class
    (tmp_path / "net.py").write_text(textwrap.dedent("""
        def pf(n, scale=1.0):
            n.flow = n.load * scale
            return n.flow

        class Network:
            def __init__(self, load=2.0):
                self.load = load
            pf = pf
    """))
    data, _ = infer(f"{tmp_path / 'net.py'}:Network", call="pf", probe=probe)
    assert "call_on_class" not in data["model"]
    assert set(data["inputs"]) == {"scale"}


def test_method_annotated_with_its_class_is_a_method(tmp_path):
    (tmp_path / "net.py").write_text(textwrap.dedent("""
        class Network:
            def __init__(self, load=2.0):
                self.load = load

            def pf(n: "Network", scale=1.0):
                return n.load * scale
    """))
    data, _ = infer(f"{tmp_path / 'net.py'}:Network", call="pf")
    assert "call_on_class" not in data["model"]


@pytest.mark.parametrize("signature, bound", [
    ("h: int = 4", False), ("h=None", False), ("h=4", False), ("h: 'int | None' = None", False),
    ("h=0.1", True), ("h: float = 0.1", True),
])
def test_h_is_a_step_size_unless_it_is_a_count(tmp_path, signature, bound):
    # neuralforecast's predict(h=None) is the forecast horizon; an ODE solver's h is the step size
    (tmp_path / "m.py").write_text(f"def f(x=1.0, {signature}):\n    return x\n")
    data, _ = infer(f"{tmp_path / 'm.py'}:f", probe=False)
    assert (data.get("time", {}).get("h") == "step_size") is bound


def test_probe_reads_getters_when_the_step_returns_nothing(tmp_path):
    # like floris' run() then get_farm_power(), vqf's getQuat6D()
    (tmp_path / "farm.py").write_text(textwrap.dedent("""
        class Farm:
            def __init__(self, speed=8.0):
                self.speed, self._power = speed, None

            def run(self):
                self._power = 100.0 * self.speed

            def get_power(self):
                if self._power is None:
                    raise RuntimeError("run() first")
                return self._power

            def getQuat(self):
                return [1.0, 0.0, 0.0, 0.0]

            def get_scaled(self, factor):   # needs an argument: not a getter to probe
                return self._power * factor

            def get_label(self):
                return {"name": "farm"}     # not an FMI value
    """))
    data, comments = infer(f"{tmp_path / 'farm.py'}:Farm", call="run", fmi_version=3)
    outputs = data["outputs"]
    assert outputs["power"] == {"from": "get:get_power"}
    assert outputs["quat"]["from"] == "get:getQuat" and outputs["quat"]["dimensions"] == [4]
    assert "scaled" not in outputs and "label" not in outputs and "get_label" not in outputs
    assert "init_call" not in data["model"]   # the FMU calls the model itself when a getter raises at initialization


def test_probe_skips_getters_when_the_step_returns_a_result(tmp_path):
    (tmp_path / "m.py").write_text(textwrap.dedent("""
        class M:
            def step(self, x=1.0):
                return x * 2

            def get_debug(self):
                return 3.0
    """))
    data, _ = infer(f"{tmp_path / 'm.py'}:M", call="step")
    assert all(not str(v.get("from", "")).startswith("get:") for v in data["outputs"].values())


def test_probe_calls_the_model_at_initialization_when_properties_fail_before_it(tmp_path):
    # like pydmd's eigs before fit()
    (tmp_path / "dmd.py").write_text(textwrap.dedent("""
        class DMD:
            def __init__(self, rank=2):
                self.rank, self._eigs = rank, None

            def fit(self, x=1.0):
                self._eigs = [x, x / 2]
                return self

            @property
            def eigs(self):
                if self._eigs is None:
                    raise ValueError("fit() first")
                return self._eigs[0]
    """))
    data, comments = infer(f"{tmp_path / 'dmd.py'}:DMD", call="fit")
    assert "eigs" in data["outputs"]
    assert data["model"]["init_call"] is True and "eigs" in comments[("model", "init_call")]


def test_probe_leaves_init_call_for_properties_that_are_only_missing(tmp_path):
    (tmp_path / "m.py").write_text(textwrap.dedent("""
        class M:
            def step(self, x=1.0):
                self._y = x

            @property
            def y(self):
                return self._y   # AttributeError before the step: the FMU skips it at initialization
    """))
    data, _ = infer(f"{tmp_path / 'm.py'}:M", call="step")
    assert "y" in data["outputs"] and "init_call" not in data["model"]


def test_probe_drops_a_time_argument_that_defaults_to_none(tmp_path):
    # like ssqueezepy's ssq_cwt(x, ..., t=None): t is the signal's time vector, not the clock
    (tmp_path / "cwt.py").write_text(textwrap.dedent("""
        def cwt(x=1.0, t=None):
            if t is not None:
                len(t)   # a vector: a float fails
            return x * 2
    """))
    data, comments = infer(f"{tmp_path / 'cwt.py'}:cwt")
    assert "time" not in data and data["outputs"]
    assert comments[("model", "constants.t")] == {"python": "None"}


def test_probe_keeps_a_none_default_time_argument_that_works(tmp_path):
    (tmp_path / "m.py").write_text("def f(x=1.0, t=None):\n    return x + (t or 0.0)\n")
    data, _ = infer(f"{tmp_path / 'm.py'}:f")
    assert data["time"] == {"t": "time"}
