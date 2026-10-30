"""The generated adapter, driven through its FMI 2 methods without UniFMU."""
import textwrap
from pathlib import Path

import pytest

from fmugen.builder import init, isolated_imports
from fmugen.config import InterfaceError
from fmugen.templates.model_fmi2 import Fmi2Status

OK, ERROR = Fmi2Status.ok, Fmi2Status.error


def write(tmp_path, name, source, config=None):
    (tmp_path / name).write_text(textwrap.dedent(source))
    if config is not None:
        (tmp_path / "fmugen.toml").write_text(textwrap.dedent(config))
    return tmp_path


def test_function_with_state_time_and_logging(tmp_path, make_fmu, adapter):
    write(tmp_path, "integrator.py", '''
        import logging
        log = logging.getLogger("integrator")

        def integrate(u, x_prev=0.0, dt=1.0, t=0.0):
            if u > 10:
                log.warning("large input %s at t=%s", u, t)
            return {"x_next": x_prev + u * dt, "time_seen": t}
    ''', '''
        [model]
        entry = "integrator.py:integrate"
        [time]
        dt = "step_size"
        t = "time"
        [inputs]
        u = { start = 1.0 }
        [states]
        x_prev = { start = 0.0, next = "return:x_next" }
        [outputs]
        x = { from = "return:x_next" }
        time_seen = {}
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.get("x_prev") == 0.0                      # initialization does not advance the state
    for t in (0.0, 0.5, 1.0):
        assert fmu.fmi2DoStep(t, 0.5, False) == OK
    assert fmu.get("x", "time_seen") == [1.5, 1.0]

    assert fmu.set("u", 20.0) == OK
    assert fmu.fmi2DoStep(1.5, 0.5, False) == OK
    status, category, message = fmu.logs[-1]
    assert (status, category) == (Fmi2Status.warning, "logStatusWarning")
    assert "large input 20.0 at t=1.5" in message


def test_exceptions_become_errors_with_traceback(tmp_path, make_fmu, adapter):
    write(tmp_path, "fragile.py", '''
        def fragile(x=1.0):
            return {"y": 1.0 / x}
    ''')
    fmu = adapter(make_fmu(tmp_path / "fragile.py"))
    fmu.initialize()
    assert fmu.set("x", 0.0) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) == ERROR
    assert "ZeroDivisionError" in fmu.logs[-1][2] and "fragile.py" in fmu.logs[-1][2]


def test_failing_computed_constant_is_an_initialization_error(tmp_path, make_fmu, adapter):
    write(tmp_path, "deriv.py", """
        def derivative(fun, x=0.5):
            return fun(x)
    """, """
        [model]
        entry = "deriv.py:derivative"
        [model.constants]
        fun = { call = "math:nosuch" }
        [inputs]
        x = { start = 0.5 }
        [outputs]
        y = { from = "return" }
    """)
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))   # instantiating doesn't compute constants
    assert fmu.fmi2SetupExperiment(0.0, None, None) == OK and fmu.fmi2EnterInitializationMode() == OK
    assert fmu.fmi2ExitInitializationMode() == ERROR
    message = fmu.logs[-1][2]
    assert "constants.fun = 'math:nosuch' failed" in message and "AttributeError" in message


def test_enumeration(tmp_path, make_fmu, adapter):
    write(tmp_path, "valve.py", '''
        import enum

        class Mode(enum.Enum):
            CLOSED = "closed"
            OPEN = "open"

        def valve(flow=1.0, mode=Mode.OPEN):
            return {"out": flow if mode is Mode.OPEN else 0.0, "mode_echo": mode}
    ''')
    fmu = adapter(make_fmu(tmp_path / "valve.py:valve"))
    td = fmu.engine.interface["type_definitions"]["Mode"]
    assert td["items"] == ["CLOSED", "OPEN"]
    fmu.initialize()
    assert fmu.get("out") == 1.0
    assert fmu.fmi2SetInteger([fmu.vr["mode"]], [1]) == OK      # CLOSED
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("out") == 0.0
    assert fmu.fmi2GetInteger([fmu.vr["mode_echo"]]) == (OK, [1])


def test_state_with_cloudpickle_when_pickle_fails(tmp_path, make_fmu, adapter):
    write(tmp_path, "lam.py", '''
        class Scaled:
            def __init__(self, k=2.0):
                self.f = lambda u: k * u            # pickle can't store a lambda; cloudpickle can
                self.total = 0.0

            def step(self, u=1.0):
                self.total += self.f(u)
                return {"total": self.total}
    ''')
    with isolated_imports():
        init(str(tmp_path / "lam.py"), probe=True)
    assert "save_state" not in (tmp_path / "fmugen.toml").read_text()
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    status, state = fmu.fmi2SerializeFmuState()
    assert status == OK and state[:1] == b"C"              # saved with cloudpickle
    assert fmu.fmi2DoStep(1.0, 1.0, False) == OK and fmu.get("total") == 4.0
    assert fmu.fmi2DeserializeFmuState(state) == OK
    assert fmu.fmi2DoStep(1.0, 1.0, False) == OK and fmu.get("total") == 4.0   # rolled back, then redone


def test_state_of_selected_attributes(tmp_path, make_fmu, adapter):
    write(tmp_path, "res.py", '''
        import threading

        class Accumulator:
            def __init__(self):
                self.lock = threading.Lock()        # a fixed resource: can't be pickled at all
                self.total = 0.0

            def step(self, u=1.0):
                with self.lock:
                    self.total += u
                return {"total": self.total}
    ''')
    with isolated_imports():
        _, data = init(str(tmp_path / "res.py"), probe=True)
    assert data["model"]["save_state"] == ["total"]
    text = (tmp_path / "fmugen.toml").read_text()
    assert "kept as they are on restore: lock" in text
    fmu_dir = make_fmu(tmp_path)
    assert 'canGetAndSetFMUstate="true"' in (fmu_dir / "modelDescription.xml").read_text()
    fmu = adapter(fmu_dir)
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK and fmu.get("total") == 1.0
    _status, state = fmu.fmi2SerializeFmuState()
    lock = fmu.engine.obj.lock
    assert fmu.fmi2DoStep(1.0, 1.0, False) == OK and fmu.get("total") == 2.0
    assert fmu.fmi2DeserializeFmuState(state) == OK
    assert fmu.engine.obj.lock is lock                     # the resource is kept, not replaced
    assert fmu.fmi2DoStep(1.0, 1.0, False) == OK and fmu.get("total") == 2.0   # rolled back, then redone


def test_unpicklable_attribute_that_changes_is_flagged(tmp_path):
    # a generator advances on every step: saving only `count` would restore it wrongly, so init
    # names the attribute it leaves out and asks to check it (or set save_state = false)
    write(tmp_path, "gen.py", '''
        class Counter:
            def __init__(self):
                self.it = (i for i in range(1000))   # generators can't be pickled
                self.count = -1

            def step(self):
                self.count = next(self.it)
    ''')
    with isolated_imports():
        _, data = init(str(tmp_path / "gen.py"), probe=True)
    assert data["model"]["save_state"] == ["count"]
    assert "kept as they are on restore: it. Check those don't change" in (tmp_path / "fmugen.toml").read_text()




def test_setup_positional_arguments_and_kind_function(tmp_path, make_fmu, adapter):
    write(tmp_path, "lib.py", '''
        UNITS = None
        SI = "SI"

        def set_units(units):
            global UNITS
            UNITS = units

        def prop(name, value, /, scale=1.0):          # positional-only, like many C extensions
            if UNITS is None:
                raise ValueError("units not set")
            return {"T": value * scale, "P": 2 * value}[name]

        class State:                                   # all the work happens in the constructor
            def __init__(self, **kwargs):
                self.T, self.P = kwargs["T"], kwargs["P"]
                self.rho = self.P / self.T
    ''')
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent('''
        [model]
        entry = "lib.py:prop"
        setup = ["lib:set_units(lib.SI)"]
        [inputs]
        name  = { start = "T", to = "pos:0" }
        value = { start = 3.0, to = "pos:1" }
        scale = { start = 2.0 }
        [outputs]
        y = { from = "return" }
    '''))
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.get("y") == 6.0
    assert fmu.fmi2SetString([fmu.vr["name"]], ["P"]) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == 6.0 and fmu.engine.module.UNITS == "SI"

    (tmp_path / "fmugen.toml").write_text(textwrap.dedent('''
        [model]
        entry = "lib.py:State"
        kind = "function"
        [inputs]
        T = { start = 300.0 }
        P = { start = 600.0 }
        [outputs]
        rho = { from = "return:rho" }
    '''))
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.get("rho") == 2.0


def test_setter_method_inputs_and_property_outputs(tmp_path, make_fmu, adapter):
    write(tmp_path, "lab.py", '''
        class Lab:                                     # a hardware-style API
            def __init__(self):
                self._power, self._temp = 0.0, 20.0

            def heater(self, value):                   # set through a method
                self._power = value

            @property
            def temperature(self):                     # read through a property
                return self._temp

            def update(self, dt):
                self._temp += 0.1 * self._power * dt
    ''')
    from fmugen.builder import init
    _, data = init(tmp_path / "lab.py", tmp_path / "inferred.toml", probe=True)
    assert data["outputs"] == {"temperature": {}}       # the property is found by inference
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent('''
        [model]
        entry = "lab.py:Lab"
        call = "update"
        [time]
        dt = "step_size"
        [inputs]
        power = { start = 10.0, to = "call:heater" }
        [outputs]
        temperature = {}
    '''))
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 2.0, False) == OK
    assert fmu.get("temperature") == 22.0


def test_accumulating_setter_runs_once_per_step(tmp_path, make_fmu, adapter):
    # like PyTCI's give_drug: each call adds a dose, so an extra call is an extra dose
    write(tmp_path, "pump.py", '''
        class Pump:
            def __init__(self):
                self.total, self.calls = 0.0, 0

            def give(self, dose):
                self.total += dose
                self.calls += 1

            def wait(self, dt):
                return self.total
    ''')
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent('''
        [model]
        entry = "pump.py:Pump"
        call = "wait"
        [time]
        dt = "step_size"
        [inputs]
        dose = { start = 2.0, to = "call:give" }
        [outputs]
        total = {}
        calls = { type = "Integer" }
    '''))
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    for k in range(3):
        assert fmu.fmi2DoStep(float(k), 1.0, False) == OK
    assert fmu.get("calls", "total") == [3, 6.0]   # 3 steps, 3 doses (not 4)
    fmu.set("dose", 5.0)
    assert fmu.fmi2DoStep(3.0, 1.0, False) == OK
    assert fmu.get("calls", "total") == [4, 11.0]


def test_setter_changed_after_initialization_reaches_the_first_step(tmp_path, make_fmu, adapter):
    write(tmp_path, "pump.py", '''
        class Pump:
            def __init__(self):
                self.doses = []

            def give(self, dose):
                self.doses.append(dose)

            def wait(self):
                return sum(self.doses)
    ''')
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent('''
        [model]
        entry = "pump.py:Pump"
        call = "wait"
        [inputs]
        dose = { start = 2.0, to = "call:give" }
        [outputs]
        y = { from = "return" }
    '''))
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    fmu.set("dose", 7.0)
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == 9.0   # 2 at initialization, then the new 7


def test_call_on_class_passes_no_instance(tmp_path, make_fmu, adapter):
    write(tmp_path, "heat.py", '''
        class Model:
            def building(load, gain=3.0):
                return load * gain
    ''')
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent('''
        [model]
        entry = "heat.py:Model"
        call = "building"
        call_on_class = true
        [inputs]
        load = { start = 2.0 }
        [outputs]
        y = { from = "return" }
    '''))
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == 6.0


def test_build_does_not_run_the_model(tmp_path, make_fmu):
    # like tclab: constructing the model needs hardware (or a network) that isn't there at build time
    write(tmp_path, "hw.py", '''
        RUNS = []

        class Heater:
            def __init__(self, port="COM3"):
                RUNS.append("constructed")
                raise RuntimeError(f"no device on {port}")

            def step(self, power=0.0):
                RUNS.append("stepped")
                return {"T": 20.0 + power}
    ''', config='''
        [model]
        entry = "hw.py:Heater"
        call = "step"
        save_state = false
        [parameters]
        port = { start = "COM3" }
        [inputs]
        power = { start = 0.0 }
        [outputs]
        T = { from = "return:T" }
    ''')
    fmu_dir = make_fmu(tmp_path)
    assert 'canGetAndSetFMUstate="false"' in (fmu_dir / "modelDescription.xml").read_text()


@pytest.mark.parametrize("text, expected", [("true", True), ("False", False), (" 1 ", True), ("0", False),
                                            ("yes", True), ("off", False)])
def test_boolean_from_strings(text, expected):
    from fmugen.templates.fmugen_runtime import coerce
    assert coerce("Boolean", text) is expected


def test_boolean_from_unknown_string_is_an_error():
    from fmugen.templates.fmugen_runtime import coerce
    with pytest.raises(ValueError):
        coerce("Boolean", "maybe")


def test_number_like_values_are_coerced():
    from decimal import Decimal
    from fractions import Fraction

    from fmugen.templates.fmugen_runtime import coerce, plain_number
    assert coerce("Real", plain_number(Decimal("1.5"))) == 1.5
    assert coerce("Real", plain_number(Fraction(1, 4))) == 0.25


def test_pint_quantities_in_and_out():
    pint = pytest.importorskip("pint")
    from fmugen.templates.fmugen_runtime import converter, plain_number
    q = converter({"convert": "pint", "unit": "kPa"})(100.0)
    assert q == pint.get_application_registry().Quantity(100.0, "kPa")
    assert plain_number(q, "Pa") == pytest.approx(1e5)
    assert plain_number(q) == 100.0


def test_dotted_paths_walk_mappings_sequences_and_attributes():
    from types import SimpleNamespace

    from fmugen.templates.fmugen_runtime import pick
    result = (0, {"rewards": {"speed": 1.5}, "a.b": 2}, SimpleNamespace(state={"T": 3}))
    assert pick(result, "1.rewards.speed") == 1.5
    assert pick(result[1], "a.b") == 2
    assert pick(result, "2.state.T") == 3


def test_nested_outputs_through_an_fmu(tmp_path, make_fmu, adapter):
    (tmp_path / "nested.py").write_text(textwrap.dedent("""
        def f(x: float = 1.0):
            return {"zone": {"T": 2 * x, "air": {"rh": 0.5}}}
    """))
    init(tmp_path / "nested.py", output=tmp_path / "fmugen.toml")
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    fmu.initialize()
    assert fmu.set("x", 3.0) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("zone.T", "zone.air.rh") == [6.0, 0.5]


def test_table_values_are_read_by_label():
    pandas = pytest.importorskip("pandas")
    from fmugen.templates.fmugen_runtime import coerce, pick, plain_number
    frame = pandas.DataFrame({"zenith": [33.3], "azimuth": [120.0]})
    assert coerce("Real", plain_number(pick(frame, "zenith"))) == 33.3
    assert coerce("Real", plain_number(pick(pandas.Series({"a": 1.5}), "a"))) == 1.5


def test_time_argument_with_an_epoch_is_a_datetime(tmp_path, make_fmu, adapter):
    (tmp_path / "clock.py").write_text(textwrap.dedent("""
        def hour(time):
            return {"hour": time.hour + time.minute / 60}
    """))
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent("""
        [model]
        entry = "clock.py:hour"
        [time]
        time = { source = "end_time", epoch = "2026-06-21T06:00:00+00:00" }
        [outputs]
        hour = {}
    """))
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 5400.0, False) == OK
    assert fmu.get("hour") == 7.5


def test_epoch_must_be_a_date_time(tmp_path, make_fmu):
    from fmugen.config import InterfaceError
    (tmp_path / "clock.py").write_text("def hour(time):\n    return time\n")
    (tmp_path / "fmugen.toml").write_text(
        '[model]\nentry = "clock.py:hour"\n[time]\ntime = { epoch = "yesterday" }\n[outputs]\ny = { from = "return" }\n')
    with pytest.raises(InterfaceError, match="ISO 8601"):
        make_fmu(tmp_path / "fmugen.toml")


def test_object_argument_fields_through_an_fmu(tmp_path, make_fmu, adapter):
    (tmp_path / "objs.py").write_text(textwrap.dedent("""
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class Plant:
            gain: float = 2.0
            name: str = "p1"
            notes: tuple = ("kept",)

        @dataclasses.dataclass
        class Weather:
            T: float

        def f(w: Weather, plant: Plant = Plant()):
            assert plant.notes == ("kept",)          # a field that isn't a variable keeps its default
            return {"y": w.T * plant.gain}
    """))
    init(f"{tmp_path / 'objs.py'}:f", output=tmp_path / "fmugen.toml", starts={"w.T": 1.0})
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    fmu.initialize()
    assert fmu.set("w_T", 3.0) == OK and fmu.set("plant_gain", 4.0) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == 12.0


def test_model_reads_data_files_by_relative_path(tmp_path, make_fmu, adapter, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "gain.txt").write_text("3.0")
    (tmp_path / "scaled.py").write_text(textwrap.dedent("""
        GAIN = float(open("data/gain.txt").read())        # read when imported

        def scaled(x: float = 1.0):
            with open("data/gain.txt") as f:               # and on every call
                return {"y": x * float(f.read()), "g": GAIN}
    """))
    init(tmp_path / "scaled.py")
    text = (tmp_path / "fmugen.toml").read_text()
    assert 'sources = ["data/gain.txt"]' in text and 'cwd = "."' in text
    fmu_dir = make_fmu(tmp_path / "fmugen.toml")
    monkeypatch.chdir(tmp_path.parent)                      # the importer runs somewhere else
    fmu = adapter(fmu_dir)
    fmu.initialize()
    assert fmu.set("x", 2.0) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y", "g") == [6.0, 3.0]
    assert Path.cwd() == tmp_path.parent                   # and gets its working directory back


def test_cwd_must_be_a_folder_inside_the_project(tmp_path):
    from fmugen.config import Config, InterfaceError
    with pytest.raises(InterfaceError, match="not a folder"):
        Config({"model": {"entry": "m.py:f", "cwd": "nope"}}, tmp_path)
    with pytest.raises(InterfaceError, match="must be inside"):
        Config({"model": {"entry": "m.py:f", "cwd": ".."}}, tmp_path)


def test_dotted_path_reaches_methods_of_a_dict():
    from fmugen.templates.fmugen_runtime import get_path
    options = {"a": 1}
    assert get_path({"options": options}, "options.update") == options.update
    assert get_path({"keys": 5}, "keys") == 5   # a key wins over a method of the same name


def test_dotted_path_reads_the_last_element():
    from fmugen.templates.fmugen_runtime import get_path

    class Wrapper(list):   # like GEKKO's GK_Value: a list-like around a list, or a single value before a solve
        def __init__(self, value):
            self.inner = value

        def __getitem__(self, i):
            return self.inner[i]

    assert get_path({"x": [1.0, 2.0, 3.0]}, "x.-1") == 3.0 and get_path({"x": (1, 2)}, "x.-2") == 1
    assert get_path({"x": Wrapper([4.0, 5.0])}, "x.-1") == 5.0
    with pytest.raises(IndexError):   # read as missing, as for an attribute not set yet
        get_path({"x": Wrapper(4.0)}, "x.-1")


def test_dotted_path_indexes_any_iterable():
    from fmugen.templates.fmugen_runtime import get_path

    class Space:   # like pymunk's Space: bodies is a dict_keys view
        def __init__(self):
            self._bodies = {"a": 1, "b": 2}

        @property
        def bodies(self):
            return self._bodies.keys()

    assert get_path(Space(), "bodies.0") == "a" and get_path(Space(), "bodies.-1") == "b"
    assert get_path({"s": frozenset([7])}, "s.0") == 7
    with pytest.raises(IndexError):
        get_path(Space(), "bodies.2")
    with pytest.raises(AttributeError):   # strings, mappings and iterators aren't indexed this way
        get_path({"s": "abc"}, "s.0")
    with pytest.raises(KeyError):   # a mapping's part is a key
        get_path({"d": {"x": 1}}, "d.0")
    with pytest.raises(AttributeError):
        get_path({"g": iter([1, 2])}, "g.0")


def test_dotted_path_reads_items_of_objects_with_getitem():
    from fmugen.templates.fmugen_runtime import get_path

    class Solution:   # like pybamm's Solution: not a mapping, variables built on demand by key
        t = [0.0, 100.0]

        def __getitem__(self, key):
            if key != "Voltage [V]":
                raise KeyError(key)
            return {"entries": [4.1, 4.0]}

    assert get_path(Solution(), "Voltage [V].entries.-1") == 4.0
    assert get_path(Solution(), "t.-1") == 100.0   # an attribute comes first
    with pytest.raises(AttributeError):   # no such key: the attribute error
        get_path(Solution(), "Current [A]")


def test_series_convert_dates_the_value_at_the_fmu_time(tmp_path):
    pd = pytest.importorskip("pandas")
    from fmugen.config import Config, normalize
    from fmugen.templates.fmugen_runtime import converter
    series = converter({"convert": "series", "epoch": "2026-06-21T00:00:00"}, lambda: 86400.0)(3.5)
    assert list(series) == [3.5] and series.index[0] == pd.Timestamp("2026-06-22")
    (tmp_path / "m.py").write_text("def f(x=1.0):\n    return x\n")

    def f(x=1.0):
        return x

    for info, error in (({"start": 1.0, "convert": "series"}, "needs an epoch"),
                        ({"start": 1.0, "convert": "series", "epoch": "noon"}, "not an ISO 8601"),
                        ({"start": 1.0, "epoch": "2026-06-21"}, "only applies with convert"),
                        ({"start": [1.0, 2.0], "dimensions": [2], "convert": "series", "epoch": "2026-06-21"},
                         "single values")):
        with pytest.raises(InterfaceError, match=error):
            normalize(Config({"model": {"entry": "m.py:f"}, "inputs": {"x": info}}, tmp_path), f, "m", [str(tmp_path)])


def test_generator_results_become_tuples():
    from fmugen.templates.fmugen_runtime import results_of
    assert results_of(x for x in (1, 2)) == (1, 2)
    values = [1, 2]
    assert results_of(values) is values   # anything else is unchanged
    it = iter(values)
    assert results_of(it) is it


def test_dotted_path_indexes_numpy_arrays():
    np = pytest.importorskip("numpy")
    from fmugen.templates.fmugen_runtime import get_path
    model = {"t_core": np.array([20.0, 21.0, 22.0]), "grid": np.array([[1.0, 2.0], [3.0, 4.0]])}
    assert get_path(model, "t_core.0") == 20.0 and get_path(model, "t_core.-1") == 22.0
    assert get_path(model, "grid.1.0") == 3.0


COUNTER = """
CALLS = 0
SCALE = 2.0

def tick(x: float = 1.0):
    global CALLS
    CALLS += 1
    return {"y": x * SCALE, "calls": CALLS}
"""


def test_module_globals_are_reset_and_saved(tmp_path, make_fmu, adapter):
    (tmp_path / "counter.py").write_text(COUNTER)
    _, data = init(tmp_path / "counter.py")
    assert data["model"]["globals"] == ["counter:CALLS"]          # SCALE is never rebound
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    fmu.initialize()
    for t in range(3):
        assert fmu.fmi2DoStep(float(t), 1.0, False) == OK
    calls = fmu.get("calls")
    _status, state = fmu.fmi2SerializeFmuState()
    assert fmu.fmi2DoStep(3.0, 1.0, False) == OK and fmu.get("calls") == calls + 1
    assert fmu.fmi2DeserializeFmuState(state) == OK
    assert fmu.fmi2DoStep(3.0, 1.0, False) == OK and fmu.get("calls") == calls + 1   # rolled back first
    assert fmu.fmi2Reset() == OK
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("calls") == calls - 2   # counted from the import-time value again


def test_globals_must_exist(tmp_path, make_fmu):
    from fmugen.config import InterfaceError
    (tmp_path / "counter.py").write_text(COUNTER)
    (tmp_path / "fmugen.toml").write_text(
        '[model]\nentry = "counter.py:tick"\nglobals = ["counter:NOPE"]\n[outputs]\ny = {}\n')
    with pytest.raises(InterfaceError, match="NOPE"):
        make_fmu(tmp_path / "fmugen.toml")


def test_rollback_repeats_random_draws(tmp_path, make_fmu, adapter):
    (tmp_path / "noisy.py").write_text(textwrap.dedent("""
        import random
        import numpy as np

        def noisy(x: float = 0.0):
            return {"a": x + random.random(), "b": x + float(np.random.normal())}
    """))
    init(tmp_path / "noisy.py")
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    fmu.initialize()
    _, state = fmu.fmi2SerializeFmuState()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    first = fmu.get("a", "b")
    assert fmu.fmi2DeserializeFmuState(state) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("a", "b") == first


def test_star_args_constructor_gets_positional_parameters(tmp_path, make_fmu, adapter):
    write(tmp_path, "affine.py", """
        class Affine:
            def __init__(self, *args, **kwargs):   # e.g. control.StateSpace(A, B, C, D)
                self.a, self.b = args

            def step(self, x=1.0):
                return self.a * x + self.b
    """)
    output, data = init(tmp_path / "affine.py", starts={"a": 2.0, "b": 0.5})
    assert data["parameters"] == {"a": {"start": 2.0, "to": "pos:0"}, "b": {"start": 0.5, "to": "pos:1"}}
    fmu = adapter(make_fmu(output))
    assert fmu.set("a", 3.0) == OK
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == 3.5


def test_star_args_constructor_and_method_split_by_the_probe(tmp_path, make_fmu, adapter):
    write(tmp_path, "normal.py", """
        import math

        class Normal:   # like SWIG's openturns.Normal(mu, sigma).computePDF(x)
            def __init__(self, *args):
                self.mu, self.sigma = args

            def pdf(self, *args):
                (x,) = args
                return math.exp(-((x - self.mu) / self.sigma) ** 2 / 2) / (self.sigma * math.sqrt(2 * math.pi))
    """)
    starts = {"mu": 0.0, "sigma": 1.0, "x": 0.5}
    output, data = init(f"{tmp_path / 'normal.py'}:Normal", call="pdf", starts=starts, probe=True)
    plain = {k: {kk: vv for kk, vv in v.items() if kk not in ("variability", "attr")}   # sigma: tunable, via self.sigma
             for k, v in data["parameters"].items()}
    assert plain == {"mu": {"start": 0.0, "to": "pos:0"}, "sigma": {"start": 1.0, "to": "pos:1"}}
    assert data["inputs"] == {"x": {"start": 0.5, "to": "pos:0"}}
    fmu = adapter(make_fmu(output))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == pytest.approx(0.3520653267642995)


def test_computed_constant_for_star_args(tmp_path, make_fmu, adapter):
    write(tmp_path, "affine.py", """
        class Affine:
            def __init__(self, *args):   # e.g. nashpy.Game(A)
                self.a, self.b = args

            def step(self, x=1.0):
                return self.a * x + self.b
    """)
    output, data = init(tmp_path / "affine.py", starts={"a": {"call": "math:sqrt(4.0)"}, "b": 0.5})
    assert data["model"]["constants"] == {"0": {"call": "math:sqrt(4.0)"}}
    assert data["parameters"] == {"b": {"start": 0.5, "to": "pos:1"}}
    assert '0 = { call = "math:sqrt(4.0)" }' in output.read_text()
    fmu = adapter(make_fmu(output))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == 2.5


def test_namedtuple_argument_fields(tmp_path, make_fmu, adapter):
    write(tmp_path, "meal.py", """
        from typing import NamedTuple

        class Action(NamedTuple):
            CHO: float
            insulin: float = 0.0

        class Patient:
            def __init__(self):
                self.glucose = 100.0

            def step(self, action: Action):
                self.glucose += action.CHO - 10 * action.insulin
                return self.glucose
    """)
    output, data = init(f"{tmp_path / 'meal.py'}:Patient", starts={"action.CHO": 1.0})
    assert data["inputs"]["action_CHO"] == {"start": 1.0, "to": "arg:action.CHO"}
    fmu = adapter(make_fmu(output))
    fmu.initialize()
    assert fmu.set("action_insulin", 0.5) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == 96.0


def test_fields_of_an_argument_of_unknown_class_are_an_error(tmp_path):
    from fmugen.config import InterfaceError
    write(tmp_path, "kw.py", """
        class Patient:
            def __init__(self, **kwargs):
                pass

            def step(self, action):
                return action.CHO
    """)
    with pytest.raises(InterfaceError, match="cannot tell the class of action"):
        init(tmp_path / "kw.py", starts={"action.CHO": 1.0})


def test_date_time_argument_gets_an_epoch(tmp_path, make_fmu, adapter):
    write(tmp_path, "sun.py", """
        import datetime

        def altitude(lat=40.0, when: datetime.datetime | None = None):
            return {"hour": when.hour + when.minute / 60}
    """)
    output, data = init(tmp_path / "sun.py")
    assert "# a date-time argument: the epoch" in output.read_text()   # annotated: not a guess
    assert data["time"]["when"]["source"] == "end_time"
    assert data["time"]["when"]["epoch"].endswith("T00:00:00")
    fmu = adapter(make_fmu(output))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 5400.0, False) == OK
    assert fmu.get("hour") == 1.5


def test_string_in_a_real_output_says_to_set_its_type(tmp_path, make_fmu, adapter):
    write(tmp_path, "chem.py", """
        def chemical(T=300.0):
            return {"CAS": "7732-18-5"}
    """, config="""
        [model]
        entry = "chem.py:chemical"
        [outputs]
        CAS = {}
    """)
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    assert fmu.fmi2SetupExperiment(0.0, None, None) == OK and fmu.fmi2EnterInitializationMode() == OK
    assert fmu.fmi2ExitInitializationMode() == ERROR
    assert 'CAS is a string' in fmu.logs[-1][2] and 'set type = "String"' in fmu.logs[-1][2]


def test_no_outputs_is_a_warning_and_a_build_note(tmp_path, make_fmu, capsys):
    write(tmp_path, "quiet.py", """
        class Quiet:
            def step(self, x=1.0):
                self.update(x)

            def update(self, x):
                pass
    """)
    output, _ = init(tmp_path / "quiet.py")
    assert "no outputs found" in capsys.readouterr().err
    assert "no outputs found" in output.read_text()
    make_fmu(output)
    assert "the FMU has no outputs" in capsys.readouterr().out


def test_call_reaches_lazily_exported_submodules(tmp_path, monkeypatch):
    from fmugen.templates.fmugen_runtime import call_reference
    package = tmp_path / "lazypkg"
    (package / "models").mkdir(parents=True)
    (package / "__init__.py").write_text(textwrap.dedent("""
        def __getattr__(name):   # as pybamm exports lithium_ion
            if name == "family":
                from .models import family
                return family
            raise AttributeError(name)
    """))
    (package / "models" / "__init__.py").write_text("")
    (package / "models" / "family.py").write_text("def SPM(n=1):\n    return ('SPM', n)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with isolated_imports():
        assert call_reference("lazypkg.family:SPM(n=2)") == ("SPM", 2)
        with pytest.raises(ModuleNotFoundError):
            call_reference("lazypkg.missing:SPM()")


def test_misspelled_field_and_unknown_argument_are_named(tmp_path):
    from fmugen.config import InterfaceError
    write(tmp_path, "objs.py", """
        import dataclasses

        @dataclasses.dataclass
        class Weather:
            T: float

        def f(w: Weather):
            return {"y": w.T}
    """)
    with pytest.raises(InterfaceError, match=r"w \(Weather\) has no field 'Tx'"):
        init(f"{tmp_path / 'objs.py'}:f", starts={"w.Tx": 1.0})
    with pytest.raises(InterfaceError, match=r"not arguments of the model: \['v.T'\]"):
        init(f"{tmp_path / 'objs.py'}:f", starts={"w.T": 1.0, "v.T": 1.0})


def test_call_start_for_the_step_method_copy_of_a_shared_name(tmp_path):
    # x0 in both: the method's one is given as solve_x0, and a call: start keeps the argument's name
    write(tmp_path, "ctrl.py", """
        class Ctrl:
            def __init__(self, x0):
                self.x0 = x0

            def solve(self, x0):
                return {"y": float(sum(x0)) + self.x0[0]}
    """)
    _, data = init(f"{tmp_path / 'ctrl.py'}:Ctrl", call="solve", probe=True,
                   starts={"x0": {"call": "builtins:list([1.0])"}, "solve_x0": {"call": "builtins:list([2.0, 3.0])"}})
    assert data["model"]["call_constants"]["x0"] == {"call": "builtins:list([2.0, 3.0])"}
    assert data["model"]["constants"]["x0"] == {"call": "builtins:list([1.0])"}


def test_init_start_for_a_callable_without_signature(tmp_path):
    # max is a C built-in: inspect.signature fails, so the --start values go by position, in order
    _, data = init("builtins:max", output=tmp_path / "fmugen.toml", starts={"a": 1.0, "b": 2.0}, probe=True)
    assert data["inputs"] == {"a": {"start": 1.0, "to": "pos:0"}, "b": {"start": 2.0, "to": "pos:1"}}
    assert data["outputs"] == {"y": {"from": "return"}}
    assert "passed by position, in the order given" in (tmp_path / "fmugen.toml").read_text()


def test_probe_retries_a_date_time_in_utc(tmp_path):
    write(tmp_path, "sun.py", """
        def altitude(when):
            if when.tzinfo is None:
                raise ValueError("needs a time zone")
            return {"hour": when.hour}
    """)
    output, data = init(tmp_path / "sun.py", probe=True)
    assert data["time"]["when"]["epoch"].endswith("T00:00:00+00:00")
    text = output.read_text()
    assert "the probe worked with one, not with a number" in text and "epoch in UTC" in text
    assert data["outputs"]["hour"] == {}


def test_warning_names_only_outputs_of_unknown_type(tmp_path, capsys):
    write(tmp_path, "chem.py", """
        def lookup(name):
            return name.upper()

        class Chemical:
            def __init__(self):
                self.level = 1.0

            def step(self, dt=1.0):
                self.level += 0.5 * dt
                self.count = self.level * 2
                self.label = lookup("x")
                return self.level
    """)
    init(f"{tmp_path / 'chem.py'}:Chemical")
    err = capsys.readouterr().err
    assert "warning: label: type not shown by the code" in err
    assert "count" not in err and "y" not in err.split(":")[1]


def test_init_warns_about_a_file_named_like_a_backend_module(tmp_path, capsys):
    write(tmp_path, "model.py", """
        def area(r=1.0):
            return 3.14 * r * r
    """)
    init(f"{tmp_path / 'model.py'}:area")
    assert "warning: model.py clashes with a UniFMU backend module" in capsys.readouterr().err


def test_numeric_strings_still_become_numbers(tmp_path, make_fmu, adapter):
    write(tmp_path, "text.py", """
        def reading(x=1.0):
            return {"v": "2.5"}
    """, config="""
        [model]
        entry = "text.py:reading"
        [outputs]
        v = {}
    """)
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    fmu.initialize()
    assert fmu.get("v") == 2.5


@pytest.mark.parametrize("body, kind, comment", [
    ("return {'y': when.hour}", "time", "the probe worked with one, not with a number"),
    ("return {'y': when * 2.0}", "inputs", "suggests a date-time, but the probe failed with one"),
    ("return {'y': len(str(when))}", "time", "taken as a date-time from its name 'when'"),
])
def test_probe_checks_a_date_time_guessed_from_the_name(tmp_path, body, kind, comment):
    write(tmp_path, "m.py", f"""
        def f(when):
            {body}
    """)
    output, data = init(tmp_path / "m.py", probe=True)
    assert "when" in data.get(kind, {}) and "when" not in data.get("time" if kind == "inputs" else "inputs", {})
    assert comment in output.read_text()


def test_call_arguments_can_be_calls_builtins_and_containers():
    import math
    import operator

    from fmugen.templates.fmugen_runtime import call_reference
    assert call_reference("operator:add(math.sqrt(16.0), -1)") == 3.0
    assert call_reference('builtins:list((float("inf"), math.pi))') == [math.inf, math.pi]
    assert call_reference("builtins:dict(a=math.floor(2.5), b={'k': [math.e]})") == {"a": 2, "b": {"k": [math.e]}}
    # without parentheses: the object itself (a function to pass, an enum member), not called
    assert call_reference("math:sin") is math.sin
    assert call_reference("operator:attrgetter") is operator.attrgetter
    with pytest.raises(ValueError, match="arguments must be"):
        call_reference("builtins:print(lambda: 0)")


def test_init_stops_when_a_computed_constant_fails(tmp_path):
    write(tmp_path, "deriv.py", """
        def derivative(fun, x=0.5):
            return fun(x)
    """)
    from fmugen.config import InterfaceError
    with pytest.raises(InterfaceError, match="constants.fun = call:math:nosuch fails: AttributeError"):
        init(tmp_path / "deriv.py", starts={"fun": {"call": "math:nosuch"}}, probe=True)
    _, data = init(tmp_path / "deriv.py", output="-", starts={"fun": {"call": "math:cos"}}, probe=True)
    assert data["outputs"] == {"y": {"from": "return"}}


def test_array_result_for_a_scalar_output(tmp_path, make_fmu, adapter):
    pytest.importorskip("numpy")
    write(tmp_path, "enu.py", """
        import numpy

        def enu(x=1.0):   # like pymap3d's geodetic2enu: a tuple; or an array of one element
            return (x, 2 * x, 3 * x) if x > 0 else numpy.array([x])
    """, """
        [model]
        entry = "enu.py:enu"
        [inputs]
        x = { start = -1.0 }
        [outputs]
        y = { from = "return" }
    """)
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    fmu.initialize()
    assert fmu.get("y") == -1.0   # one element: its value
    assert fmu.set("x", 1.0) == OK
    assert fmu.fmi2DoStep(0.0, 1.0, False) == ERROR
    assert "y is a tuple of shape [3], not a single value: give it dimensions = [3]" in fmu.logs[-1][2]


def test_a_step_that_changes_nothing_also_runs_at_initialization(tmp_path, make_fmu, adapter):
    write(tmp_path, "option.py", """
        class Call:   # like blackscholes.BlackScholesCall(...).price()
            def __init__(self, S=55.0, K=50.0):
                self.S, self.K = S, K

            def price(self):
                return max(self.S - self.K, 0.0)

        class Counter:
            def __init__(self):
                self.n = 0

            def step(self, u=1.0):
                self.n += 1
                return self.n * u
    """)
    output, data = init(f"{tmp_path / 'option.py'}:Call", call="price", probe=True)
    assert data["model"]["init_call"] is True
    fmu = adapter(make_fmu(output))
    fmu.initialize()
    assert fmu.get("y") == 5.0   # at the start time, before any step
    output, data = init(f"{tmp_path / 'option.py'}:Counter", output=tmp_path / "counter.toml", probe=True)
    assert "init_call" not in data["model"]   # the step changes the object: running it twice would count twice


def test_numbers_are_retried_as_one_element_arrays(tmp_path, make_fmu, adapter):
    pytest.importorskip("numpy")
    write(tmp_path, "heat.py", """
        import numpy as np

        def heat_index(t2_k, rh):   # like thermofeel: numpy.nonzero refuses 0-d arrays
            t2_k = np.asarray(t2_k)
            out = t2_k.copy()
            out[np.nonzero(rh > 40)] += 0.01 * rh
            return out
    """)
    output, data = init(tmp_path / "heat.py", starts={"t2_k": 300.0, "rh": 50.0}, probe=True)
    assert data["inputs"]["t2_k"] == {"start": 300.0, "convert": "numpy:atleast_1d"}
    assert data["outputs"] == {"y": {"from": "return"}}
    fmu = adapter(make_fmu(output))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.get("y") == pytest.approx(300.5)


def test_debug_logging_of_the_backend_is_raised_to_info(tmp_path, make_fmu, adapter, monkeypatch):
    # UniFMU's main.py sets the root logger to DEBUG; numba then writes megabytes while compiling
    # (antropy), which crashes UniFMU 0.14 and hangs the importer
    import logging
    write(tmp_path, "chatty.py", """
        import logging

        def f(x=1.0):
            logging.getLogger("numba.core.ssa").debug("Running <numba.core.ssa._FreshVarHandler>")
            logging.getLogger("chatty").info("step")
            return {"y": x}
    """)
    fmu_dir = make_fmu(tmp_path / "chatty.py")
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.DEBUG)
    monkeypatch.setenv("UNIFMU_DISPATCHER_ENDPOINT", "tcp://127.0.0.1:1")
    fmu = adapter(fmu_dir)
    assert root.level == logging.INFO
    fmu.initialize()
    messages = [entry[2] for entry in fmu.logs]
    assert any("[chatty] step" in m for m in messages) and not any("numba" in m for m in messages)


PID_TUNABLE = """
[model]
entry = "simple_pid:PID"
[model.constants]
sample_time = {{ python = "None" }}
setpoint = 1.0
[time]
dt = "step_size"
[parameters]
Kp = {{ start = 2.0, variability = "tunable", {binding} }}
[inputs]
measurement = {{ start = 0.0, to = "arg:input_" }}
[outputs]
output = {{ from = "return" }}
"""


def test_a_tunable_parameter_passed_by_position_needs_attr(tmp_path, make_fmu):
    (tmp_path / "fmugen.toml").write_text(PID_TUNABLE.format(binding='to = "pos:0"'))
    with pytest.raises(InterfaceError, match="needs attr"):
        make_fmu(tmp_path)


def test_tuning_an_attribute_the_model_does_not_have_is_an_error(tmp_path, make_fmu, adapter):
    # PID keeps Kp in self.Kp: attr = "gain" would only add an attribute that PID never reads
    (tmp_path / "fmugen.toml").write_text(PID_TUNABLE.format(binding='attr = "gain"'))
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.set("Kp", 100.0) == ERROR
    assert "no attribute 'gain'" in fmu.logs[-1][2]
    assert fmu.get("Kp") == 2.0 and not hasattr(fmu.engine.obj, "gain")


def test_a_function_taking_the_step_size_is_not_called_with_zero_at_initialization(tmp_path, make_fmu, adapter):
    write(tmp_path, "rate.py", """
        def rate(x: float = 1.0, dt: float = 1.0):
            return {"y": x / dt}
        """, """
        [model]
        entry = "rate.py:rate"
        [time]
        dt = "step_size"
        [inputs]
        x = { start = 1.0 }
        [outputs]
        y = {}
        """)
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()                                       # no ZeroDivisionError
    assert fmu.get("y") == 0.0                             # not computed until the first step
    assert fmu.fmi2DoStep(0.0, 0.5, False) == OK
    assert fmu.get("y") == 2.0


def test_constants_the_model_changes_are_restored_by_reset_and_rollback(tmp_path, make_fmu, adapter):
    write(tmp_path, "history.py", """
        def record(x: float = 0.0, history=None):
            history.append(x)
            return {"n": len(history)}
        """, """
        [model]
        entry = "history.py:record"
        init_call = false
        [model.constants]
        history = []
        [inputs]
        x = { start = 0.0 }
        [outputs]
        n = {}
        """)
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK and fmu.get("n") == 1
    status, saved = fmu.fmi2SerializeFmuState()
    assert status == OK
    assert fmu.fmi2DoStep(1.0, 1.0, False) == OK and fmu.get("n") == 2
    for _ in range(2):                                     # the same state, restored twice
        assert fmu.fmi2DeserializeFmuState(saved) == OK
        assert fmu.fmi2DoStep(1.0, 1.0, False) == OK and fmu.get("n") == 2
    assert fmu.fmi2Reset() == OK
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK and fmu.get("n") == 1


PID_BINDING = """
[model]
entry = "simple_pid:PID"
[model.constants]
sample_time = {{ python = "None" }}
setpoint = 1.0
{constants}
[parameters]
{parameter}
[inputs]
measurement = {{ start = 0.0, to = "arg:input_" }}
[outputs]
output = {{ from = "return" }}
"""

# (binding kind, extra constants, the tunable parameter, its new value, output of the second step)
# PID(Kp, Ki, Kd, setpoint=1) with measurement 0: error 1, output = Kp + Ki * sum of the dt's
TUNABLE_BINDINGS = [
    ("init", "Ki = 0.0", 'Kp = { start = 2.0, variability = "tunable" }', 4.0, 4.0),
    ("initpos", "Ki = 0.0", 'Kp = { start = 2.0, variability = "tunable", to = "pos:0", attr = "Kp" }', 4.0, 4.0),
    ("attr", "Ki = 0.0", 'Kp = { start = 2.0, variability = "tunable", to = "attr:Kp" }', 4.0, 4.0),
    # set_auto_mode(False): the PID holds its last output (2.0) instead of computing 5.0
    ("call", "Kp = 5.0\nKi = 0.0",
     'auto = { start = true, variability = "tunable", to = "call:set_auto_mode" }', False, None),
    # __call__(input_, dt): Kp = 0, Ki = 1, so the output is the sum of the dt's: 1 + 0.5
    ("arg", "Kp = 0.0\nKi = 1.0", 'step = { start = 1.0, variability = "tunable", to = "arg:dt" }', 0.5, 1.5),
]


@pytest.mark.parametrize("kind, constants, parameter, value, expected", TUNABLE_BINDINGS,
                         ids=[case[0] for case in TUNABLE_BINDINGS])
def test_a_tunable_parameter_changed_between_steps_reaches_the_model(tmp_path, make_fmu, adapter, kind, constants,
                                                                       parameter, value, expected):
    (tmp_path / "fmugen.toml").write_text(PID_BINDING.format(constants=constants, parameter=parameter))
    fmu = adapter(make_fmu(tmp_path))
    name = parameter.split()[0]
    assert fmu.engine.by_name[name]["to"]["kind"] == kind
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK, fmu.logs
    first = fmu.get("output")
    assert fmu.set(name, value) == OK, fmu.logs
    assert fmu.fmi2DoStep(1.0, 1.0, False) == OK, fmu.logs
    if kind == "call":   # the method ran: the PID is in manual mode and repeats its output
        assert fmu.engine.obj.auto_mode is False and fmu.get("output") == first
    else:
        assert fmu.get("output") == pytest.approx(expected)
