"""Models assembled on an object: keyed attribute paths, getter outputs, a step function taking the
object, and repeated sub-steps (``[model] substeps`` without ``integrate``)."""
import pytest
from test_runtime import write

from fmugen.config import InterfaceError
from fmugen.templates.fmugen_runtime import get_path, object_call, set_path
from fmugen.templates.model_fmi2 import Fmi2Status

OK = Fmi2Status.ok


class _Plain:
    pass


def test_set_path_sets_attributes_as_before():
    obj = _Plain()
    obj.inner = _Plain()
    set_path(obj, "inner.x", 1.0)              # a new attribute
    set_path(obj, "y", 2.0)
    assert obj.inner.x == 1.0 and obj.y == 2.0


def test_set_path_prefers_an_attribute_over_a_key():
    class Both(dict):
        value = 0
    obj = _Plain()
    obj.d = Both(value=5)
    set_path(obj, "d.value", 7)
    assert obj.d.value == 7 and obj.d["value"] == 5


def test_set_path_sets_keys_and_indexes_where_there_is_no_attribute():
    obj = _Plain()
    obj.table = {"p_set": {"l": 1.0}}
    obj.qpos = [0.0, 0.0]
    set_path(obj, "table.p_set.l", 3.0)       # parents that are keys, then a key
    set_path(obj, "qpos.1", 4.0)
    set_path(obj, "qpos.-1", 5.0)
    assert obj.table["p_set"]["l"] == 3.0 and obj.qpos == [0.0, 5.0]
    with pytest.raises(AttributeError):        # not a key yet: set as an attribute, which fails
        set_path(obj, "table.new", 6.0)
    assert obj.table == {"p_set": {"l": 3.0}}


def test_set_path_still_raises_on_a_missing_parent():
    with pytest.raises(AttributeError):
        set_path(_Plain(), "missing.x", 1.0)


def test_set_path_on_pandas_labels():
    pandas = pytest.importorskip("pandas")
    obj = _Plain()
    obj.loads = pandas.DataFrame({"p_set": [1.0, 2.0]}, index=["l", "m"])
    set_path(obj, "loads.p_set.l", 9.0)
    assert get_path(obj, "loads.p_set.l") == 9.0 and obj.loads.loc["m", "p_set"] == 2.0


def test_object_call_passes_the_object_and_its_attributes():
    obj = _Plain()
    obj.data = [1, 2]
    call = object_call("builtins:sorted(self.data, reverse=True)", obj)
    assert call() == [2, 1]
    obj.data = [3, 5, 4]                         # read on every call
    assert call() == [5, 4, 3]
    obj.limit = 3
    assert object_call("builtins:max(self.limit)", obj)(7) == 7         # bound arguments come after
    assert object_call("builtins:sorted(self.data)", obj)(key=lambda v: -v) == [5, 4, 3]
    assert object_call("builtins:id(self)", obj)() == id(obj)


STEPPER = '''
    class Filter:
        """Integrates its input with a fixed internal step, like an IMU filter."""
        def __init__(self, gain=1.0):
            self.gain, self.x, self.calls, self.times = gain, 0.0, 0, []
            self.state = {"q": [1.0, 0.0]}

        def update(self, u=0.0, t=0.0, dt=0.0):
            self.x += self.gain * u * dt
            self.calls += 1
            self.times.append(t)

        def quaternion(self):
            return [self.x, -self.x]

    def advance(f, scale, u=0.0):
        f.x += scale * u
        f.calls += 1
'''


def test_substeps_without_integrate_repeat_the_step(tmp_path, make_fmu, adapter):
    write(tmp_path, "filt.py", STEPPER, '''
        [model]
        entry = "filt.py:Filter"
        call = "update"
        substeps = 4
        [time]
        t = "time"
        dt = "step_size"
        [inputs]
        u = { start = 2.0 }
        [outputs]
        x = {}
        calls = { type = "Integer" }
        q = { from = "get:quaternion", dimensions = [2] }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK, fmu.logs
    assert fmu.get("x") == pytest.approx(2.0)          # 4 runs of 0.25 s
    assert fmu.engine.obj.calls == 4
    assert fmu.engine.obj.times == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert fmu.engine.values["q"] == pytest.approx([2.0, -2.0])


def test_without_substeps_the_step_runs_once(tmp_path, make_fmu, adapter):
    write(tmp_path, "filt.py", STEPPER, '''
        [model]
        entry = "filt.py:Filter"
        call = "update"
        [time]
        dt = "step_size"
        [inputs]
        u = { start = 2.0 }
        [outputs]
        x = {}
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert fmu.engine.obj.calls == 1 and fmu.get("x") == pytest.approx(2.0)
    assert "repeat" not in fmu.engine.interface


def test_step_call_is_a_function_taking_the_object(tmp_path, make_fmu, adapter):
    write(tmp_path, "filt.py", STEPPER, '''
        [model]
        entry = "filt.py:Filter"
        call = "filt:advance(self, 10.0)"
        setup = ["update(1.0, 0.0, 0.5)"]
        [parameters]
        gain = { start = 2.0 }
        [inputs]
        u = { start = 0.5, to = "arg:u" }
        g = { start = 3.0, to = "attr:gain" }
        q0 = { start = 1.0, to = "attr:state.q.0" }
        [outputs]
        x = {}
        q = { from = "attr:state.q.0" }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    start = fmu.get("x")
    assert start == pytest.approx(1.0)                  # setup: gain 2 * 1.0 * 0.5
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK, fmu.logs
    assert fmu.get("x") == pytest.approx(start + 5.0)
    assert fmu.engine.obj.gain == 3.0                   # attr: still set as an attribute
    assert fmu.get("q") == 1.0


@pytest.mark.parametrize("model, message", [
    ('entry = "filt.py:Filter"\ncall = "filt:advance(self"', "missing"),
    ('entry = "filt.py:Filter"\ncall = "filt:nothing(self)"', "cannot import"),
    ('entry = "filt.py:Filter"\ncall = "filt:advance(self, 1.0)"\ncall_on_class = true', "exclude"),
    ('entry = "filt.py:Filter"\ncall = "update"\nsubsteps = 0', "positive integer"),
    ('entry = "filt.py:Filter"\ncall = false\nsubsteps = 2\n[clocks.c]\ninterval = 1.0\ncall = "update"', "exclude"),
])
def test_invalid_object_steps(tmp_path, make_fmu, model, message):
    write(tmp_path, "filt.py", STEPPER, f'''
        [model]
        fmi_version = 3
        {model.replace(chr(10), chr(10) + "        ")}
        [outputs]
        x = {{}}
    ''')
    with pytest.raises(InterfaceError, match=message):
        make_fmu(tmp_path)


def test_get_needs_a_class(tmp_path, make_fmu):
    write(tmp_path, "m.py", '''
        def f(x=1.0):
            return x
    ''', '''
        [model]
        entry = "m.py:f"
        [outputs]
        y = { from = "get:quaternion" }
    ''')
    with pytest.raises(InterfaceError, match="return:"):
        make_fmu(tmp_path)


GETTER = '''
    class Farm:
        def __init__(self):
            self.power, self.runs = None, 0

        def run(self):
            self.power, self.runs = 5.0, self.runs + 1

        def get_power(self):
            if self.power is None:
                raise RuntimeError("run() first")
            return self.power
'''


def test_a_getter_that_fails_at_initialization_calls_the_model_once(tmp_path, make_fmu, adapter):
    write(tmp_path, "farm.py", GETTER, '''
        [model]
        entry = "farm.py:Farm"
        call = "run"
        [outputs]
        power = { from = "get:get_power" }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.get("power") == 5.0 and fmu.engine.obj.runs == 1
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK and fmu.engine.obj.runs == 2


def test_a_getter_that_works_at_initialization_doesnt_call_the_model(tmp_path, make_fmu, adapter):
    write(tmp_path, "farm.py", GETTER.replace("None, 0", "1.0, 0"), '''
        [model]
        entry = "farm.py:Farm"
        call = "run"
        [outputs]
        power = { from = "get:get_power" }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.get("power") == 1.0 and fmu.engine.obj.runs == 0
