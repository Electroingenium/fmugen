"""FMI 3: the generated adapter driven directly, and built FMUs driven through UniFMU."""
import shutil
import textwrap
from ctypes import c_double, c_int, c_uint64

import pytest
from fmpy import read_model_description, simulate_fmu
from fmpy.fmi3 import FMU3Slave, fmi3ValueReference
from fmpy.validation import validate_fmu

from fmugen.builder import build
from fmugen.config import InterfaceError
from fmugen.templates.fmugen_runtime import Status

OK, WARNING, ERROR = Status.ok, Status.warning, Status.error


def write(tmp_path, source, config):
    (tmp_path / "m.py").write_text(textwrap.dedent(source))
    (tmp_path / "fmugen.toml").write_text(textwrap.dedent(config))
    return tmp_path


# ---------------- the adapter, directly ----------------

def test_arrays_binary_and_integer_types(tmp_path, make_fmu, adapter):
    write(tmp_path, '''
        def f(v=(1.0, 2.0, 3.0), m=((1, 2), (3, 4)), key=b"\\x0f", n=1):
            return {"total": sum(v), "row_sums": [sum(r) for r in m], "masked": bytes(b ^ 0xff for b in key),
                    "double": 2 * n}
    ''', '''
        [model]
        entry = "m.py:f"
        fmi_version = 3
        [inputs]
        v   = { dimensions = [3], start = [1.0, 2.0, 3.0] }
        m   = { dimensions = [2, 2], type = "Int16", start = 7 }      # scalar start broadcast
        key = { type = "Binary", start = "0f" }
        n   = { type = "UInt8", start = 1 }
        [outputs]
        total    = {}
        row_sums = { dimensions = [2], type = "Int32" }
        masked   = { type = "Binary" }
        double   = { type = "UInt16" }
    ''')
    fmu = adapter(make_fmu(tmp_path), fmi=3)
    fmu.initialize()
    assert fmu.get("m") == 7                                   # first of the flattened values
    assert fmu.engine.get_values([fmu.vr["m"]]) == (OK, [7, 7, 7, 7])
    assert fmu.get("total", "row_sums", "masked", "double") == [6.0, 14, 14, b"\xf0", 2]

    assert fmu.fmi3SetFloat64([fmu.vr["v"]], [1.0, 1.0, 1.0]) == OK
    assert fmu.fmi3SetInt16([fmu.vr["m"]], [1, 2, 3, 4]) == OK
    assert fmu.fmi3SetBinary([fmu.vr["key"]], [2], [b"\x00\x01"]) == OK
    assert fmu.fmi3SetUInt8([fmu.vr["n"]], [256]) == ERROR      # out of range for UInt8
    assert fmu.fmi3DoStep(0.0, 1.0, False)[0] == OK
    assert fmu.engine.get_values([fmu.vr["row_sums"], fmu.vr["total"]]) == (OK, [3, 7, 3.0])
    assert fmu.fmi3GetBinary([fmu.vr["masked"]]) == (OK, [b"\xff\xfe"])


def test_structural_parameter_resizes_arrays(tmp_path, make_fmu, adapter):
    write(tmp_path, '''
        import numpy as np

        class Smoother:
            def __init__(self, n=3):
                self.window = np.zeros(n)

            def step(self, u):
                self.window = np.roll(self.window, 1)
                self.window[0] = u
                return float(self.window.mean())
    ''', '''
        [model]
        entry = "m.py:Smoother"
        call = "step"
        fmi_version = 3
        [structural_parameters]
        n = { start = 3 }
        [inputs]
        u = { start = 1.0 }
        [outputs]
        mean = { from = "return" }
        window = { dimensions = ["n"] }
    ''')
    fmu = adapter(make_fmu(tmp_path), fmi=3)
    assert fmu.set("n", 4) == ERROR                            # only in configuration mode
    assert fmu.fmi3EnterConfigurationMode() == OK
    assert fmu.fmi3SetUInt64([fmu.vr["n"]], [4]) == OK
    assert fmu.fmi3ExitConfigurationMode() == OK
    assert fmu.engine.get_values([fmu.vr["window"]]) == (OK, [0.0] * 4)
    fmu.initialize()
    assert fmu.fmi3DoStep(0.0, 1.0, False)[0] == OK
    assert fmu.get("mean") == 0.25
    assert fmu.engine.get_values([fmu.vr["window"]]) == (OK, [1.0, 0.0, 0.0, 0.0])


def test_output_clock_and_terminate(tmp_path, make_fmu, adapter):
    write(tmp_path, '''
        class Tank:
            def __init__(self, capacity=3.0):
                self.capacity, self.level, self.overflowed = capacity, 0.0, False

            def fill(self, inflow, dt):
                self.level += inflow * dt
                self.overflowed = self.level > self.capacity
                self.full = self.level >= 2 * self.capacity
    ''', '''
        [model]
        entry = "m.py:Tank"
        call = "fill"
        fmi_version = 3
        [time]
        dt = "step_size"
        [parameters]
        capacity = { start = 3.0 }
        [inputs]
        inflow = { start = 1.0 }
        [clocks.overflow]
        causality = "output"
        from = "attr:overflowed"
        [outputs]
        level = {}
        spill = { from = "attr:level", clocks = ["overflow"] }
        [events]
        terminate = "attr:full"
    ''')
    fmu = adapter(make_fmu(tmp_path), fmi=3, event_mode_used=True)
    fmu.initialize()
    assert fmu.fmi3EnterStepMode() == OK
    results = [fmu.fmi3DoStep(t, 1.0, False) for t in (0.0, 1.0, 2.0)]
    assert [r[1] for r in results] == [False, False, False]
    status, event_needed, terminate, early_return, last_time = fmu.fmi3DoStep(3.0, 1.0, False)
    assert (status, event_needed, terminate, early_return, last_time) == (OK, True, False, False, 4.0)
    assert fmu.fmi3EnterEventMode() == OK
    assert fmu.fmi3GetClock([fmu.vr["overflow"]]) == (OK, [True])
    assert fmu.fmi3GetClock([fmu.vr["overflow"]]) == (OK, [False])   # reset once read
    assert fmu.get("spill") == 4.0
    assert fmu.fmi3EnterStepMode() == OK
    assert fmu.fmi3DoStep(4.0, 2.0, False)[2] is True                 # level 6 = 2 * capacity


@pytest.mark.fmu
def test_fmi2_arrays_are_one_scalar_per_element(tmp_path):
    write(tmp_path, "def f(v=(1.0, 2.0)):\n    return {'s': sum(v), 'w': [x * 10 for x in v]}\n", '''
        [model]
        entry = "m.py:f"
        [inputs]
        v = { dimensions = [2], start = [1.0, 2.0] }
        [outputs]
        s = {}
        w = { dimensions = [2] }
    ''')
    build(tmp_path, tmp_path / "out.fmu")
    assert validate_fmu(str(tmp_path / "out.fmu")) == []
    md = read_model_description(str(tmp_path / "out.fmu"))
    assert md.variableNamingConvention == "structured"
    assert [v.name for v in md.modelVariables] == ["v[1]", "v[2]", "s", "w[1]", "w[2]"]
    result = simulate_fmu(str(tmp_path / "out.fmu"), stop_time=1.0, output_interval=1.0, start_values={"v[2]": 5.0})
    assert list(result[-1])[1:] == [6.0, 10.0, 50.0]


def test_fmi3_only_features_are_rejected_for_fmi2(tmp_path):
    write(tmp_path, "def f(b=b'x'):\n    return {'n': len(b)}\n", '''
        [model]
        entry = "m.py:f"
        [inputs]
        b = { type = "Binary", start = "78" }
        [outputs]
        n = {}
    ''')
    with pytest.raises(InterfaceError, match="requires FMI 3"):
        build(tmp_path, tmp_path / "out.fmu")
    build(tmp_path, tmp_path / "out3.fmu", fmi_version=3)
    assert validate_fmu(str(tmp_path / "out3.fmu")) == []


# ---------------- through UniFMU ----------------


ALL_TYPES = {
    "Float32": 1.5, "Float64": 2.25, "Int8": -8, "UInt8": 200, "Int16": -1600, "UInt16": 60000,
    "Int32": -2**31, "UInt32": 2**32 - 1, "Int64": -2**62, "UInt64": 2**64 - 1,
    "Boolean": True, "String": "héllo", "Binary": b"\x00\xff",
}


@pytest.mark.fmu
def test_every_type_clock_setting_and_state_function_through_unifmu(tmp_path):
    names = {t: t.lower() for t in ALL_TYPES}
    args = ", ".join(f"{n}_in" for n in names.values())
    returns = ", ".join(f'"{n}_out": {n}_in' for n in names.values())
    inputs = "\n".join(f'{n}_in = {{ type = "{t}" }}' for t, n in names.items())
    outputs = "\n".join(f'{n}_out = {{ type = "{t}" }}' for t, n in names.items())
    write(tmp_path, f"def echo({args}):\n    return {{{returns}}}\n", f'''
[model]
entry = "m.py:echo"
fmi_version = 3
[clocks.fixed_clock]
interval_variability = "fixed"
interval = 1.0
[clocks.tunable_clock]
interval_variability = "tunable"
interval = 2.0
[inputs]
{inputs}
[outputs]
{outputs}
''')
    out = build(tmp_path, tmp_path / "echo.fmu")[0]
    assert validate_fmu(str(out)) == []
    unzipdir = tmp_path / "echo"
    shutil.unpack_archive(out, unzipdir, "zip")
    md = read_model_description(str(unzipdir))
    vr = {v.name: v.valueReference for v in md.modelVariables}
    fmu = FMU3Slave(guid=md.guid, unzipDirectory=str(unzipdir), modelIdentifier="unifmu", instanceName="echo")
    assert fmu.getVersion() == "3.0"
    fmu.instantiate(eventModeUsed=True)
    try:
        fixed = (fmi3ValueReference * 1)(vr["fixed_clock"])
        tunable = (fmi3ValueReference * 1)(vr["tunable_clock"])
        fmu.enterInitializationMode(startTime=0.0)
        fmu.setIntervalDecimal(fixed, (c_double * 1)(0.5))
        fmu.fmi3SetShiftDecimal(fmu.component, fixed, 1, (c_double * 1)(0.25))
        fmu.setIntervalFraction(tunable, (c_uint64 * 1)(3), (c_uint64 * 1)(4))
        for t, value in ALL_TYPES.items():
            getattr(fmu, f"set{t}")([vr[f"{names[t]}_in"]], [value])
        fmu.exitInitializationMode()

        intervals, qualifiers, shifts = (c_double * 2)(), (c_int * 2)(), (c_double * 2)()
        both = (fmi3ValueReference * 2)(vr["fixed_clock"], vr["tunable_clock"])
        fmu.getIntervalDecimal(both, intervals, qualifiers)
        fmu.getShiftDecimal(both, shifts)
        assert list(intervals) == [0.5, 0.75] and list(shifts) == [0.25, 0.0]
        fmu.setIntervalDecimal(tunable, (c_double * 1)(1.25))      # tunable: allowed in event mode
        fmu.getIntervalDecimal(tunable, intervals, qualifiers)
        assert (intervals[0], qualifiers[0]) == (1.25, 2)
        with pytest.raises(Exception):
            fmu.setIntervalDecimal(fixed, (c_double * 1)(2.0))     # fixed: only before initialization ends

        fmu.enterStepMode()
        fmu.doStep(0.0, 0.5)
        for t, value in ALL_TYPES.items():
            assert getattr(fmu, f"get{t}")([vr[f"{names[t]}_out"]]) == [value], t

        state = fmu.getFMUState()
        serialized = fmu.serializeFMUState(state)
        fmu.setFloat64([vr["float64_in"]], [9.0])
        fmu.doStep(0.5, 0.5)
        assert fmu.getFloat64([vr["float64_out"]]) == [9.0]
        restored = fmu.deserializeFMUState(serialized)
        fmu.setFMUState(restored)
        assert fmu.getFloat64([vr["float64_out"]]) == [2.25]
        fmu.freeFMUState(state)
        fmu.freeFMUState(restored)
        fmu.terminate()
    finally:
        fmu.freeInstance()


def test_fraction_fits_in_uint64():
    from fmugen.templates.fmugen_runtime import UINT64_MAX, fraction
    assert fraction(0.1) == (1, 10)
    counter, resolution = fraction(1e-20)
    assert 0 < resolution <= UINT64_MAX and counter / resolution == pytest.approx(1e-20, rel=1e-6)


def test_array_output_empty_until_the_first_step(tmp_path, make_fmu, adapter):
    # like pyloudnorm's blockwise_loudness: empty after construction, full after each step
    write(tmp_path, '''
        class Meter:
            def __init__(self):
                self.blocks = []

            def step(self, level=1.0):
                self.blocks = [level, 2 * level, 3 * level]
                return level
    ''', '''
        [model]
        entry = "m.py:Meter"
        fmi_version = 3
        call = "step"
        [inputs]
        level = { start = 1.0 }
        [outputs]
        blocks = { dimensions = [3] }
    ''')
    fmu = adapter(make_fmu(tmp_path), fmi=3)
    fmu.initialize()   # the empty list is skipped, not an error
    assert fmu.fmi3DoStep(0.0, 1.0, False)[0] == OK
    assert fmu.engine.get_values([fmu.vr["blocks"]]) == (OK, [1.0, 2.0, 3.0])


def test_array_output_of_the_wrong_size_still_fails_a_step(tmp_path, make_fmu, adapter):
    write(tmp_path, '''
        class Meter:
            def __init__(self):
                self.blocks = [0.0, 0.0, 0.0]

            def step(self, level=1.0):
                self.blocks = [level]
                return level
    ''', '''
        [model]
        entry = "m.py:Meter"
        fmi_version = 3
        call = "step"
        [inputs]
        level = { start = 1.0 }
        [outputs]
        blocks = { dimensions = [3] }
    ''')
    fmu = adapter(make_fmu(tmp_path), fmi=3)
    fmu.initialize()
    assert fmu.fmi3DoStep(0.0, 1.0, False)[0] == ERROR
