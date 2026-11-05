"""How a step runs: discard, ODE integration, time sources, async models, logging categories, reset."""
import math

import pytest
from test_runtime import write

from fmugen.config import InterfaceError, summary
from fmugen.templates.model_fmi2 import Fmi2Status

OK, DISCARD, ERROR = Fmi2Status.ok, Fmi2Status.discard, Fmi2Status.error


def test_discard_on_rolls_the_step_back(tmp_path, make_fmu, adapter):
    write(tmp_path, "growth.py", '''
        import warnings

        class Growth:
            def __init__(self, rate=1.0):
                self.rate, self.x = rate, 1.0

            def step(self, dt):
                self.x += self.rate * self.x * dt
                if dt > 0.5:
                    raise ArithmeticError("step too large")
                if dt > 0.2:
                    warnings.warn("inaccurate step", RuntimeWarning)
    ''', '''
        [model]
        entry = "growth.py:Growth"
        call = "step"
        discard_on = ["builtins:ArithmeticError", "builtins:RuntimeWarning"]
        [time]
        dt = "step_size"
        [outputs]
        x = {}
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 0.1, False) == OK
    assert fmu.get("x") == pytest.approx(1.1)
    for step in (1.0, 0.3):                                   # an exception, then a warning: both discard
        assert fmu.fmi2DoStep(0.1, step, False) == DISCARD
        assert fmu.get("x") == pytest.approx(1.1) and fmu.engine.obj.x == pytest.approx(1.1)   # undone
        assert fmu.engine.time == pytest.approx(0.1)
        assert fmu.logs[-1][1] == "logStatusDiscard"
    assert fmu.fmi2DoStep(0.1, 0.1, False) == OK              # a smaller step is accepted
    assert fmu.get("x") == pytest.approx(1.21)


def test_discard_in_fmi3_reports_the_time_reached(tmp_path, make_fmu, adapter):
    write(tmp_path, "m.py", '''
        def f(x=1.0):
            if x < 0:
                raise ValueError("negative")
            return {"y": x}
    ''', '''
        [model]
        entry = "m.py:f"
        fmi_version = 3
        discard_on = ["builtins:ValueError"]
        [inputs]
        x = { start = 1.0 }
        [outputs]
        y = {}
    ''')
    fmu = adapter(make_fmu(tmp_path), fmi=3)
    fmu.initialize()
    assert fmu.set("x", -1.0) == OK
    status, event_needed, _terminate, early_return, last_time = fmu.fmi3DoStep(0.0, 1.0, False)
    assert (status, event_needed, early_return, last_time) == (DISCARD, False, False, 0.0)


def test_discard_on_must_name_exception_classes(tmp_path, make_fmu):
    write(tmp_path, "m.py", "def f(x=1.0):\n    return {'y': x}\n", '''
        [model]
        entry = "m.py:f"
        discard_on = ["math:pi"]
        [inputs]
        x = {}
        [outputs]
        y = {}
    ''')
    with pytest.raises(InterfaceError, match="not an exception"):
        make_fmu(tmp_path)


def test_discard_on_needs_the_fmu_state(tmp_path, make_fmu):
    write(tmp_path, "m.py", "class M:\n    def step(self, x=1.0):\n        return x\n", '''
        [model]
        entry = "m.py:M"
        call = "step"
        save_state = false
        discard_on = ["builtins:ValueError"]
        [inputs]
        x = {}
        [outputs]
        y = { from = "return" }
    ''')
    with pytest.raises(InterfaceError, match="discard_on needs the FMU state"):
        make_fmu(tmp_path)


def test_time_sources_start_stop_and_tolerance(tmp_path, make_fmu, adapter):
    write(tmp_path, "m.py", '''
        def f(t0, t1, rtol=None):
            return {"t0": t0, "t1": t1, "rtol": -1.0 if rtol is None else rtol}
    ''', '''
        [model]
        entry = "m.py:f"
        [time]
        t0 = "start_time"
        t1 = "stop_time"
        rtol = "tolerance"
        [outputs]
        t0 = {}
        t1 = {}
        rtol = {}
    ''')
    fmu = adapter(make_fmu(tmp_path))
    assert fmu.fmi2SetupExperiment(2.0, 10.0, 1e-6) == OK
    assert fmu.fmi2EnterInitializationMode() == OK and fmu.fmi2ExitInitializationMode() == OK
    assert fmu.fmi2DoStep(2.0, 1.0, False) == OK
    assert fmu.get("t0", "t1", "rtol") == [2.0, 10.0, 1e-6]


ODE = '''
    def decay(t, y, k=0.5, u=0.0):
        return -k * y + u
'''


@pytest.mark.parametrize("method, substeps, rel", [("euler", 1000, 1e-3), ("rk4", 1, 1e-4),
                                                   ("scipy:RK45", None, 1e-3)])
def test_integrate_an_ode_right_hand_side(tmp_path, make_fmu, adapter, method, substeps, rel):
    write(tmp_path, "ode.py", ODE, f'''
        [model]
        entry = "ode.py:decay"
        integrate = "{method}"
        {f"substeps = {substeps}" if substeps else ""}
        [time]
        t = "time"
        [parameters]
        k = {{ start = 0.5 }}
        [inputs]
        u = {{ start = 0.0 }}
        [states]
        y = {{ start = 2.0 }}
        [outputs]
        dy = {{ from = "return" }}
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.get("dy") == -1.0                                # the derivative at the start
    for t in range(4):
        assert fmu.fmi2DoStep(t * 0.5, 0.5, False) == OK, fmu.logs
    assert fmu.get("y") == pytest.approx(2.0 * math.exp(-1.0), rel=rel)
    assert fmu.get("dy") == pytest.approx(-0.5 * fmu.get("y"))


def test_integrate_a_vector_state_with_several_derivatives(tmp_path, make_fmu, adapter):
    write(tmp_path, "osc.py", '''
        def oscillator(t, position, velocity, omega=1.0):
            return {"dx": velocity, "dv": -omega ** 2 * position}
    ''', '''
        [model]
        entry = "osc.py:oscillator"
        integrate = "rk4"
        substeps = 10
        [time]
        t = "time"
        [states]
        position = { start = 1.0, der = "return:dx" }
        velocity = { start = 0.0, der = "return:dv" }
        [outputs]
        acceleration = { from = "return:dv" }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    for k in range(10):
        assert fmu.fmi2DoStep(k * math.pi / 10, math.pi / 10, False) == OK, fmu.logs
    assert fmu.get("position", "velocity") == pytest.approx([-1.0, 0.0], abs=1e-6)   # half a period
    assert fmu.get("acceleration") == pytest.approx(1.0, abs=1e-6)


def test_a_failed_solve_is_a_discard(tmp_path, make_fmu, adapter):
    write(tmp_path, "ode.py", ODE, '''
        [model]
        entry = "ode.py:decay"
        integrate = "euler"
        [time]
        t = "time"
        [parameters]
        k = { start = 1e300 }
        [states]
        y = { start = 2.0 }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1e10, False) == DISCARD
    assert fmu.get("y") == 2.0


@pytest.mark.parametrize("model, state, message", [
    ('integrate = "rk5"', "", "must be one of"),
    ('integrate = "scipy:Nope"', "", "not a solve_ivp method"),
    ('integrate = "rk4"\nsubsteps = 0', "", "positive integer"),
    ('integrate = "rk4"', ', next = "return"', "not next"),
    ("", ', next = "return", der = "return"', "only applies with"),
])
def test_integrate_config_errors(tmp_path, make_fmu, model, state, message):
    write(tmp_path, "ode.py", ODE,
          f'[model]\nentry = "ode.py:decay"\n{model}\n[states]\ny = {{ start = 2.0{state} }}\n')
    with pytest.raises(InterfaceError, match=message):
        make_fmu(tmp_path)


def test_debug_logging_categories_filter_messages(tmp_path, make_fmu, adapter):
    write(tmp_path, "m.py", '''
        import logging
        logging.getLogger("m").setLevel(logging.INFO)

        def f(x=1.0):
            logging.getLogger("m").warning("x is %s", x)
            logging.getLogger("m").info("info %s", x)
            return {"y": 1.0 / x}
    ''')
    fmu = adapter(make_fmu(tmp_path / "m.py"))
    fmu.initialize()
    assert fmu.fmi2SetDebugLogging(["logStatusWarning"], True) == OK
    fmu.logs.clear()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    assert [c for _, c, _ in fmu.logs] == ["logStatusWarning"]        # not logAll
    assert fmu.fmi2SetDebugLogging([], True) == OK                    # no categories: all of them
    fmu.logs.clear()
    assert fmu.fmi2DoStep(1.0, 1.0, False) == OK
    assert [c for _, c, _ in fmu.logs] == ["logStatusWarning", "logAll"]
    assert fmu.fmi2SetDebugLogging(["logAll"], True) == OK
    assert fmu.set("x", 0.0) == OK
    assert fmu.fmi2DoStep(2.0, 1.0, False) == ERROR
    assert fmu.logs[-1][1] == "logStatusError"                        # errors always reach the importer


def test_reset_terminates_the_model_and_a_time_jump_is_a_warning(tmp_path, make_fmu, adapter):
    write(tmp_path, "port.py", '''
        CLOSED = []

        class Port:
            def __init__(self):
                self.value = 0.0

            def read(self):
                self.value += 1.0

            def close(self):
                CLOSED.append(self)
    ''', '''
        [model]
        entry = "port.py:Port"
        call = "read"
        terminate = "close"
        [outputs]
        value = {}
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK
    fmu.logs.clear()
    assert fmu.fmi2DoStep(5.0, 1.0, False) == OK                      # the importer skipped 1..5
    assert fmu.logs[-1][0] == Fmi2Status.warning and "at time 1.0" in fmu.logs[-1][2]
    obj = fmu.engine.obj
    assert fmu.fmi2Reset() == OK
    assert obj in fmu.engine.module.CLOSED                           # closed by reset, not left to the GC


def test_async_step_method(tmp_path, make_fmu, adapter):
    write(tmp_path, "aio.py", '''
        import asyncio

        class Sensor:
            def __init__(self):
                self.loop, self.total = None, 0.0

            async def poll(self, x=1.0):
                await asyncio.sleep(0)
                loop = asyncio.get_running_loop()
                assert self.loop in (None, loop)      # the same event loop on every step
                self.loop = loop
                self.total += x
                return self.total
    ''', '''
        [model]
        entry = "aio.py:Sensor"
        call = "poll"
        save_state = false
        [inputs]
        x = { start = 1.0 }
        [outputs]
        total = { from = "return" }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.initialize()
    for t in range(3):
        assert fmu.fmi2DoStep(float(t), 1.0, False) == OK, fmu.logs
    assert fmu.get("total") == 3.0


def test_generator_steps_are_read_whole(tmp_path, make_fmu, adapter):
    # like kokoro's KPipeline or piper's synthesize: each call yields its results in chunks
    write(tmp_path, "gen.py", '''
        def chunks(n=1.0):
            for i in range(3):
                yield n * i
    ''', '''
        [model]
        entry = "gen.py:chunks"
        [inputs]
        n = {}
        [outputs]
        last = { from = "return:2" }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    fmu.set("n", 2.0)
    fmu.initialize()
    assert fmu.fmi2DoStep(0.0, 1.0, False) == OK, fmu.logs
    assert fmu.get("last") == 4.0


def test_endless_generator_steps_fail_with_a_hint(tmp_path, make_fmu, adapter, monkeypatch):
    from fmugen.templates.fmugen_runtime import _helpers
    monkeypatch.setattr(_helpers, "GENERATOR_LIMIT", 10)
    write(tmp_path, "gen.py", '''
        def counter(n=1.0):
            while True:
                yield n
    ''', '''
        [model]
        entry = "gen.py:counter"
        [inputs]
        n = {}
        [outputs]
        y = { from = "return:0" }
    ''')
    fmu = adapter(make_fmu(tmp_path))
    try:   # the step may run when initialization ends, to give the outputs their first values
        fmu.initialize()
        status = fmu.fmi2DoStep(0.0, 1.0, False)
    except AssertionError:
        status = ERROR
    assert status == ERROR
    assert "next() once per step" in str(fmu.logs)


def test_fmi2_takes_wider_types_and_checks_ranges(tmp_path, make_fmu, capsys):
    source = "def f(n=1, flag=True, x=1.0):\n    return {'y': x * n}\n"
    write(tmp_path, "m.py", source, '''
        [model]
        entry = "m.py:f"
        [inputs]
        n = { type = "Int64", start = 1 }
        x = { type = "Float32", start = 1.0, min = 0.0, max = 10.0 }
        [outputs]
        y = {}
    ''')
    make_fmu(tmp_path)
    assert "32-bit Integer" in capsys.readouterr().out
    for line, message in [("x = { start = 11.0, min = 0.0, max = 10.0 }", "outside min..max"),
                          ("x = { start = 1.0, min = 5.0, max = 2.0 }", "larger than max"),
                          ('flag = { type = "Boolean", start = 1 }', r"use true\)")]:
        write(tmp_path, "m.py", source, f'[model]\nentry = "m.py:f"\n[inputs]\n{line}\n[outputs]\ny = {{}}\n')
        with pytest.raises(InterfaceError, match=message):
            make_fmu(tmp_path)


def test_description_is_the_first_line_of_the_docstring():
    assert summary("Solve e.g. the heat equation. Uses SI units.\nMore.") == "Solve e.g. the heat equation. Uses SI units."
    assert len(summary("x" * 500)) == 200 and summary("") == ""
