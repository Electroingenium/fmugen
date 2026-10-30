"""fmugen build --capture-output: a model that prints a lot must not crash UniFMU.

UniFMU 0.14 panics (zeromq rep.rs: not yet implemented) and the importer hangs when the
backend writes more than about 4 KB to the console. With --capture-output, the runtime
captures that output and forwards it to the importer's log instead. The FMU runs in a
subprocess with a timeout, so a regression fails instead of hanging the test run.
"""
import subprocess
import sys
import textwrap

import pytest

from fmugen.builder import build

SIMULATE = textwrap.dedent("""
    import sys
    from fmpy import simulate_fmu
    logs = []
    result = simulate_fmu(sys.argv[1], stop_time=2, output=["y"],
                          logger=lambda *args: logs.append(args[-1]), debug_logging=True)
    printed = [m.decode() if isinstance(m, bytes) else m for m in logs]
    printed = [m for m in printed if m.startswith("[output]")]
    # each call's output once: the capture file is emptied after each read, and nothing is left from before
    longest = max(len(m) for m in printed)
    clean = all(set(m.removeprefix("[output] ")) <= set("pe\\r\\n") for m in printed)
    print("OK", result["y"][-1], len(printed), longest, clean)
""")


@pytest.mark.fmu
def test_large_console_output_reaches_the_log_instead_of_crashing(tmp_path):
    (tmp_path / "noisy.py").write_text(textwrap.dedent("""
        import sys
        def noisy(u=1.0):
            print("p" * 20000)                      # stdout
            sys.stderr.write("e" * 20000 + "\\n")    # stderr, like warnings and progress bars
            return {"y": 2 * u}
    """))
    fmu = build(tmp_path / "noisy.py", tmp_path / "noisy.fmu", capture_output=True)[0]
    (tmp_path / "simulate.py").write_text(SIMULATE)
    log = tmp_path / "simulate.log"
    with log.open("w") as out:   # a file, not a pipe: a hung backend would keep a pipe open past the timeout
        try:
            subprocess.run([sys.executable, str(tmp_path / "simulate.py"), str(fmu)],
                           stdout=out, stderr=subprocess.STDOUT, timeout=120)
        except subprocess.TimeoutExpired:
            pass
    text = log.read_text(errors="replace")
    ok = [line for line in text.splitlines() if line.startswith("OK")]
    assert ok, "the FMU hung or failed:\n" + text[-2000:]
    _, y, printed, longest, clean = ok[0].split()
    assert float(y) == 2.0 and int(printed) > 1
    assert int(longest) < 2 * 20000 + 100 and clean == "True"   # one call's output, not a growing file


INSTANTIATE = textwrap.dedent("""
    import sys
    from fmpy import simulate_fmu
    logs = []
    try:
        simulate_fmu(sys.argv[1], stop_time=1, logger=lambda *args: logs.append(args[-1]), debug_logging=True)
    except Exception:
        pass
    messages = [m.decode() if isinstance(m, bytes) else m for m in logs]
    # UniFMU cuts the backend's output short: the runtime writes the exception's line before the traceback
    print("FOUND" if any("model import broke" in m and "Traceback" in m for m in messages) else "MISSING", flush=True)
""")


@pytest.mark.fmu
def test_an_import_error_at_instantiation_is_not_captured_away(tmp_path):
    # timm's compiled FMU: the model failed to import, the backend exited with status 1, and its
    # traceback was in the capture file, so UniFMU reported "Backend emitted no output"
    (tmp_path / "broken.py").write_text(textwrap.dedent("""
        import os
        if "UNIFMU_DISPATCHER_ENDPOINT" in os.environ:   # only inside the FMU, not when building
            raise ImportError("model import broke")
        def broken(u=1.0):
            return {"y": u}
    """))
    fmu = build(tmp_path / "broken.py", tmp_path / "broken.fmu", capture_output=True)[0]
    (tmp_path / "simulate.py").write_text(INSTANTIATE)
    log = tmp_path / "simulate.log"
    with log.open("w") as out:   # a file, not a pipe: a backend left running would keep a pipe open
        try:
            subprocess.run([sys.executable, str(tmp_path / "simulate.py"), str(fmu)],
                           stdout=out, stderr=subprocess.STDOUT, timeout=120)
        except subprocess.TimeoutExpired:
            pass
    text = log.read_text(errors="replace")
    assert "FOUND" in text, text[-2000:]


def test_output_goes_to_the_console_by_default(tmp_path):
    import json
    import zipfile
    (tmp_path / "quiet.py").write_text("def quiet(u=1.0):\n    return {'y': u}\n")
    for flag in (False, True):
        fmu = build(tmp_path / "quiet.py", tmp_path / f"quiet{flag}.fmu", capture_output=flag)[0]
        with zipfile.ZipFile(fmu) as z:
            assert json.loads(z.read("resources/interface.json"))["capture_output"] is flag


PERCENT_LOG = textwrap.dedent("""
    import sys
    from fmpy import simulate_fmu
    logs = []
    # FMPy formats FMI 2 messages with their variadic arguments: an unescaped "%s" reads a missing argument
    # and crashes. Two steps: UniFMU's backend also echoes each record to its console, which must stay
    # under the ~4 KB that hangs UniFMU 0.14 (docs/limitations/unifmu.md section 1).
    result = simulate_fmu(sys.argv[1], stop_time=1, output_interval=0.5, output=["y"],
                          logger=lambda *args: logs.append(args[-1]), debug_logging=True)
    messages = [m.decode() if isinstance(m, bytes) else m for m in logs]
    print("OK", result["y"][-1], any("done 50%s of %d" in m for m in messages))
""")


@pytest.mark.fmu
def test_fmi2_log_messages_with_percent_signs_reach_the_importer(tmp_path):
    # UniFMU 0.14 passes FMI 2 messages to the importer as printf format strings (limitations/unifmu.md
    # section 11); fmugen's FMI 2 adapter escapes them
    (tmp_path / "progress.py").write_text(textwrap.dedent("""
        import logging
        def progress(x=3.0):
            logging.getLogger("progress").warning("done 50%s of %d")
            return {"y": x}
    """))
    fmu = build(tmp_path / "progress.py", tmp_path / "progress.fmu")[0]
    (tmp_path / "simulate.py").write_text(PERCENT_LOG)
    log = tmp_path / "simulate.log"
    with log.open("w") as out:   # a file, not a pipe: a backend left running would keep a pipe open
        try:
            subprocess.run([sys.executable, str(tmp_path / "simulate.py"), str(fmu)],
                           stdout=out, stderr=subprocess.STDOUT, timeout=120)
        except subprocess.TimeoutExpired:
            pass
    text = log.read_text(errors="replace")
    ok = [line for line in text.splitlines() if line.startswith("OK")]
    assert ok, "the importer crashed or hung:\n" + text[-2000:]
    assert ok[0].split()[1:] == ["3.0", "True"]
