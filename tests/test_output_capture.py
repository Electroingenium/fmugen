"""A model's console output, and backends that exit, through UniFMU and FMPy.

UniFMU 1.x drains the backend's console while it runs and relays it to the importer's stdout
(0.14 hung once the backend had written about 4 KB), and turns a backend that exits during a
call into an error status (0.14 could abort the importer). fmugen build --capture-output sends
the output to the importer's FMI log instead. Each FMU runs in a subprocess with a timeout, so
a regression fails instead of hanging or killing the test run.
"""
import subprocess
import sys
import textwrap

import pytest

from fmugen.builder import build


def _run(tmp_path, script, fmu):
    """Run ``script`` with the FMU's path in a subprocess; return its exit code and output.

    :return: ``(exit code, or None after a timeout; output)``.
    :rtype: tuple[int or None, str]
    """
    (tmp_path / "simulate.py").write_text(script)
    log = tmp_path / "simulate.log"
    with log.open("w") as out:   # a file, not a pipe: a backend left running would keep a pipe open past the timeout
        try:
            code = subprocess.run([sys.executable, str(tmp_path / "simulate.py"), str(fmu)],
                                  stdout=out, stderr=subprocess.STDOUT, timeout=120, check=False).returncode
        except subprocess.TimeoutExpired:
            code = None
    return code, log.read_text(errors="replace")


NOISY = textwrap.dedent("""
    import sys
    def noisy(u=1.0):
        print("p" * 20000)                      # stdout
        sys.stderr.write("e" * 20000 + "\\n")    # stderr, like warnings and progress bars
        return {"y": 2 * u}
""")

SIMULATE_CAPTURED = textwrap.dedent("""
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
def test_large_console_output_reaches_the_log(tmp_path):
    (tmp_path / "noisy.py").write_text(NOISY)
    fmu = build(tmp_path / "noisy.py", tmp_path / "noisy.fmu", capture_output=True)[0]
    _, text = _run(tmp_path, SIMULATE_CAPTURED, fmu)
    ok = [line for line in text.splitlines() if line.startswith("OK")]
    assert ok, "the FMU hung or failed:\n" + text[-2000:]
    _, y, printed, longest, clean = ok[0].split()
    assert float(y) == 2.0 and int(printed) > 1
    assert int(longest) < 2 * 20000 + 100 and clean == "True"   # one call's output, not a growing file


SIMULATE = textwrap.dedent("""
    import sys
    from fmpy import simulate_fmu
    result = simulate_fmu(sys.argv[1], stop_time=2, output=["y"])
    print("OK", result["y"][-1], flush=True)
""")


@pytest.mark.fmu
@pytest.mark.parametrize("fmi_version", [2, 3])
def test_large_console_output_without_capture_simulates(tmp_path, fmi_version):
    # 40 KB per call through UniFMU's console pipe: 0.14 hung after about 4 KB
    (tmp_path / "noisy.py").write_text(NOISY)
    fmu = build(tmp_path / "noisy.py", tmp_path / "noisy.fmu", fmi_version=fmi_version)[0]
    code, text = _run(tmp_path, SIMULATE, fmu)
    ok = [line for line in text.splitlines() if line.startswith("OK")]
    assert code == 0 and ok, f"exit code {code}; the FMU hung or failed:\n" + text[-2000:]
    assert float(ok[0].split()[1]) == 2.0
    assert "p" * 1000 in text   # relayed to the importer's console


INSTANTIATE = textwrap.dedent("""
    import sys
    from fmpy import simulate_fmu
    logs = []
    try:
        simulate_fmu(sys.argv[1], stop_time=1, logger=lambda *args: logs.append(args[-1]), debug_logging=True)
        print("SIMULATED", flush=True)
    except Exception as e:
        print("RAISED", type(e).__name__, e, flush=True)
    messages = [m.decode() if isinstance(m, bytes) else m for m in logs]
    for m in messages:
        print("LOG", m, flush=True)
    # the runtime writes the exception's line before the traceback, so it survives a long output
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
    _, text = _run(tmp_path, INSTANTIATE, fmu)
    assert "FOUND" in text, text[-2000:]


@pytest.mark.fmu
@pytest.mark.parametrize("fmi_version", [2, 3])
def test_a_backend_that_exits_gives_the_importer_an_error(tmp_path, fmi_version):
    # the backend process ends without a reply, as a frozen backend missing a file did:
    # UniFMU 0.14 could panic and abort the importer (0xC0000409), 1.x returns an error status
    (tmp_path / "quits.py").write_text(textwrap.dedent("""
        import os
        if "UNIFMU_DISPATCHER_ENDPOINT" in os.environ:   # only inside the FMU, not when building
            print("backend going away", flush=True)
            os._exit(1)
        def quits(u=1.0):
            return {"y": u}
    """))
    fmu = build(tmp_path / "quits.py", tmp_path / "quits.fmu", fmi_version=fmi_version)[0]
    code, text = _run(tmp_path, INSTANTIATE, fmu)
    assert code == 0, f"the importer died (exit code {code}):\n" + text[-2000:]
    assert "RAISED" in text and "SIMULATED" not in text, text[-2000:]
    assert "exited unexpectedly" in text or "disconnected" in text, text[-2000:]


PERCENT_LOG = textwrap.dedent("""
    import sys
    from fmpy import simulate_fmu
    logs = []
    # FMPy formats FMI 2 messages with their variadic arguments: a "%s" used as the format would read
    # a missing argument and crash
    result = simulate_fmu(sys.argv[1], stop_time=1, output_interval=0.5, output=["y"],
                          logger=lambda *args: logs.append(args[-1]), debug_logging=True)
    messages = [m.decode() if isinstance(m, bytes) else m for m in logs]
    exact = any("done 50%s of %d" in m for m in messages)
    doubled = any("%%" in m for m in messages)
    print("OK", result["y"][-1], exact, doubled)
""")


@pytest.mark.fmu
def test_fmi2_log_messages_with_percent_signs_reach_the_importer(tmp_path):
    # UniFMU 0.14 passed FMI 2 messages to the importer as printf format strings and fmugen escaped
    # them; 1.x passes "%s" as the format, so the text must arrive unchanged, with no "%%"
    (tmp_path / "progress.py").write_text(textwrap.dedent("""
        import logging
        def progress(x=3.0):
            logging.getLogger("progress").warning("done 50%s of %d")
            return {"y": x}
    """))
    fmu = build(tmp_path / "progress.py", tmp_path / "progress.fmu")[0]
    _, text = _run(tmp_path, PERCENT_LOG, fmu)
    ok = [line for line in text.splitlines() if line.startswith("OK")]
    assert ok, "the importer crashed or hung:\n" + text[-2000:]
    assert ok[0].split()[1:] == ["3.0", "True", "False"]


def test_output_goes_to_the_console_by_default(tmp_path):
    import json
    import zipfile
    (tmp_path / "quiet.py").write_text("def quiet(u=1.0):\n    return {'y': u}\n")
    for flag in (False, True):
        fmu = build(tmp_path / "quiet.py", tmp_path / f"quiet{flag}.fmu", capture_output=flag)[0]
        with zipfile.ZipFile(fmu) as z:
            assert json.loads(z.read("resources/interface.json"))["capture_output"] is flag
