"""init TARGET --probe with plain timers (no profiler) around fmugen's steps; prints one row.

Also times importing the model's top package in a fresh process (python -X importtime would be similar).
"""
import collections
import contextlib
import functools
import io
import subprocess
import sys
import time
import warnings

warnings.simplefilter("ignore")
target = sys.argv[1:]
top = target[0].split(":")[0].split(".")[0]
t = time.perf_counter(); subprocess.run([sys.executable, "-W", "ignore", "-c", f"import {target[0].split(':')[0]}"], check=True)
import_alone = time.perf_counter() - t - 0.09   # minus a bare interpreter start (0.09 s here)

import fmugen.examples
import fmugen.interface
import fmugen.probe._calling as calling
import fmugen.static

spent, calls, depth = collections.Counter(), collections.Counter(), collections.Counter()
def timed(owner, name, label):
    fn = getattr(owner, name)
    @functools.wraps(fn)
    def wrapper(*a, **k):
        depth[label] += 1; t = time.perf_counter()
        try: return fn(*a, **k)
        finally:
            depth[label] -= 1; calls[label] += 1
            if not depth[label]: spent[label] += time.perf_counter() - t
    setattr(owner, name, wrapper)
timed(fmugen.static, "written_globals", "globals")
timed(fmugen.interface, "_with_usage", "examples")
timed(fmugen.static, "packages_distributions", "scan")
timed(fmugen.examples, "packages_distributions", "scan")
timed(calling, "_probe_once", "probe calls")
timed(fmugen.interface, "load_module", "import")
timed(fmugen.interface.importlib, "import_module", "import")
from fmugen.__main__ import main

t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try: main(["init", *target, "-o", "-", "--probe"])
    except SystemExit: pass
wall = time.perf_counter() - t
print(f"{' '.join(target)[:44]:44} | init {wall:5.1f}s | {top} alone {import_alone:4.1f}s | " + " | ".join(
    f"{k} {spent[k]:4.1f}s" + (f" ({calls[k]})" if k in ("probe calls", "globals") else "") for k in ("import", "examples", "scan", "globals", "probe calls")))
