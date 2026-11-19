"""init TARGET --probe as is, or (variant no-torch-unless-loaded) with the probe's torch:tensor retry only when torch is already imported.

Usage: torch_retry.py {as-is|no-torch-unless-loaded} TARGET [OPTIONS]; writes VARIANT.toml and prints the time.
"""
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
variant, target = sys.argv[1], sys.argv[2:]
import fmugen.probe._calling as c

if variant == "no-torch-unless-loaded":
    orig = c.ARRAY_RETRIES
    class Lazy(tuple):
        def __iter__(self):
            return iter(r for r in orig if r == "numpy" or r.partition(":")[0] in sys.modules)
    c.ARRAY_RETRIES = Lazy(orig)
from fmugen.__main__ import main

out = io.StringIO(); t = time.perf_counter()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    try: main(["init", *target, "-o", "-", "--probe"])
    except SystemExit: pass
Path(f"{variant}.toml").write_text(out.getvalue(), encoding="utf-8")
print(f"{variant:24} {time.perf_counter()-t:5.1f}s  torch loaded: {'torch' in sys.modules}")
