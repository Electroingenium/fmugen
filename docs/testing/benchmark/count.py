import contextlib
import cProfile
import io
import pstats
import sys
import time

from fmugen.__main__ import main

target = sys.argv[1:]
pr = cProfile.Profile(); t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
    pr.enable()
    try: main(["init", *target, "-o", "-", "--probe"])
    except SystemExit: pass
    pr.disable()
wall = time.perf_counter() - t
st = pstats.Stats(pr).stats
def get(fn, file=""):
    n = c = 0
    for (f, l, name), (cc, nc, tt, ct, callers) in st.items():
        if name == fn and file in f: n += nc; c = max(c, ct)
    return n, c
pd_n = sum(v[1] for k, v in st.items() if k[2] == "packages_distributions" and "importlib" in k[0])
pd_t = sum(v[3] for k, v in st.items() if k[2] == "packages_distributions" and "importlib" in k[0])
once = get("_infer_config_once", "interface")
print(f"{' '.join(target)[:45]:45} wall={wall:5.2f}s  _infer_config_once calls={once[0]:3} cum={once[1]:5.2f}s  pkg_dists scans={pd_n} {pd_t:5.2f}s  probe_once={get('_probe_once')[0]}")
