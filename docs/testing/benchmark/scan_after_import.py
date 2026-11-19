"""Time importlib.metadata.packages_distributions() after importing MODULE (or - for none), with the GC on or off.

Usage: scan_after_import.py {MODULE|-} {on|off}
"""
import gc
import importlib
import importlib.metadata as m
import sys
import time
import warnings

warnings.simplefilter("ignore")
mod, g = sys.argv[1], sys.argv[2]
if mod != "-": importlib.import_module(mod)
gc.collect()
if g == "off": gc.disable()
t = time.perf_counter(); m.packages_distributions()
print(f"scan after importing {mod:17} gc {g:3}: {time.perf_counter()-t:.2f}s  objects tracked {len(gc.get_objects())}")
