"""Time os.path.exists on every file of every installed distribution, before and after importing MODULE (or -), per package.

Usage: stat_per_package.py {MODULE|-}
"""
import collections
import importlib
import importlib.metadata as m
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
def measure():
    per = collections.Counter(); n = collections.Counter()
    for d in m.distributions():
        name = d.metadata["Name"]
        for f in d.files or []:
            p = str(d.locate_file(f)); t = time.perf_counter(); os.path.exists(p); per[name] += time.perf_counter() - t; n[name] += 1
    return per, n
before, n = measure()
if sys.argv[1] != "-": importlib.import_module(sys.argv[1])
after, _ = measure()
print(f"total stat time: before import {sum(before.values()):.2f}s  after import {sum(after.values()):.2f}s")
for name in sorted(after, key=lambda k: -after[k])[:6]:
    print(f"  {name:22} {n[name]:6} files  before {before[name]*1000:7.0f} ms  after {after[name]*1000:7.0f} ms  ({after[name]/n[name]*1e6:.0f} vs {before[name]/n[name]*1e6:.0f} us/file)")
