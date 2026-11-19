import importlib.metadata as md
import runpy
import sys
import types

import fmugen.examples as ex
import fmugen.static as st

shim = types.SimpleNamespace(**{k: getattr(md, k) for k in dir(md) if not k.startswith("__")})
shim.packages_distributions = st.packages_distributions
ex.importlib = types.SimpleNamespace(metadata=shim)
script = sys.argv.pop(1)   # count.py: run as if started directly, with the patch in place
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
