import sys, types, importlib.metadata as md
import fmugen.examples as ex, fmugen.static as st
shim = types.SimpleNamespace(**{k: getattr(md, k) for k in dir(md) if not k.startswith("__")})
shim.packages_distributions = st.packages_distributions
ex.importlib = types.SimpleNamespace(metadata=shim)
sys.argv = [sys.argv[0]] + sys.argv[1:]
exec(open(sys.argv.pop(1)).read())
