"""Turn ordinary Python models into UniFMU (FMI 2.0 or 3.0 Co-Simulation) FMUs.

The model code is not changed. A sidecar fmugen.toml (written by `fmugen init`, or
inferred by `fmugen build model.py`) says which function or class to call and how
its arguments, return value and attributes map to FMU variables.
See README.md and docs/.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fmugen")   # from pyproject.toml, as installed
except PackageNotFoundError:   # run from a source tree that isn't installed
    __version__ = "0+unknown"

#: The only UniFMU version fmugen works with.
UNIFMU_VERSION = "1.1.0"
#: ``generationTool`` in modelDescription.xml: which fmugen and UniFMU built an FMU.
GENERATION_TOOL = f"fmugen {__version__} + unifmu {UNIFMU_VERSION}"
