"""fmugen runtime — split into focused modules for development, concatenated into one file at build time.

Modules:
    _constants  — imports, FMI constants, Status, StepRejected
    _helpers    — standalone utility functions (coerce, get_path, resolve_reference, ...)
    _engine     — Engine class and _ForwardingHandler
    _composite  — CompositeEngine, make_engine, check_spec_version

Every public name is re-exported here so that ``from fmugen.templates.fmugen_runtime import X``
works identically whether X lives in _constants, _helpers, _engine or _composite.
"""
from fmugen.templates.fmugen_runtime._composite import *
from fmugen.templates.fmugen_runtime._constants import *
from fmugen.templates.fmugen_runtime._engine import *
from fmugen.templates.fmugen_runtime._helpers import *
