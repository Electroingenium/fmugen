"""fmugen runtime engine, shared by the FMI 2 and FMI 3 adapters (resources/model.py).

This file is identical for every model and every FMI version. It reads
resources/interface.json, imports the user's code (resources/fmugen_model/) and runs it: constructing the model
object, calling it once per step or per clock tick, and moving values between FMU
variables and the arguments, return values and attributes of the user's code.

The user's code is never modified. The FMI-version-specific adapters translate
FMI calls into the methods of `Engine`.
"""

#: The interface.json format this runtime reads; fmugen build writes the same number.
SPEC_VERSION = 1

#: Start value of a variable without ``start``, by FMI type (FMI 2 and FMI 3 names).
DEFAULT_VALUES = {
    "Real": 0.0, "Float32": 0.0, "Float64": 0.0,
    "Integer": 0, "Int8": 0, "UInt8": 0, "Int16": 0, "UInt16": 0,
    "Int32": 0, "UInt32": 0, "Int64": 0, "UInt64": 0,
    "Boolean": False, "String": "", "Binary": b"", "Enumeration": 1,
}
#: (min, max) of each FMI integer type; FMI 2 ``Integer`` is 32-bit.
INT_RANGES = {
    "Int8": (-2**7, 2**7 - 1), "UInt8": (0, 2**8 - 1),
    "Int16": (-2**15, 2**15 - 1), "UInt16": (0, 2**16 - 1),
    "Int32": (-2**31, 2**31 - 1), "UInt32": (0, 2**32 - 1),
    "Int64": (-2**63, 2**63 - 1), "UInt64": (0, 2**64 - 1),
    "Integer": (-2**31, 2**31 - 1),
}
#: FMI floating-point types (FMI 2 ``Real``, FMI 3 ``Float32`` / ``Float64``).
FLOAT_TYPES = ("Real", "Float32", "Float64")
#: Strings that a Boolean variable accepts as true (lower-cased).
TRUE_STRINGS = ("true", "1", "yes", "on")
#: Strings that a Boolean variable accepts as false (lower-cased).
FALSE_STRINGS = ("false", "0", "no", "off", "")

#: FMI 3 interval qualifiers returned by ``fmi3GetInterval*``.
INTERVAL_NOT_YET_KNOWN, INTERVAL_UNCHANGED, INTERVAL_CHANGED = 0, 1, 2


class Status:
    """FMI status codes; identical numbering in FMI 2 and FMI 3.

    :cvar ok: all well.
    :cvar warning: something to report, but the computation continues.
    :cvar discard: the step's result is invalid (e.g. ``discard_on``); the importer may retry.
    :cvar error: this FMU instance failed.
    :cvar fatal: every FMU instance is corrupted.
    """
    ok = 0
    warning = 1
    discard = 2
    error = 3
    fatal = 4


class StepRejected(Exception):
    """A step the FMU can't complete (``[model] integrate``: the solver failed).

    Raised by :meth:`Engine._integrate`; it is added to ``discard_on``, so doStep restores the
    state from before the step and returns ``Status.discard``.
    """
