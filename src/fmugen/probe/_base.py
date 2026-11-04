"""Shared state of a probe: the setup calls, the time arguments, ``--read`` outputs and constant errors."""
import datetime

from fmugen.arguments import _fmi_value, _scalar_of, _typed
from fmugen.config import InterfaceError


#: Step size (and end time) passed to a probe call's time arguments.
PROBE_STEP_SIZE = 1.0


#: ``init --read``: ``{output name: "attr:<path>" or "return[:<path>]"}``, set by
#: :func:`fmugen.interface.infer_config` while it runs.
REQUESTED_READS = {}


#: Why reading a requested output failed in the latest probe: ``{name: text}``.
READ_ERRORS = {}


def _requested_outputs(data, comments, arrays, obj=None, result=None):
    """Add the ``init --read`` outputs, typed by what their path gives after the probe's step.

    A path that fails, or gives no FMI value, is recorded in :data:`READ_ERRORS` instead; the
    caller of the probe reports it if no later attempt reads it.

    :param data: the config being inferred; updated in place.
    :type data: dict
    :param comments: ``(section, key) -> text`` for :func:`render_toml`; updated in place.
    :type comments: dict
    :param arrays: whether arrays may be FMU variables (FMI 3).
    :type arrays: bool
    :param obj: the model object after the step (classes), or None.
    :param result: what the step returned.
    """
    from fmugen.templates.fmugen_runtime import get_path
    for name, source in REQUESTED_READS.items():
        where, _, path = source.partition(":")
        try:
            if where == "attr" and obj is None:
                raise InterfaceError("attr: reads the model object, and this model is called as a function: "
                                     "read its result with return:...")
            start = obj if where == "attr" else result
            value = get_path(start, path) if path else start
        except InterfaceError as e:
            READ_ERRORS[name] = str(e)
            continue
        except Exception as e:
            READ_ERRORS[name] = f"reading it failed after the probe's step: {type(e).__name__}: {e}"
            continue
        if not _fmi_value(value, arrays):
            READ_ERRORS[name] = (f"it gives {type(value).__name__}, not a number, bool or string"
                                 + ("" if arrays else " (arrays and bytes need --fmi 3)"))
            continue
        READ_ERRORS.pop(name, None)
        for section in ("outputs", "locals"):
            data.get(section, {}).pop(name, None)
        data.setdefault("outputs", {})[name] = _typed({"from": source}, value, arrays)
        shown = repr(_scalar_of(value))
        comments[("outputs", name)] = f"--read; {shown if len(shown) <= 40 else shown[:37] + '...'} in the probe"


def _run_setup(data, obj=None):
    """Run ``[model] setup`` steps for the probe: functions before construction, methods after.

    The same order as :meth:`Engine._run_setup`, so the probe sees the model as the FMU will.

    :param data: the config being inferred.
    :type data: dict
    :param obj: the constructed object (runs the method steps), or None (runs the function steps).
    """
    from fmugen.config import _setup
    from fmugen.templates.fmugen_runtime import (
        get_path,
        resolve_constant,
        resolve_reference,
    )
    for step in _setup(data["model"].get("setup", []), is_class=True):
        is_method = ":" not in step["call"]
        if is_method != (obj is not None):
            continue
        target = get_path(obj, step["call"]) if is_method else resolve_reference(step["call"])
        target(*[resolve_constant(a) for a in step["args"]],
               **{k: resolve_constant(v) for k, v in step["kwargs"].items()})


class ConstantError(InterfaceError):
    """A constant (e.g. ``--const NAME=call:...``) that can't be computed: the FMU would fail the same way.

    Re-raised instead of being turned into a comment, because no config could fix it.
    """


def _time_kwargs(time_args):
    """Values of the ``[time]`` arguments for a probe call.

    Date-times get their epoch; ``step_size`` and ``end_time`` get :data:`PROBE_STEP_SIZE`;
    everything else 0.0.

    :param time_args: the ``[time]`` table.
    :type time_args: dict
    :rtype: dict
    """
    kwargs = {}
    for arg, source in time_args.items():
        if isinstance(source, dict):   # { source = ..., epoch = ... }: a date-time
            kwargs[arg] = datetime.datetime.fromisoformat(source["epoch"])
        else:
            kwargs[arg] = {"step_size": PROBE_STEP_SIZE, "end_time": PROBE_STEP_SIZE}.get(source, 0.0)
    return kwargs
