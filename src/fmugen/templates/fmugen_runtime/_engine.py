"""Engine: the runtime for a single-model FMU."""
import asyncio
import contextlib
import copy
import datetime
import inspect
import json
import logging
import math
import os
import pickle
import sys
import traceback
import warnings
from collections.abc import Mapping
from pathlib import Path

from fmugen.templates.fmugen_runtime._constants import (
    DEFAULT_VALUES, INTERVAL_CHANGED, INTERVAL_NOT_YET_KNOWN, INTERVAL_UNCHANGED,
    Status, StepRejected,
)
from fmugen.templates.fmugen_runtime._helpers import (
    _capture_output, _release_output, _clock_of, _is_number, _quiet_debug_logging,
    assemble_items, coerce, converter, describe, dump_state,
    flatten, get_path, is_literal_constant, keyword_constants,
    load_entry, load_state, object_call, outside_bundle, pick, plain_number, reshape,
    resolve_constant, resolve_reference, results_of, rng_states, set_path,
    set_rng_states, split_global,
)


class _ForwardingHandler(logging.Handler):
    """`logging` handler that forwards the user's log records to the FMI importer.

    Installed on the root logger only while user code runs (see :meth:`Engine._forward_logs`),
    because UniFMU can only carry a log message while it is processing an FMI call. Records are
    mapped to an FMI status and log category by level: ERROR and above become ``Status.error`` /
    ``logStatusError``, WARNING becomes ``Status.warning`` / ``logStatusWarning``, the rest are
    ``Status.ok`` in the engine's info category (``logAll`` for FMI 2, ``logEvents`` for FMI 3).

    :ivar engine: the :class:`Engine` whose :meth:`Engine.log` receives the records.
    """
    def __init__(self, engine):
        """Create a handler that accepts every level (filtering happens in :meth:`emit`).

        :param engine: the engine that sends the messages to the importer.
        :type engine: Engine
        """
        super().__init__(level=logging.DEBUG)
        self.engine = engine

    def emit(self, record):
        """Forward one log record to the importer, unless logging is off.

        Nothing is sent while ``engine.logging_on`` is false (FMI 3 ``loggingOn = false`` at
        instantiation, or ``fmi2/3SetDebugLogging(false)``). An attached exception is appended as a
        formatted traceback.

        :param record: the record emitted by the user's code or a library it uses.
        :type record: logging.LogRecord
        """
        if not self.engine.logging_on:
            return
        if record.levelno >= logging.ERROR:
            status, category = Status.error, "logStatusError"
        elif record.levelno >= logging.WARNING:
            status, category = Status.warning, "logStatusWarning"
        else:
            status, category = Status.ok, self.engine.info_category
        message = f"[{record.name}] {record.getMessage()}"
        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))
        self.engine.log(message, status, category)


class Engine:
    """Runtime of a single-model FMU: runs the user's unmodified code as ``interface.json`` describes.

    The FMI adapters (``model_fmi2.py`` / ``model_fmi3.py``) translate every FMI call into one
    method of this class; values are exchanged as flat lists by value reference, and every
    method returns an FMI :class:`Status` (some also return extra values, as the adapters need).

    Lifecycle and the methods that drive it (``self.mode`` in brackets):

    1. **instantiate** (``fmi2Instantiate`` / ``fmi3InstantiateCoSimulation``) -> :meth:`__init__`:
       reads ``interface.json``, imports the user's module and entry (class or function), resolves
       enum types, converters, ``globals`` and ``discard_on``, then calls :meth:`reset`
       [``"instantiated"``]. The model object is *not* constructed yet.
    2. **configuration** (FMI 3 only, models with structural parameters):
       :meth:`enter_configuration_mode` / :meth:`exit_configuration_mode`
       [``"configuration"``, or ``"reconfiguration"`` from step mode]. Setting a structural
       parameter resizes the arrays that depend on it (:meth:`_resize_arrays`).
    3. **setup** (FMI 2 ``fmi2SetupExperiment``; FMI 3 passes the same data to
       ``fmi3EnterInitializationMode``) -> :meth:`setup_experiment`.
    4. **initialization mode** -> :meth:`enter_initialization_mode` [``"initialization"``];
       parameters and start values are set with :meth:`set_values`.
       :meth:`exit_initialization_mode` resolves ``constants``, runs ``setup`` calls, constructs
       the model object (classes), writes attribute-bound variables, optionally calls the model
       once (``entry.init_call``) and reads the initial outputs [``"step"`` or FMI 3 ``"event"``].
    5. **step mode** -> :meth:`do_step` calls the model once per communication step (or
       integrates it, ``integrate``), reads outputs and next states, ticks output clocks and
       checks ``events``. Inputs and tunable parameters arrive through :meth:`set_values`;
       outputs leave through :meth:`get_values`. A step that raises one of ``discard_on`` is
       undone from a state saved before it (:meth:`serialize` / :meth:`deserialize`).
    6. **event mode** (FMI 3 clocks) -> :meth:`enter_event_mode`, :meth:`set_clocks`,
       :meth:`update_discrete_states` (runs the code of active input clocks),
       :meth:`get_clocks` / :meth:`get_intervals`, then :meth:`enter_step_mode`.
    7. **terminate** -> :meth:`terminate` calls the model's ``entry.terminate`` method
       [``"terminated"``]; **reset** -> :meth:`reset` puts everything back to the
       instantiated state (``globals``, literal constants, start values).

    Every call into user code goes through :meth:`run`, which turns exceptions into
    ``Status.error`` (or ``Status.discard`` for ``discard_on``), forwards the model's ``logging``
    records and printed output to the importer, and runs inside ``[model] cwd``.

    Value flow: FMU values live in :attr:`values` (by variable name, in FMI form: flat lists for
    arrays, 1-based ints for enumerations). Before a call, variables bound with ``to``
    (``pos:N``, ``arg:name``, ``init:name``, ``initpos:N``, ``attr:path``, ``call:method``) are
    converted to Python (:meth:`_to_python`) and passed; after it, variables with ``from``
    (``return`` / ``attr`` / ``get``) and states with ``next`` are read back (:meth:`_read_results`,
    :meth:`_from_python`). Field meanings: ``docs/interface.md``.

    :ivar resources_dir: the FMU's ``resources/`` folder (holds ``interface.json`` and ``fmugen_model/``).
    :ivar interface: the parsed ``interface.json``.
    :ivar variables: ``interface["variables"]``, in value-reference order.
    :ivar by_name: variable name -> variable entry.
    :ivar by_reference: value reference -> variable entry.
    :ivar elements: FMI 2 only: value reference of one array element -> (array variable, index);
        FMI 2 has no arrays, so element k of an array gets the array's reference + k.
    :ivar clocks: FMI 3 clock name -> clock entry (``interface["clocks"]``).
    :ivar clock_by_reference: clock value reference -> clock entry.
    :ivar events: ``interface["events"]`` (``terminate`` / ``next_event_time`` sources).
    :ivar experiment: ``interface["experiment"]`` (default ``start_time``, ``step_size``, ``fixed_step``).
    :ivar log_categories: categories chosen with ``SetDebugLogging``; None means all.
    :ivar info_category: category of informational messages (``logAll`` FMI 2, ``logEvents`` FMI 3).
    :ivar logging_on: whether non-error messages (model logs and output) are sent at all.
    :ivar cwd: absolute ``[model] cwd`` inside the FMU, or None to leave the working directory.
    :ivar module: the user's imported module (``entry.module``).
    :ivar entry: the user's class or function (``entry.name`` in that module).
    :ivar enums: declared type name -> Python Enum class, for ``type_definitions`` with ``enum``.
    :ivar constants: resolved ``constants`` (None until initialization).
    :ivar call_constants: resolved ``call_constants`` (None until initialization).
    :ivar converters: variable name -> function applied before the model gets the value (``convert``).
    :ivar globals: (module, name) pairs of ``[model] globals``.
    :ivar global_starts: deep copies of those globals at import time, restored by :meth:`reset`.
    :ivar discard_on: exception/warning classes that make a step return discard (plus
        :class:`StepRejected` when ``integrate`` is used and state can be saved).
    :ivar is_class: whether the entry is a class (constructed at initialization) or a function.
    :ivar obj: the model object (classes only; None before initialization and for functions).
    :ivar mode: FMI state: ``instantiated``, ``configuration``, ``reconfiguration``,
        ``initialization``, ``step``, ``event`` or ``terminated``.
    :ivar values: variable name -> current FMI value.
    :ivar time: current simulation time (start of the next step).
    :ivar start_time: start time from setup (``time_args`` source ``start_time``).
    :ivar stop_time: stop time from setup, or None.
    :ivar tolerance: tolerance from setup, or None (also used as ``rtol`` for scipy integration).
    :ivar clock_active: clock name -> whether it is ticking now.
    :ivar clock_interval: clock name -> current interval in seconds, or None if not yet known.
    :ivar clock_shift: clock name -> shift in seconds.
    :ivar interval_reported: clock name -> last interval returned by :meth:`get_intervals`,
        used to compute the FMI 3 interval qualifier.
    :ivar clocks_handled: input clocks whose code already ran in the current event.
    :ivar clock_last_tick: clock name -> time of its last tick (gives the clock call's step size).
    :ivar terminate_requested: the model asked to end the simulation (``events.terminate``).
    :ivar next_event_time: the model's next time event (``events.next_event_time``), or None.
    """
    def __init__(self, resources_dir, log, info_category="logAll"):
        """Load ``interface.json`` and import the user's model (FMI instantiate).

        Called once per FMU instance from the adapter's ``Model.__init__``, i.e. on
        ``fmi2Instantiate`` / ``fmi3InstantiateCoSimulation``. Imports run inside ``[model] cwd``
        because a model may read files at import time. ``constants`` are *not* resolved here
        (a ``{ call = ... }`` may download or compute): that happens at initialization, where a
        failure reaches the importer with its traceback. Ends with :meth:`reset`, so all start
        values are in place and ``mode`` is ``"instantiated"``.

        Side effects: may install output capture (``capture_output``, UniFMU only), raises the root
        logger's level from DEBUG to INFO inside UniFMU, sets Hugging Face offline environment variables when
        ``resources/hf_cache`` exists, prepends ``sys_path`` entries to ``sys.path``.

        :param resources_dir: the FMU's ``resources/`` folder (or a composite part's folder).
        :type resources_dir: str or pathlib.Path
        :param log: ``log(message, status, category)``, forwards a message to the importer.
        :type log: callable
        :param info_category: log category of informational messages (``logAll`` for FMI 2,
            ``logEvents`` for FMI 3).
        :type info_category: str
        :raises Exception: any import error of the user's module, ``globals`` or ``discard_on``
            references; UniFMU then reports the instantiation as failed.
        """
        self.resources_dir = Path(resources_dir)
        self._send_log = log
        self.log_categories = None   # fmi2/3SetDebugLogging: None = every category
        self.info_category = info_category
        self.interface = json.loads((self.resources_dir / "interface.json").read_text())
        self._output, self._position = _capture_output(self.interface.get("capture_output", False)), 0
        self.variables = self.interface["variables"]
        self.by_name = {v["name"]: v for v in self.variables}
        self.by_reference = {v["valueReference"]: v for v in self.variables}
        # FMI 2 has no arrays: element k of an array has its own reference, the array's + k
        self.elements = {} if self.interface.get("fmi_version") != 2 else {
            v["valueReference"] + k: (v, k) for v in self.variables if v.get("dimensions")
            for k in range(math.prod(v["dimensions"]))}
        self.clocks = {c["name"]: c for c in self.interface.get("clocks", [])}
        self.clock_by_reference = {c["valueReference"]: c for c in self.clocks.values()}
        self.events = self.interface.get("events", {})
        self.experiment = self.interface.get("experiment", {})

        self.logging_on = True
        _quiet_debug_logging()
        hf_cache = outside_bundle(self.resources_dir) / "hf_cache"
        if hf_cache.is_dir():   # Hugging Face models bundled by fmugen build: load them from the FMU, offline
            os.environ.update(HF_HUB_CACHE=str(hf_cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        cwd = self.interface.get("cwd")
        # kept outside a compiled backend's bundle, at the same place under resources/
        self.cwd = outside_bundle(self.resources_dir) / cwd if cwd is not None else None
        try:   # an import error here ends the backend: its traceback must reach the console
            with self._in_model_dir():   # the model may read files when it is imported
                self.module, self.entry = load_entry(self.interface, self.resources_dir)
                self.enums = {
                    name: resolve_reference(td["enum"])
                    for name, td in self.interface.get("type_definitions", {}).items() if td.get("enum")
                }
                # constants ({ call = ... } may download or compute) are resolved when the FMU initializes,
                # where a failure is reported to the importer with its traceback (resolve_constants)
                self.constants = self.call_constants = None
                # "series" dates its value at the FMU time (the start of the step being made)
                self.converters = {v["name"]: converter(v, lambda: self.time) for v in self.variables if v.get("convert")}
                # [model] globals: module-level state, put back to its import-time value on reset
                self.globals = [split_global(ref) for ref in self.interface.get("globals", [])]
                self.global_starts = copy.deepcopy(self._read_globals())
                # [model] discard_on: exceptions and warnings that make doStep return discard (the step is undone)
                self.discard_on = tuple(resolve_reference(ref) for ref in self.interface.get("discard_on", []))
            if self.interface.get("integrate") and self.interface.get("save_state", True) is not False:
                self.discard_on += (StepRejected,)   # a failed solve rejects the step (undone from the saved state)
            self._loop = None   # the event loop of an async model, kept for the whole run
            self.is_class = self.interface["entry"]["kind"] == "class"
            self.reset()
        except BaseException as e:
            # the exception's own line first: UniFMU cuts the backend's output short, and the
            # warnings printed while importing and a long traceback would push it out
            _release_output(self._output, f"[fmugen] the model failed to load: {type(e).__name__}: {e}\n")
            raise
        self._forward_output()   # e.g. warnings printed while the model was imported

    # ================= life cycle =================

    def reset(self):
        """Put the FMU back into the state right after instantiation (``fmi2Reset`` / ``fmi3Reset``).

        Also the last step of :meth:`__init__`. If a model object exists and was not terminated, its
        ``entry.terminate`` method runs first, so it releases resources (ports, files) now rather than
        when garbage-collected. Then ``globals`` get their import-time values back, literal
        ``constants`` (not ``{ call = ... }`` / ``ref`` ones, which stay shared) are rebuilt from
        ``interface.json`` in case the model changed them in place, every variable gets its ``start``
        value (arrays last, because their sizes may depend on structural parameters), the model
        object is dropped and all clock/event bookkeeping is cleared. ``time`` goes back to
        ``experiment.start_time``.

        :return: always ``Status.ok`` (a failing terminate method is logged by :meth:`run`).
        :rtype: int
        """
        method = self.interface["entry"].get("terminate")
        if method and getattr(self, "obj", None) is not None and self.mode != "terminated":
            # release what the model holds (serial ports, files) now, not when it is garbage collected
            self.run("terminate (reset)", lambda: self._invoke(getattr(self.obj, method)))
        if getattr(self, "global_starts", None):
            self._write_globals(copy.deepcopy(self.global_starts))
        if getattr(self, "constants", None) is not None:   # constants the model may have changed in place
            self._write_literal_constants({
                table: {name: resolve_constant(copy.deepcopy(value))
                        for name, value in self.interface.get(table, {}).items() if is_literal_constant(value)}
                for table in ("constants", "call_constants")})
        self.values = {}
        for v in self.variables:
            if v["causality"] != "independent" and not v.get("dimensions"):
                self.values[v["name"]] = self._start(v)
        for v in self.variables:  # arrays last: their sizes may depend on structural parameters
            if v.get("dimensions"):
                self.values[v["name"]] = self._start(v)
        self.obj = None
        self.mode = "instantiated"
        self.time = self.experiment.get("start_time", 0.0)
        self.start_time, self.stop_time, self.tolerance = self.time, None, None
        # call: setters: value last passed, run at initialization and not yet by a step, run this step
        self._setter_last, self._setter_fresh, self._setter_called = {}, set(), set()
        self.clock_active = {name: False for name in self.clocks}
        self.clock_interval = {name: c.get("interval") for name, c in self.clocks.items()}
        self.clock_shift = {name: c.get("shift", 0.0) for name, c in self.clocks.items()}
        self.interval_reported = {}
        self.clocks_handled = set()
        self.clock_last_tick = {}
        self.terminate_requested = False
        self.next_event_time = None
        return Status.ok

    def setup_experiment(self, start_time, stop_time=None, tolerance=None):
        """Store the experiment's start time, stop time and tolerance.

        FMI 2: ``fmi2SetupExperiment``. FMI 3 has no such call: the adapter calls this from
        ``fmi3EnterInitializationMode`` with the same data. The values are later passed to the model
        through ``time_args`` (``start_time``, ``stop_time``, ``tolerance``) and the tolerance is the
        ``rtol`` of ``integrate = "scipy:..."``.

        :param start_time: simulation start; also becomes the current ``time``.
        :type start_time: float
        :param stop_time: simulation end, or None when the importer did not define one.
        :type stop_time: float or None
        :param tolerance: relative tolerance, or None when not defined.
        :type tolerance: float or None
        :return: ``Status.ok``.
        :rtype: int
        """
        self.time = start_time
        self.start_time, self.stop_time, self.tolerance = start_time, stop_time, tolerance
        return Status.ok

    def enter_initialization_mode(self):
        """Enter initialization mode (``fmi2/3EnterInitializationMode``).

        Only changes ``mode`` to ``"initialization"``, which (see :meth:`_settable`) still allows
        fixed parameters and ``initial = "exact"`` locals to be set. Nothing of the model runs yet.

        :return: ``Status.ok``.
        :rtype: int
        """
        self.mode = "initialization"
        return Status.ok

    def exit_initialization_mode(self, next_mode="step"):
        """Build and initialize the model (``fmi2/3ExitInitializationMode``).

        This is where the user's code first runs, inside :meth:`run` (so any exception becomes
        ``Status.error`` with its traceback). See the nested ``initialize`` for the order of steps.
        On success ``mode`` becomes ``next_mode``.

        :param next_mode: mode after initialization: ``"step"``, or ``"event"`` for an FMI 3 importer
            that set ``eventModeUsed`` (the FMI 3 standard leaves it in event mode then).
        :type next_mode: str
        :return: the status of initialization.
        :rtype: int
        """
        def initialize():
            """Initialize the model, in order.

            1. Resolve ``constants`` / ``call_constants`` (:meth:`_resolve_constants`).
            2. Run ``setup`` calls that are functions (``"module:function"``).
            3. Classes: construct the object -- the class itself, or its ``entry.create`` factory --
               with keyword constants, ``init:`` and ``initpos:`` bound parameters; run ``setup``
               calls that are methods; write ``attr:`` / ``call:`` bound variables (unclocked and of
               every clock).
            4. With ``entry.init_call``: call the model once at the start time (step size
               ``experiment.step_size``) and read its outputs; otherwise read only the outputs that
               are available without a call (attributes); when a ``get:`` output raises there, the
               model is called once as with ``init_call``. Missing values are tolerated here.
            5. Read the outputs of every input clock that are available as attributes.
            """
            self._resolve_constants()
            self._run_setup(after_construction=False)
            if self.is_class:
                kwargs = keyword_constants(self.constants)
                kwargs.update(self._bound("init"))
                create = self.interface["entry"].get("create")
                self.obj = self._invoke(getattr(self.entry, create) if create else self.entry,
                                        *self._positional(kind="initpos"), **kwargs)
                self._run_setup(after_construction=True)
                self._setter_last, self._setter_fresh, self._setter_called = {}, set(), set()
                self._apply_attributes(clock=None, initial=True)
                for name in self.clocks:
                    self._apply_attributes(clock=name, initial=True)
            init_call = self.interface["entry"].get("init_call")
            if not init_call:
                try:
                    self._read_results(None, have_result=False, update_states=False, missing_ok=True)
                except Exception as e:
                    # a get: output whose method raises until the model has run (floris' get_farm_power()):
                    # call the model once, as init_call = true does, then read again
                    getters = [v["name"] for v in self.variables if v.get("from", {}).get("kind") == "get"]
                    if not getters or not self.interface["entry"].get("step", True) or self.interface.get("integrate"):
                        raise
                    self.log(f"reading {', '.join(getters)} failed before the first step ({e!r}): the model is "
                             "called once at initialization, as with [model] init_call = true",
                             Status.ok, self.info_category)
                    init_call = True
            if init_call:
                result = self._call(self.time, self.experiment.get("step_size", 0.0))
                self._read_results(result, have_result=True, update_states=False, missing_ok=True)
            for name, clock in self.clocks.items():
                if clock["causality"] == "input":
                    self._read_results(None, have_result=False, update_states=False, missing_ok=True, clock=name)

        status = self.run("initialization", initialize)
        if status == Status.ok:
            self.mode = next_mode
        return status

    def do_step(self, current_time, step_size):
        """Advance the model by one communication step (``fmi2DoStep`` / ``fmi3DoStep``).

        Valid only in step mode. With ``experiment.fixed_step``, any other step size is an error. A
        ``current_time`` that differs from the FMU's time (the importer skipped or repeated time
        without restoring a state) is only a warning: the model goes on from its current state.

        When ``discard_on`` is set and ``save_state`` allows it, the whole FMU state is serialized
        first; if the step then raises (or warns) one of the ``discard_on`` classes -- or
        :class:`StepRejected` from a failed integration -- the state is restored and the step
        returns ``Status.discard``, so the importer can retry with a smaller step. ``time`` only
        advances on ``Status.ok``.

        :param current_time: the communication point at the start of the step.
        :type current_time: float
        :param step_size: the communication step size.
        :type step_size: float
        :return: ``(status, event_handling_needed)``; the second is true when an output clock
            ticked during the step (FMI 3 ``eventHandlingNeeded``; ignored by FMI 2).
        :rtype: tuple[int, bool]
        """
        if self.mode != "step":
            return self.error(f"doStep called in mode {self.mode!r}"), False
        fixed = self.experiment.get("step_size") if self.experiment.get("fixed_step") else None
        if fixed is not None and not math.isclose(step_size, fixed, rel_tol=1e-9, abs_tol=1e-12):
            return self.error(f"this FMU only supports a communication step size of {fixed}, got {step_size}"), False
        if not math.isclose(current_time, self.time, rel_tol=1e-9, abs_tol=1e-9):
            # the importer skipped or repeated time without restoring a saved FMU state
            self.log(f"doStep at time {current_time}, but the FMU is at time {self.time} (no FMU state was "
                     "restored): the model continues from its current state", Status.warning, "logStatusWarning")
        # [model] discard_on: the state before the step, to undo a step the model rejects
        saved = self.serialize()[1] if self.discard_on and self.interface.get("save_state", True) else None

        ticked = []
        self._setter_called.clear()   # setters run once in this step

        def advance():
            """Run one step of the user's code and collect its results.

            Three cases, by ``interface.json``:

            * ``integrate``: the model returns derivatives; :meth:`_integrate` advances the states
              over the step, then the model is called once more at the end time to read outputs.
            * ``entry.step`` true (the default): call the model with the step's inputs, once or
              ``repeat`` times over equal parts of the step (``[model] substeps``), then read outputs
              (``from``) and next states (``next``).
            * ``entry.step`` false (``[model] call = false``): only clocks run code; read attribute
              outputs only.

            Afterwards output clocks are ticked (names collected in ``ticked``) and ``events`` checked.
            """
            if self.interface.get("integrate"):   # the model returns derivatives: integrate, then read outputs
                self._integrate(current_time, step_size)
                result, have_result = self._call(current_time + step_size, step_size), True
                self._read_results(result, have_result=True, update_states=False, missing_ok=False)
            elif self.interface["entry"].get("step", True):
                repeat = self.interface.get("repeat", 1)   # [model] substeps without integrate
                h = step_size / repeat
                for i in range(repeat):
                    result = self._call(current_time + i * h, h)
                have_result = True
                self._read_results(result, have_result=True, update_states=True, missing_ok=False)
            else:  # [model] call = false: only clocks run code
                result, have_result = None, False
                self._read_results(None, have_result=False, update_states=False, missing_ok=True)
            ticked.extend(self._check_output_clocks(result, have_result=have_result))
            self._check_events(result, have_result=have_result)

        status = self.run("doStep", advance, discard=self.discard_on)
        if status == Status.discard:
            if saved:
                self.deserialize(saved)
            return status, False
        if status == Status.ok:
            self.time = current_time + step_size
        return status, bool(ticked)

    def terminate(self):
        """End the simulation (``fmi2Terminate`` / ``fmi3Terminate``).

        Calls the model object's ``entry.terminate`` method (e.g. ``close``) when one is configured
        and the object exists, then sets ``mode`` to ``"terminated"`` so :meth:`reset` doesn't call
        it a second time.

        :return: the status of the terminate method, or ``Status.ok``.
        :rtype: int
        """
        method = self.interface["entry"].get("terminate")
        status = Status.ok
        if method and self.obj is not None:
            status = self.run("terminate", lambda: self._invoke(getattr(self.obj, method)))
        self.mode = "terminated"
        return status

    # ================= FMI 3 modes =================

    def enter_configuration_mode(self):
        """Enter FMI 3 (re)configuration mode (``fmi3EnterConfigurationMode``).

        From ``"instantiated"`` the FMU enters ``"configuration"`` (structural parameters can be
        set); from ``"step"`` it enters ``"reconfiguration"`` (only tunable structural parameters).
        FMI 2 has no configuration mode.

        :return: ``Status.ok``, or ``Status.error`` if the FMU has no structural parameters or the
            mode does not allow it.
        :rtype: int
        """
        if not any(v["causality"] == "structuralParameter" for v in self.variables):
            return self.error("this FMU has no structural parameters")
        if self.mode == "instantiated":
            self.mode = "configuration"
        elif self.mode == "step":
            self.mode = "reconfiguration"
        else:
            return self.error(f"cannot enter configuration mode from mode {self.mode!r}")
        return Status.ok

    def exit_configuration_mode(self):
        """Leave FMI 3 (re)configuration mode (``fmi3ExitConfigurationMode``).

        Goes back to the mode configuration was entered from: ``"instantiated"`` or ``"step"``.

        :return: ``Status.ok``, or ``Status.error`` when not in a configuration mode.
        :rtype: int
        """
        if self.mode == "configuration":
            self.mode = "instantiated"
        elif self.mode == "reconfiguration":
            self.mode = "step"
        else:
            return self.error(f"not in configuration mode (mode {self.mode!r})")
        return Status.ok

    def enter_event_mode(self):
        """Enter FMI 3 event mode (``fmi3EnterEventMode``).

        Starts a new event: no input clock's code has run in it yet (``clocks_handled`` is
        cleared), so :meth:`update_discrete_states` will run each active one once.

        :return: ``Status.ok``.
        :rtype: int
        """
        self.mode = "event"
        self.clocks_handled = set()
        return Status.ok

    def enter_step_mode(self):
        """Leave FMI 3 event mode for step mode (``fmi3EnterStepMode``).

        Every clock stops ticking: ``clock_active`` is set to false for all of them.

        :return: ``Status.ok``.
        :rtype: int
        """
        self.mode = "step"
        for name in self.clock_active:
            self.clock_active[name] = False
        return Status.ok

    def update_discrete_states(self):
        """Run the code of every active input clock that hasn't run in this event yet.

        FMI 3 only: ``fmi3UpdateDiscreteStates``, in event mode, after the importer activated
        clocks with :meth:`set_clocks`. Each clock's code runs at most once per event
        (``clocks_handled``), so repeated calls in the same event are harmless.

        :return: ``(status, terminate_simulation, next_event_time or None)``, from
            ``terminate_requested`` and ``next_event_time`` as set by the clocks' ``events``.
        :rtype: tuple[int, bool, float or None]
        """
        def update():
            """For each active, not yet handled input clock with a ``call``: call it and read its results.

            The clock call gets as step size the time since that clock last ticked (its interval on the
            first tick). Then its clocked outputs are read, then unclocked attribute outputs (the clock's
            code may have changed them), output clocks and ``events`` are checked and the clock's next
            interval is taken from ``interval_from``.
            """
            for name, clock in self.clocks.items():
                if clock["causality"] != "input" or not self.clock_active[name] or name in self.clocks_handled:
                    continue
                self.clocks_handled.add(name)
                if not clock.get("call"):
                    continue
                # step size of a clock call: time since this clock last ticked (its interval the first time)
                last = self.clock_last_tick.get(name)
                step_size = self.time - last if last is not None else (self.clock_interval.get(name) or 0.0)
                self.clock_last_tick[name] = self.time
                self._setter_called.clear()   # setters run once in this tick
                result = self._call(self.time, step_size, clock=name)
                self._read_results(result, have_result=True, update_states=False, missing_ok=False, clock=name)
                # the clock's code may also have changed attributes behind unclocked outputs
                self._read_results(None, have_result=False, update_states=False, missing_ok=True)
                self._check_output_clocks(result, have_result=True)
                self._check_events(result, have_result=True)
                self._remember_interval(name, result, have_result=True)

        status = self.run("updateDiscreteStates", update)
        return status, self.terminate_requested, self.next_event_time

    # ================= values =================

    def size(self, var):
        """Number of scalar values of a variable, or None for a scalar.

        Dimensions that are names of structural parameters use their current value, so the result
        changes when such a parameter is set.

        :param var: an entry of ``interface["variables"]``.
        :type var: dict
        :return: the product of its ``dimensions``, or None without dimensions.
        :rtype: int or None
        """
        dims = var.get("dimensions")
        if not dims:
            return None
        n = 1
        for d in dims:
            n *= int(self.values[d]) if isinstance(d, str) else d
        return n

    def get_values(self, references):
        """Return the current values of the variables with these references (``fmi2/3Get*``).

        Used for every type (``GetReal``, ``GetFloat64``, ``GetString``, ``GetBinary``, ...): values
        are already in FMI form. An array contributes all its elements (FMI 3); an FMI 2 element
        reference gives one element; the independent variable gives the current time. Nothing of the
        model runs: outputs were read at the end of the last step or event.

        :param references: value references, in the order the importer asked for them.
        :type references: list[int]
        :return: ``(status, flat list of values)``; ``Status.error`` and ``[]`` for an unknown reference.
        :rtype: tuple[int, list]
        """
        values = []
        for r in references:
            if r in self.elements:
                var, k = self.elements[r]
                values.append(self.values[var["name"]][k])
                continue
            var = self.by_reference.get(r)
            if var is None:
                return self.error(f"unknown value reference {r}"), []
            if var["causality"] == "independent":
                values.append(self.time)
            elif var.get("dimensions"):
                values.extend(self.values[var["name"]])
            else:
                values.append(self.values[var["name"]])
        return Status.ok, values

    def set_values(self, references, values):
        """Set variables from the importer (``fmi2/3Set*``).

        Values are consumed from the flat list in order: one per scalar, :meth:`size` per array, one
        per FMI 2 element reference. Each value is checked against what may be set in the current
        mode (:meth:`_settable`) and coerced to the variable's FMI type. Setting a structural
        parameter resizes the arrays that depend on it.

        Tuning a parameter after initialization (step, event or reconfiguration mode) also writes it
        straight onto the model object, through its ``attr``, and refreshes ``calculatedParameter``
        outputs read from attributes. The attribute must already exist: ``setattr`` would otherwise
        add one the model never reads, so this is an error and the old value is kept. Other
        variables only reach the model on the next call (see :meth:`_call`).

        :param references: value references.
        :type references: list[int]
        :param values: the values, flattened.
        :type values: list
        :return: ``Status.ok``, or the first error (unknown reference, too few values, not settable
            now, wrong type, failing setattr).
        :rtype: int
        """
        values = list(values)
        cursor = 0
        for r in references:
            var, element = self.elements.get(r, (self.by_reference.get(r), None))
            if var is None:
                return self.error(f"unknown value reference {r}")
            n = self.size(var) if element is None else None
            if len(values) - cursor < (n or 1):
                return self.error(f"not enough values for {var['name']}")
            raw = values[cursor:cursor + n] if n is not None else values[cursor]
            cursor += n or 1
            if element is not None:   # one element of an FMI 2 array
                current = self.values[var["name"]]
                raw = [*current[:element], raw, *current[element + 1:]]
            problem = self._settable(var)
            if problem:
                return self.error(f"{var['name']} ({var['causality']}) cannot be set {problem}")
            try:
                value = [coerce(var["type"], x) for x in raw] if isinstance(raw, list) else coerce(var["type"], raw)
            except (TypeError, ValueError) as e:
                return self.error(f"{var['name']}: {e}")
            name = var["name"]
            previous, self.values[name] = self.values.get(name), value

            if var["causality"] == "structuralParameter":
                self._resize_arrays(name)
            if (self.mode in ("step", "event", "reconfiguration") and self.obj is not None and var.get("attr")
                    and var["causality"] in ("parameter", "structuralParameter")):
                try:
                    get_path(self.obj, var["attr"])
                except (AttributeError, KeyError, IndexError):
                    # setattr would add an attribute the model never reads: the new value would be lost
                    self.values[name] = previous
                    return self.error(f"{name}: the model object has no attribute {var['attr']!r} to tune; "
                                      f"set attr = \"...\" in the config to the attribute that holds {name}")
                status = self.run(f"setting {name}",
                                  lambda: setattr(self.obj, var["attr"], self._to_python(var, self.values[name])))
                if status != Status.ok:
                    return status
                self.run("updating calculated parameters", self._refresh_calculated_parameters)
        return Status.ok

    def _settable(self, var):
        """Return why ``var`` can't be set in the current mode, or None if it can.

        Implements the FMI causality/variability rules: inputs always; parameters before or during
        initialization, and tunable ones afterwards; structural parameters in configuration or
        initialization mode, tunable ones also in reconfiguration; ``initial = "exact"`` locals before
        initialization. Everything else (outputs, calculated parameters) never.

        :param var: an entry of ``interface["variables"]``.
        :type var: dict
        :return: the end of an error message, or None.
        :rtype: str or None
        """
        causality, mode = var["causality"], self.mode
        initializing = mode in ("instantiated", "initialization", "configuration")
        if causality == "input":
            return None
        if causality == "parameter":
            if initializing or var["variability"] == "tunable":
                return None
            return "after initialization (it is a fixed parameter)"
        if causality == "structuralParameter":
            if mode in ("configuration", "initialization") or (
                    mode == "reconfiguration" and var["variability"] == "tunable"):
                return None
            return "outside configuration mode"
        if causality == "local" and var.get("initial") == "exact" and initializing:
            return None
        return "by the importer" if not initializing else "before initialization"

    def _start(self, var):
        """The start value of a variable, in FMI form, sized for its current dimensions.

        Uses the variable's ``start`` or the type's default; Binary starts are hex strings in
        ``interface.json``. For arrays, a single start value is repeated over all elements.

        :param var: an entry of ``interface["variables"]``.
        :type var: dict
        :return: a scalar, or a list of :meth:`size` values.
        :raises ValueError: an array start with a wrong number of values (other than one).
        """
        start = var.get("start", DEFAULT_VALUES[var["type"]])
        if var["type"] == "Binary":
            start = [bytes.fromhex(s) for s in start] if isinstance(start, list) else bytes.fromhex(start or "")
        n = self.size(var)
        if n is None:
            return start
        if isinstance(start, list):
            if len(start) == n:
                return list(start)
            if len(start) == 1:
                return start * n
            raise ValueError(f"{var['name']}: start has {len(start)} values, expected {n}")
        return [start] * n

    def _resize_arrays(self, structural_name):
        """Give new start values to every array whose dimensions use a structural parameter.

        Called by :meth:`set_values` after ``structural_name`` changed. If an array's own start no
        longer fits the new size, it falls back to the type's default (with a warning).

        :param structural_name: name of the structural parameter that was set.
        :type structural_name: str
        """
        for v in self.variables:
            if structural_name in (v.get("dimensions") or []):
                try:
                    self.values[v["name"]] = self._start(v)
                except ValueError:
                    default = DEFAULT_VALUES[v["type"]]
                    self.values[v["name"]] = [default] * self.size(v)
                    self.log(f"{v['name']} was reset to {default!r} because {structural_name} changed its size; "
                             "set it before initialization", Status.warning, "logStatusWarning")

    # ================= clocks =================

    def drop_output_ticks(self):
        """Forget output clock ticks the importer won't read.

        FMI 3 only: called by the adapter's ``fmi3DoStep`` when an output clock ticked but the importer
        does not use event mode (``eventModeUsed = false``), so it would never call ``fmi3GetClock``.

        :return: ``Status.ok``.
        :rtype: int
        """
        for name, clock in self.clocks.items():
            if clock["causality"] == "output":
                self.clock_active[name] = False
        return Status.ok

    def set_clocks(self, references, values):
        """Activate or deactivate input clocks (``fmi3SetClock``, event mode).

        The clock's code runs later, in :meth:`update_discrete_states`.

        :param references: clock value references.
        :type references: list[int]
        :param values: one bool per reference.
        :type values: list[bool]
        :return: ``Status.ok``, or ``Status.error`` for a reference that isn't an input clock.
        :rtype: int
        """
        for r, value in zip(references, values):
            clock = self.clock_by_reference.get(r)
            if clock is None or clock["causality"] != "input":
                return self.error(f"value reference {r} is not an input clock")
            self.clock_active[clock["name"]] = bool(value)
        return Status.ok

    def get_clocks(self, references):
        """Return whether clocks are ticking (``fmi3GetClock``).

        Reading an output clock resets it, as FMI 3 requires (each tick is reported once).

        :param references: clock value references.
        :type references: list[int]
        :return: ``(status, list of bool)``.
        :rtype: tuple[int, list[bool]]
        """
        values = []
        for r in references:
            clock = self.clock_by_reference.get(r)
            if clock is None:
                return self.error(f"value reference {r} is not a clock"), []
            values.append(self.clock_active[clock["name"]])
            if clock["causality"] == "output":
                self.clock_active[clock["name"]] = False  # output clocks reset once read
        return Status.ok, values

    def get_intervals(self, references):
        """Return clock intervals and their qualifiers (``fmi3GetIntervalDecimal`` / ``Fraction``).

        The qualifier says whether the interval is not yet known, unchanged or changed since it was
        last read. For ``changing``, ``countdown`` and ``tunable`` clocks it compares with the
        interval last reported; for the others only the first read is "changed".

        :param references: clock value references.
        :type references: list[int]
        :return: ``(status, intervals, qualifiers)``; an unknown interval is 0.0 with
            ``INTERVAL_NOT_YET_KNOWN``.
        :rtype: tuple[int, list[float], list[int]]
        """
        intervals, qualifiers = [], []
        for r in references:
            clock = self.clock_by_reference.get(r)
            if clock is None:
                return self.error(f"value reference {r} is not a clock"), [], []
            name = clock["name"]
            interval = self.clock_interval.get(name)
            if interval is None:
                intervals.append(0.0)
                qualifiers.append(INTERVAL_NOT_YET_KNOWN)
                continue
            if clock["interval_variability"] in ("changing", "countdown", "tunable"):
                qualifier = INTERVAL_UNCHANGED if self.interval_reported.get(name) == interval else INTERVAL_CHANGED
            else:
                qualifier = INTERVAL_CHANGED if name not in self.interval_reported else INTERVAL_UNCHANGED
            self.interval_reported[name] = interval
            intervals.append(float(interval))
            qualifiers.append(qualifier)
        return Status.ok, intervals, qualifiers

    def set_intervals(self, references, intervals):
        """Set clock intervals from the importer (``fmi3SetIntervalDecimal`` / ``Fraction``).

        Allowed for ``fixed`` clocks before initialization ends and for ``tunable`` ones also in
        event mode, as FMI 3 specifies.

        :param references: clock value references.
        :type references: list[int]
        :param intervals: intervals in seconds.
        :type intervals: list[float]
        :return: ``Status.ok``, or ``Status.error`` when not a clock or not settable now.
        :rtype: int
        """
        for r, interval in zip(references, intervals):
            clock = self.clock_by_reference.get(r)
            if clock is None:
                return self.error(f"value reference {r} is not a clock")
            variability = clock["interval_variability"]
            allowed = {"fixed": ("instantiated", "initialization"), "tunable": ("instantiated", "initialization", "event")}
            if self.mode not in allowed.get(variability, ()):
                return self.error(f"the interval of {clock['name']} ({variability}) cannot be set in mode {self.mode!r}")
            self.clock_interval[clock["name"]] = float(interval)
        return Status.ok

    def get_shifts(self, references):
        """Return clock shifts in seconds (``fmi3GetShiftDecimal`` / ``Fraction``).

        :param references: clock value references.
        :type references: list[int]
        :return: ``(status, shifts)``.
        :rtype: tuple[int, list[float]]
        """
        shifts = []
        for r in references:
            clock = self.clock_by_reference.get(r)
            if clock is None:
                return self.error(f"value reference {r} is not a clock"), []
            shifts.append(float(self.clock_shift[clock["name"]]))
        return Status.ok, shifts

    def set_shifts(self, references, shifts):
        """Set clock shifts (``fmi3SetShiftDecimal`` / ``Fraction``).

        Only for ``fixed`` and ``tunable`` clocks, before initialization ends.

        :param references: clock value references.
        :type references: list[int]
        :param shifts: shifts in seconds.
        :type shifts: list[float]
        :return: ``Status.ok``, or ``Status.error``.
        :rtype: int
        """
        for r, shift in zip(references, shifts):
            clock = self.clock_by_reference.get(r)
            if clock is None:
                return self.error(f"value reference {r} is not a clock")
            if clock["interval_variability"] not in ("fixed", "tunable") or self.mode not in ("instantiated", "initialization"):
                return self.error(f"the shift of {clock['name']} cannot be set now")
            self.clock_shift[clock["name"]] = float(shift)
        return Status.ok

    def _check_output_clocks(self, result, have_result):
        """Tick output clocks whose source is truthy, and read their clocked variables.

        Called after every step and every input clock call. An output clock's ``from`` source (a
        return value or attribute) is evaluated; when it is truthy the clock becomes active and the
        variables in that clock are read from the same result.

        :param result: what the model call returned (ignored when ``have_result`` is false).
        :param have_result: whether ``result`` comes from a call made just now.
        :type have_result: bool
        :return: names of the clocks that ticked.
        :rtype: list[str]
        """
        ticked = []
        for name, clock in self.clocks.items():
            if clock["causality"] != "output":
                continue
            found, value = self._source_value(clock["from"], result, have_result, missing_ok=True)
            if found and value:
                self.clock_active[name] = True
                ticked.append(name)
                self._read_results(result, have_result=have_result, update_states=False, missing_ok=False, clock=name)
        return ticked

    def _remember_interval(self, name, result, have_result):
        """Update a clock's interval from its ``interval_from`` source, when it has one.

        Lets a model set the time until its clock ticks next (``countdown`` / ``changing`` clocks).
        A missing or None value keeps the current interval.

        :param name: clock name.
        :type name: str
        :param result: what the model call returned.
        :param have_result: whether ``result`` comes from a call made just now.
        :type have_result: bool
        """
        source = self.clocks[name].get("interval_from")
        if source:
            found, value = self._source_value(source, result, have_result, missing_ok=True)
            if found and value is not None:
                self.clock_interval[name] = float(value)

    def _check_events(self, result, have_result):
        """Read the model's event requests after a step or clock call.

        Updates every input clock's interval (``interval_from``), ``terminate_requested`` from
        ``events.terminate`` and ``next_event_time`` from ``events.next_event_time``. FMI 3 reports
        them in ``fmi3DoStep`` (``terminateSimulation``) and ``fmi3UpdateDiscreteStates``.

        :param result: what the model call returned.
        :param have_result: whether ``result`` comes from a call made just now.
        :type have_result: bool
        """
        for name, clock in self.clocks.items():
            if clock["causality"] == "input":
                self._remember_interval(name, result, have_result)
        if "terminate" in self.events:
            found, value = self._source_value(self.events["terminate"], result, have_result, missing_ok=True)
            self.terminate_requested = bool(found and value)
        if "next_event_time" in self.events:
            found, value = self._source_value(self.events["next_event_time"], result, have_result, missing_ok=True)
            self.next_event_time = float(value) if found and value is not None else None

    # ================= state =================

    def serialize(self):
        """The FMU state as bytes (``fmi2/3GetFMUstate`` + ``SerializeFMUstate`` in UniFMU).

        Holds the variable values, the model object, time, mode, all clock bookkeeping, ``globals``,
        the states of the global random number generators (so a rolled-back step draws the same
        numbers) and literal constants. With ``save_state = [attributes]`` only those attribute
        paths of the model object are saved; the rest (e.g. an ONNX session or a device handle)
        stays as it is. Pickled with pickle, cloudpickle or dill (:func:`dump_state`). Also used
        internally by :meth:`do_step` to undo discarded steps.

        :return: ``(status, bytes)``; ``Status.error`` and ``b""`` when the state can't be pickled.
        :rtype: tuple[int, bytes]
        """
        save = self.interface.get("save_state", True)
        if isinstance(save, list):
            obj = None if self.obj is None else {path: get_path(self.obj, path) for path in save}
        else:
            obj = self.obj
        state = {
            "values": self.values, "obj": obj, "time": self.time, "mode": self.mode,
            "clock_active": self.clock_active, "clock_interval": self.clock_interval,
            "clock_shift": self.clock_shift, "terminate_requested": self.terminate_requested,
            "clock_last_tick": self.clock_last_tick, "globals": self._read_globals(), "rng": rng_states(),
            "constants": self._literal_constants(),
        }
        try:
            return Status.ok, dump_state(state)
        except Exception as e:
            self.log(f"cannot serialize FMU state: {e!r}", Status.error, "logStatusError")
            return Status.error, b""

    def deserialize(self, data):
        """Restore a state made by :meth:`serialize` (``fmi2/3SetFMUstate`` + ``DeSerializeFMUstate``).

        With ``save_state = [attributes]`` the saved attributes are written back onto the current
        model object; otherwise the whole object is replaced. Each load produces new copies, so the
        same state can be restored several times.

        :param data: bytes from :meth:`serialize`.
        :type data: bytes
        :return: ``Status.ok``, or ``Status.error`` when the bytes can't be unpickled.
        :rtype: int
        """
        try:
            state = load_state(data)
        except Exception as e:
            return self.error(f"cannot deserialize FMU state: {e!r}")
        self.values = dict(state["values"])
        if isinstance(self.interface.get("save_state", True), list):
            if self.obj is not None and state["obj"] is not None:
                for path, value in state["obj"].items():
                    set_path(self.obj, path, value)
        else:
            self.obj = state["obj"]
        self.time, self.mode = state["time"], state["mode"]
        self.clock_active = dict(state.get("clock_active", self.clock_active))
        self.clock_interval = dict(state.get("clock_interval", self.clock_interval))
        self.clock_shift = dict(state.get("clock_shift", self.clock_shift))
        self.terminate_requested = state.get("terminate_requested", False)
        self.clock_last_tick = dict(state.get("clock_last_tick", {}))
        self._write_globals(copy.deepcopy(state.get("globals", {})))
        # the state's own copies (each load is new), so restoring the same state twice works too
        self._write_literal_constants(state.get("constants", {}))
        set_rng_states(state.get("rng", {}))
        return Status.ok

    def _read_globals(self):
        """Current values of the ``[model] globals``.

        :return: ``"module:NAME"`` -> value.
        :rtype: dict
        """
        return {f"{module.__name__}:{name}": getattr(module, name) for module, name in self.globals}

    def _write_globals(self, values):
        """Set ``[model] globals`` back from a dict made by :meth:`_read_globals`.

        Keys missing from ``values`` are left alone (a state saved by an FMU without them).

        :param values: ``"module:NAME"`` -> value.
        :type values: dict
        """
        for module, name in self.globals:
            key = f"{module.__name__}:{name}"
            if key in values:
                setattr(module, name, values[key])

    # ================= running user code =================

    def error(self, message):
        """Log an error to the importer and return ``Status.error``.

        :param message: the error message.
        :type message: str
        :return: ``Status.error``.
        :rtype: int
        """
        self.log(message, Status.error, "logStatusError")
        return Status.error

    def run(self, what, fn, discard=()):
        """Run model code, turning exceptions into an FMI status.

        Every call into the user's code goes through here. While ``fn`` runs, the model's ``logging``
        records are forwarded, the working directory is ``[model] cwd``, and warning classes in
        ``discard`` are raised as errors (so they can discard a step). Printed output is forwarded
        afterwards.

        :param what: what is running, for messages (``"doStep"``, ``"initialization"``, ...).
        :type what: str
        :param fn: the code to run, without arguments.
        :type fn: callable
        :param discard: exception and warning classes that give ``Status.discard`` instead of an
            error (``[model] discard_on``).
        :type discard: tuple[type, ...]
        :return: ``Status.ok``, ``Status.discard`` or ``Status.error`` (logged with the traceback).
        :rtype: int
        """
        try:
            with self._forward_logs(), self._in_model_dir(), warnings.catch_warnings():
                for cls in discard:
                    if issubclass(cls, Warning):
                        warnings.simplefilter("error", cls)
                fn()
        except Exception as e:
            self._forward_output()
            if discard and isinstance(e, discard):
                self.log(f"{what} discarded: {e!r}", Status.discard, "logStatusDiscard")
                return Status.discard
            return self.error(f"{what} failed: {e!r}\n{traceback.format_exc()}")
        self._forward_output()
        return Status.ok

    def log(self, message, status, category):
        """Send a message to the importer.

        Errors always go; other messages only in the categories the importer chose with
        ``fmi2/3SetDebugLogging`` (all when it named none).

        :param message: the text.
        :type message: str
        :param status: the FMI status the message is logged with.
        :type status: int
        :param category: the FMI log category (``logStatusError``, ``logAll``, ...).
        :type category: str
        """
        if status >= Status.error or self.log_categories is None or category in self.log_categories:
            self._send_log(message, status, category)

    def set_debug_logging(self, logging_on, categories=()):
        """Turn logging on or off and choose categories (``fmi2/3SetDebugLogging``).

        :param logging_on: whether model logs and printed output are sent at all.
        :type logging_on: bool
        :param categories: categories to send; empty means all.
        :type categories: collections.abc.Iterable[str]
        :return: ``Status.ok``.
        :rtype: int
        """
        self.logging_on = bool(logging_on)
        self.log_categories = set(categories) or None
        return Status.ok

    def _invoke(self, fn, *args, **kwargs):
        """Call ``fn(*args, **kwargs)``, running an ``async def`` to completion.

        A coroutine function (a step method, classmethod factory or terminate method) runs on one
        event loop kept for the whole run, so objects bound to that loop keep working between steps.
        What other functions return is never awaited, even when it could be (a Future).

        :param fn: the user's function or method.
        :type fn: callable
        :param args: positional arguments.
        :param kwargs: keyword arguments.
        :return: what ``fn`` returned (awaited when ``fn`` is async).
        """
        if not inspect.iscoroutinefunction(fn):
            return fn(*args, **kwargs)
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(fn(*args, **kwargs))

    @contextlib.contextmanager
    def _in_model_dir(self):
        """Context manager: run model code in ``[model] cwd``, then go back.

        So relative paths such as ``open("weather.csv")`` find the files copied with ``sources``.
        Does nothing when ``cwd`` is not set.
        """
        if self.cwd is None:
            yield
            return
        previous = os.getcwd()
        os.chdir(self.cwd)
        try:
            yield
        finally:
            os.chdir(previous)

    def _forward_output(self):
        """Send what the model printed (stdout/stderr, also from C code) to the importer's log.

        Only with ``capture_output`` (see :func:`_capture_output`). The capture file is emptied after
        each read, so a long run that prints a lot doesn't fill the temp disk.
        """
        if self._output is None:
            return
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        self._output.seek(self._position)
        text = self._output.read().decode("utf-8", errors="replace")
        # empty the file once read, so a long run that prints a lot doesn't fill the temp disk.
        # Descriptors 1 and 2 share its offset (dup2), so they write from the start again.
        self._output.seek(0)
        self._output.truncate()
        self._position = 0
        if text.strip() and self.logging_on:
            self.log(f"[output] {text.rstrip()}", Status.ok, self.info_category)

    @contextlib.contextmanager
    def _forward_logs(self):
        """Context manager: forward ``logging`` records emitted by the user's code to the importer.

        The handler is only installed while user code runs, because UniFMU can only
        carry a log message while it is processing a command.
        """
        handler = _ForwardingHandler(self)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            yield
        finally:
            root.removeHandler(handler)

    def _to_python(self, var, value):
        """Convert an FMI value into what the model receives.

        Applies :meth:`_to_python_value`, then the variable's ``convert`` function (e.g. a pint
        quantity), if any.

        :param var: an entry of ``interface["variables"]``.
        :type var: dict
        :param value: the FMI value from :attr:`values`.
        :return: the Python value passed to the model.
        """
        value = self._to_python_value(var, value)
        converter = self.converters.get(var["name"])
        return converter(value) if converter is not None else value

    def _to_python_value(self, var, value):
        """Shape an FMI value for Python: enumeration members, nested lists, numpy arrays.

        Enumerations with an ``enum`` class become its members (FMI value 1 is the first member);
        arrays are reshaped from flat to their ``dimensions`` and turned into numpy arrays when the
        variable has ``numpy``.

        :param var: an entry of ``interface["variables"]``.
        :type var: dict
        :param value: the FMI value.
        :return: the reshaped value.
        """
        enum = self.enums.get(var.get("declared_type"))
        if var.get("dimensions"):
            if enum is not None:
                value = [list(enum)[x - 1] for x in value]
            value = reshape(value, [int(self.values[d]) if isinstance(d, str) else d for d in var["dimensions"]])
            if var.get("numpy"):
                import numpy
                value = numpy.array(value)
            return value
        if enum is not None:
            return list(enum)[value - 1]
        return value

    def _from_python(self, var, value):
        """Convert a value the model returned into the variable's FMI form.

        Arrays are converted from a quantity to plain numbers, flattened and checked against
        :meth:`size`; every element goes through :meth:`_scalar_from_python`.

        :param var: an entry of ``interface["variables"]``.
        :type var: dict
        :param value: a return value or attribute of the model.
        :return: a scalar, or a flat list for arrays.
        :raises ValueError: an array with the wrong number of values.
        """
        if var.get("dimensions"):
            value = plain_number(value, var.get("unit"))   # an array quantity (pint, astropy)
            flat = flatten(value.tolist() if hasattr(value, "tolist") else value)
            n = self.size(var)
            if len(flat) != n:
                raise ValueError(f"{var['name']} has {len(flat)} values, expected {n}")
            return [self._scalar_from_python(var, x) for x in flat]
        return self._scalar_from_python(var, value)

    def _scalar_from_python(self, var, value):
        """Convert one value from the model to the variable's FMI type.

        Enum members become their 1-based index; quantities are converted to the variable's
        ``unit``; one-element arrays give their element. A non-numeric string for a numeric
        variable gets an error that suggests ``type = "String"``.

        :param var: an entry of ``interface["variables"]``.
        :type var: dict
        :param value: one value from the model.
        :return: the coerced FMI value.
        :raises TypeError: None, a non-numeric string, or a multi-element array.
        :raises ValueError: a value out of the type's range.
        """
        if var["type"] == "Enumeration":
            enum = self.enums.get(var.get("declared_type"))
            if enum is not None and isinstance(value, enum):
                return list(enum).index(value) + 1
        if value is None:
            raise TypeError(f"{var['name']} is None")
        value = self._single_value(var, value)
        if isinstance(value, str) and var["type"] not in ("String", "Boolean", "Binary") and not _is_number(value):
            raise TypeError(f"{var['name']} is a string ({value[:80]!r}), not a {var['type']}: "
                            f"set type = \"String\" for it in the config")
        return coerce(var["type"], plain_number(value, var.get("unit")))

    @staticmethod
    def _single_value(var, value):
        """The value of a scalar variable: a one-element array, list or tuple gives its element.

        :param var: an entry of ``interface["variables"]`` (for the error message).
        :type var: dict
        :param value: what the model gave for the variable.
        :return: the single value inside, or ``value`` itself.
        :raises TypeError: a longer array, list or tuple: the variable needs ``dimensions``.
        """
        if isinstance(value, (str, bytes, Mapping)):
            return value
        shape = getattr(value, "shape", None)
        if shape is not None and hasattr(value, "tolist") and not isinstance(shape, Mapping):
            shape = tuple(int(n) for n in shape)
            if shape == () or math.prod(shape) == 1:
                return flatten(value.tolist())[0] if shape else value
        elif isinstance(value, (list, tuple)):
            shape = (len(value),)
            if len(value) == 1:
                return value[0]
        else:
            return value
        kind = "a tuple" if isinstance(value, tuple) else "an array" if hasattr(value, "tolist") else "a list"
        raise TypeError(f"{var['name']} is {kind} of shape {list(shape)}, not a single value: give it "
                        f"dimensions = {list(shape)} (FMI 3) or read one element (from = \"...:0\"); "
                        "fmugen init --probe finds them")

    def _resolve_constants(self):
        """Compute ``[model] constants`` and ``call_constants`` as Python values, once.

        Runs at initialization (from ``initialize``), not at instantiation, so a ``{ call = ... }``
        that downloads or computes reports failures to the importer. Literal values are deep-copied
        because the model may change a list or dict constant in place.

        :raises RuntimeError: a constant that can't be resolved, naming it.
        """
        if self.constants is not None:
            return
        resolved = {}
        for table in ("constants", "call_constants"):
            resolved[table] = {}
            for name, value in self.interface.get(table, {}).items():
                try:
                    # a copy: the model may change a list or dict constant in place
                    resolved[table][name] = resolve_constant(copy.deepcopy(value))
                except Exception as e:
                    shown = value["call"] if isinstance(value, dict) and "call" in value else value
                    raise RuntimeError(f"[model] {table}.{name} = {shown!r} failed: {e!r}") from e
        self.constants, self.call_constants = resolved["constants"], resolved["call_constants"]

    def _literal_constants(self):
        """The resolved constants that were written as values, by table.

        Those (``{ python = ... }``, numbers, lists) are saved with the FMU state and rebuilt on
        reset; ``{ call = ... }`` and references are objects shared by the whole run (a loaded
        model) and stay as they are on reset and rollback.

        :return: ``{"constants": {...}, "call_constants": {...}}``, or ``{}`` before initialization.
        :rtype: dict
        """
        if self.constants is None:
            return {}
        resolved = {"constants": self.constants, "call_constants": self.call_constants}
        return {table: {name: resolved[table][name] for name, value in self.interface.get(table, {}).items()
                        if is_literal_constant(value)}
                for table in resolved}

    def _write_literal_constants(self, values):
        """Overwrite resolved constants in place (reset and :meth:`deserialize`).

        Nothing happens before the constants were resolved.

        :param values: ``{table: {name: value}}``, as from :meth:`_literal_constants`.
        :type values: dict
        """
        if self.constants is None:
            return
        resolved = {"constants": self.constants, "call_constants": self.call_constants}
        for table, constants in values.items():
            resolved[table].update(constants)

    def _run_setup(self, after_construction):
        """Run ``[model] setup`` calls: functions before construction, methods after it.

        A ``"module:function"`` call runs before the model object is built; a bare method name (or
        dotted attribute path) runs on the object after construction. Arguments are resolved
        like constants.

        :param after_construction: whether to run the method calls (True) or the function calls.
        :type after_construction: bool
        """
        for step in self.interface.get("setup", []):
            is_method = ":" not in step["call"]
            if is_method != after_construction:
                continue
            target = get_path(self.obj, step["call"]) if is_method else resolve_reference(step["call"])
            target(*[resolve_constant(a) for a in step.get("args", [])],
                   **{k: resolve_constant(v) for k, v in step.get("kwargs", {}).items()})

    def _positional(self, clock=None, kind="pos"):
        """Values of variables bound to positional arguments, in order.

        ``kind = "pos"`` gives ``to = "pos:N"`` call arguments; ``"initpos"`` gives a class's
        constructor arguments. Without a clock, constants named ``"0"``, ``"1"``, ... are added at
        those positions (``*args``).

        :param clock: the clock whose call this is, or None for the step call / constructor.
        :type clock: str or None
        :param kind: ``"pos"`` or ``"initpos"``.
        :type kind: str
        :return: the positional arguments.
        :rtype: list
        """
        bound = {
            int(v["to"]["name"]): self._to_python(v, self.values[v["name"]])
            for v in self.variables
            if v.get("to", {}).get("kind") == kind and _clock_of(v) == clock
        }
        if clock is None:   # constants named "0", "1", ...: positions of *args
            constants = self.call_constants if self.is_class and kind == "pos" else self.constants
            bound.update((int(k), value) for k, value in constants.items() if k.isdigit())
        return [bound[i] for i in sorted(bound)]

    def _bound_target(self, kind, clock):
        """The function whose arguments ``kind`` variables are bound to.

        Needed to build object arguments (``arg:u.temp``) from that function's signature.

        :param kind: ``"init"`` (the constructor or ``entry.create``) or ``"arg"`` (the call).
        :type kind: str
        :param clock: the clock whose call this is, or None.
        :type clock: str or None
        :return: the class, factory, method or function.
        :rtype: callable
        """
        if kind == "init":
            create = self.interface["entry"].get("create")
            return getattr(self.entry, create) if create else self.entry
        return self._target(clock)

    def _bound(self, kind, clock=None):
        """``{python name: value}`` of every variable bound to a constructor or call argument.

        Variables bound to items (``to = "arg:pair[0]"``, ``"arg:params[key]"``) are put together
        into a tuple or dict argument; fields (``"arg:u.temp"``) into an object built from the
        target's signature (:func:`assemble_items`).

        :param kind: ``"init"`` (constructor) or ``"arg"`` (call).
        :type kind: str
        :param clock: the clock whose call this is, or None.
        :type clock: str or None
        :return: keyword arguments.
        :rtype: dict
        """
        bound = {
            v["to"]["name"]: self._to_python(v, self.values[v["name"]])
            for v in self.variables
            if v.get("to", {}).get("kind") == kind and _clock_of(v) == clock
        }
        objects = any("." in name and "[" not in name for name in bound)
        return assemble_items(bound, self._bound_target(kind, clock) if objects else None)

    def _apply_attributes(self, clock=None, initial=False):
        """Write variables bound to attributes or setter methods onto the model object.

        ``to = "attr:path"`` sets the attribute; ``to = "call:method"`` calls the method with the
        value. Runs after construction and before every call of that clock (or of the step).

        A setter method runs once per model call, as setters that accumulate need (PyTCI's
        ``give_drug`` adds a dose each call): the first call after initialization skips it when the
        value hasn't changed since initialization ran it, and the integration stages of one step
        skip it unless the value changed.

        :param clock: only variables of this clock (None: unclocked ones).
        :type clock: str or None
        :param initial: the run right after construction.
        :type initial: bool
        """
        for v in self.variables:
            kind = v.get("to", {}).get("kind")
            if kind not in ("attr", "call") or _clock_of(v) != clock:
                continue
            if kind == "attr":
                set_path(self.obj, v["to"]["name"], self._to_python(v, self.values[v["name"]]))
                continue
            name, raw = v["name"], self.values[v["name"]]
            unchanged = name in self._setter_last and self._setter_last[name] == raw
            fresh = name in self._setter_fresh
            self._setter_fresh.discard(name)
            if not initial and unchanged and (fresh or name in self._setter_called):
                self._setter_called.add(name)
                continue
            get_path(self.obj, v["to"]["name"])(self._to_python(v, raw))
            self._setter_last[name] = copy.deepcopy(raw)
            if initial:
                self._setter_fresh.add(name)
            else:
                self._setter_called.add(name)

    def _call(self, time, step_size, clock=None):
        """Call the model once: the step method/function, or a clock's call.

        Builds the arguments: keyword constants (``call_constants`` for classes, ``constants`` for
        functions; none for clock calls), ``arg:`` bound variables, ``time_args`` (time, step size,
        end time, ...; a ``datetime`` for those in ``time_epochs``) and positional ones. Writes
        attribute-bound variables first.

        :param time: the time passed as ``time``.
        :type time: float
        :param step_size: the step size passed as ``step_size``.
        :type step_size: float
        :param clock: the clock whose call to make, or None for the step call.
        :type clock: str or None
        :return: what the model returned.
        """
        if clock is None:
            kwargs = keyword_constants(self.call_constants if self.is_class else self.constants)
            kwargs.update(self._bound("arg"))
        else:
            kwargs = self._bound("arg", clock=clock)
        sources = {"time": time, "step_size": step_size, "end_time": time + step_size,
                   "start_time": self.start_time, "stop_time": self.stop_time, "tolerance": self.tolerance}
        epochs = self.interface.get("time_epochs", {})
        for arg, source in self.interface.get("time_args", {}).items():
            if clock is None or arg in self.clocks[clock].get("time_args", ()):
                kwargs[arg] = sources[source]
                if arg in epochs:   # a date-time: the epoch plus the FMU time in seconds
                    kwargs[arg] = datetime.datetime.fromisoformat(epochs[arg]) + datetime.timedelta(seconds=kwargs[arg])
        self._apply_attributes(clock=clock)
        return results_of(self._invoke(self._target(clock), *self._positional(clock), **kwargs))

    def _integrate(self, time, step_size):
        """``[model] integrate``: advance the states over the step, with the model as right-hand side.

        The model, called with the states (and inputs) at each stage, returns the derivatives of the
        variables that have ``der``. Methods: ``euler`` and ``rk4`` with ``integrate.substeps`` fixed
        substeps, or ``scipy:<method>`` through :func:`scipy.integrate.solve_ivp` with the
        experiment's tolerance. The final states are left in :attr:`values`.

        :param time: start of the step.
        :type time: float
        :param step_size: length of the step.
        :type step_size: float
        :raises StepRejected: the solver failed or the states became non-finite; with
            ``discard_on`` in effect, :meth:`do_step` undoes the step and returns discard.
        :raises ValueError: a derivative with the wrong number of values.
        """
        states = [v for v in self.variables if "der" in v]
        sizes = [self.size(v) or 1 for v in states]

        def load(y):
            """Write a flat state vector into the state variables' values.

            :param y: the states, flattened in variable order.
            :type y: list[float]
            """
            i = 0
            for v, n in zip(states, sizes):
                self.values[v["name"]] = [float(x) for x in y[i:i + n]] if v.get("dimensions") else float(y[i])
                i += n

        def rhs(t, y):
            """Right-hand side: the derivatives at time ``t`` for states ``y``.

            :param t: stage time.
            :type t: float
            :param y: stage states, flattened.
            :type y: list[float]
            :return: derivatives, flattened in the same order.
            :rtype: list[float]
            :raises ValueError: a derivative with the wrong number of values.
            """
            load(y)
            result = self._call(t, step_size)
            derivatives = []
            for v, n in zip(states, sizes):
                found, value = self._source_value(v["der"], result, True)
                value = plain_number(value)
                flat = flatten(value.tolist() if hasattr(value, "tolist") else value)
                if len(flat) != n:
                    raise ValueError(f"the derivative of {v['name']} has {len(flat)} values, expected {n}")
                derivatives += [float(x) for x in flat]
            return derivatives

        y = [float(x) for v in states for x in (self.values[v["name"]] if v.get("dimensions")
                                                 else [self.values[v["name"]]])]
        method, substeps = self.interface["integrate"]["method"], self.interface["integrate"]["substeps"]
        if method.startswith("scipy:"):
            from scipy.integrate import solve_ivp
            options = {"rtol": self.tolerance} if self.tolerance else {}
            solution = solve_ivp(lambda t, x: rhs(t, list(x)), (time, time + step_size), y,
                                 method=method.partition(":")[2], **options)
            if not solution.success:
                raise StepRejected(f"solve_ivp ({method}) failed: {solution.message}")
            y = [float(x) for x in solution.y[:, -1]]
        else:
            h, t = step_size / substeps, time
            for _ in range(substeps):
                if method == "euler":
                    y = [a + h * d for a, d in zip(y, rhs(t, y))]
                else:   # classic Runge-Kutta, 4th order
                    k1 = rhs(t, y)
                    k2 = rhs(t + h / 2, [a + h / 2 * k for a, k in zip(y, k1)])
                    k3 = rhs(t + h / 2, [a + h / 2 * k for a, k in zip(y, k2)])
                    k4 = rhs(t + h, [a + h * k for a, k in zip(y, k3)])
                    y = [a + h / 6 * (p + 2 * q + 2 * r + s) for a, p, q, r, s in zip(y, k1, k2, k3, k4)]
                t += h
            if not all(math.isfinite(x) for x in y):
                raise StepRejected(f"the states are not finite after the step ({method}): try a smaller step "
                                   "or more substeps")
        load(y)

    def _target(self, clock):
        """The callable that a step or clock call invokes.

        :param clock: a clock name, or None for the step call (``entry.call`` method of the object,
            or the entry function itself).
        :type clock: str or None
        :return: the method or function. A clock's ``call`` may be ``"module:function"``, a method
            of the object, or a function of the user's module.
        :rtype: callable
        """
        if clock is None:
            if not self.is_class:
                return self.entry
            call = self.interface["entry"]["call"]
            if ":" in call:   # "module:function(self, ...)": a function taking the model object
                return object_call(call, self.obj)
            owner = type(self.obj) if self.interface["entry"].get("call_on_class") else self.obj
            return getattr(owner, call)
        call = self.clocks[clock]["call"]
        if ":" in call:
            return resolve_reference(call)
        return getattr(self.obj, call) if self.is_class else getattr(self.module, call)

    def _source_value(self, source, result, have_result, missing_ok=False):
        """Read a value from a ``{"kind": "return" | "attr" | "get", "name": ...}`` source.

        ``return`` picks from the call's result (whole result when ``name`` is absent), ``attr``
        follows a path on the model object, ``get`` calls the getter method at that path.

        :param source: a ``from``, ``next``, ``der`` or event source.
        :type source: dict
        :param result: what the model call returned.
        :param have_result: whether ``result`` comes from a call made just now; if not, return
            sources are not found.
        :type have_result: bool
        :param missing_ok: return ``(False, None)`` instead of raising when the path is missing.
        :type missing_ok: bool
        :return: ``(found, value)``.
        :rtype: tuple[bool, object]
        :raises AttributeError: missing attribute (or KeyError/IndexError), unless ``missing_ok``.
        """
        try:
            if source["kind"] == "return":
                if not have_result:
                    return False, None
                return True, pick(result, source.get("name"))
            if source["kind"] == "get":   # a getter method of the object, called after the step
                return True, get_path(self.obj, source["name"])()
            return True, get_path(self.obj, source["name"])
        except (AttributeError, KeyError, IndexError):
            if missing_ok:
                return False, None
            raise

    def _read_results(self, result, have_result, update_states, missing_ok, clock=None):
        """Read output variables (``from``) and, optionally, next states (``next``) after a call.

        :param result: what the model call returned.
        :param have_result: whether ``result`` comes from a call made just now.
        :type have_result: bool
        :param update_states: also read ``next`` sources into the state variables (normal steps;
            not with ``integrate``, which updates the states itself).
        :type update_states: bool
        :param missing_ok: skip outputs that can't be read, are None or are arrays of the wrong size
            (initialization, before the model set them) instead of failing.
        :type missing_ok: bool
        :param clock: read the variables of this clock, or None for unclocked ones.
        :type clock: str or None
        :raises LookupError: an output or state that can't be read, unless ``missing_ok``.
        """
        for v in self.variables:
            if "from" not in v or _clock_of(v) != clock:
                continue
            try:
                found, value = self._source_value(v["from"], result, have_result)
            except (AttributeError, KeyError, IndexError) as e:
                if missing_ok:
                    continue
                raise LookupError(f"{v['name']}: cannot read {describe(v['from'])} ({e!r})") from e
            if found and value is None and missing_ok:
                continue  # e.g. an attribute the model only sets after its first step
            if found and missing_ok and v.get("dimensions"):
                try:
                    self.values[v["name"]] = self._from_python(v, value)
                except ValueError:
                    continue   # e.g. an array that stays empty until the first step (pyloudnorm's blockwise_loudness)
            elif found:
                self.values[v["name"]] = self._from_python(v, value)
        if update_states:
            for v in self.variables:
                if "next" in v:
                    try:
                        _, value = self._source_value(v["next"], result, True)
                    except (AttributeError, KeyError, IndexError) as e:
                        raise LookupError(f"state {v['name']}: cannot read {describe(v['next'])} ({e!r})") from e
                    self.values[v["name"]] = self._from_python(v, value)

    def _refresh_calculated_parameters(self):
        """Re-read ``calculatedParameter`` variables taken from attributes.

        Called after a tunable parameter was set on the model object, since the model may derive
        other values from it.
        """
        for v in self.variables:
            if v["causality"] == "calculatedParameter" and v["from"]["kind"] == "attr":
                self.values[v["name"]] = self._from_python(v, get_path(self.obj, v["from"]["name"]))
