"""CompositeEngine: several models stepped in order, and the factory that picks the right engine."""
import json
import math
import os
import pickle
import subprocess
import sys
import threading
from collections import defaultdict
from multiprocessing.connection import Listener
from pathlib import Path

from fmugen.templates.fmugen_runtime._constants import SPEC_VERSION, Status
from fmugen.templates.fmugen_runtime._engine import Engine
from fmugen.templates.fmugen_runtime._helpers import outside_bundle

#: This machine's OS, as an isolated part's ``launch.json`` names it.
_THIS_OS = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")


def make_engine(resources_dir, log, info_category="logAll"):
    """Create the engine for an FMU: one model, or several connected ones.

    Called by the adapters' ``Model.__init__`` at instantiation. Reads ``interface.json``, checks
    its ``spec_version`` and returns a :class:`CompositeEngine` when it has ``composite`` set,
    else an :class:`Engine`.

    :param resources_dir: the FMU's ``resources/`` folder.
    :type resources_dir: str or pathlib.Path
    :param log: ``log(message, status, category)``, forwards a message to the importer.
    :type log: callable
    :param info_category: log category for informational messages.
    :type info_category: str
    :return: the engine; both kinds have the same methods.
    :rtype: Engine or CompositeEngine
    :raises RuntimeError: the ``spec_version`` doesn't match this runtime.
    """
    interface = json.loads((Path(resources_dir) / "interface.json").read_text())
    check_spec_version(interface)
    if interface.get("composite"):
        return CompositeEngine(resources_dir, log, info_category)
    return Engine(resources_dir, log, info_category)


def check_spec_version(interface):
    """Refuse an ``interface.json`` written for another runtime.

    A part or file from another fmugen version could be read wrongly, so the version must
    match :data:`SPEC_VERSION` exactly (see ``docs/interface.md``, Versioning).

    :param interface: a parsed ``interface.json``.
    :type interface: dict
    :raises RuntimeError: the versions differ.
    """
    found = interface.get("spec_version")
    if found != SPEC_VERSION:
        raise RuntimeError(f"interface.json has spec_version {found!r}, but this fmugen runtime reads version "
                           f"{SPEC_VERSION}: rebuild the FMU with one fmugen version")


#: What an engine method returns when its part's process is gone, for those that don't return a status alone.
_FAILED = {"get_values": (Status.error, []), "do_step": (Status.error, False),
           "update_discrete_states": (Status.error, False, None), "serialize": (Status.error, b""),
           "get_clocks": (Status.error, []), "get_intervals": (Status.error, [], []), "get_shifts": (Status.error, [])}


class PartProxy:
    """One isolated part of a composite FMU: an :class:`Engine` running in a process of its own.

    Has what :class:`CompositeEngine` uses of an engine; each call goes to the part's process
    (``fmugen_part_main.py``, the part's ``main.py``), started with the command in its
    ``launch.json``, so the part has its own Python, packages and modules. The two talk over a local
    socket with a random key, which keeps the model's printing out of the protocol. Log records come
    back before each answer, so they reach the importer during the call. When the process dies, the
    call returns an error status, and so does every later one.

    :ivar name: the part's name.
    :ivar by_reference: the part's value reference -> variable entry.
    :ivar discard_on: whether the part can discard a step.
    """

    def __init__(self, name, folder, log, info_category):
        """Start the part's process and wait for its engine.

        :param name: the part's name.
        :type name: str
        :param folder: the part's folder, ``resources/parts/<name>``.
        :type folder: str or pathlib.Path
        :param log: ``log(message, status, category)`` to the importer.
        :type log: callable
        :param info_category: log category for informational messages.
        :type info_category: str
        :raises RuntimeError: the process exited, or the part's model failed to import.
        """
        self.name, self.log, self.connection = name, log, None
        folder = outside_bundle(folder).resolve()
        command = json.loads((folder / "launch.json").read_text())[_THIS_OS]
        if command[0].startswith("dist/"):   # Windows looks for a relative executable outside the new cwd
            command = [str(folder / command[0]), *command[1:]]
        key = os.urandom(32)
        env = {**os.environ, "FMUGEN_PART_KEY": key.hex(), "FMUGEN_PART_CATEGORY": info_category}
        if "LD_LIBRARY_PATH_ORIG" in env:   # a frozen backend's own libraries aren't the part's
            env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH_ORIG"]
        else:
            env.pop("LD_LIBRARY_PATH", None) if getattr(sys, "frozen", False) else None
        with Listener(("127.0.0.1", 0), authkey=key) as listener:
            self.process = subprocess.Popen([*command, listener.address[0], str(listener.address[1])], cwd=folder, env=env)
            accepted = []
            thread = threading.Thread(target=lambda: accepted.append(listener.accept()), daemon=True)
            thread.start()
            while thread.is_alive():   # a vendored part's first run installs its packages: no time limit
                thread.join(0.2)
                if self.process.poll() is not None and not accepted:
                    raise RuntimeError(f"part {name}: its process exited with status {self.process.returncode} "
                                       "before connecting")
            self.connection = accepted[0]
        frame = self.connection.recv()
        while frame[0] == "log":
            self.log(*frame[1:])
            frame = self.connection.recv()
        if frame[0] != "ready":
            raise RuntimeError(f"part {name} failed to start:\n{frame[1]}")
        self.discard_on = frame[1]["discard_on"]
        self.by_reference = frame[1]["by_reference"]
        self._logging_on = frame[1]["logging_on"]

    def _ask(self, kind, method, args, failed):
        """Send one request and wait for its answer, passing on log records that come first.

        :param kind: ``"call"``, ``"get"`` or ``"set"``.
        :type kind: str
        :param method: the engine's method or attribute.
        :type method: str
        :param args: the call's arguments, or the value to set.
        :param failed: what to return when the process is gone.
        :return: the answer.
        """
        if self.connection is None:
            return failed
        try:
            self.connection.send((kind, method, args))
            while True:
                frame = self.connection.recv()
                if frame[0] == "log":
                    self.log(*frame[1:])
                elif frame[0] == "result":
                    return frame[1]
                else:
                    self.log(f"part {self.name}: {frame[1]}", Status.error, "logStatusError")
                    return failed
        except (EOFError, OSError):
            self.connection = None
            code = self.process.wait()
            self.log(f"part {self.name}: its process ended with status {code} during {method}",
                     Status.error, "logStatusError")
            return failed

    def __getattr__(self, name):
        """An engine method, called in the part's process.

        :param name: the method's name.
        :type name: str
        :rtype: callable
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args: self._ask("call", name, args, _FAILED.get(name, Status.error))

    def size(self, var):
        """Number of scalar values of a variable, asked of the part only for an array.

        :param var: an entry of the part's variables.
        :type var: dict
        :rtype: int or None
        """
        return self._ask("call", "size", (var,), None) if var.get("dimensions") else None

    @property
    def logging_on(self):
        """Whether the part sends its log records.

        :rtype: bool
        """
        return self._logging_on

    @logging_on.setter
    def logging_on(self, on):
        """Turn the part's log records on or off.

        :param on: the new value.
        :type on: bool
        """
        self._logging_on = on
        self._ask("set", "logging_on", on, None)

    @property
    def terminate_requested(self):
        """Whether the part asked to end the simulation.

        :rtype: bool
        """
        return self._ask("get", "terminate_requested", (), False)

    @property
    def clock_active(self):
        """The part's clock name -> whether it is ticking.

        :rtype: dict[str, bool]
        """
        return self._ask("get", "clock_active", (), defaultdict(bool))

    def close(self):
        """End the part's process: closing the connection makes it exit."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class CompositeEngine:
    """Several models (parts), each run by its own :class:`Engine`, stepped in order.

    Built by ``fmugen build`` from a ``[composite]`` config (``compose.combine``). Each part has
    its own ``interface.json`` in ``resources/parts/<name>/``. The FMU's variables are the parts'
    unconnected ones, named ``<part>.<variable>``; each composite variable has ``part`` and
    ``local`` (the reference inside its part), and the importer's references are mapped to
    those (:meth:`_target`). Clocks work the same way (:meth:`_clock`).

    Routing: life-cycle calls go to every part (:meth:`_all`, which returns the worst status);
    value and clock calls go to the part that owns the reference. ``connections`` copy a part's
    output into another part's input (:meth:`_feed`) right before the receiving part
    initializes or steps, so parts later in order see the current step's outputs of earlier
    ones. A discarded step in any part undoes the whole step for all parts.

    :ivar resources_dir: the FMU's ``resources/`` folder.
    :ivar log: ``log(message, status, category)`` to the importer.
    :ivar interface: the composite ``interface.json``.
    :ivar parts: part name -> its :class:`Engine`, in stepping order.
    :ivar variables: the composite's variables.
    :ivar by_reference: composite value reference -> variable entry.
    :ivar elements: FMI 2 only: element reference -> (array variable, index).
    :ivar connections: ``((src part, src reference), (dst part, dst reference), factor, offset)``.
    :ivar sized_later: the connections with an end sized by a structural parameter.
    :ivar clocks: composite clock name (``<part>.<clock>``) -> clock entry.
    :ivar clock_by_reference: composite clock reference -> clock entry.
    :ivar time: the composite's current time (for the independent variable).
    """

    def __init__(self, resources_dir, log, info_category="logAll"):
        """Load the composite ``interface.json`` and instantiate every part's engine.

        Like :meth:`Engine.__init__`, at FMI instantiation. Each part's own ``interface.json`` is
        version-checked before its engine imports the part's model.

        :param resources_dir: the FMU's ``resources/`` folder.
        :type resources_dir: str or pathlib.Path
        :param log: ``log(message, status, category)``, shared by all parts.
        :type log: callable
        :param info_category: log category for informational messages.
        :type info_category: str
        :raises RuntimeError: a part's ``spec_version`` doesn't match.
        """
        self.resources_dir = Path(resources_dir)
        self.log = log
        self.interface = json.loads((self.resources_dir / "interface.json").read_text())
        hf_cache = outside_bundle(self.resources_dir) / "hf_cache"
        if hf_cache.is_dir():   # Hugging Face models bundled by fmugen build, shared by the parts
            os.environ.update(HF_HUB_CACHE=str(hf_cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if self.interface.get("isolated"):   # each part checks its own spec_version, in its own process
            self.parts = {}
            for p in self.interface["parts"]:
                self.parts[p["name"]] = PartProxy(p["name"], self.resources_dir / p["dir"], log, info_category)
        else:
            for p in self.interface["parts"]:
                check_spec_version(json.loads((self.resources_dir / p["dir"] / "interface.json").read_text()))
            self.parts = {p["name"]: Engine(self.resources_dir / p["dir"], log, info_category)
                          for p in self.interface["parts"]}
        self.variables = self.interface["variables"]
        self.by_reference = {v["valueReference"]: v for v in self.variables}
        self.elements = {} if self.interface.get("fmi_version") != 2 else {
            v["valueReference"] + k: (v, k) for v in self.variables if v.get("dimensions")
            for k in range(math.prod(v["dimensions"]))}
        self.connections = [(tuple(c["from"]), tuple(c["to"]), c.get("factor", 1.0), c.get("offset", 0.0))
                            for c in self.interface["connections"]]
        self.sized_later = [c for c in self.interface["connections"] if "check_size" in c]
        # the parts' clocks, <part>.<clock>: each reference is passed on to its part
        self.clocks = {c["name"]: c for c in self.interface.get("clocks", [])}
        self.clock_by_reference = {c["valueReference"]: c for c in self.clocks.values()}
        self.time = self.interface.get("experiment", {}).get("start_time", 0.0)

    def close(self):
        """End the processes of isolated parts (they also end when this process does)."""
        for engine in self.parts.values():
            if isinstance(engine, PartProxy):
                engine.close()

    # ---- helpers ----

    @property
    def logging_on(self):
        """Whether logging is on: true only if it is on in every part.

        Set by the FMI 3 adapter from ``loggingOn``; setting it sets it in every part.

        :rtype: bool
        """
        return all(e.logging_on for e in self.parts.values())

    @logging_on.setter
    def logging_on(self, on):
        """Turn logging on or off in every part.

        :param on: the new value for every part.
        :type on: bool
        """
        for engine in self.parts.values():
            engine.logging_on = on

    def set_debug_logging(self, logging_on, categories=()):
        """``fmi2/3SetDebugLogging`` for every part.

        :param logging_on: whether model logs and output are sent.
        :type logging_on: bool
        :param categories: categories to send; empty means all.
        :type categories: collections.abc.Iterable[str]
        :return: the worst status of the parts.
        :rtype: int
        """
        return self._all("set_debug_logging", logging_on, categories)

    @property
    def terminate_requested(self):
        """Whether any part asked to end the simulation (``events.terminate``).

        :rtype: bool
        """
        return any(e.terminate_requested for e in self.parts.values())

    def _all(self, method, *args):
        """Call one engine method on every part, in order.

        :param method: name of the :class:`Engine` method.
        :type method: str
        :param args: its arguments.
        :return: the worst (highest) status, or ``Status.ok`` without parts.
        :rtype: int
        """
        return max((getattr(e, method)(*args) for e in self.parts.values()), default=Status.ok)

    def _feed(self, name):
        """Copy the connected outputs into the inputs of part ``name``.

        Runs right before that part initializes or steps. Values pass through the parts'
        ``get_values`` / ``set_values``, so the receiving part checks and coerces them. A connection
        between different units converts them, element by element for an array.

        :param name: the receiving part.
        :type name: str
        :return: ``Status.ok``, or the first failing status.
        :rtype: int
        """
        for (src, src_ref), (dst, dst_ref), factor, offset in self.connections:
            if dst != name:
                continue
            status, values = self.parts[src].get_values([src_ref])
            if status == Status.ok:
                if factor != 1.0 or offset != 0.0:
                    values = [factor * v + offset for v in values]
                status = self.parts[dst].set_values([dst_ref], values)
            if status != Status.ok:
                return status
        return Status.ok

    def _check_sizes(self):
        """Check the connections whose size depends on a structural parameter.

        :return: ``Status.ok``, or ``Status.error`` naming the first connection whose ends differ.
        :rtype: int
        """
        for connection in self.sized_later:
            sizes = []
            for part, reference in (connection["from"], connection["to"]):
                engine = self.parts[part]
                sizes.append(engine.size(engine.by_reference[reference]))
            if sizes[0] != sizes[1]:
                return self.error(f"connection {connection['check_size']}: the arrays have different sizes, "
                                  f"{sizes[0]} and {sizes[1]}")
        return Status.ok

    def _target(self, r):
        """Map one of the FMU's value references to a part.

        :param r: a composite value reference (FMI 2: possibly an array element).
        :type r: int
        :return: ``(part engine, its value reference, element index or None)``, or
            ``(None, None, None)`` for an unknown reference.
        :rtype: tuple
        """
        if r in self.elements:
            var, k = self.elements[r]
            return self.parts[var["part"]], var["local"], k
        var = self.by_reference.get(r)
        if var is None or "part" not in var:
            return None, None, None
        return self.parts[var["part"]], var["local"], None

    def error(self, message):
        """Log an error to the importer and return ``Status.error``.

        :param message: the error message.
        :type message: str
        :return: ``Status.error``.
        :rtype: int
        """
        self.log(message, Status.error, "logStatusError")
        return Status.error

    # ---- life cycle ----

    def reset(self):
        """``fmi2/3Reset``: reset every part and the composite's time.

        :return: the worst status of the parts.
        :rtype: int
        """
        self.time = self.interface.get("experiment", {}).get("start_time", 0.0)
        return self._all("reset")

    def setup_experiment(self, start_time, stop_time=None, tolerance=None):
        """Pass the experiment setup to every part (see :meth:`Engine.setup_experiment`).

        :param start_time: simulation start.
        :type start_time: float
        :param stop_time: simulation end, or None.
        :type stop_time: float or None
        :param tolerance: relative tolerance, or None.
        :type tolerance: float or None
        :return: the worst status of the parts.
        :rtype: int
        """
        self.time = start_time
        return self._all("setup_experiment", start_time, stop_time, tolerance)

    def enter_initialization_mode(self):
        """``fmi2/3EnterInitializationMode`` for every part.

        :return: the worst status of the parts.
        :rtype: int
        """
        return self._all("enter_initialization_mode")

    def exit_initialization_mode(self, next_mode="step"):
        """Initialize the parts in order, feeding each its connected inputs first.

        So a part's initial outputs reach the parts initialized after it. Stops at the first error.

        :param next_mode: ``"step"`` or ``"event"``, as for :meth:`Engine.exit_initialization_mode`.
        :type next_mode: str
        :return: the worst status.
        :rtype: int
        """
        status = self._check_sizes()
        if status >= Status.error:
            return status
        for name, engine in self.parts.items():
            status = max(status, self._feed(name))
            if status < Status.error:
                status = max(status, engine.exit_initialization_mode(next_mode))
            if status >= Status.error:
                return status
        return status

    def do_step(self, current_time, step_size):
        """Step every part in order, feeding each its connected inputs first.

        When any part has ``discard_on`` and the FMU can save state, the state of all parts is
        saved before the step; a part that discards then undoes the whole step (the parts before it
        are put back too) and the step returns ``Status.discard``.

        :param current_time: the communication point at the start of the step.
        :type current_time: float
        :param step_size: the communication step size.
        :type step_size: float
        :return: ``(status, event_handling_needed)``; the second is true when any part's output
            clock ticked.
        :rtype: tuple[int, bool]
        """
        saved = None
        if any(e.discard_on for e in self.parts.values()) and self.interface.get("can_get_and_set_state", True):
            saved = self.serialize()[1]
        status, ticked = Status.ok, False
        for name, engine in self.parts.items():
            status = max(status, self._feed(name))
            if status < Status.error:
                part_status, part_ticked = engine.do_step(current_time, step_size)
                status, ticked = max(status, part_status), ticked or part_ticked
            if status == Status.discard:
                if saved:
                    self.deserialize(saved)
                return status, False
            if status >= Status.error:
                return status, ticked
        self.time = current_time + step_size
        return status, ticked

    def terminate(self):
        """``fmi2/3Terminate`` for every part.

        :return: the worst status of the parts.
        :rtype: int
        """
        return self._all("terminate")

    def _configurable(self):
        """The parts with structural parameters (only they have a configuration mode).

        :return: their engines.
        :rtype: list[Engine]
        """
        return [e for e in self.parts.values() if any(v["causality"] == "structuralParameter" for v in e.variables)]

    def enter_configuration_mode(self):
        """``fmi3EnterConfigurationMode`` for the parts that have structural parameters.

        :return: the worst status, or ``Status.error`` when no part has structural parameters.
        :rtype: int
        """
        parts = self._configurable()
        if not parts:
            return self.error("this FMU has no structural parameters")
        return max(e.enter_configuration_mode() for e in parts)

    def exit_configuration_mode(self):
        """``fmi3ExitConfigurationMode`` for the parts that have structural parameters.

        Connected arrays must have the same size by then.

        :return: the worst status, or ``Status.error`` when no part has structural parameters or
            connected arrays differ in size.
        :rtype: int
        """
        parts = self._configurable()
        if not parts:
            return self.error("this FMU has no structural parameters")
        status = max(e.exit_configuration_mode() for e in parts)
        return max(status, self._check_sizes())

    def enter_event_mode(self):
        """``fmi3EnterEventMode`` for every part.

        :return: the worst status of the parts.
        :rtype: int
        """
        return self._all("enter_event_mode")

    def enter_step_mode(self):
        """``fmi3EnterStepMode`` for every part.

        :return: the worst status of the parts.
        :rtype: int
        """
        return self._all("enter_step_mode")

    def update_discrete_states(self):
        """``fmi3UpdateDiscreteStates`` for every part, results combined.

        :return: ``(worst status, whether any part requests termination, earliest next event time
            or None)``.
        :rtype: tuple[int, bool, float or None]
        """
        results = [e.update_discrete_states() for e in self.parts.values()]
        times = [t for _, _, t in results if t is not None]
        return max(s for s, _, _ in results), any(t for _, t, _ in results), (min(times) if times else None)

    # ---- values ----

    def size(self, var):
        """Number of scalar values of a composite variable, or None for a scalar.

        Asked of its part, which knows the current values of its structural parameters.

        :param var: an entry of the composite's ``variables``.
        :type var: dict
        :return: the number of elements, or None.
        :rtype: int or None
        """
        if not var.get("dimensions"):
            return None
        engine, local, _ = self._target(var["valueReference"])
        return engine.size(engine.by_reference[local])   # its part knows the structural parameters

    def get_values(self, references):
        """``fmi2/3Get*``: read each reference from its part.

        :param references: composite value references.
        :type references: list[int]
        :return: ``(status, flat list of values)``; the independent variable gives the time.
        :rtype: tuple[int, list]
        """
        values = []
        for r in references:
            var = self.by_reference.get(r)
            if var is not None and var["causality"] == "independent":
                values.append(self.time)
                continue
            engine, local, element = self._target(r)
            if engine is None:
                return self.error(f"unknown value reference {r}"), []
            status, part_values = engine.get_values([local if element is None else local + element])
            if status != Status.ok:
                return status, []
            values.extend(part_values)
        return Status.ok, values

    def set_values(self, references, values):
        """``fmi2/3Set*``: pass each value on to its part.

        :param references: composite value references.
        :type references: list[int]
        :param values: the values, flattened.
        :type values: list
        :return: ``Status.ok``, or the first failing status.
        :rtype: int
        """
        values, cursor = list(values), 0
        for r in references:
            engine, local, element = self._target(r)
            if engine is None:
                return self.error(f"unknown value reference {r}")
            n = 1 if element is not None else (self.size(self.by_reference[r]) or 1)
            status = engine.set_values([local if element is None else local + element], values[cursor:cursor + n])
            cursor += n
            if status != Status.ok:
                return status
        return Status.ok

    # ---- clocks: each one belongs to a part ----

    def _clock(self, r):
        """Map one of the FMU's clock references to its part.

        :param r: a composite clock value reference.
        :type r: int
        :return: ``(part engine, the clock's reference in that part)``, or ``(None, None)``.
        :rtype: tuple
        """
        clock = self.clock_by_reference.get(r)
        return (self.parts[clock["part"]], clock["local"]) if clock else (None, None)

    def _clock_values(self, method, references, columns=1):
        """Call a part's clock get method for each clock reference and join the results.

        :param method: ``"get_clocks"``, ``"get_intervals"`` or ``"get_shifts"``.
        :type method: str
        :param references: the FMU's (composite) clock value references.
        :type references: list[int]
        :param columns: how many lists the method returns after its status (2 for intervals and
            qualifiers).
        :type columns: int
        :return: ``(status, *lists)``; empty lists on failure.
        :rtype: tuple
        """
        lists = [[] for _ in range(columns)]
        for r in references:
            engine, local = self._clock(r)
            if engine is None:
                return (self.error(f"value reference {r} is not a clock"), *([] for _ in lists))
            status, *values = getattr(engine, method)([local])
            if status != Status.ok:
                return (status, *([] for _ in lists))
            for target, part_values in zip(lists, values):
                target.extend(part_values)
        return (Status.ok, *lists)

    def _clock_set(self, method, references, values):
        """Call a part's clock set method for each clock reference with its value.

        :param method: ``"set_clocks"``, ``"set_intervals"`` or ``"set_shifts"``.
        :type method: str
        :param references: the FMU's (composite) clock value references.
        :type references: list[int]
        :param values: one value per reference.
        :type values: list
        :return: ``Status.ok``, or the first failing status.
        :rtype: int
        """
        for r, value in zip(references, values):
            engine, local = self._clock(r)
            if engine is None:
                return self.error(f"value reference {r} is not a clock")
            status = getattr(engine, method)([local], [value])
            if status != Status.ok:
                return status
        return Status.ok

    @property
    def clock_active(self):
        """Composite clock name -> whether it is ticking, from its part.

        :rtype: dict[str, bool]
        """
        return {name: self.parts[c["part"]].clock_active[c["name"].partition(".")[2]]
                for name, c in self.clocks.items()}

    def drop_output_ticks(self):
        """Forget output clock ticks in every part (see :meth:`Engine.drop_output_ticks`).

        :return: the worst status of the parts.
        :rtype: int
        """
        return self._all("drop_output_ticks")

    def set_clocks(self, references, values):
        """``fmi3SetClock``, routed to each clock's part (see :meth:`Engine.set_clocks`).

        :param references: the FMU's (composite) clock value references.
        :type references: list[int]
        :param values: one bool per reference.
        :type values: list[bool]
        :return: ``Status.ok``, or the first failing status.
        :rtype: int
        """
        return self._clock_set("set_clocks", references, values)

    def get_clocks(self, references):
        """``fmi3GetClock``, routed to each clock's part (see :meth:`Engine.get_clocks`).

        :param references: the FMU's (composite) clock value references.
        :type references: list[int]
        :return: ``(status, list of bool)``.
        :rtype: tuple
        """
        return self._clock_values("get_clocks", references)

    def get_intervals(self, references):
        """``fmi3GetInterval*``, routed to each clock's part (see :meth:`Engine.get_intervals`).

        :param references: the FMU's (composite) clock value references.
        :type references: list[int]
        :return: ``(status, intervals, qualifiers)``.
        :rtype: tuple
        """
        return self._clock_values("get_intervals", references, columns=2)

    def set_intervals(self, references, intervals):
        """``fmi3SetInterval*``, routed to each clock's part (see :meth:`Engine.set_intervals`).

        :param references: the FMU's (composite) clock value references.
        :type references: list[int]
        :param intervals: intervals in seconds.
        :type intervals: list[float]
        :return: ``Status.ok``, or the first failing status.
        :rtype: int
        """
        return self._clock_set("set_intervals", references, intervals)

    def get_shifts(self, references):
        """``fmi3GetShift*``, routed to each clock's part (see :meth:`Engine.get_shifts`).

        :param references: the FMU's (composite) clock value references.
        :type references: list[int]
        :return: ``(status, shifts)``.
        :rtype: tuple
        """
        return self._clock_values("get_shifts", references)

    def set_shifts(self, references, shifts):
        """``fmi3SetShift*``, routed to each clock's part (see :meth:`Engine.set_shifts`).

        :param references: the FMU's (composite) clock value references.
        :type references: list[int]
        :param shifts: shifts in seconds.
        :type shifts: list[float]
        :return: ``Status.ok``, or the first failing status.
        :rtype: int
        """
        return self._clock_set("set_shifts", references, shifts)

    # ---- state ----

    def serialize(self):
        """The state of every part, as bytes.

        Format: ``b"M"`` followed by a pickle of ``{"parts": {name: part state bytes}, "time": ...}``.

        :return: ``(status, bytes)``; the first failing part's status and ``b""`` on failure.
        :rtype: tuple[int, bytes]
        """
        states = {}
        for name, engine in self.parts.items():
            status, data = engine.serialize()
            if status != Status.ok:
                return status, b""
            states[name] = data
        return Status.ok, b"M" + pickle.dumps({"parts": states, "time": self.time})

    def deserialize(self, data):
        """Restore a state made by :meth:`serialize`.

        :param data: bytes from :meth:`serialize`.
        :type data: bytes
        :return: the worst status of the parts, or ``Status.error`` for bytes not from a composite.
        :rtype: int
        """
        if data[:1] != b"M":
            return self.error("not an FMU state saved by this composite FMU")
        state = pickle.loads(data[1:])
        self.time = state["time"]
        return max((self.parts[name].deserialize(part) for name, part in state["parts"].items()),
                   default=Status.ok)
