# FMI behaviour

fmugen builds UniFMU FMUs for **FMI 2.0** (the default) or **FMI 3.0** (`[model] fmi_version = 3` or `fmugen build --fmi 3`). Both are Co-Simulation FMUs.

The native UniFMU library in `binaries/` starts a Python process (`resources/main.py`), which forwards each FMI call to the adapter in `resources/model.py`. fmugen generates that adapter: `model_fmi2.py` or `model_fmi3.py`, depending on the version. Both are thin and delegate to the same engine (`resources/fmugen_runtime.py`). The engine reads `resources/interface.json` ([reference](interface.md)) to find out how each FMU variable maps onto your code.

- [Support overview](#support-overview)
- [FMI 2](#fmi-2)
- [FMI 3](#fmi-3)
- [Clocks (FMI 3)](#clocks-fmi-3)
- [Shared behaviour](#shared-behaviour): setting variables, FMU state, logging, errors
- [Unsupported functions](#unsupported-functions)
- [Out of scope](#out-of-scope)

---

## Support overview

These tables follow UniFMU's published support matrix for its Python backend. "UniFMU" is what the backend supports; "fmugen" is what a fmugen FMU does with it.

### FMI 2

| Function | UniFMU | fmugen |
|---|---|---|
| `fmi2GetTypesPlatform`, `fmi2GetVersion` | ✓ | answered by the native library |
| `fmi2SetDebugLogging` | ✓ | turns forwarding of the model's log records on/off; the categories given limit the messages sent (errors always go) |
| `fmi2Instantiate`, `fmi2FreeInstance` | ✓ | imports the model / ends the Python process |
| `fmi2SetupExperiment` | ✓ | stores start, stop and tolerance |
| `fmi2EnterInitializationMode`, `fmi2ExitInitializationMode` | ✓ | exit builds the model object and computes initial outputs |
| `fmi2Terminate`, `fmi2Reset` | ✓ | optional terminate method; reset to start values |
| `fmi2Get/Set{Real,Integer,Boolean,String}` | ✓ | see [Setting variables](#setting-variables) |
| `fmi2GetFMUstate`, `fmi2SetFMUstate`, `fmi2FreeFMUstate`, `fmi2SerializedFMUstateSize`, `fmi2SerializeFMUstate`, `fmi2DeSerializeFMUstate` | ✓ | the Python object is pickled, or the attributes in `[model] save_state` ([FMU state](#fmu-state)) |
| `fmi2DoStep` | ✓ | runs the step function/method |
| `fmi2CancelStep` | x | not needed: `doStep` is synchronous |
| `fmi2GetStatus`, `fmi2GetRealStatus`, `fmi2GetIntegerStatus`, `fmi2GetBooleanStatus`, `fmi2GetStringStatus` | x | not needed: no asynchronous steps; the FMU never asks to stop in FMI 2 |
| `fmi2SetRealInputDerivatives`, `fmi2GetRealOutputDerivatives` | x | not declared (`canInterpolateInputs`, `maxOutputDerivativeOrder` = 0) |
| `fmi2GetDirectionalDerivative` | x | not declared (`providesDirectionalDerivative` = false) |
| `fmi2EnterEventMode`, `fmi2NewDiscreteStates`, `fmi2EnterContinuousTimeMode`, `fmi2CompletedIntegratorStep`, `fmi2SetTime`, `fmi2SetContinuousStates`, `fmi2GetDerivatives`, `fmi2GetEventIndicators`, `fmi2GetContinuousStates`, `fmi2GetNominalsOfContinuousStates` | x | Model Exchange: not available |

### FMI 3

| Function | UniFMU | fmugen |
|---|---|---|
| `fmi3GetVersion` | ✓ | answered by the native library |
| `fmi3SetDebugLogging` | ✓ | as FMI 2 |
| `fmi3InstantiateCoSimulation`, `fmi3FreeInstance` | ✓ | imports the model; remembers `eventModeUsed` |
| `fmi3EnterInitializationMode`, `fmi3ExitInitializationMode` | ✓ | as FMI 2; exit goes to event mode if the importer uses it |
| `fmi3EnterEventMode`, `fmi3EnterStepMode` | ✓ | step mode deactivates all clocks |
| `fmi3EnterConfigurationMode`, `fmi3ExitConfigurationMode` | ✓ | change [structural parameters](#structural-parameters-and-arrays) |
| `fmi3Terminate`, `fmi3Reset` | ✓ | as FMI 2 |
| `fmi3Get/Set{Float32,Float64,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Boolean,String,Binary}` | ✓ | scalars and arrays; integers are range-checked |
| `fmi3GetClock`, `fmi3SetClock` | ✓ | see [Clocks](#clocks-fmi-3) |
| `fmi3Get/SetIntervalDecimal`, `fmi3Get/SetIntervalFraction`, `fmi3Get/SetShiftDecimal`, `fmi3Get/SetShiftFraction` | ✓ | see [Clocks](#clocks-fmi-3) |
| `fmi3UpdateDiscreteStates` | ✓ | runs the code of the clocks that ticked |
| `fmi3GetFMUState`, `fmi3SetFMUState`, `fmi3FreeFMUState`, `fmi3SerializedFMUStateSize`, `fmi3SerializeFMUState`, `fmi3DeserializeFMUState` | ✓ | as FMI 2, plus clock state |
| `fmi3DoStep` | ✓ | runs the step, then reports output clocks and `terminateSimulation` |
| `fmi3InstantiateModelExchange`, `fmi3InstantiateScheduledExecution`, `fmi3ActivateModelPartition` | x | Model Exchange and Scheduled Execution: not available |
| `fmi3EvaluateDiscreteStates`, `fmi3EnterContinuousTimeMode`, `fmi3CompletedIntegratorStep`, `fmi3SetTime`, `fmi3SetContinuousStates`, `fmi3GetContinuousStateDerivatives`, `fmi3GetEventIndicators`, `fmi3GetContinuousStates`, `fmi3GetNominalsOfContinuousStates`, `fmi3GetNumberOfEventIndicators`, `fmi3GetNumberOfContinuousStates` | x | Model Exchange: not available |
| `fmi3GetNumberOfVariableDependencies`, `fmi3GetVariableDependencies` | x | dependencies are only declared in `modelDescription.xml` |
| `fmi3GetDirectionalDerivative`, `fmi3GetAdjointDerivative`, `fmi3GetOutputDerivatives` | x | not declared |

---

## FMI 2

```
fmi2Instantiate ─► [set parameters, inputs, state starts] ─► fmi2SetupExperiment
   ─► fmi2EnterInitializationMode ─► [set ...] ─► fmi2ExitInitializationMode
   ─► loop: [set inputs / tunable parameters] ─► fmi2DoStep(t, h) ─► [get outputs]
   ─► fmi2Terminate ─► fmi2FreeInstance
```

| FMI call | What the adapter does |
|---|---|
| `fmi2Instantiate` | Adds the model's directories to `sys.path`, imports the entry, and sets every variable to its start value. A class is **not** constructed yet, because parameters may still change. |
| `fmi2SetupExperiment(start, stop, tol)` | Stores them; the FMU's time becomes `start`. |
| `fmi2EnterInitializationMode` | Nothing to do. |
| `fmi2ExitInitializationMode` | **Class:** constructs the object with the parameters and `[model.constants]`, then writes `attr:`-bound variables onto it. **Function** (or `init_call = true`): calls the model once at the start time to compute initial outputs. States are not advanced. Then reads outputs, locals and calculated parameters; attributes that don't exist yet keep their start values. |
| `fmi2DoStep(t, h, …)` | Rejects `h` if `fixed_step` is set and `h` differs. Writes `attr:` inputs, then calls the function or step method with the inputs, parameters bound to `arg:`, states, time arguments and `call_constants`. Reads outputs, locals and calculated parameters, then replaces each state with its `next` value (with `[model] integrate`: integrates the states over the step, then calls the model once more at `t + h` for the outputs). The FMU's time becomes `t + h`. An exception listed in `[model] discard_on` (or a failed solve) returns `discard` and puts the FMU back to `t`. A `t` that isn't the FMU's time (the importer skipped or repeated time without restoring a state) logs a warning. |
| `fmi2Get*` / `fmi2Set*` | Current values / see [Setting variables](#setting-variables). An Enumeration uses `fmi2GetInteger`. |
| `fmi2Terminate` | Calls `[model] terminate` on the object, if set. |
| `fmi2Reset` | Calls `[model] terminate` on the object if the FMU wasn't terminated, discards the object, and restores start values and the start time. |
| `fmi2SetDebugLogging` | Turns forwarding of the model's log records on or off. With categories, only messages in them are sent; error messages always are. |

**Step semantics:** `doStep(t, h)` takes the inputs at `t` and produces outputs at `t + h`.

`[events]` has no FMI 2 equivalent, so it is ignored, with a note at build time.

---

## FMI 3

```
InstantiateCoSimulation ─► [EnterConfigurationMode ─► set structural parameters ─► ExitConfigurationMode]
   ─► EnterInitializationMode(start, stop, tol) ─► [set ...] ─► ExitInitializationMode
   ─► event mode (if eventModeUsed) ─► EnterStepMode
   ─► loop: DoStep(t, h) ─► if eventHandlingNeeded or an input clock is due:
            EnterEventMode ─► [GetClock / SetClock, set clocked inputs] ─► UpdateDiscreteStates ─► EnterStepMode
   ─► Terminate ─► FreeInstance
```

FMI 3 works like FMI 2, with these differences:

| FMI call | What the adapter does |
|---|---|
| `fmi3InstantiateCoSimulation` | As `fmi2Instantiate`. Remembers whether the importer uses event mode (`eventModeUsed`). |
| `fmi3EnterConfigurationMode` / `fmi3ExitConfigurationMode` | Allowed when the model has structural parameters: before initialization (configuration mode), and during step mode for `tunable` ones (reconfiguration). Arrays sized by a changed structural parameter are resized to their start values. If a list start no longer fits, it is reset to the type's default with a warning. |
| `fmi3EnterInitializationMode(tol, start, stop)` | Stores the experiment; the FMU's time becomes `start`. |
| `fmi3ExitInitializationMode` | As FMI 2, then enters event mode if `eventModeUsed`, else step mode. |
| `fmi3DoStep(t, h, …)` | As FMI 2 (with `[model] call = false`, no code runs; outputs are re-read from attributes). Afterwards it checks output clocks and `[events]` (see below). Returns `eventHandlingNeeded` (an output clock ticked), `terminateSimulation` (from `[events] terminate`), `earlyReturn = false` and `lastSuccessfulTime = t + h`. A Python step can't be interrupted, so steps never return early. |
| `fmi3EnterEventMode` / `fmi3EnterStepMode` | Change mode. Entering step mode deactivates all clocks. |
| `fmi3UpdateDiscreteStates` | Runs the code of each active input clock (once per event), then re-reads outputs. Returns `discreteStatesNeedUpdate = false` and `terminateSimulation`. `nextEventTime` comes from `[events] next_event_time` when configured. |
| `fmi3Get/Set<type>` | Scalars and arrays (values flattened in row-major order). Integer setters reject values outside the type's range. `fmi3SetBinary` takes `bytes`. |
| Serialize / deserialize | As FMI 2, plus clock activity, intervals and shifts. |

A fmugen FMI 3 FMU always has the independent variable `time` (`Float64`, `causality="independent"`), as FMI 3 requires.

### Events and stopping the simulation

`[events]` lets the model influence the importer through values it already computes:

- `terminate = "attr:done"` (or `"return:<key>"`): when the value is true after a step or clock tick, `fmi3DoStep`/`fmi3UpdateDiscreteStates` return `terminateSimulation = true`.
- `next_event_time = "attr:t_next"`: reported by `fmi3UpdateDiscreteStates` as the time of the model's next event.

`hasEventMode="true"` is declared when the model has clocks or `[events]`.

### Structural parameters and arrays

Variables with `dimensions` are FMI 3 arrays (in FMI 2, one scalar per element: `x[1]`, `x[2]`, …). A dimension can be a fixed size or a **structural parameter**: a `UInt64` (by default) `structuralParameter`, typically a constructor argument such as `n_nodes`. Structural parameters can only be set in configuration mode, or in reconfiguration mode if `tunable`. See [models.md](models.md#arrays-and-structural-parameters-fmi-3).

---

## Clocks (FMI 3)

A clock is an event the importer and the FMU agree on. fmugen maps clocks onto your code:

| | Input clock | Output clock |
|---|---|---|
| **Who ticks it** | the importer (`fmi3SetClock` in event mode) | the model, when a value it computes is true |
| **What runs** | `call`: a method of the model object, or a function, run during `fmi3UpdateDiscreteStates` | nothing extra; the step that ticked it already ran |
| **Clocked inputs** | passed to `call` instead of the step method | — |
| **Clocked outputs** | read from `call`'s return value or the object after the tick | read from the step's result or the object after the tick |
| **Interval variability** | `constant`, `fixed`, `tunable`, `changing`, `countdown`, `triggered` | `triggered` |

**Interval variability (input clocks):**

| `interval_variability` | Interval comes from | Importer may change it |
|---|---|---|
| `constant` | `interval` in the config | never |
| `fixed` | `interval` (and `shift`) | interval and shift: before initialization ends (`fmi3SetInterval*`, `fmi3SetShift*`) |
| `tunable` | `interval` (and `shift`) | as `fixed`, and the interval also in event mode |
| `changing`, `countdown` | the model: `interval_from` after each tick | — |
| `triggered` | none (aperiodic; the importer ticks it when it wants) | — |

**Rules:**
- `fmi3GetIntervalDecimal` returns qualifier `2` (changed) the first time and when the interval changed, `1` (unchanged) otherwise, and `0` (not yet known) for triggered clocks. `fmi3GetIntervalFraction` / `fmi3GetShiftFraction` give the same values as fractions.
- An **output clock** reads as active once after it ticks: `fmi3GetClock` resets it, as FMI 3 requires. If the importer doesn't use event mode, a tick can't be reported; it is dropped with a warning.
- **Time arguments in clock calls:** `[time]` arguments the clock's `call` accepts get `time` = the event time, and `step_size` = the time since that clock last ticked (its interval on the first tick).
- **State:** clock activity, intervals and shifts are part of the FMU state, so rollback restores them.

Examples: [models.md](models.md#clocks-fmi-3) shows how to configure periodic and triggered input clocks and output clocks; simple-pid's `PID` (periodic) and filterpy's `KalmanFilter` (triggered, with array inputs) were tested with them ([tested-models.md](tested-models.md)).

---

## Shared behaviour

### Setting variables

| Variable | Before initialization is done | After (step / event mode) |
|---|---|---|
| input | ✓ | ✓, used by the next step (clocked inputs: by the next tick) |
| parameter, `fixed` | ✓ | ✗ error. Fixed parameters can't change after initialization. |
| parameter, `tunable` | ✓ | ✓. For classes the value is also written to the object attribute (`attr`), and calculated parameters are re-read. |
| structural parameter (FMI 3) | in configuration mode | `tunable` ones in reconfiguration mode |
| state (`local`, `initial="exact"`) | ✓ sets its initial value | ✗ |
| output, local, calculated parameter, `time` | ✗ | ✗ |

"Before initialization is done" covers both before and during initialization mode.

### FMU state

`canGetAndSetFMUstate` / `canGetAndSetFMUState` and `canSerializeFMUstate` / `canSerializeFMUState` follow `[model] save_state`: `true` unless it is `false`. The state is pickled, with `cloudpickle` as a fallback (lambdas, local functions). With a list of attributes, only those are saved and restored; the rest of the object (an ONNX session, a device handle) stays as it is. `fmugen build` does not run the model: `fmugen init` decides `save_state` when it makes its probe call, by trying to pickle the object, then each attribute.

Restoring a state restores the whole Python object, so classes that keep internal state (integrators, filters, controllers) roll back correctly too. `[model] constants` and `call_constants` written as values (numbers, lists, `{ python = ... }`) are saved with the state and put back by reset, so a model that changes one in place (appends to a list) repeats its results. Constants computed by `{ call = ... }` (a loaded model, a downloaded file) are made once per FMU instance and shared by reset and rollback.

**Only restore states from a trusted source.** A serialized FMU state is a pickle, and unpickling runs code: `fmi2DeSerializeFMUstate` / `fmi3DeserializeFMUState` with bytes from someone else can run anything on the machine. Restore states this FMU saved, as importers do for rollback; don't load states received from others.

### Logging

Your code uses the standard `logging` module. While the adapter runs your code, a handler forwards every record to the importer through UniFMU:

| Record level | Status | Category |
|---|---|---|
| ≥ `ERROR` | error | `logStatusError` |
| `WARNING` | warning | `logStatusWarning` |
| below | OK | `logAll` (FMI 2) / `logEvents` (FMI 3) |

- **When forwarding happens:** only during an FMI call. UniFMU can only carry a log message while it is processing a command, so records from background threads aren't forwarded.
- **Filtering:** `fmi2SetDebugLogging` / `fmi3SetDebugLogging` with a list of categories limits the messages to those categories (errors are always sent); with none, every category is sent. `modelDescription.xml` declares the standard categories plus `logUnifmuMessages`.

### Errors

Any exception raised by your code makes the call return an error status. The traceback is logged under `logStatusError`. The importer decides what happens next; most stop the simulation.

### Capabilities written to `modelDescription.xml`

| Attribute | Value |
|---|---|
| `needsExecutionTool` | `true`: a Python interpreter is needed (see [packaging.md](packaging.md#the-fmus-python-environment)) |
| `canHandleVariableCommunicationStepSize` | `true`, or `false` with `[experiment] fixed_step` |
| `canGetAndSetFMUstate`/`FMUState`, `canSerializeFMUstate`/`FMUState` | `[model] save_state` |
| `canNotUseMemoryManagementFunctions` (FMI 2) | `true` |
| `hasEventMode` (FMI 3) | `true` with clocks or `[events]` |
| `canReturnEarlyAfterIntermediateUpdate`, `providesIntermediateUpdate` (FMI 3) | `false` |
| `canBeInstantiatedOnlyOncePerProcess` (FMI 3) | `false` |
| derivative and interpolation capabilities | not declared, so they take their FMI defaults: `false` / `0` |

---

## Unsupported functions

Everything marked **x** in the [support overview](#support-overview) is never declared in `modelDescription.xml`, so a compliant importer won't call it. None of it is needed for Co-Simulation of ordinary Python models:

| Group | What it is for | Why it isn't needed here |
|---|---|---|
| `fmi2CancelStep`, `fmi2Get*Status` | Asynchronous `doStep` (returning `fmi2Pending`), and asking whether the FMU wants to stop | `doStep` always finishes before returning. In FMI 2 the FMU never asks to stop (use FMI 3 `[events] terminate`). |
| Directional / adjoint derivatives | Jacobians for Model Exchange solvers or co-simulation algorithms that solve algebraic loops | Your model would have to compute derivatives, which ordinary Python models don't. |
| Input and output derivatives | Interpolating inputs / extrapolating outputs within a step | Accuracy refinements for large steps. Importers hold inputs constant unless the FMU says otherwise. |
| Model Exchange and Scheduled Execution | The importer integrates the model's equations, or schedules model partitions | UniFMU's Python backend implements Co-Simulation only. |
| `fmi3Get(NumberOf)VariableDependencies` | Dependencies at runtime | They are declared statically in `<ModelStructure>` (`depends_on`). |

> **Caveat:** if a non-compliant importer calls one of these anyway, UniFMU's Python backend stops with "unrecognized command", and the simulation fails rather than getting an error status back.

## Out of scope

UniFMU can do more than fmugen currently targets:

- **Other backends:** C# and Java.

See the [UniFMU repository](https://github.com/INTO-CPS-Association/unifmu) for these.
