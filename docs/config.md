# `fmugen.toml` reference

`fmugen.toml` describes how an unmodified Python model becomes an FMU. It sits in the model's folder, and every path in it is relative to that folder. This page is the reference for every key; it is the only one, and the other pages link here.

- `fmugen init` writes a first version: what it infers, and a walk through a generated config, are in [init.md](init.md).
- [models.md](models.md) has a worked example for each kind of model; the [complete example](#complete-example) below uses most keys at once.
- The `fmugen` command's options are in [cli.md](cli.md).

Unknown keys are rejected with an error, so typos don't go unnoticed. Keys marked **FMI 3** are only accepted when building FMI 3 (`[model] fmi_version = 3` or `fmugen build --fmi 3`). Building such a config as FMI 2 fails with an error that says so. Otherwise the same config builds either version.

```toml
[model]                     # what to call
[model.constants]           # fixed non-FMI arguments (constructor / function)
[model.call_constants]      # fixed non-FMI arguments (step method)
[experiment]                # default experiment and step-size capability
[time]                      # arguments that receive the FMU time / step size
[structural_parameters]     # FMI 3: FMU variables, one inline table each ...
[parameters]
[calculated_parameters]
[inputs]
[states]
[outputs]
[locals]
[clocks.<name>]             # FMI 3: one table per clock
[events]                    # FMI 3: terminate / next event time
```

---

## `[model]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `entry` | string | **required** | `"path/file.py:Name"` (a file under this folder) or `"package.module:Name"` (an installed module). `Name` is a function or a class. |
| `fmi_version` | `2` or `3` | `2` | FMI version to build. `fmugen build --fmi` overrides it. |
| `call` | string or `false` | `"__call__"` if the class is callable | Classes: the method run on each step. A **function** entry with `call` set is a factory: it is called once, at the end of initialization, with the parameters (like a constructor), and `call` is the method run on each step on the object it returns, e.g. `entry = "silero_vad:load_silero_vad"`, `call = "__call__"`. `"module:function(self, ...)"` (classes): the step is a function taking the model object, for libraries stepped that way, e.g. `"pandapipes:pipeflow(self)"` or `"mujoco:mj_step(self.model, self.data)"`; `self` is the object and `self.<path>` a path on it, read on every step, other arguments are as in a computed constant, and arguments bound with `arg:`/`pos:` come after them. `false` (classes and functions; needs `[clocks]`): nothing runs on `doStep`, only clocks run code. |
| `name` | string | the entry's name | `modelName` in `modelDescription.xml`. `fmugen build --name` overrides it. |
| `description` | string | first line of the entry's docstring, else the module's (cut at 200 characters) | `description` in `modelDescription.xml`. |
| `author` | string | `""` | `author` in `modelDescription.xml`. `fmugen build --author` overrides it. |
| `sources` | list of strings | `[]` | Extra files and directories to copy into the FMU, keeping their paths. A file entry is always copied. Data files (CSV, EPW, JSON, …) too. Their folders go on `sys.path`, so flat and package imports both work. |
| `globals` | list of `"module:NAME"` | `[]` | Module-level variables the model changes while it runs (`global CALLS; CALLS += 1`), e.g. pythermalcomfort's `"pythermalcomfort.jos3_functions.thermoregulation:PRE_SHIV"`. On reset they go back to their value when the model was imported, and they are saved and restored with the FMU state. `fmugen init` lists the ones holding numbers, strings or arrays that the model's modules rebind. Module globals are shared by every instance of the FMU in one process. |
| `cwd` | string | none | A folder, relative to `fmugen.toml`, that the model's code runs in (its working directory), inside the FMU's copy of `sources`. For models that open files by a relative path, such as `open("data/weather.csv")`: use `"."` and list the files in `sources`. It applies to the import, setup, construction and every call, and the importer's working directory is put back afterwards. `fmugen init` sets both when a string in the model or a `--start` value names a file next to the config. |
| `requirements` | list of strings | `[]` | pip requirement specifiers the model needs at runtime, e.g. `"simple-pid==2.0.1"`. They must be installed in the environment fmugen runs in. `fmugen init` writes them: the installed distributions that provide the entry's package, the modules the model's own files import, and the modules named in `setup`, computed constants, `convert` and `enum`, pinned to the installed versions. See [packaging.md](packaging.md#the-fmus-python-environment). |
| `kind` | `"function"` | none | Treat a class as a function: construct it with the inputs on every step and read outputs from the new object (`return:<attr>`). For classes whose constructor does all the work. |
| `create` | classmethod name | none | Classes only: build the object with this classmethod instead of calling the class, e.g. `"from_pretrained"`. Parameters bound to `init:` (the default) become its arguments. For models loaded from saved weights or files. |
| `call_on_class` | bool | `false` | Classes only: call the step method on the class, not on the object: `Model.building(...)` instead of `Model().building(...)`, for methods written without `self`. `init` sets it when the step method is written in the class body and its first parameter isn't `self` (or annotated with the class); not for a function defined elsewhere and attached to the class, such as pypsa's `pf(n, ...)`, which gets the object as `n`. |
| `save_state` | `true`, `false` or list of attribute names | `true` | How the FMU saves and restores its state (rollback). `true`: the whole model object is pickled (with `cloudpickle` when `pickle` can't, e.g. lambdas). `false`: not supported (`canGetAndSetFMUState="false"`), e.g. for hardware. A list such as `["_state", "_context"]`: only those attributes are saved and put back on restore; the rest of the object (an ONNX session, a device handle) is kept as it is. `fmugen init` sets it by trying to pickle the object. |
| `setup` | list | `[]` | Calls run each time the FMU initializes (also after a reset), before the model is used. `"module:function"` or `"module:function(args)"` runs before the class is constructed (e.g. `"psychrolib:SetUnitSystem(psychrolib.SI)"`). A bare `"method"` or `"method(args)"` runs on the object right after construction (e.g. `"reset"`). Arguments are Python literals or dotted names of importable objects. The table form `{ call = "...", args = [...], kwargs = {...} }` takes the same values as `[model.constants]`. |
| `init_call` | bool | `true` for functions with a step (except when the function takes the step size and `[experiment]` has no `step_size`), `false` otherwise | Call the function or step method when leaving initialization mode, to compute the initial outputs. States are not advanced by this call. Without it, a class's outputs read from the return value are 0 at the start time. `init --probe` sets it for a class whose step method changed nothing on the object in the probe (e.g. blackscholes' `price()`). Without it, when a `get:` output's method raises at initialization (floris' `get_farm_power()` before the first `run()`), the model is called once anyway, as with `true`. |
| `terminate` | string | none | Classes only: a method called on `fmi2Terminate`/`fmi3Terminate` (for example `"close"`), and on reset when the FMU wasn't terminated first, so the resources the object holds (a serial port, a file) are released then rather than when it is garbage collected. |
| `discard_on` | list of `"module:Class"` | `[]` | Exceptions and warnings that reject a step: when the step raises one (a warning in the list is raised as an exception during the step), `doStep` returns `discard` and the FMU is put back to the start of the step, so a variable-step importer can retry with a smaller step. E.g. `["scipy.integrate:IntegrationWarning", "builtins:ArithmeticError"]`. Undoing the step saves the FMU state before every step, so it needs `save_state` (`build` refuses `false`). Other exceptions are still errors. |
| `integrate` | `"euler"`, `"rk4"` or `"scipy:<method>"` | none | The model is an ODE right-hand side: it returns the **derivatives** of the `[states]`, and the FMU integrates them over each step. `<method>` is a `scipy.integrate.solve_ivp` method (`RK45`, `RK23`, `DOP853`, `Radau`, `BDF`, `LSODA`), run with the importer's tolerance as `rtol`. Each state has `der` instead of `next`; `[time]` gives the model the time of each stage. A failed solve returns `discard`. See [models.md](models.md#ode-right-hand-sides). |
| `substeps` | positive integer | `1` | With `integrate = "euler"` or `"rk4"`: fixed sub-steps per communication step. Without `integrate`: the step runs this many times per communication step, each with an equal part of it (`[time]` gives each run its own time and step size), for models stepped at a fixed rate of their own, such as an IMU filter at its sample rate; outputs are read after the last run. |
| `annotations` | XML string | none | `<Tool name="...">` elements added to `modelDescription.xml`: inside `<VendorAnnotations>` in FMI 2, and as `<Annotation type="...">` inside `<Annotations>` in FMI 3. |

### `[model.constants]` and `[model.call_constants]`

These are fixed keyword arguments that are not FMU variables, typically values that FMI can't represent.
- `constants` go to the constructor (classes) or to the function (functions).
- `call_constants` go to the step method (classes only).

| Value | Passed to the model as |
|---|---|
| a TOML string, number, bool, array or table | that value (an array becomes a list) |
| `{ python = "<literal>" }` | `ast.literal_eval(literal)`, e.g. `"None"`, `"(0, 100)"`, `"{'a': 1}"` |
| `{ ref = "module:attr.path" }` or `{ ref = "module.attr" }` | the imported object, e.g. a class or function |
| `{ call = "module:function(args)" }` | the result of calling it when the FMU initializes. Arguments are literals, dotted names of importable objects or builtins, calls of those (`numpy.linspace(0, 20, 200)`, `float("inf")`), and lists, tuples and dicts of these; keyword arguments allowed. Without parentheses, `"module:attr.path"` is the object itself, not called: a function to pass (`"numpy:sin"`) or an enum member (`"pyfluids:FluidsList.Water"`). `init --probe` stops if the call fails. Example: `"huggingface_sb3:load_from_hub(repo_id='sb3/demo-hf-CartPole-v1', filename='ppo-CartPole-v1.zip')"`. `init --start "NAME=call:..."` writes it. |

Every key must be an argument of the constructor, function or method (unless it accepts `**kwargs`). A key that is a number (`0 = ...`, `1 = ...`) is a positional argument instead, for callables that only take `*args` such as `nashpy.Game(A)`; positions are counted together with variables bound to `to = "pos:N"` and must be numbered 0, 1, 2, ... without gaps. `init --start NAME=call:...` writes one for a `*args` callable.

---

## `[experiment]`

All keys are optional.

| Key | Type | Meaning |
|---|---|---|
| `start_time` | number | `DefaultExperiment startTime`; also the FMU's time before the importer sets one. |
| `stop_time` | number | `DefaultExperiment stopTime`. |
| `step_size` | number | `DefaultExperiment stepSize`. Also used for the initialization call. |
| `tolerance` | number | `DefaultExperiment tolerance`. |
| `fixed_step` | bool, default `false` | The model only works with `step_size`: writes `canHandleVariableCommunicationStepSize="false"`, and a `doStep` with any other step size returns an error. Requires `step_size`. |

---

## `[time]`

Each key is an argument of the step function or method (or of a clock's `call` that accepts it); its value says what to pass:

| Value | Passed on a step | Passed on a clock tick (FMI 3) |
|---|---|---|
| `"time"` | `t`, the communication point at the start of the step | the event time |
| `"step_size"` | `h`, the communication step size | time since the clock last ticked (its interval on the first tick) |
| `"end_time"` | `t + h` | the event time + the above |
| `"start_time"` | the simulation's start time (`fmi2SetupExperiment` / `fmi3EnterInitializationMode`) | same |
| `"stop_time"` | its stop time, or `None` when the importer gives none | same |
| `"tolerance"` | the importer's relative tolerance, or `None` when it gives none, e.g. for `solve_ivp(rtol=...)` | same |

```toml
[time]
dt = "step_size"
```

For a model that takes a date-time (pvlib, pandas, weather data), write the entry as a table with an `epoch`: the argument then gets `epoch + t` seconds as a `datetime.datetime`. FMU time 0 is the epoch. An epoch with a time zone (`+00:00`) gives aware date-times; one without gives naive ones.

```toml
[time]
time = { source = "end_time", epoch = "2026-06-21T06:00:00+00:00" }
```

`fmugen init` writes this for arguments annotated `datetime`/`date` (also `Optional[...]`), and for unannotated ones named `when`, `date`, `datetime`, `dateandtime`, `date_time`, `timestamp` or `utc_time`, with `source = "end_time"` and today's date at 00:00 as the epoch, without a time zone. With `--probe`, if the model rejects that, `init` adds `+00:00` (UTC). For an argument matched only by its name, `--probe` also calls the model with `0.0` there: if only the date-time works, the comment says it was confirmed; if only the number works, the argument becomes an ordinary input; otherwise it stays a date-time marked as a guess. Set the epoch you need.

`source` defaults to `"time"`. For a function of time with no state (a solar position, a weather lookup), use `"end_time"` so each output belongs to the time it is reported at.

---

## Variables

Each variable is one line: `name = { key = value, ... }`. The name is the FMU variable name and must be a valid Python identifier; in FMI 3, `time` is reserved for the independent variable. Variables get value references in this order: structural parameters, parameters, calculated parameters, inputs, states, outputs, locals, then (FMI 3) `time`, then clocks.

| Section | FMI causality | Value flows | Default binding |
|---|---|---|---|
| `[structural_parameters]` **FMI 3** | `structuralParameter` | importer → model | class: `init:<name>`; function: `arg:<name>` |
| `[parameters]` | `parameter` | importer → model | class: constructor argument `init:<name>`; function: `arg:<name>` |
| `[calculated_parameters]` | `calculatedParameter` | model → importer | `attr:<name>` (classes only) |
| `[inputs]` | `input` | importer → model | `arg:<name>` |
| `[states]` | `local`, `initial="exact"` | fed back each step | `arg:<name>`, plus the required `next` (`der` with `[model] integrate`) |
| `[outputs]` | `output` | model → importer | class: `attr:<name>`; function: `return:<name>` |
| `[locals]` | `local` | model → importer | as outputs |

### Types

| `type` | FMI 2 | FMI 3 | Python value |
|---|---|---|---|
| `Real` | `Real` | `Float64` | `float` |
| `Float64` | `Real` | `Float64` | `float` |
| `Float32` | `Real` | `Float32` | `float` |
| `Integer` | `Integer` | `Int32` | `int` |
| `Int32` | `Integer` | `Int32` | `int` |
| `Int8`, `UInt8`, `Int16`, `UInt16`, `UInt32`, `Int64`, `UInt64` | `Integer` (32-bit: `build` notes that `UInt32`, `Int64` and `UInt64` values outside its range are an error) | same | `int` (range-checked) |
| `Boolean` | `Boolean` | `Boolean` | `bool` |
| `String` | `String` | `String` | `str` |
| `Binary` | — | `Binary` | `bytes`; `start` is a hex string, e.g. `"0aff"` |
| `Enumeration` | `Enumeration` | `Enumeration` | `int`, or the `Enum` member with `enum` |

Without `type`, it comes from `start`: `bool` → Boolean, `int` → Integer, `str` → String, otherwise Real. `items` or `enum` make it an Enumeration. Structural parameters default to `UInt64`.

### Keys for every variable

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | see [Types](#types) | The FMI type. |
| `start` | value, or list for arrays | see below | Start value. An Enumeration takes an item name or a 1-based index. An array takes a flat list (row-major) or one value for every element. |
| `unit` | string | none | Floating-point only. Added to `<UnitDefinitions>`, with its SI `<BaseUnit>` (exponents, factor, offset) when fmugen recognises it: SI units and the usual derived ones with prefixes (`km`, `kW`, `mbar`), `degC`, `degF`, `deg`, `h`, `min`, `l`, `bar`, `Wh`, `rpm`, combined as in Modelica (`m/s2`, `W/(m2.K)`, `kg.m-3`). Importers use it to check and convert units. `C`, `F`, `S` and `T` on their own get none: they are as often written for Celsius, Fahrenheit, seconds and temperature as for coulomb, farad, siemens and tesla (use `degC`, `degF`, `s`). |
| `description` | string | none | Variable description. |
| `quantity` | string | none | Physical quantity, e.g. `"ThermodynamicTemperature"`. |
| `min`, `max` | number | none | Numeric types and Enumeration. Declared range; not enforced by the adapter. `build` checks `min <= max` and that `start` is in the range. |
| `nominal` | number | none | Floating-point only. |
| `variability` | string | parameters: `fixed`; others: `continuous` for unclocked floating-point, `discrete` otherwise | parameters and structural parameters: `fixed`/`tunable`; inputs: `discrete`/`continuous`; outputs and locals: also `constant`. |
| `initial` | string | see below | Overrides the FMI `initial` attribute where FMI allows a choice. |
| `items` | list of strings | none | Enumeration items, numbered from 1. The model receives the integer. |
| `enum` | string | none | `"module:EnumClass"`: items are the enum's members, and the model receives and returns members. |
| `dimensions` | list | none | Makes the variable an array. Each entry is a size (`3`) or, in FMI 3, the name of a structural parameter (`"n"`). FMI 2 has no arrays: the FMU has one scalar per element, `x[1]`, `x[2]`, … (`x[1,2]` for a matrix), and the model still gets the whole array. |
| `numpy` | bool | `false` | Arrays only: pass the value to the model as a `numpy.ndarray` instead of nested lists. Values coming back are accepted either way. |
| `convert` | `"module:function"`, `"pint"` or `"series"` | none | Values passed to the model (parameters, inputs, states): call this function on the value first, e.g. `"torch:tensor"`, `"jax.numpy:asarray"`. `init` sets it from `torch.Tensor`, JAX and TensorFlow annotations. `"pint"` passes a pint quantity in the variable's `unit` (from pint's application registry), for models such as `fluids.units` that only accept quantities. `"series"` passes a one-value pandas Series whose index is the date `epoch` + the FMU time at the start of the step (in seconds), for models that read the date from their inputs' index, such as pyet. `init --probe` writes it when the call fails with plain numbers and with arrays, with today's date at 00:00 as the epoch. |
| `epoch` | ISO 8601 date-time | none | With `convert = "series"` (required there): the date at FMU time 0, e.g. `"2026-06-21T00:00:00"`. |
| `clocks` **FMI 3** | list with one clock name | none | Inputs, outputs and locals: the variable belongs to that clock (see [`[clocks.<name>]`](#clocksname-fmi-3)). |
| `extra` | table | none | More XML attributes of the variable, written as given, e.g. `extra = { displayUnit = "degC", relativeQuantity = true, unbounded = true }` (in FMI 2, `displayUnit`, `relativeQuantity`, `unbounded`, `reinit` and `derivative` go on the type element). A `displayUnit` also gets its `<DisplayUnit>` in the unit definition, with the conversion when both units are recognised. Attributes fmugen writes from other keys (`start`, `unit`, `min`, …) can't be set here. |

**Start values and `initial`:**
- **Structural parameters, parameters, inputs, states:** always have a start value. If you don't give one, it is 0 / `false` / `""` / empty / the first item.
- **Outputs without `start`:** `initial="calculated"`. **With `start`:** `initial="exact"`, and the start value is reported until the model produces a value (e.g. before the first step).
- **Locals and calculated parameters without `start`:** `initial="calculated"`. **With `start`:** `initial="approx"`.

### Binding keys

| Key | Sections | Value | Meaning |
|---|---|---|---|
| `to` | structural parameters, parameters, inputs, states | `"init:<arg>"` | Constructor argument (classes). |
| | | `"arg:<arg>"` | Argument of the function or step method; for a clocked input, of the clock's `call`. |
| | | `"call:<method>"` | A setter method called with the value after construction and before every step (classes), e.g. tclab's `Q1(value)`. |
| | | `"pos:<N>"` | Positional argument number N (0, 1, … without gaps) of the function or step method, for positional-only arguments such as those of C extensions. In a class's `[parameters]` and `[structural_parameters]`, a position of the constructor instead (numbered separately), e.g. for `control.StateSpace(*args)`. |
| | | `"attr:<name>"` | Attribute set on the object after construction and before every step (classes); clocked: before every tick. Dotted paths allowed. Where a part isn't an attribute, it's a key or an index: `"attr:loads.p_set.l"` sets `obj.loads.p_set["l"]` (a key the table already has), `"attr:data.qpos.0"` the first element. |
| | | `"arg:<arg>[<i>]"`, `"arg:<arg>[<key>]"` | Item `i` of a tuple argument, or key `key` of a dict argument: the variables bound to the items of one argument are put together into a tuple (items 0, 1, … without gaps) or a dict, e.g. torchani's `forward((species, coordinates))`. `init` splits a tuple or dict `--start` value this way. Also `"init:<arg>[…]"`. |
| | | `"arg:<arg>.<field>"` | Field `field` of an object argument (a dataclass, a pydantic model, an attrs class, a namedtuple): the variables bound to its fields build it. If the argument has a default instance, that instance is copied with these fields replaced (`dataclasses.replace`, `model_copy`, `attrs.evolve`), so its other fields keep their values; otherwise the class from its annotation is called with the fields. `init` writes one variable per field, `<arg>_<field>`. Also `"init:<arg>.<field>"`. |
| `from` | outputs, locals, calculated_parameters | `"return"` | The whole return value. |
| | | `"return:<key>"` | `result[key]` for a dict, `result[int(key)]` for a tuple or list, else `result.key`. A dotted path (`"return:zone.T"`) takes these steps in turn. Where there's no such attribute, a number also picks an element of any other iterable (a `dict_keys` view, a set: `"attr:bodies.0"`; `-1` is the last), and on an object that takes keys but isn't a dict, a part is tried as a key (pybamm's `"return:Voltage [V].entries.-1"` reads `solution["Voltage [V]"]`). |
| | | `"attr:<name>"` | An attribute of the object after the step (classes). Dotted paths allowed, read as `return:` paths are (keys and indexes too, e.g. `"attr:res_junction.p_bar"`). Calculated parameters must use this form. |
| | | `"get:<method>"` | What a method of the object returns, called without arguments after the step (classes), for results read through getters, e.g. vqf's `"get:getQuat6D"`. Dotted paths allowed (`"get:sim.state"`). |
| `next` | states (**required**, except with `integrate`) | same forms as `from` | Where the state's value for the next step comes from. |
| `der` | states, with `[model] integrate` | `"return"` (one state); required for several | Where the state's derivative comes from in the model's result, e.g. `"return:dx"`, `"return:0"`. |
| `attr` | parameters, structural parameters | string or `false` | Tunable class parameters: the attribute written when the importer changes the value after initialization. Defaults to the parameter's `to` name; required for a tunable parameter passed by position (`to = "pos:N"`). It must be an attribute the object has: setting the parameter to a new value gives an error status otherwise, instead of adding an attribute the model never reads. `false` means it is passed only to the constructor. |
| `depends_on` | outputs | list of input names | `<ModelStructure>` dependencies of the output. Without it, the output may depend on every input. |

Constructor and call arguments named by `init:`, `arg:` and `[time]` must exist in the signature (unless it accepts `**kwargs`).

---

## `[composite]`

A config with `[composite]` instead of `[model]` builds one FMU from several fmugen configs. See [models.md](models.md#several-models-in-one-fmu).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `parts` | table `{ name = "path" }` | required | The parts, in the order they run each step. A path is a `fmugen.toml` or a folder containing one, relative to this file. Two or more. |
| `connections` | list of strings | `[]` | `"part.output -> part.input"`. |
| `name`, `description`, `author` | string | the folder name, a list of the parts, empty | As in `[model]`. |
| `fmi_version` | 2 or 3 | 2 | As in `[model]`. |

`[experiment]` may also be given. Other top-level tables are not allowed in a composite config.

Parts may have clocks, structural parameters and `[events]` (FMI 3): the FMU's clocks are the parts' clocks, named `<part>.<clock>`, and a structural parameter sizes the arrays of its own part. Clocked variables and arrays sized by a structural parameter can't be connected.

---

## `[clocks.<name>]` (FMI 3)

One table per clock; `<name>` is the FMU clock name. See [fmi.md](fmi.md#clocks-fmi-3) for how clocks behave at runtime.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `causality` | `"input"` or `"output"` | `"input"` | Input: the importer ticks it. Output: the model ticks it. |
| `interval_variability` | string | `"constant"` with `interval`, else `"triggered"` | `constant`, `fixed`, `tunable` (periodic: need `interval`), `changing`, `countdown` (need `interval_from`), `triggered` (aperiodic). Output clocks must be `triggered`. |
| `interval` | number (s) | none | Periodic input clocks: the interval. |
| `shift` | number (s) | `0.0` | Periodic input clocks: offset of the first tick. |
| `interval_from` | `"attr:..."` / `"return..."` | none | `changing`/`countdown`: where the model's next interval is read after each tick. |
| `call` | string | none | Input clocks: what runs on a tick. A method name (classes), a function in the entry's module (functions), or `"module:function"`. Required if the clock has clocked inputs. |
| `from` | `"attr:..."` / `"return..."` | **required** for output clocks | The value that ticks the clock when it is true after a step or tick. |
| `description` | string | none | Clock description. |

---

## `[events]` (FMI 3)

| Key | Value | Meaning |
|---|---|---|
| `terminate` | `"attr:..."` / `"return..."` | When true after a step or tick, the FMU returns `terminateSimulation = true`. |
| `next_event_time` | `"attr:..."` / `"return..."` | Reported by `fmi3UpdateDiscreteStates` as the time of the next event. |

`[events]` is ignored, with a note, when building FMI 2.

---

## Validation rules

`fmugen build` rejects a config with a message naming the section and key (e.g. `[inputs] x: unknown key(s) ['strat']`) when:

- it has an unknown section or key, or is missing `[model] entry`;
- a binding names an argument the target doesn't have;
- two variables, or a variable and a clock, share a name;
- a type, `variability` or `initial` doesn't fit the section or FMI version;
- a start value is invalid for its type (wrong kind, out of range, bad hex);
- `pos:` indices have gaps, a state has no `next`, or a clocked input's clock has no `call`;
- `fixed_step` is set without `step_size`, or `call = false` is set without clocks;
- it defines no variables at all;
- an FMI 3-only feature is used in an FMI 2 build;
- the model's module (the entry file or a file in `sources`) has the name of a module the FMU's backend already uses (`model`, `backend`, `main`, `abstract_backend`, `schemas`, `fmugen_runtime`, `fmugen_launch`) or of a Python standard library module (e.g. `queue.py`, `random.py`), which would shadow it: rename the file. Your code goes into `resources/fmugen_model/` inside the FMU, which is first on `sys.path`.

The model itself is not run: errors in the model show up when the FMU runs, or earlier in `fmugen init --probe`.

---

## Complete example

An FMI 3 config that uses most sections. It is illustrative: `plant.py` is not part of the repository. For real configs, see the hand configs linked from [tested-models.md](testing/tested-models.md).

```toml
[model]
entry = "plant.py:Plant"            # your file and class
fmi_version = 3
call = "step"
setup = ["reset"]                   # plant.reset() after construction
sources = ["lib"]                   # local helper package
requirements = ["numpy>=2"]
name = "plant"
description = "Example plant"
author = "Me"

[model.constants]
solver = "rk4"                      # a constructor argument that isn't an FMU variable

[experiment]
step_size = 0.1
stop_time = 60.0

[time]
dt = "step_size"

[structural_parameters]
n = { start = 3, description = "number of zones" }

[parameters]
area = { start = 2.0, unit = "m2" }                              # Plant(area=...)
gain = { start = 1.0, variability = "tunable" }                  # also plant.gain = ... later
temps0 = { dimensions = ["n"], start = 20.0, unit = "degC", to = "attr:temps", numpy = true }

[calculated_parameters]
volume = { unit = "m3" }                                         # plant.volume after construction

[inputs]
heat = { start = 0.0, unit = "W" }                               # step(heat=...)
valve = { start = 1, type = "UInt8", to = "call:set_valve" }     # plant.set_valve(value)
sensor = { start = 0.0, clocks = ["sample"] }                    # passed to plant.sample(sensor=...)

[states]
level_prev = { start = 0.5, unit = "m", next = "return:level" }

[outputs]
level = { from = "return:level", unit = "m" }
temps = { dimensions = ["n"], unit = "degC" }                    # plant.temps
alarm = { start = false }

[locals]
energy = { unit = "J" }

[clocks.sample]
interval = 1.0
call = "sample"

[clocks.overheat]
causality = "output"
from = "attr:overheated"

[events]
terminate = "attr:done"
```

---

## How the config maps onto `modelDescription.xml`

| Config | FMI 2 XML | FMI 3 XML |
|---|---|---|
| `[model] name / description / author` | `modelName / description / author` | same |
| — | `guid` | `instantiationToken` |
| `[experiment]` | `<DefaultExperiment>` | same |
| `fixed_step` | `canHandleVariableCommunicationStepSize` | same |
| `[model] save_state` is not `false` | `canGetAndSetFMUstate`, `canSerializeFMUstate` | `canGetAndSetFMUState`, `canSerializeFMUState` |
| clocks or `[events]` | — | `hasEventMode="true"` |
| each variable | `<ScalarVariable>` + `<Real>`/`<Integer>`/... child with start, unit, min, max, nominal, quantity | `<Float64>`/`<Int32>`/`<Binary>`/... element with the same attributes; String and Binary starts as `<Start value=…/>` |
| `dimensions` | — | `<Dimension start=…/>` or `<Dimension valueReference=…/>` |
| `clocks` on a variable | — | `clocks="<clock valueReference>"` |
| `[clocks.<name>]` | — | `<Clock causality intervalVariability intervalDecimal shiftDecimal/>` |
| — | — | `<Float64 name="time" causality="independent"/>` |
| `unit` values | `<UnitDefinitions>` | same |
| `items` / `enum` | `<TypeDefinitions><SimpleType><Enumeration>` | `<TypeDefinitions><EnumerationType>` |
| outputs, `depends_on` | `<ModelStructure><Outputs>` | `<ModelStructure><Output>`, including output clocks |
| unclocked outputs without `start`, calculated parameters | `<InitialUnknowns>` | `<InitialUnknown>` |
