# `interface.json` reference

- [What it is](#what-it-is)
- [Top-level keys](#top-level-keys)
- [`entry`](#entry)
- [Constants and setup calls](#constants-and-setup-calls)
- [`variables`](#variables)
- [Bindings](#bindings)
- [`clocks` and `events` (FMI 3)](#clocks-and-events-fmi-3)
- [Composite FMUs](#composite-fmus)
- [Versioning](#versioning)
- [Example](#example)

---

## What it is

`resources/interface.json` is fmugen's normalised description of one model. `fmugen build` writes it from the validated `fmugen.toml` and the imported model. Inside the FMU, fmugen's engine reads it to map every FMI variable onto your code, and `modelDescription.xml` is generated from the same data. You never edit it. It's documented here for anyone debugging an FMU or writing tools around fmugen.

The JSON Schema is [`src/fmugen/interface.schema.json`](../src/fmugen/interface.schema.json) (JSON Schema 2020-12, shipped in the package). `tests/test_interface_schema.py` checks every example FMU, FMI 2 and 3, and a composite FMU against it.

| | Where |
|---|---|
| Built by | `config.normalize()` (`src/fmugen/config.py`); for composite FMUs, `compose.combine()` (`src/fmugen/compose.py`) |
| Written by | `builder.write_interface()` (`src/fmugen/builder.py`), as indented JSON |
| Read by | the engine (`templates/fmugen_runtime/`: `make_engine`, `Engine`, `CompositeEngine`), `description.py` (`modelDescription.xml`), `distribute.py` (`--vendor`, `--compile`, `--hf-weights`) |

With `--compile`, the file is frozen into the executable and isn't in `resources/`.

---

## Top-level keys

Every key is always present in a single-model file.

| Key | Type | Meaning | From |
|---|---|---|---|
| `spec_version` | `1` | Format version, see [Versioning](#versioning) | fixed |
| `fmi_version` | `2` or `3` | FMI version of the FMU | `--fmi`, `[model] fmi_version` |
| `model_name` | string | `modelName` | `--name`, `[model] name`, else the entry's name |
| `description` | string | `description` | `[model] description`, else the first line of the entry's docstring (≤ 200 characters) |
| `author` | string | `author` | `--author`, `[model] author` |
| `entry` | object | The model's function or class, see [`entry`](#entry) | `[model] entry`, `call`, `create`, ... |
| `sys_path` | string[] | Folders prepended to `sys.path`, relative to `resources/` (usually `["fmugen_model"]`) | `[model] sources` |
| `cwd` | string or null | Working directory of the model's code, relative to `resources/` (`fmugen_model/...`); null leaves it unchanged | `[model] cwd` |
| `globals` | string[] | `"module:NAME"` module-level variables saved with the FMU state | `[model] globals` |
| `setup` | object[] | Calls run at initialization, see [setup calls](#constants-and-setup-calls) | `[model] setup` |
| `constants` | object | Constructor (class) or call (function) arguments that aren't FMU variables | `[model.constants]` |
| `call_constants` | object | Constant arguments of the step method (classes only) | `[model.call_constants]` |
| `time_args` | object | Argument name → `time`, `step_size`, `end_time`, `start_time`, `stop_time` or `tolerance` | `[time]` |
| `time_epochs` | object | Argument name → ISO 8601 date-time: the argument gets a `datetime` (epoch + seconds) instead of a float | `[time] x = { epoch = ... }` |
| `experiment` | object | Optional `start_time`, `stop_time`, `step_size`, `tolerance`, `fixed_step` (needs `step_size`) | `[experiment]` |
| `type_definitions` | object | Enumeration type name → `{items: [...], enum: "module:Class" or null}`. FMI value 1 is the first item | `items` / `enum` of variables |
| `variables` | object[] | The FMU variables, in value-reference order, see [`variables`](#variables) | variable sections |
| `clocks` | object[] | FMI 3 clocks; `[]` for FMI 2 | `[clocks.*]` |
| `events` | object | FMI 3 `terminate` / `next_event_time` bindings; `{}` for FMI 2 | `[events]` |
| `has_event_mode` | bool | `clocks` or `events` is non-empty (FMI 3 `hasEventMode`) | derived |
| `can_get_and_set_state` | bool | `save_state` isn't `false` | derived |
| `save_state` | `true`, `false` or string[] | `true`: the whole model object is pickled (cloudpickle as a fallback); a list: only those attribute paths; `false`: no FMU state | `[model] save_state` |
| `discard_on` | string[] | `"module:Exception"` classes (or warnings) that make a step return `Discard` and roll back | `[model] discard_on` |
| `integrate` | null or object | `{method, substeps}`: the model returns derivatives and the engine integrates the states. `method` is `euler`, `rk4` or `scipy:<RK45, RK23, DOP853, Radau, BDF, LSODA>` | `[model] integrate`, `substeps` |
| `repeat` | int (only when set) | how many times the step runs per communication step, each with an equal part of it | `[model] substeps` without `integrate` |
| `annotations` | string or null | XML of `<Tool name="...">` elements, copied into `<Annotations>` | `[model] annotations` |
| `capture_output` | bool | Inside UniFMU, send the model's stdout/stderr to a file and forward it to the importer's log (UniFMU 0.14 crashes on more than ~4 KB of console output) | `--capture-output` |

The notes `config.normalize()` produces (e.g. "the FMU has no outputs") are printed by `fmugen build` and aren't written to the file.

---

## `entry`

| Key | Type | Meaning |
|---|---|---|
| `module` | string | Module imported from `sys_path` |
| `name` | string | Function or class in that module |
| `kind` | `function` or `class` | A factory counts as `class`. A class with `[model] kind = "function"` is `function`: it's constructed on every step |
| `create` | string or null | Classmethod that builds the object instead of the constructor, e.g. `from_pretrained` |
| `factory` | bool | The entry is a function called once with the parameters; `call` runs on what it returns |
| `call` | string or null | Method run on each step. null for functions and when `step` is false |
| `step` | bool | `false` (`[model] call = false`): `doStep` runs no model code, only clocks do |
| `call_on_class` | bool | Call `call` on the class, not on the object (a method written without `self`) |
| `init_call` | bool | Call the model once on leaving initialization mode, so outputs are valid at the start time |
| `terminate` | string or null | Method called on `fmiTerminate` |

---

## Constants and setup calls

A constant (in `constants`, `call_constants`, and setup `args` / `kwargs`) is one of:

| Form | Meaning |
|---|---|
| a JSON literal | passed as is |
| `{"python": "<source>"}` | a Python literal JSON can't hold (tuple, complex, ...), evaluated with `ast.literal_eval` |
| `{"ref": "pkg.mod.NAME"}` | an importable object, resolved when the FMU initializes |
| `{"call": "module:fn(...)"}` | the result of a call made when the FMU initializes |

Keys `"0"`, `"1"`, ... are positional arguments.

`setup` entries are `{"call": "module:function" or "method", "args": [...], "kwargs": {...}}`. They run when the FMU initializes: `module:function` calls before the model object is built, bare method names (classes only) on the object after.

---

## `variables`

Variables are in value-reference order: structural parameters, parameters, calculated parameters, inputs, states, outputs, locals, then (FMI 3) `time`. In FMI 2 an array takes one value reference per element, so the next variable's reference skips ahead; in FMI 3 each variable takes one.

| Key | Present | Meaning |
|---|---|---|
| `name` | always | FMU name. Outputs and locals may be dotted (`zone.T`) |
| `causality` | always | `structuralParameter`, `parameter`, `calculatedParameter`, `input`, `output`, `local` (also states), `independent` (FMI 3 `time` only) |
| `type` | always | FMI 2: `Real`, `Integer`, `Boolean`, `String`, `Enumeration`. FMI 3: `Float32`, `Float64`, `Int8`...`UInt64`, `Boolean`, `String`, `Binary`, `Enumeration`. Config types are mapped per version (`Float32` → `Real` in FMI 2, `Real` → `Float64` in FMI 3) |
| `variability` | always | parameters: `fixed`, `tunable`; inputs: `discrete`, `continuous`; outputs and locals: `constant`, `discrete`, `continuous` |
| `valueReference` | always | See above |
| `start` | values passed in, and results with a start | Scalar, or a list for arrays. Enumerations: the item's 1-based index |
| `initial` | all but inputs | `exact`, `approx` or `calculated` |
| `unit`, `description`, `quantity` | optional | strings |
| `min`, `max`, `nominal` | optional | numbers |
| `extra` | optional | More XML attributes, string → string |
| `dimensions` | arrays | Sizes; in FMI 3 also names of structural parameters |
| `numpy` | arrays, optional | `true`: pass a numpy array instead of a list |
| `convert` | optional | `"module:function"` applied before the model gets the value, or `"pint"` (a quantity in `unit`) |
| `clocks` | FMI 3, optional | One-element list: the clock the variable belongs to |
| `declared_type` | Enumerations | Key of `type_definitions` |
| `depends_on` | outputs, optional | Input names the output depends on (`<ModelStructure>`) |
| `attr` | tunable class parameters | Attribute set when the parameter changes after construction |
| `to`, `from`, `next`, `der` | see [Bindings](#bindings) | |

The FMI 3 `time` variable is `{"name": "time", "causality": "independent", "variability": "continuous", "type": "Float64", "description": "Simulation time", "valueReference": N}`.

---

## Bindings

A binding is `{"kind": ..., "name": ...}` and says where a value goes in, or comes from, your code.

**`to`**: structural parameters, parameters, inputs and states (values passed to the model).

| `kind` | `name` | The value is |
|---|---|---|
| `arg` | argument name | a keyword argument of the step call (or of the clock's call, for a clocked input) |
| `pos` | `"0"`, `"1"`, ... | a positional argument of the step call |
| `init` | argument name | a keyword argument of the constructor (classes) |
| `initpos` | `"0"`, `"1"`, ... | a positional argument of the constructor (classes) |
| `attr` | attribute path | set as an attribute of the model object (classes); a part that isn't an attribute is a key or an index |
| `call` | method name | passed to a setter method, `obj.method(value)` (classes) |

A `name` may pick an item: `arg:pair[0]`, `arg:params[key]` or `arg:u.temp`. The argument then gets a list, dict or object that collects every variable bound to it.

**`from`**: outputs, locals and calculated parameters. **`next`** (states): the state's next value. **`der`** (states with `integrate`): its derivative.

| `kind` | `name` | The value is read from |
|---|---|---|
| `return` | null | the whole return value |
| `return` | key, index or path | part of the return value (`return:T`, `return:0`, `return:zone.T`) |
| `attr` | attribute path | an attribute of the model object (classes) |
| `get` | method path | what that method of the model object returns, called without arguments (classes) |

Function models only allow `arg`, `pos` and `return`.

---

## `clocks` and `events` (FMI 3)

Clocks get value references after all variables.

| Key | Present | Meaning |
|---|---|---|
| `name` | always | |
| `causality` | always | `input` (ticked by the importer) or `output` (ticked by the model) |
| `interval_variability` | always | `constant`, `fixed`, `tunable`, `changing`, `countdown`, `triggered`. Output clocks are always `triggered` |
| `valueReference` | always | |
| `description` | optional | |
| `interval`, `shift` | `constant` / `fixed` / `tunable` | seconds |
| `interval_from` | `changing` / `countdown` | `from`-style binding giving the next interval |
| `call` | input clocks, optional | Function or method run when the clock ticks |
| `time_args` | with `call` | Which `time_args` the call accepts |
| `from` | output clocks | `from`-style binding that is true when the clock ticks |

`events` may hold `terminate` (true asks the importer to stop) and `next_event_time` (the next time event), each a `from`-style binding.

---

## Composite FMUs

A composite FMU (`[composite]`) has one full single-model file per part, in `resources/parts/<part>/interface.json`, plus a top-level file with these keys only:

| Key | Meaning |
|---|---|
| `spec_version`, `fmi_version`, `model_name`, `description`, `author` | As above |
| `composite` | `true` |
| `parts` | `[{"name": part, "dir": "parts/<part>"}]` |
| `connections` | `[{"from": [part, vr], "to": [part, vr]}]`, value references inside each part |
| `variables` | The parts' variables not fed by a connection, named `<part>.<name>`, with `part` and `local` (the reference inside the part) added. `declared_type`, `depends_on`, `clocks` and named `dimensions` get the `<part>.` prefix too |
| `type_definitions` | The parts' types, keys prefixed `<part>.` |
| `experiment` | `[composite]` experiment; a part's fixed step size is taken if none is set |
| `clocks` | The parts' clocks, `<part>.<clock>`, with `part` and `local` |
| `events` | always `{}` |
| `has_event_mode`, `can_get_and_set_state`, `capture_output` | Combined from the parts (any, all, any) |

---

## Versioning

`spec_version` is `SPEC_VERSION` in `templates/fmugen_runtime/_constants.py`. The engine refuses any other version when it loads the file (`check_spec_version`), so an FMU never runs with an engine that reads its file differently. The engine is copied into each FMU, so this only matters for tools reading the file. Change the version, and the schema's `spec_version` constant, when a key is renamed or removed or its meaning changes. Adding an optional key doesn't need a new version.

---

## Example

From a psychrometry model (FMI 2, a function model), shortened to one input and one output:

```json
{
  "spec_version": 1,
  "fmi_version": 2,
  "model_name": "compute_balances_simplified",
  "description": "Simplified mass and energy balance calculator for an air-based drying process.",
  "author": "",
  "entry": {
    "module": "psychrometry", "name": "compute_balances_simplified", "kind": "function",
    "create": null, "factory": false, "call": null, "step": true, "init_call": true, "terminate": null
  },
  "sys_path": ["fmugen_model"],
  "cwd": null,
  "globals": [], "setup": [], "constants": {}, "call_constants": {},
  "time_args": {}, "time_epochs": {}, "experiment": {}, "type_definitions": {},
  "variables": [
    {
      "name": "temp_1", "causality": "input", "type": "Real", "variability": "continuous",
      "start": 0.0, "to": {"kind": "arg", "name": "temp_1"}, "valueReference": 0
    },
    {
      "name": "Q_in", "causality": "output", "type": "Real", "variability": "continuous",
      "initial": "calculated", "from": {"kind": "return", "name": "Q_in"}, "valueReference": 1
    }
  ],
  "clocks": [], "events": {}, "has_event_mode": false,
  "can_get_and_set_state": true, "save_state": true, "discard_on": [],
  "integrate": null, "annotations": null, "capture_output": false
}
```

Each step, the engine calls `compute_balances_simplified(temp_1=<input>)` and reads `Q_in` from the returned dict.
