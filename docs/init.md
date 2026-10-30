# `fmugen init`: what it infers, and editing its config

`fmugen init` reads your model and writes a first `fmugen.toml`. That file is a draft. `init` reads the code but, unless you pass `--probe`, it never calls the model, so some things are guesses, and the comments in the file tell you which ones. This page explains what `init` infers and how, then walks through the file section by section and shows how to turn the draft into a config you can build.

- [cli.md](cli.md#fmugen-init) lists every `init` option.
- [config.md](config.md) is the reference for every key.
- [models.md](models.md) has a worked example for each kind of model.

Contents:

1. [The workflow](#1-the-workflow)
2. [How `init` infers the config](#2-how-init-infers-the-config)
3. [A first look: a function](#3-a-first-look-a-function)
4. [A class with state: simple-pid](#4-a-class-with-state-simple-pid)
5. [Section by section](#5-section-by-section)
6. [The comments `init` writes, and what to do about each](#6-the-comments-init-writes-and-what-to-do-about-each)
7. [Editing checklist](#7-editing-checklist)
8. [Common edits](#8-common-edits)
9. [FAQ](#9-faq)

---

## 1. The workflow

```
model ──► fmugen init ──► fmugen.toml ──► review and edit ──► fmugen build ──► model.fmu
```

```bash
fmugen init simple_pid:PID -o pid/fmugen.toml
```

```bash
fmugen build pid/fmugen.toml -o out/pid.fmu
```

Rules that hold everywhere in the file:

- **You never change the model's Python.** Everything the FMU needs to know goes in the config.
- **Paths are relative to the folder that holds `fmugen.toml`.**
- **Unknown keys are errors.** A typo such as `strat = 1.0` stops the build with a message listing the allowed keys, so mistakes don't go unnoticed.
- **One variable per line:** `name = { key = value, ... }`. The name on the left is the variable name the FMU shows to the importer.
- **Lines starting with `#` are comments.** `init` writes many commented-out lines for you to *uncomment* (delete the `# `) when they apply.

Use `-o -` to print the config without writing it, and `--force` to overwrite an existing one. Once you have edited a config by hand, rerunning `init` with `--force` throws your edits away, so keep a copy, or use `-o -` and copy the parts you need.

---

## 2. How `init` infers the config

`init` imports the module, picks the entry, reads the signatures and **the source code**, and writes a commented config. **It does not call the model**: no object is built, no setup call or step is run, so models that need a device, a network or a licence can be configured anywhere. (Importing the module runs its top-level code, as any import does.) `fmugen build model.py` runs the same inference in memory.

Reading the code is enough for most plain-Python models. What it can't know, the config says in comments ([§6](#6-the-comments-init-writes-and-what-to-do-about-each)).

### From the code

| | Inferred from |
|---|---|
| **Entry** | The name after `:`, else the only public function or class defined in the module |
| **Step method** (classes) | `step`, `do_step`, `update`, `__call__`, the method the `--start` names fit, or the only public method. When still unclear, `init` asks in a terminal; otherwise it picks the likeliest by name (`predict`, `forecast`, `simulate`, `solve`, `run`, …; `fit`, `reset`, `save` last) and warns with the candidates |
| **Inputs (functions) / parameters (classes)** | Arguments with a `bool`/`int`/`float`/`str` default, or with no default |
| **Start values and types** | The default value. With no default: a default the docstring gives (`Defaults to 0.5`), a typical value for a docstring unit (`293.15` for `[K]`), a value the code's own checks accept (`if n < 3: raise` gives `10`), the first member of an annotated Enum, or else a made-up `1.0` (the Integer `1` for names like `n_layers`, `hidden_size`). Each is commented so you can check it |
| **Time arguments** | Arguments named `dt`, `h`, `step_size`, `stepSize`, `time_step` (step size); `t`, `time`, `currentTime`, `t_now` (time at the start of the step); `until`, `end_time`, `t_end`, `t_next` (time at its end). Case and underscores don't matter. An `h` that is a count (annotated `int`, or defaulting to an integer or `None`) is left as an argument. A `--start` for one of them makes it a fixed input instead |
| **Date-time arguments** | Annotated `datetime`/`date`, or named `when`, `date`, `dateandtime`, …: [`[time]` with an `epoch`](config.md#time) |
| **Outputs from the return value** | Dict keys (also a dict built in a variable), tuple positions, a single value (`y`), NamedTuple/dataclass fields, or the return annotation |
| **Outputs and locals of classes** | Public attributes the step method assigns, following the methods it calls on `self` (also assigned in `__init__`: local; else output), and annotated properties |
| **Output types** | Annotations and what the code shows (`True`, comparisons, strings, arithmetic); everything else is assumed Real, with a warning |
| **States** | An input `x_prev` (or `x_old`, `x_last`, `prev_x`, `last_x`, `old_x`) whose next value is returned or stored as `x_next`, `next_x`, `x_new`, `new_x` or `x` |
| **Units** | The unit a numpydoc docstring states in square brackets (`rho : float` / `Density, [kg/m^3]`), with a comment. `[-]` and units fmugen doesn't recognise are left out |
| **Sources and `cwd`** | Local modules under the config's folder that the model imports; `cwd = "."` if the code opens a file by a relative path |
| **Requirements** | The installed packages the model imports, pinned to their installed versions |
| **Constants** | Non-FMI defaults, written as commented-out examples |
| **Positional-only arguments** | Bound with [`to = "pos:N"`](models.md#positional-only-arguments) |
| **Arrays and Binary** (`--fmi 3`) | List, tuple and numpy defaults become `dimensions` (`numpy = true` for numpy inputs); `bytes` becomes Binary; numpy `float32` becomes Float32 |

**Usage examples.** `init` also reads the `>>>` examples in the model's docstrings, the Python code blocks of its README (from the installed package's metadata, or a `README.md` next to a model file), and the scripts in an `examples` folder. The first example that builds the class (or calls the function) gives the options you didn't pass: the step method, the classmethod that builds the object (`--create`), the calls between building it and the step (`--setup`), and the values of arguments with no default (`--start`). Nothing in an example is run, except computed values (`np.linspace(0, 1, 4)`) with `--probe`. Each value is commented `start from the example in README.md: check it`. An argument no example of the model shows takes the value the package's examples pass to an argument of the same name elsewhere. Options you give always win.

**Where `--start` and `--const` values go.** A name that appears in both the constructor and the step method goes to the one without a default; on a tie, to the step method. Names neither declares are passed through `**kwargs`, if one accepts it. For a constructor that only declares `(*args, **kwargs)` (`control.StateSpace`) they are passed by position (`to = "pos:N"`), in the order given. If the constructor and the step method both have an argument `NAME`, the step method's variable is named `<method>_NAME` (`decay_units`), bound with `to = "arg:NAME"`. A variable named `time` becomes `model_time`, bound with `to` / `from`: FMI 3 reserves the name, and FMI 2 importers name the simulation time `time` in their results. `ARG.FIELD` names go wherever `ARG` goes.

### Not inferred from the code

Use `--probe`, or write them by hand; the config says so in comments:

- **Array sizes.** With `--fmi 3`, array results appear as commented lines with `dimensions = [...]` to fill in.
- **Results built at runtime** (filled in loops, set by helper functions, returned from library calls) and **code without Python source** (C extensions, compiled models).
- **Unannotated properties**: commented lines, to uncomment if they are numbers (reading a property runs code).
- **`save_state`**: a commented `save_state = false` line to uncomment if the model holds a device, file, socket or ONNX session.
- **Factories whose class the code doesn't show**: the step method's inputs and outputs. The class is found from the return annotation, or from return statements that call a class, or a function that does; otherwise only by calling the factory.

Never inferred: descriptions, states with other names, clocks, events, setter-method inputs (`call:`).

### With `--probe`

`--probe` also calls the model once with the start values (setup calls, construction, one step) and adds what that shows: array sizes, exact types, results built at runtime, every readable numeric property, what the step created or changed, and whether the object can be pickled (`save_state`). Results the code shows but the probe didn't reach are kept, commented `from the code; not seen in the probe`. It also:

- **Finds states** the step returns under the input's own name (`integrate(level, inflow)` returning `{"level": ...}`), and **tunable parameters**: a parameter kept as an attribute is changed on a copy of the model between two steps, and is made `variability = "tunable"` if the results change.
- **Builds objects** an argument is annotated with (`atoms: Atoms`) with no arguments, as a `call` constant with a comment. Replace it with `--const atoms=call:...` when the model needs a real one.
- **Warns about `--compile`** when the model's libraries load their own `.py` files by path (py-pde).

The probe only works if the model accepts the start values, so give realistic ones with `--start`. When a call fails, `init` reads the error and tries again before giving up. Every value it ends up with is commented `made up as …: check it`:

| When | It tries |
|---|---|
| The constructor fails | Classmethods that build an object with no arguments (`default()`, `create()`, `from_...()`) |
| The guessed step method fails or gives no outputs | The next two candidates |
| The step fails until `reset()` is called | `setup = ["reset"]` (gymnasium environments) |
| The step needs another method first ("call connect() first", or one that sets an attribute the step reads) | That method in `setup`, with its arguments made up |
| The step needs training (`fit`, `train`, `learn`) | Trains a throwaway object on made-up data to find the inputs and outputs; writes `# setup = ["fit(...)"]` commented out, for your data |
| A number is refused | Other numbers (4, 16, 100), an Integer, arrays of ones in the shape the docstring or the error names, then `(3,)`, `(1, 1)`, `(3, 3)`; torch tensors for a torch module; lists; one-element numpy arrays (`convert = "numpy:atleast_1d"`); text when the error asks for a string |
| The error names what the argument should be | A dict (with the keys a `KeyError` names), an object with the attributes an `AttributeError` names, a class of the package named in the error or docstring, an Enum's SI member, a dated pandas Series, an astropy `Quantity`, `numpy.sin` for a function, a jax `PRNGKey(0)`, a value the error lists (`Supported sampling rates: [8000, 16000]`), and a few domain-specific objects (a water molecule for atoms and positions, an aluminium crystal for a pymatgen `Structure`) |
| Several optional arguments are needed (`Reynolds(V, D, rho=None, mu=None, nu=None)`) | `None`-default arguments made up as `1.0`, one at a time, then two together, and those the error names |
| An input of an object with gymnasium spaces | The start of a `Discrete` space, the middle of a `Box` |
| The step returns nothing for a constant input | 8 values going up and down |

If the probe call still fails, `init` writes the config with what the code shows and says why in a comment.

---

## 3. A first look: a function

`fluids.friction.Colebrook(Re, eD)`, from the [fluids](https://pypi.org/project/fluids/) package, takes the Reynolds number and the pipe's relative roughness and returns the Darcy friction factor:

```bash
fmugen init fluids.friction:Colebrook --fmi 3 --start Re=1e5 --start eD=1e-4
```

```toml
# fmugen.toml: how this Python model becomes an FMU. Reference: docs/config.md
# Generated by `fmugen init`; check the start values, add units, and remove what you don't need.

[model]
entry = "fluids.friction:Colebrook"
fmi_version = 3

# Arguments whose code default is not an FMI value; the default is used.
# To override one, uncomment the line and the table header.
# [model.constants]
# tol = { python = "None" }

[inputs]
Re = { start = 100000.0 }  # start from --start
eD = { start = 0.0001 }  # start from --start

[outputs]
y = { from = "return" }

# FMI 3 clocks run model code on events; they can't be inferred. Example (docs/models.md#clocks-fmi-3):
# [clocks.sample]
# ...
```

What each part means:

| Line | Meaning |
|---|---|
| `entry = "fluids.friction:Colebrook"` | What to call: `module:name`, or `path/file.py:name` for a local file. |
| `fmi_version = 3` | Written because of `--fmi 3`. Without it, FMI 2. |
| The commented `[model.constants]` block | `Colebrook` also takes `tol=None`. `None` isn't an FMI value, so it isn't an input, and the model uses its default. |
| `[inputs]` | Function arguments. The FMU passes them on every step. |
| `start = 100000.0` | The value before the importer sets one. `--start Re=1e5` is a float, so the variable is Real. |
| `[outputs] y = { from = "return" }` | The function returns a single value, so `init` names it `y`. |
| The commented `[clocks.sample]` block | An FMI 3 feature that can't be inferred. Delete it if you don't need clocks. |

On every `doStep`, the FMU calls `Colebrook(Re=..., eD=...)` and stores the result in `y`.

`init` also writes `[model] requirements`: the installed packages the model needs, pinned to their installed versions (left out of the listing above). They go into the FMU's `requirements.txt` and `fmugen build --vendor`; check them.

To finish it, rename the output and add units:

```toml
[model]
entry = "fluids.friction:Colebrook"
requirements = ["fluids==1.3.1"]
fmi_version = 3

[inputs]
Re = { start = 100000.0, unit = "1", min = 0.0 }
eD = { start = 0.0001, unit = "1", min = 0.0 }

[outputs]
fd = { from = "return", unit = "1", description = "Darcy friction factor" }
```

The name on the left is free (`fd` instead of `y`). What ties it to the model is `from = "return"`.

---

## 4. A class with state: simple-pid

```bash
fmugen init simple_pid:PID -o -
```

`init` prints warnings, then the config:

```
warning: setpoint: typed Integer because the default is an int; write a float start if it is Real
warning: input_: no default in the code: the start value is made up, check it
warning: components, tunings, output_limits: properties of unknown type left commented out: uncomment the numbers
warning: y: type not shown by the code, assumed a single Real; it may be an array or a tuple, which fails the step: run init --probe, or set type = ... / dimensions = [...]
```

```toml
[model]
entry = "simple_pid:PID"
# save_state = false  # not checked without --probe: set it if the model holds a device, file, socket or ONNX session

# Arguments whose code default is not an FMI value; the default is used.
# To override one, uncomment the line and the table header.
# [model.constants]
# output_limits = { python = "(None, None)" }
# error_map = { python = "None" }
# time_fn = { python = "None" }

[time]
dt = "step_size"

[parameters]
Kp = { start = 1.0 }
Ki = { start = 0.0 }
Kd = { start = 0.0 }
setpoint = { start = 0 }  # Integer because the default is an int; write a float start for Real
sample_time = { start = 0.01 }  # kept as an attribute: add variability = "tunable" if the step reads it
auto_mode = { start = true }
proportional_on_measurement = { start = false }  # kept as an attribute: add variability = "tunable" if the step reads it
differential_on_measurement = { start = true }  # kept as an attribute: add variability = "tunable" if the step reads it
starting_output = { start = 0.0 }

[inputs]
input_ = { start = 1.0 }  # no default in the code: check the start value

[outputs]
y = { from = "return" }
# components = {}  # a property of the object (type unknown): uncomment if it is a number, or run init --probe
# tunings = {}  # a property of the object (type unknown): uncomment if it is a number, or run init --probe
# output_limits = {}  # a property of the object (type unknown): uncomment if it is a number, or run init --probe
```

How this runs inside the FMU:

1. **At initialization:** `PID(Kp=..., Ki=..., ..., setpoint=...)` is built from `[parameters]`. For a class, parameters are constructor arguments.
2. **On every `doStep(t, h)`:** `pid(input_, dt=h)` is called. `PID` is callable, so the step method is `__call__` and `init` doesn't need to write `call`. `[time] dt = "step_size"` passes the communication step size as `dt`.
3. The return value goes to `y`.

Going through the warnings:

- **`setpoint = { start = 0 }` is Integer.** A PID setpoint is a real number, so write `start = 0.0`.
- **`input_` has no default**, so `1.0` is made up (not `0.0`, which models often refuse or divide by). It's fine here. For models whose values must be specific (`Re` in a range, a matrix matching the constructor's sizes), give a realistic `--start`; with `--probe`, `init` also tries other values when `1.0` fails ([§2](#with---probe)).
- **`y` is assumed Real.** Correct for a PID. Rename it to `output` if you like.
- **Properties left commented out:** `tunings` and `output_limits` are tuples, not numbers, so leave them commented. `components` is a tuple too.
- **`sample_time = 0.01`:** simple-pid then skips calls closer together than 0.01 s. In an FMU that is usually unwanted: uncomment `[model.constants]` and set `sample_time = { python = "None" }`, and delete `sample_time` from `[parameters]`. A name can't be both a variable and a constant.
- **Tunable gains:** to let the importer change `Kp` during the simulation, add `variability = "tunable"`. The FMU then writes the new value to the attribute `pid.Kp`.

The edited config:

```toml
[model]
entry = "simple_pid:PID"
requirements = ["simple-pid==2.0.1"]
description = "PID controller"

[model.constants]
output_limits = { python = "(-100.0, 100.0)" }
sample_time = { python = "None" }

[time]
dt = "step_size"

[parameters]
Kp       = { start = 2.0, variability = "tunable" }
Ki       = { start = 0.5, variability = "tunable" }
Kd       = { start = 0.0, variability = "tunable" }
setpoint = { start = 1.0, variability = "tunable" }

[inputs]
measurement = { start = 0.0, to = "arg:input_" }

[outputs]
output = { from = "return" }
```

Notice `measurement = { ..., to = "arg:input_" }`: the FMU variable is called `measurement`, but the model gets it as its argument `input_`. `to` (going into the model) and `from` (coming out of it) let you name variables however you like.

---

## 5. Section by section

Not every config has every section; `init` writes only those that apply. The order of the sections doesn't matter.

### `[model]`: what to call

| Key | What `init` writes | When to edit |
|---|---|---|
| `entry` | The function or class. | Rarely. |
| `call` | The step method, if it's not `__call__`: `step`, `do_step`, `update`, the method the `--start` names fit, the only public method, `forward` for a torch module returned by a factory, or `--call`. When it's still unclear, `init` asks which one in a terminal; otherwise it picks the likeliest (`predict`, `forecast`, `simulate`, `solve`, `run`, …; `fit`, `reset`, `save` last) and warns with the candidates. | If `init` picked the wrong method. `pykalman`'s `KalmanFilter` needs `--call filter_update`. |
| `fmi_version` | `3` with `--fmi 3`. | To switch versions. `fmugen build --fmi` also overrides it. |
| `requirements` | The installed packages the model needs, pinned to the installed versions. | When a version should be looser, or a package is missing (`init` says so in a comment). |
| `sources` | Local modules the model imports, under the config's folder. | Add data files (CSV, JSON, weights) the model reads. |
| `cwd` | `"."` if the code opens a file by a relative path next to the config. | When the model opens `data/x.csv` and fails to find it. |
| `setup` | `--setup` calls. | Libraries that need a call before use (`psychrolib:SetUnitSystem(psychrolib.SI)`). |
| `save_state` | Commented out without `--probe`; set with `--probe` if pickling the object failed. | `false` if the model holds a device, file, socket or session. |
| `globals` | Module variables the model changes. | Rarely; `init` finds them. |
| `name`, `description`, `author` | Not written (defaults: the entry name and its docstring). | Add them: importers display them. |

### `[model.constants]` and `[model.call_constants]`: fixed arguments

Arguments that aren't FMU variables: `None`, tuples, objects, classes, arrays you don't want to expose. `constants` go to the constructor (classes) or the function; `call_constants` go to the step method.

`init` writes every argument whose default is not a plain number, bool or string here, **commented out**, because leaving it out already uses the code's default. Uncomment a line only to pass a different value. If the `[model.constants]` header is commented out too, uncomment it as well.

Value forms:

```toml
[model.constants]
n_points   = 50                                          # a plain TOML value
limits     = { python = "(-100.0, 100.0)" }              # a Python literal
solver     = { ref = "scipy.integrate:RK45" }            # an imported object (class, function)
weights    = { call = "numpy:eye(3)" }                   # the result of a call, made at initialization
```

Lines marked `# computed when the FMU initializes` are `call =` constants `init` wrote for you, either from `--const NAME=call:...` or for arguments that need an array to work (pykalman's `filtered_state_mean = { call = "numpy:zeros(1)" }`). They are active, not commented.

### `[experiment]`: default run settings

`init` doesn't write it. Add it when the model has a natural time step:

```toml
[experiment]
step_size  = 3600.0
stop_time  = 172800.0
fixed_step = true      # the model only works at step_size; any other doStep is an error
```

### `[time]`: passing time and step size

Arguments that receive the simulation time instead of being inputs. `init` recognizes them by name (`dt`, `t`, `end_time`, …; the full list is in [§2](#from-the-code)).

```toml
[time]
dt = "step_size"     # h, the communication step
t  = "time"          # t, the start of the step
# t_end = "end_time" # t + h
```

For a model that takes a date (`pvlib`, weather lookups), `init` writes an `epoch`: FMU time 0 is that date.

```toml
[time]
when = { source = "end_time", epoch = "2026-10-08T00:00:00" }
```

`init` uses today's date at midnight. **Set the epoch you need**, with its time zone (`+00:00`) if the model cares. Details: [config.md](config.md#time).

### Variables: `[parameters]`, `[inputs]`, `[states]`, `[outputs]`, `[locals]`, `[calculated_parameters]`

| Section | Who sets it | When | Default binding |
|---|---|---|---|
| `[parameters]` | importer | before initialization (or later, if `tunable`) | class: constructor argument; function: argument |
| `[inputs]` | importer | before every step | step argument |
| `[states]` | the model | fed back from one step to the next | step argument + `next` |
| `[outputs]` | the model | after every step | class: attribute; function: return value |
| `[locals]` | the model | after every step | like outputs; visible but not meant for connecting |
| `[calculated_parameters]` | the model | after initialization | attribute of the constructed object |

For a function, `init` makes every argument an input. For a class, constructor arguments become parameters and step arguments inputs. Attributes the step method assigns become outputs, or locals if `__init__` also assigns them.

The keys you will touch most:

| Key | Example | Meaning |
|---|---|---|
| `start` | `start = 20.0` | Start value; also sets the type (`20` → Integer, `20.0` → Real, `true` → Boolean, `"x"` → String). |
| `type` | `type = "Real"` | Set the type explicitly. |
| `unit` | `unit = "degC"` | Informational. `init` only adds the units a numpydoc docstring states. |
| `description` | `description = "..."` | Shown by importers. |
| `min`, `max` | `min = 0.0, max = 1.0` | Declared range, not enforced. |
| `variability` | `variability = "tunable"` | Parameters that may change during the simulation. |
| `to` | `to = "arg:input_"` | Where an input or parameter goes in the model. |
| `from` | `from = "return:1"` | Where an output comes from. |
| `next` | `next = "attr:t_m_next"` | States only: where next step's value comes from. |
| `dimensions` | `dimensions = [3]` | An array. FMI 2 shows it as `x[1]`, `x[2]`, `x[3]`. |

`from` forms you'll see in generated configs:

| `from` | Reads |
|---|---|
| `"return"` | The whole return value. |
| `"return:0"`, `"return:1"` | Items of a returned tuple (`init` names them `y0`, `y1`, …). |
| `"return:temperature"` | A key of a returned dict, or a field of a returned object. |
| `"return:zone.T"` | A dotted path. |
| `"attr:x"` | An attribute of the object after the step (classes). Outputs named like the attribute don't need it: it's the default. |

### `[clocks.<name>]` and `[events]` (FMI 3)

Clocks run model code on events instead of on every step. They can't be inferred, so `init` always writes the commented example at the end of an FMI 3 config. **Delete it** if you don't need clocks. [models.md](models.md#clocks-fmi-3) shows how to use them.

---

## 6. The comments `init` writes, and what to do about each

Inline comments mark guesses and things to check. A comment never changes the build, so you can delete them once you've dealt with them.

| Comment | What it means | What to do |
|---|---|---|
| `# start from --start` | You passed this start value. | Nothing. |
| `# no default in the code: check the start value` | `1.0` is made up. | Put a realistic value. |
| `# no default in the code: its docstring's default, …` / `a typical value in K, …` / `… passes the code's checks on it, 1.0 doesn't` | Taken from the docstring or the code's checks, not made up. | Check it fits your case. |
| `# start from the example in …: check it` | Taken from the package's usage example. | Check it fits your case. |
| `# no default in the code: made up, as arrays of ones, shape (…) (the probe failed with 1.0)` | The probe failed with `1.0` and worked with this. | Put a realistic value (and size, for an array). |
| `# Integer because the default is an int; write a float start for Real` | The code's default is `0`, not `0.0`. | Write `0.0` if the quantity is real. |
| `# kept as an attribute: add variability = "tunable" if the step reads it` | The constructor stores this value on the object. | Add `variability = "tunable"` if changing it mid-simulation should have an effect. |
| `# start from --start, passed by position to *args: check the order` | The constructor only takes `*args`; your `--start` order is the argument order. | Check it matches the model's documentation. |
| `# start from --start (passed through **kwargs)` | The name isn't in the signature; it's passed as a keyword. | Check the model really accepts it. |
| `# a property of the object (type unknown): uncomment if it is a number, or run init --probe` | A property without a type annotation. Reading it runs code. | Uncomment it if it returns a number, or rerun with `--probe`. |
| `# a property of the object` | An annotated or probed property. | Delete it if you don't need it. |
| `# from the code; not seen in the probe` | The code shows this result, but the `--probe` call didn't produce it (e.g. a branch not taken). | Keep it if the model can reach that code; it fails the step if it isn't there. |
| `# set by method()` | An attribute set by another method, not by the step. | Keep it only if that method runs. |
| `# computed when the FMU initializes` | A `call =` constant. | Check the call is what you want. |
| `# an array: set its dimensions and uncomment, or run init --probe` | An array result whose size the code doesn't show. | Fill in `dimensions = [...]` and uncomment, or use `--probe`. |
| `# an int in the probe; Real, since results often vary: type = "Integer" if it is a count` | The probe returned an int. | Add `type = "Integer"` if it's always an integer. |
| `# NAME: None in the probe, so its type is unknown; ...` | The probe returned `None`. | Set `type = ...` and uncomment, or delete it. |
| `# NAME: complex in the probe, and FMI has no complex type: ...` | FMI has no complex numbers. | Uncomment the `_real` / `_imag` lines that follow. |
| `# NAME: not in what the probe's call returned; ...` | The output wasn't in the result. | Usually delete it. |
| `# skipped 'NAME': not an FMI value or not a valid name` | An argument or result `init` couldn't represent. | Use a constant, or bind it by hand. |
| `# save_state = false  # not checked without --probe: ...` | `init` doesn't know if the object can be pickled. | Uncomment it if the model holds a device, file, socket or session. |
| `save_state = false  # the model object can't be pickled (...)` | `--probe` tried and failed. | Nothing; the FMU can't roll back. See [FAQ](#the-fmu-says-it-cant-save-its-state). |
| `# module-level state the model changes while it runs: ...` | A `globals` entry. | Nothing. |
| `probe call failed (...)` | The model raised an error with the start values. | Rerun `init --probe` with realistic `--start` values. |

The warnings `init` prints on the terminal say the same things, grouped by variable.

---

## 7. Editing checklist

Before `fmugen build`:

- [ ] **Entry and step method.** For a class, is `call` the method that advances the model by one step?
- [ ] **Start values.** Every variable `# no default in the code` has a value the model accepts.
- [ ] **Integer vs Real.** Every `# Integer because the default is an int` is fixed if the quantity is real.
- [ ] **States.** If the model takes its previous result as an argument under a name `init` doesn't recognise (`x_prev`, `x_old`, `last_x`, … — see [models.md](models.md#state-fed-back-between-steps)), add a `[states]` entry by hand (see [below](#add-a-state-init-didnt-detect)).
- [ ] **Outputs.** Delete the ones you don't need, rename `y`, `y0`, `y1` to meaningful names, and resolve every commented line (uncomment or delete).
- [ ] **Units and descriptions.** Add them; `init` only adds units a docstring states, and never descriptions.
- [ ] **Tunable parameters.** Add `variability = "tunable"` where it makes sense.
- [ ] **Constants.** Uncomment only those you want to change.
- [ ] **Requirements.** Check the pinned versions `init` wrote.
- [ ] **Data files.** Listed in `sources`, with `cwd = "."` if the model opens them by a relative path.
- [ ] **`[experiment]`.** Add `step_size` (and `fixed_step = true` if the model only works at one step).
- [ ] **FMI 3 clock example.** Deleted if unused.

---

## 8. Common edits

### Rename a variable

The left-hand name is the FMU name; keep the link to the model with `to` or `from`:

```toml
[inputs]
T_outdoor = { start = 10.0, to = "arg:t_out" }

[outputs]
heat_index = { from = "return" }
power      = { from = "attr:p_el" }
```

### Make a parameter changeable during the simulation

```toml
[parameters]
setpoint = { start = 21.0, variability = "tunable" }
```

For a class, the FMU writes the new value to `obj.setpoint`. If the attribute has another name, add `attr = "_setpoint"`.

### Add a state `init` didn't detect

A model that takes its own previous result (Madgwick's `updateIMU(q, gyr, acc)` returns the new `q`):

```toml
[states]
q = { dimensions = [4], start = [1.0, 0.0, 0.0, 0.0], next = "return" }
```

`init` only detects previous-value names such as `x_prev`, `x_old` or `last_x`, fed back from `x_next`, `x_new` or `x`.

### Expose a fixed argument as a parameter, or hide one

Move the line between `[model.constants]` and `[parameters]`. A name can be in only one place.

### Give the model a value FMI can't represent

```toml
[model.constants]
fluid = { python = "('Water', 'Air')" }
model = { call = "huggingface_sb3:load_from_hub(repo_id='sb3/demo-hf-CartPole-v1', filename='ppo-CartPole-v1.zip')" }
```

### Add an output from a returned dict

```toml
[outputs]
pressure = { from = "return:p", unit = "Pa" }
```

### Arrays

With FMI 3, arrays are real arrays; with FMI 2, `x[1]`, `x[2]`, … scalars. Both need a size:

```toml
[inputs]
coords = { dimensions = [3], start = [0.0, 0.0, 0.0], numpy = true }   # numpy: pass a numpy array, not a list
```

---

## 9. FAQ

### Do I have to keep the comments?

No. They are notes for you. Delete them when you've dealt with them.

### Should I always use `--probe`?

Use it when the model returns arrays, comes from a C extension or compiled code, builds results at run time, or may hold something that can't be pickled. It calls the model once with the start values, so give realistic ones with `--start`. Without `--probe`, the model is never called, which is what you want for models that need hardware, a network or a licence.

### `init` found no outputs. What now?

The results are built in a way the code doesn't show (loops, helper functions, C code). Rerun with `--probe`, or write `[outputs]` by hand with `from = "return..."` or `from = "attr:..."`.

### `init` picked the wrong step method.

When it wasn't sure, it said so (a terminal prompt, or a warning with the candidates). Rerun with `--call METHOD`, or edit `[model] call`.

### The class's constructor does all the work; there's no step method.

Rerun with `--kind function`: the class is built with the inputs on every step and outputs are read from the new object.

### The class is built with `from_pretrained` (or another classmethod).

`init` uses a classmethod when the constructor can't take your `--start`/`--const` names and exactly one public classmethod can, when a usage example builds it that way, or, with `--probe`, when the constructor fails and a no-argument classmethod works. Otherwise rerun with `--create from_pretrained`.

### The step fails until `reset()` (or another set-up) is called.

`init --probe` finds the usual cases ([§2](#with---probe)). Otherwise rerun with `--setup reset`, or the call the failure comment names. Training (`fit(...)` on your data) is left commented out for you to fill in.

### Why is my variable Integer?

Its start value is an `int` (`0`, `1`). Write `0.0`, or add `type = "Real"`.

### Can I rename `y`, `y0`, `y1`?

Yes. The name is free; `from` is what reads the value.

### What is the difference between outputs and locals?

Both are read after every step. Outputs are meant to be connected to other FMUs; locals are for inspecting internal values. `init` makes attributes that `__init__` also sets into locals. Move lines between the two sections freely.

### What is the difference between `[model.constants]` and `[parameters]`?

Parameters are FMU variables: the importer can see and set them. Constants are fixed in the config, and can hold any Python value (tuples, `None`, objects).

### A line in `[model.constants]` is commented out. Is it used?

The code's default is used. Uncomment it only to change the value.

### Should I use FMI 2 or FMI 3?

FMI 2 is supported by more tools. FMI 3 adds real arrays, more integer types, Binary, clocks and events. The same config builds either, unless it uses FMI 3-only keys; `fmugen build --fmi 3` switches without editing the file.

### The model needs a call before it can be used.

```toml
[model]
setup = ["psychrolib:SetUnitSystem(psychrolib.SI)"]
```

Or pass `--setup` to `init`.

### The model opens a file next to it.

```toml
[model]
sources = ["data"]
cwd = "."
```

### The FMU says it can't save its state.

The object can't be pickled (it holds a device, file or session), so `save_state = false`. The FMU still simulates; it just can't roll back (`fmi2GetFMUstate`). If only some attributes hold the state, list them: `save_state = ["_state"]`.

### The model only works at one time step.

```toml
[experiment]
step_size  = 3600.0
fixed_step = true
```

### I edited the config and rerun `init`. Where did my edits go?

`init --force` rewrites the file. Use `-o -` to print a new draft and copy the parts you need.

### The build fails, or the FMU fails at run time.

Error messages and their fixes (`unknown key(s)`, `requires FMI 3`, import errors, hangs) are in [troubleshooting.md](troubleshooting.md).

### Where's the full list of keys?

[config.md](config.md).
