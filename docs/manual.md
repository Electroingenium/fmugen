# fmugen manual

The reference for the `fmugen` command: installation, every option, recipes and troubleshooting. Every `fmugen.toml` key is in [config.md](config.md); worked examples for each kind of model are in [models.md](models.md); runtime FMI behaviour is in [fmi.md](fmi.md).

**Contents**

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Workflow](#3-workflow)
4. [Commands](#4-commands)
   - [`fmugen init`](#fmugen-init)
   - [`fmugen build`](#fmugen-build)
   - [Model targets](#model-targets)
   - [Errors and exit codes](#errors-and-exit-codes)
5. [Configuration: `fmugen.toml`](#5-configuration-fmugentoml)
6. [Recipes](#6-recipes)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Overview

`fmugen` turns an unmodified Python model into a [UniFMU](https://github.com/INTO-CPS-Association/unifmu) FMU: FMI 2.0 or FMI 3.0, Co-Simulation.

- **The model** is any importable function or class: your own file or an installed package. It is never edited.
- **`fmugen.toml`** says how the model maps onto FMU variables: which arguments are inputs or parameters, where outputs come from, units, the experiment, and so on.
- **`fmugen init`** writes a first `fmugen.toml` by inspecting the model.
- **`fmugen build`** packages the model and its config into an `.fmu`.

---

## 2. Installation

Install fmugen into your model's virtual environment, the one that already has the model's packages:

fmugen is not on PyPI yet, so `pip install fmugen` doesn't work. Install it from a clone of this repository:

```bash
pip install path/to/2025_Inkindcontributions
```

(or `uv add path/to/2025_Inkindcontributions`). This also installs `protobuf==5.27.3` and `pyzmq`, which UniFMU's Python backend needs to run the FMU. Run `fmugen` from that environment: `init` and `build` import your model, and the FMU runs with the environment's Python.

You also need [UniFMU](https://github.com/INTO-CPS-Association/unifmu/releases) **0.14.0** (exactly this version) for `fmugen build`. fmugen runs `unifmu generate` to get each FMU's native binaries and Python backend. Put `unifmu` on `PATH`, or set the environment variable `FMUGEN_UNIFMU` to the executable. `fmugen init` works without it.

To simulate and validate FMUs as in the examples below, install [FMPy](https://github.com/CATIA-Systems/FMPy) (`pip install fmpy`).

To work on fmugen itself, clone the repository and run `uv sync`; then use `uv run fmugen …` and `uv run pytest` (`-m "not slow and not fmu"` for the in-process tests only, about a minute).

## 3. Workflow

```
your model ──► fmugen init ──► fmugen.toml ──► (review & edit) ──► fmugen build ──► model.fmu
```

1. Infer a config:

   ```bash
   fmugen init path/to/model.py
   ```

2. Review `path/to/fmugen.toml`. Check the start values, add units, and remove what you don't need.
3. Build:

   ```bash
   fmugen build path/to/fmugen.toml -o out/model.fmu
   ```

4. Check the FMU:

   ```bash
   fmpy validate out/model.fmu
   ```

   ```bash
   fmpy simulate out/model.fmu --output-file out/result.csv
   ```

For a quick try, `fmugen build path/to/model.py -o out/model.fmu` skips step 2: the config is inferred in memory.

---

## 4. Commands

```
fmugen init  MODEL [options]
fmugen build MODEL -o OUTPUT [options]
```

### `fmugen init`

Imports the model, reads its signatures and **its source code**, and writes a commented `fmugen.toml`. **The model is not called**: no object is built and no step is run, so models that need a device, a network or a licence can be configured anywhere. Importing the module still runs its top-level code, as any import does. With `--probe`, `init` also calls the model once, to find what the code doesn't show.

```
fmugen init MODEL [-o OUTPUT] [--call METHOD] [--fmi {2,3}]
                  [--start NAME=VALUE ...] [--const NAME=VALUE ...] [--setup CALL ...] [--kind function] [--create CLASSMETHOD]
                  [--probe] [--convert NAME=module:function ...] [--force]
```

| Option | Default | Description |
|---|---|---|
| `MODEL` | required | A [model target](#model-targets): `model.py`, `model.py:Name` or `package.module:Name`. |
| `-o`, `--output PATH` | `fmugen.toml` next to the model (current directory for an installed module) | Where to write the config; missing folders are created. `-o -` prints it instead. A model file must be inside the config's folder. |
| `--call METHOD` | `step`, `do_step`, `update`, `__call__`, or the only public method | Classes: the method run on each step. |
| `--fmi {2,3}` | none (FMI 2) | Writes `fmi_version` into the config. With `3`, list/tuple/numpy values are inferred as arrays, `bytes` as Binary and numpy `float32` as Float32. |
| `--start NAME=VALUE` | none | Start value of an argument that becomes an FMU variable (and, with `--probe`, the value it is called with); repeatable. `VALUE` is a Python literal (`1e5`, `"Water"`, `[1.0, 0.0]`); a bare word is taken as a string. `true`/`false` are booleans. A tuple or dict value is split into one variable per item. Use it for arguments without a default, or whose default breaks the model. A value that can't be an FMU variable (a `call:` value, a dict for `**kwargs`, None) is refused: give it with `--const`. |
| `--const NAME=VALUE` | none | Fixed value of an argument, not an FMU variable: written to [`[model.constants]`](config.md#modelconstants-and-modelcall_constants) (or `call_constants` for the step method), for what only the model's user knows: a file, a model name, a mode (`--const model=yolo11n.pt`, `--const task=sentiment-analysis`). `VALUE` is read as for `--start`; `call:module:function(args)` is computed when the FMU initializes (`call:module:attr` without parentheses passes the object itself, e.g. a function), for objects: `--const "net=call:pandapower.networks:example_simple()"`. A file it names next to the config is copied into the FMU, as for `--start`. Repeatable; a name can't be given to both. `--start` and `--const` values keep the order they're given in: for a callable without a signature it's the order of its positional arguments (`--const model=… --const data=… --start q=…` for pinocchio's `rnea(model, data, q, …)`). |
| `--setup CALL` | none | A [setup call](config.md#model), written to `[model] setup`; with `--probe` it is also run before the probe. Repeatable. E.g. `"psychrolib:SetUnitSystem(psychrolib.SI)"`, `reset`. |
| `--kind function` | none | Treat a class whose constructor does the work as a function called on every step (writes `[model] kind`). |
| `--create CLASSMETHOD` | none | Classes built by a factory: the classmethod that creates the object, e.g. `from_pretrained`. Its arguments become parameters (writes `[model] create`). |
| `--probe` | off | Also **call the model once**: run the setup calls, build the object, call the step with the start values (retrying array inputs as numpy arrays, then tensors, and number inputs as one-element numpy arrays, `convert = "numpy:atleast_1d"`, for libraries that take only arrays), and try to pickle the object. Time arguments from `[time]` get `1.0` for `step_size` and `end_time`, `0.0` otherwise. Finds what the code doesn't show: array sizes, exact types, results built at runtime, unannotated properties, and `save_state`. Results the code shows but the probe didn't reach are added too. Without `--probe`, the model is never called. |
| `--convert NAME=module:function` | none | How an argument is passed to the model, e.g. `x=torch:tensor`, `x=numpy` for a numpy array, or `x=pint` for a pint quantity (needs a `unit` on the variable); writes `convert` / `numpy`. Repeatable. Not needed when the argument is annotated (`torch.Tensor`, `np.ndarray`). |
| `--force` | off | Overwrite an existing config. |
| `--probe-timeout SECONDS` | no limit | With `--probe`: stop (exit code 2) if the probe hasn't finished by then, e.g. a model waiting for a device. The model's code can't be interrupted, so `fmugen` exits. |
| `--debug` | off | Show the full traceback of an unexpected error instead of a one-line message. |

What `init` infers from the code, what only `--probe` finds, where `--start` values go, and what to check afterwards: [models.md](models.md#what-fmugen-init-infers). [init-tutorial.md](init-tutorial.md) walks through a generated config.

**Examples:**

```bash
fmugen init simple_pid:PID -o -
```

```bash
fmugen init simple_pid:PID -o pid/fmugen.toml
```

```bash
fmugen init fluids.friction:friction_factor --start Re=1e5 -o -
```

```bash
fmugen init psychrolib:GetHumRatioFromRelHum --setup "psychrolib:SetUnitSystem(psychrolib.SI)" --start TDryBulb=25.0 --start RelHum=0.5 --start Pressure=101325.0 -o -
```

```bash
fmugen init filterpy.kalman:KalmanFilter --call predict --fmi 3 --start dim_x=2 --start dim_z=1 -o -
```

```bash
fmugen init ultralytics:YOLO --call predict --fmi 3 --probe --const model=yolo11n.pt --start source=bus.jpg -o -
```

### `fmugen build`

Packages the model into an FMU. Before writing the FMU, it runs the packaged model once through the real adapter: initialization, one tick of each input clock (FMI 3), one step, and saving and restoring state. A model that fails there fails the build with the model's own error.

```
fmugen build MODEL -o OUTPUT [--fmi {2,3}] [--format {fmu,folder}]
                   [--name NAME] [--author AUTHOR] [--call METHOD] [--probe]
                   [--vendor [--platform TAG ...] [--python-version X.Y ...] | --compile {pyinstaller,nuitka}]
                   [--probe] [--capture-output] [--hf-weights | --no-hf-weights]
                   [--distributed [--endpoint IP] [--private-output PATH]]
```

| Option | Default | Description |
|---|---|---|
| `MODEL` | required | A `fmugen.toml`, a directory containing one, or a [model target](#model-targets). For a model target, the config is inferred in memory, as `init` would without options. |
| `-o`, `--output PATH` | required | Output path, used exactly as given. Parent directories are created. |
| `--fmi {2,3}` | `[model] fmi_version`, else `2` | FMI version to build. |
| `--format {fmu,folder}` | `fmu` | `fmu`: zipped `.fmu` archive. `folder`: unzipped UniFMU folder. |
| `--name NAME` | `[model] name`, else the entry's name | `modelName` in `modelDescription.xml`. |
| `--author AUTHOR` | `[model] author`, else empty | `author` in `modelDescription.xml`. |
| `--call METHOD` | as `init` | Only when `MODEL` is a model target with a class. |
| `--probe` | off | Only when `MODEL` is a model target: infer the config as `init --probe` would, calling the model once. `--probe-timeout` as in `init`. |
| `--debug` | off | Show the full traceback of an unexpected error instead of a one-line message. |
| `--vendor` | off | Put wheels of every requirement (`[model] requirements` and the backend's) into `resources/wheels/`. On its first run on a machine, the FMU installs them into a cached virtual environment, offline. The target needs Python. See [packaging.md](packaging.md#fmus-for-other-machines). |
| `--platform TAG` | this machine | With `--vendor`: also vendor wheels for this platform, e.g. `win_amd64`, `manylinux2014_x86_64`, `macosx_11_0_arm64`. Repeatable. |
| `--python-version X.Y` | this Python | With `--vendor`: vendor wheels for this Python version. Repeatable. |
| `--compile {pyinstaller,nuitka}` | off | Freeze the model, its packages and Python into an executable (`resources/dist/main/`). The FMU contains no source code and needs no Python, but only runs on the OS it was built on. Needs the `pyinstaller` or `nuitka` extra, e.g. `pip install "path/to/2025_Inkindcontributions[pyinstaller]"`. |
| `--hf-weights` | on with `--vendor` / `--compile` | Put the Hugging Face models the model loads into the FMU, which then loads them offline. See [packaging.md](packaging.md#hugging-face-models---hf-weights). |
| `--capture-output` | off | Send what the model prints (stdout and stderr, also from C code) to the importer's log, prefixed `[output]`, instead of the console. Use it for models that print more than a few KB (warnings, progress bars): UniFMU 0.14 crashes when the FMU's Python writes that much to the console, and the importer hangs. The importer may only show these messages with debug logging on. |
| `--distributed` | off | Make `OUTPUT` a proxy FMU (no Python, no model) and write the model to a separate private folder, which runs on another host or container and connects to the proxy over TCP. The proxy's port changes on every run and can't be set (UniFMU 0.14): it's in the importer's log, under `logUnifmuMessages`. See [packaging.md](packaging.md#distributed-fmus---distributed). |
| `--endpoint IP` | `127.0.0.1` | With `--distributed`: IP address of the host running the proxy FMU, written to the private folder's `endpoint.toml`. |
| `--private-output PATH` | `<OUTPUT stem>_private` next to `OUTPUT` | With `--distributed`: the private folder. |

On success it prints the FMU path, the FMI version, and the number of variables per causality (and clocks):

```
Built C:\…\out\pid.fmu (FMI 2.0; 9 parameters, 1 inputs, 1 outputs)
```

It may also print `note:` lines, e.g. when `[events]` is ignored for FMI 2.

`build` imports the model to check the config against its real signatures, but **does not run it**: no object is constructed and no step is made, so models that need hardware, a network or a licence build anywhere. Only `fmugen init --probe` (and `build model.py --probe`, which infers the config the same way) calls the model.

**Examples:**

With `pid/fmugen.toml` written by `fmugen init simple_pid:PID -o pid/fmugen.toml` (the `pid` folder must exist):

```bash
fmugen build pid -o out/pid.fmu
```

```bash
fmugen build pid -o out/pid_fmi3.fmu --fmi 3
```

```bash
fmugen build pid -o out/pid_folder --format folder
```

Straight from an installed model, with the config inferred in memory:

```bash
fmugen build fluids.core:Reynolds -o out/reynolds.fmu
```

For another machine with Python (offline install from the FMU):

```bash
fmugen build pid -o out/pid_vendored.fmu --vendor
```

For another machine without Python, with no source code in the FMU:

```bash
fmugen build pid -o out/pid_compiled.fmu --compile pyinstaller
```

### Model targets

| Form | Meaning |
|---|---|
| `path/model.py` | A file; the entry is its only public function or class. |
| `path/model.py:Name` | A file and the function or class `Name` in it. |
| `package.module:Name` | `Name` in an installed module, e.g. `simple_pid:PID`, `filterpy.kalman:KalmanFilter`. |

In `fmugen.toml` the same forms appear as `[model] entry`, with file paths relative to the config's folder.

### Errors and exit codes

| Exit code | Meaning |
|---|---|
| `0` | Success. |
| `1` | An unexpected error, e.g. the model's own code raising while it is imported: printed as `fmugen: error: <type>: <message>`. Add `--debug` for the traceback. |
| `2` | A usage or config error, printed as `fmugen: error: …`: an unknown key, a bad value, an import failure, an `init` probe failure or timeout, or an existing config without `--force`. |

### Environment variables

| Variable | Used by | Effect |
|---|---|---|
| `FMUGEN_UNIFMU` | `build` | Path of the UniFMU executable, instead of `unifmu` on `PATH` |
| `SOURCE_DATE_EPOCH` | `build` | Build time (Unix seconds) written into `modelDescription.xml` and the zip, for [reproducible builds](packaging.md#reproducible-builds) |
| `FMUGEN_ENV_DIR` | `--vendor` FMUs | Where the FMU's cached environments go ([details](packaging.md#--vendor)) |
| `XDG_CACHE_HOME` | `--vendor` FMUs, Linux | Base of the default cache folder |

Each error names the section and key, e.g. `[inputs] x: unknown key(s) ['strat']`. See [Troubleshooting](#7-troubleshooting) and the [validation rules](config.md#validation-rules).

---

## 5. Configuration: `fmugen.toml`

Every section and key, with defaults, value forms, validation rules and a complete example, is in [config.md](config.md). The sections at a glance:

```toml
[model]                     # what to call, and how
[model.constants]           # fixed non-FMI arguments: constructor (classes) / function
[model.call_constants]      # fixed non-FMI arguments: step method (classes)
[experiment]                # default experiment, fixed step
[time]                      # arguments that receive the FMU time / step size

[structural_parameters]     # FMI 3 ─┐
[parameters]                #        │
[calculated_parameters]     #        │ FMU variables,
[inputs]                    #        │ one inline table each:
[states]                    #        │ name = { key = value, ... }
[outputs]                   #        │
[locals]                    #       ─┘

[clocks.<name>]             # FMI 3: one table per clock
[events]                    # FMI 3: terminate / next event time
[composite]                 # instead of [model]: several configs in one FMU
```

---

## 6. Recipes

| I want to… | Use |
|---|---|
| Package a plain function | `fmugen init model.py` → `build` |
| Use a class with a step method | `--call METHOD` (or `[model] call`) |
| Give arguments without defaults sensible values | `init --start NAME=VALUE` / `start = …` |
| Pass a file, a model name or an object that isn't an FMU variable | `init --const NAME=VALUE` (`call:…` for an object) / `[model.constants]` |
| Call a library setup function first | `[model] setup = ["module:function(args)"]` |
| Reset a gym-style environment before stepping | `setup = ["reset"]` |
| Feed a result back into the next step | `[states] x_prev = { next = "return:x" }` |
| Pass the FMU step size or time | `[time] dt = "step_size"`, `t = "time"` |
| Change a parameter during the simulation | `variability = "tunable"` |
| Expose a value the constructor computes | `[calculated_parameters]` |
| Bind a C extension's positional arguments | `to = "pos:N"` |
| Set an input through a method | `to = "call:method"` |
| Wrap a class whose constructor does everything | `[model] kind = "function"` |
| Use a model from PyPI | install it in the environment, `entry = "pkg.module:Name"`, `requirements` |
| Use matrices/vectors | FMI 3 `dimensions`, `numpy = true` |
| Size arrays from a constructor argument | FMI 3 `[structural_parameters]` |
| Run code on events, not every step | FMI 3 `[clocks.<name>]`, `clocks = [...]`, `call = false` |
| Let the model stop the simulation | FMI 3 `[events] terminate` |
| Only allow one step size | `[experiment] fixed_step = true` |
| Build both FMI versions from one config | `build --fmi 2` / `build --fmi 3` |

---

## 7. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `cannot import …: No module named …` | The model's package isn't in the environment running fmugen. | `pip install <pkg>` in that environment. |
| `probe call failed (…)` in the config written by `init --probe` | The model raised an error at the start values (division by zero, `log(0)`, out of range…). | Rerun `init` with realistic `--start` values. |
| `… must be called first` / `has not been defined` (raised by the model) | The library needs setup. | `[model] setup` / `init --setup`. |
| `--start NAME: arrays and bytes need --fmi 3` | An array or Binary value in an FMI 2 build. | Use `--fmi 3`. |
| `--start names that are not arguments of the model` | A typo, or the argument isn't in the inspected signature. | Check the names. (For C extensions without a signature, `--start` values are passed by position instead.) |
| `the config defines no FMU variables` | Nothing could be inferred (e.g. `**kwargs` only, or an uninspectable signature). | Add `[inputs]` and `[outputs]`, or use `--start`, `--kind function`. |
| `requires FMI 3 (…)` | FMI 3-only feature in an FMI 2 build. | `fmi_version = 3` or `build --fmi 3`. |
| `cannot inspect the signature of …` (comment in the generated config) | A C extension without signature information. | Give the arguments with `--start`, in order (they are passed by position), or write variables by hand with `to = "pos:N"`. |
| `unknown key(s) [...]` | A typo in a key. | The message lists the allowed keys. |
| `duplicate variable or clock names` | The same name in two sections, or a variable named like a clock. | Rename one; bind it with `to`/`from`. |
| `save_state = false` or `save_state = [...]` written by `init`, with a comment | The object holds something that can't be pickled: an ONNX session, a file, a socket, a generator. | Fine for most uses. With a list, check that the attributes named as not saved don't change during a simulation (a session or device handle doesn't; a generator does): if they do, set `save_state = false`. |
| The FMU fails at runtime with `ModuleNotFoundError` | The environment the FMU was built in has changed or was deleted. | Reinstall the package there, or rebuild the FMU from the environment that has it. |
| The importer hangs; the console shows `panicked at ... zeromq-0.4.1/src/rep.rs:168:40: not yet implemented` | The model printed more than about 4 KB (warnings, progress bars); UniFMU 0.14 crashes on that. | Rebuild with `--capture-output`. |
| The simulation hangs or the importer reports an unknown command | The importer called an FMI function the UniFMU backend lacks. | See [fmi.md](fmi.md#unsupported-functions). |
