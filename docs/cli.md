# `fmugen` command reference

Every option of `fmugen init` and `fmugen build`. Installation and a first run are in [getting-started.md](getting-started.md); every `fmugen.toml` key is in [config.md](config.md).

- [`fmugen init`](#fmugen-init)
- [`fmugen build`](#fmugen-build)
- [Model targets](#model-targets)
- [Errors and exit codes](#errors-and-exit-codes)
- [Environment variables](#environment-variables)

```
fmugen init  MODEL [options]
fmugen build MODEL -o OUTPUT [options]
```
```
fmugen init  MODEL [options]
fmugen build MODEL -o OUTPUT [options]
```

## `fmugen init`

Imports the model, reads its signatures and **its source code**, and writes a commented `fmugen.toml`. **The model is not called**: no object is built and no step is run, so models that need a device, a network or a licence can be configured anywhere. Importing the module still runs its top-level code, as any import does. With `--probe`, `init` also calls the model once, to find what the code doesn't show.

```
fmugen init MODEL [-o OUTPUT] [--call METHOD] [--fmi {2,3}]
                  [--start NAME=VALUE ...] [--const NAME=VALUE ...] [--setup CALL ...] [--kind function] [--create CLASSMETHOD]
                  [--probe [--probe-timeout SECONDS]] [--convert NAME=module:function ...]
                  [--force] [--debug]
```

| Option | Default | Description |
|---|---|---|
| `MODEL` | required | A [model target](#model-targets): `model.py`, `model.py:Name` or `package.module:Name`. |
| `-o`, `--output PATH` | `fmugen.toml` next to the model (current directory for an installed module) | Where to write the config; missing folders are created. `-o -` prints it instead. A model file must be inside the config's folder. |
| `--call METHOD` | `step`, `do_step`, `update`, `__call__`, the method the `--start` names fit, or the only public method; when still unclear, `init` asks in a terminal (candidates numbered, best first; Enter takes the first), and otherwise picks the first and warns with the candidates | Classes: the method run on each step. |
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

What `init` infers from the code, what only `--probe` finds, where `--start` values go, and what to check afterwards: [init.md](init.md), which also walks through a generated config.

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

## `fmugen build`

Packages the model into an FMU. Before writing the FMU, it runs the packaged model once through the real adapter: initialization, one tick of each input clock (FMI 3), one step, and saving and restoring state. A model that fails there fails the build with the model's own error.

```
fmugen build MODEL -o OUTPUT [--fmi {2,3}] [--format {fmu,folder}]
                   [--name NAME] [--author AUTHOR] [--call METHOD] [--probe]
                   [--vendor [--platform TAG ...] [--python-version X.Y ...] | --compile {pyinstaller,nuitka}]
                   [--probe-timeout SECONDS] [--capture-output] [--hf-weights | --no-hf-weights]
                   [--distributed [--endpoint IP] [--private-output PATH]] [--debug]
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

## Model targets

| Form | Meaning |
|---|---|
| `path/model.py` | A file; the entry is its only public function or class. |
| `path/model.py:Name` | A file and the function or class `Name` in it. |
| `package.module:Name` | `Name` in an installed module, e.g. `simple_pid:PID`, `filterpy.kalman:KalmanFilter`. |

In `fmugen.toml` the same forms appear as `[model] entry`, with file paths relative to the config's folder.

## Errors and exit codes

| Exit code | Meaning |
|---|---|
| `0` | Success. |
| `1` | An unexpected error, e.g. the model's own code raising while it is imported: printed as `fmugen: error: <type>: <message>`. Add `--debug` for the traceback. |
| `2` | A usage or config error, printed as `fmugen: error: …`: an unknown key, a bad value, an import failure, an `init` probe failure or timeout, or an existing config without `--force`. |

## Environment variables

| Variable | Used by | Effect |
|---|---|---|
| `FMUGEN_UNIFMU` | `build` | Path of the UniFMU executable, instead of `unifmu` on `PATH` |
| `SOURCE_DATE_EPOCH` | `build` | Build time (Unix seconds) written into `modelDescription.xml` and the zip, for [reproducible builds](packaging.md#reproducible-builds) |
| `FMUGEN_ENV_DIR` | `--vendor` FMUs | Where the FMU's cached environments go ([details](packaging.md#--vendor)) |
| `XDG_CACHE_HOME` | `--vendor` FMUs, Linux | Base of the default cache folder |

Each error names the section and key, e.g. `[inputs] x: unknown key(s) ['strat']`. See [troubleshooting.md](troubleshooting.md) and the [validation rules](config.md#validation-rules).
