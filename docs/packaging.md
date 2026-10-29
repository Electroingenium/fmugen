# Packaging and runtime

- [CLI](#cli)
- [What `fmugen build` does](#what-fmugen-build-does)
- [Generated FMU layout](#generated-fmu-layout)
- [The FMU's Python environment](#the-fmus-python-environment)
- [FMUs for other machines](#fmus-for-other-machines)
- [Simulating](#simulating)
- [Project layout](#project-layout)
- [UniFMU version](#unifmu-version)

---

## CLI

Install fmugen into your model's virtual environment (fmugen is not on PyPI yet: `pip install path/to/2025_Inkindcontributions`) and run it from there. `fmugen build` also needs the UniFMU 0.14.0 CLI (see [UniFMU version](#unifmu-version)). To work on fmugen itself, `uv sync` installs it in editable mode with the test dependencies.

Every option of `fmugen init` and `fmugen build` is in [manual.md](manual.md#4-commands). This page explains what `build` does and the options for other machines (`--vendor`, `--compile`, `--hf-weights`).

---

## What `fmugen build` does

1. Reads and validates `fmugen.toml`, or infers it from a `.py` model.
2. Runs `unifmu generate python <tmp> fmi2` (or `fmi3`) to get the UniFMU boilerplate for the FMI version. Replaces `resources/model.py` with fmugen's adapter for that version (`src/fmugen/templates/model_fmi2.py` or `model_fmi3.py`), and adds the shared engine `resources/fmugen_runtime.py`.
3. Copies the entry file and `sources` into `resources/fmugen_model/`, keeping their paths. Writes `resources/requirements.txt`: the backend's packages and `[model] requirements`.
4. Imports the entry from the copied files, checks the config against it (argument names, function vs class), and writes `resources/interface.json`.
5. Writes `modelDescription.xml` and `launch.toml`, then zips the result (or copies the folder).

`build` imports the model but **never runs it**: no object is constructed, no setup call or step is made. Models that need hardware (TCLab), a network or a licence server build anywhere. `canGetAndSetFMUState` comes from `[model] save_state`, which `fmugen init --probe` sets. `fmugen build model.py`, without a `fmugen.toml`, infers the config like `init`: from the code, without calling the model, unless you pass `build --probe`.

---

## Generated FMU layout

```
model.fmu
├── binaries/                            # UniFMU native libraries (from `unifmu generate`)
├── modelDescription.xml                 # generated from fmugen.toml
└── resources/
    ├── main.py, backend.py, ...         # UniFMU Python backend (from `unifmu generate`)
    ├── schemas/                         # UniFMU protobuf messages
    ├── launch.toml                      # how UniFMU starts the backend per OS
    ├── requirements.txt                 # backend requirements + [model] requirements (a record)
    ├── model.py                         # fmugen's FMI 2 or FMI 3 adapter
    ├── fmugen_runtime.py                # fmugen's engine, shared by both adapters
    ├── interface.json                   # variables, value references, bindings
    ├── fmugen_model/                    # your files, unchanged
    │   └── ...
    └── fmugen_launch.py, wheels/        # only with --vendor
```

With `--compile`, `resources/` holds only `launch.toml` and `dist/main/`, the frozen executable that contains all of the above.

Your code is never rewritten.

`interface.json` is fmugen's normalised description of the model: the FMI version, the entry, `sys.path`, constants, time arguments, the experiment, type definitions, every variable with its binding, and the clocks and events. The engine reads it at runtime, and `modelDescription.xml` is generated from it. Every key is described in [interface.md](interface.md), with a JSON Schema.

### Reproducible builds

Rebuilding an unchanged model gives a byte-identical FMU:

- The GUID (FMI 2) / `instantiationToken` (FMI 3) is derived from the FMU's file contents, not random, so it only changes when the FMU does.
- Zip entries are sorted and dated 1980-01-01, or `SOURCE_DATE_EPOCH` when that environment variable is set.
- `generationDateAndTime` in `modelDescription.xml` is `SOURCE_DATE_EPOCH` when set, else the build time. Set it for byte-identical rebuilds.

---

## The FMU's Python environment

fmugen is installed in the model's virtual environment, and the FMU runs with that environment's Python. `launch.toml`, which UniFMU uses to start the backend, gets the interpreter running fmugen for the current OS, and `python3` (Linux, macOS) or `python` (Windows) for the others:

```toml
linux = ["python3", "main.py"]
macos = ["python3", "main.py"]
windows = ["C:\\path\\to\\model\\.venv\\Scripts\\python.exe", "main.py"]
```

That environment has everything the FMU needs:

- your model's packages, because you developed the model there
- `protobuf==5.27.3`, `pyzmq` and `cloudpickle`, the UniFMU backend's packages (cloudpickle for FMU state), installed as fmugen's dependencies. `--distributed` adds `toml`, `coloredlogs` and `colorama`.

The FMU therefore runs on the machine where it was built, as long as that environment exists and keeps those packages. `resources/requirements.txt` records what it must contain: the backend's packages and `[model] requirements`.

---

## FMUs for other machines

An FMU built by default points at your virtual environment, so it doesn't run anywhere else. Two options make FMUs to hand to someone else:

| | `--vendor` | `--compile pyinstaller` / `--compile nuitka` |
|---|---|---|
| **The target needs** | a Python interpreter (`python3` on Linux and macOS, `python` on Windows) | nothing |
| **Source code in the FMU** | yes, readable | no |
| **Runs on** | every platform and Python version whose wheels were vendored | only the OS (and CPU architecture) it was built on |
| **First run on a machine** | installs the wheels into a cached environment, offline | starts immediately |
| **Size** | the wheels | the interpreter and every package, typically 20–100 MB |

### `--vendor`

`fmugen build --vendor` puts wheels of the backend's packages and `[model] requirements`, with their dependencies, into `resources/wheels/`. `launch.toml` then runs `fmugen_launch.py` with the system Python:

```toml
linux = ["python3", "fmugen_launch.py"]
macos = ["python3", "fmugen_launch.py"]
windows = ["python", "fmugen_launch.py"]
```

On the FMU's first run on a machine, `fmugen_launch.py` creates a virtual environment, installs the wheels into it without a network connection (`--no-index`), and caches it. Then it starts UniFMU's `main.py` with that environment. Later runs, and other FMUs with the same wheels, reuse it. The cache is in `FMUGEN_ENV_DIR` if set, else `%LOCALAPPDATA%\fmugen\envs` (Windows), `~/Library/Caches/fmugen/envs` (macOS) or `$XDG_CACHE_HOME/fmugen/envs`, default `~/.cache/fmugen/envs` (Linux). One environment is kept per set of requirements, wheels and interpreter. It's created with `uv venv` / `uv pip` when `uv` is on `PATH`, else `venv` and `pip`. FMU instances starting at the same time are safe: the environment is built in a temporary folder and renamed into place.

If the install fails (usually no wheels for this platform or Python version), the launcher prints why and exits with code 1, so the FMU fails to instantiate. Command-line arguments are passed on to `main.py` (a distributed FMU's private side takes the proxy's port).

`[model] requirements` must list every package the model needs from PyPI: only those are vendored, not everything in your environment.

By default, wheels are made for this machine and Python (`pip wheel`; packages published only as source are built here). For other targets, list them. Only published wheels can be used for those:

```bash
fmugen build pid -o out/pid.fmu --vendor --platform win_amd64 --platform manylinux2014_x86_64 --platform macosx_11_0_arm64 --python-version 3.11 --python-version 3.12 --python-version 3.13
```

Vendoring uses the environment's `pip`, or `uvx pip` when the environment has none (uv venvs).

### Hugging Face models (`--hf-weights`)

Models such as Chronos, TTM or an SB3 policy download their weights from Hugging Face on first use. With `--vendor` or `--compile` (or `--hf-weights` on any build), `fmugen build` puts them into the FMU, in `resources/hf_cache/`:

- every string parameter or constant that is a Hugging Face model id, such as `"amazon/chronos-bolt-tiny"` (`snapshot_download`). A string counts when it looks like `owner/name` and doesn't end in a data-file suffix (`.csv`, `.json`, `.npy`, `.h5`, `.fmu`, ... : `DATA_SUFFIXES` in `src/fmugen/distribute.py`), so `"data/input.csv"` is left alone, and
- whatever a `{ call = "…" }` constant downloads, such as `huggingface_sb3.load_from_hub(...)`: the call is made once at build time, with Hugging Face's cache inside the FMU.

The FMU then sets `HF_HUB_CACHE` to that folder and `HF_HUB_OFFLINE=1` before importing the model, so it runs without a network connection or the user's cache. The FMU grows by the size of the weights. `--no-hf-weights` turns it off.

### `--compile`

`fmugen build --compile pyinstaller` (or `nuitka`) freezes UniFMU's backend, fmugen's adapter and runtime, your model and every package it imports, together with the Python interpreter, into `resources/dist/main/`. All `.py` files, `fmugen_model/`, `interface.json` and `requirements.txt` are removed from the FMU; `resources/` keeps only `dist/` and `launch.toml` (the exceptions inside `dist/` are listed [below](#what-the-compiler-is-told)):

```toml
linux = ["sh", "-c", "chmod +x dist/main/main && ./dist/main/main"]
macos = ["sh", "-c", "chmod +x dist/main/main && ./dist/main/main"]
windows = ["powershell", "-command", "./dist/main/main.exe"]
```

Install the compiler in your environment first: `pip install "path/to/2025_Inkindcontributions[pyinstaller]"` or `[nuitka]` (fmugen is not on PyPI yet).

| | PyInstaller | Nuitka |
|---|---|---|
| **How the code ships** | Python bytecode inside the bundle | compiled to C, then machine code |
| **Hiding the source** | light: bytecode can be decompiled to readable Python | strong: as hard to reverse as other compiled FMUs |
| **Build time** | about a minute | several minutes or more; needs a C compiler (on Windows, Nuitka downloads one) |

Neither can cross-compile: the executable runs only on the OS and CPU architecture of the machine that built it. To ship for several OSes, build on each one.

With `cwd` set, `fmugen_model/` stays, without its `.py` files, so the model still finds its data files.

#### What the compiler is told

A freezer only follows `import` statements. Libraries also read files next to their code, find plugins through package metadata, and import modules by name, and none of that is followed. So fmugen also tells the compiler:

- **The modules the runtime imports by name:** the model's entry, `setup` calls, enums, clock calls, and the modules in constants (`ref` and `{ call = "module:fn(...)" }`, including dotted names in the arguments, such as `numpy.linspace`).
- **The model's packages, with every submodule:** the packages of those modules and of `[model] requirements`.
- **Their whole dependency tree, with every package's data files, binaries and metadata.** The tree is read from the installed packages' metadata, and also includes the optional dependencies of the model's own packages that are installed. This is how, for example, iapws's `VERSION`, ahrs's `WMM.COF`, mendeleev's `elements.db`, pymsis's `msis21.parm`, silero-vad's ONNX file and pybamm's parameter sets get into the FMU.
- **The packages the model's packages depend on, imported by name** (pybamm imports `anytree` that way).
- **Folders of `.py` files inside the model's packages that aren't packages** (no `__init__.py`), e.g. pyromat's `registry`. Libraries load these by path, so they ship as files.
- **numpy's `numpy.core` compatibility modules**, which older pickles name, e.g. data loaded with `numpy.load`.
- **The `.py` sources of packages built on transformers** (PyInstaller only). transformers reads the source of the classes it decorates, as PyInstaller's own hook already arranges for transformers itself. These sources are in the FMU.

A model that unpickles objects, e.g. `joblib.load` of a scikit-learn model, needs the packages of those objects listed in `[model] requirements` (`init` can't see them): then their submodules are collected too. The FMU grows by the collected files: in the tests, 0 to 25 MB more than before this change (pvlib, mesa) for models that already worked. The model's non-Python files are bundled next to its modules. If something is still missed, the build's error or the FMU's log names it. When the importer shows nothing, see [unifmu-limitations.md §13](unifmu-limitations.md#13-a-backend-that-exits-during-a-call-can-abort-the-importer) for running the frozen backend directly to read its error.

### Distributed FMUs (`--distributed`)

```bash
fmugen build fmugen.toml -o out/model.fmu --distributed --endpoint 10.0.0.5
```

This builds the FMU in two halves with UniFMU's `generate-distributed` (FMI 2 and FMI 3):

- **The proxy**, `out/model.fmu` (or a folder with `--format folder`): UniFMU's native binaries, the `modelDescription.xml`, and a `resources/launch.toml` that only says `location = "Remote"`. It contains no Python and none of the model. The importer loads it like any other FMU.
- **The private folder**, `out/model_private/` (`--private-output PATH` to choose it): what is normally the FMU's `resources/`, i.e. the model, `interface.json`, fmugen's adapter and runtime, and UniFMU's backend. Its `main.py` is UniFMU's `main_remote.py`, which connects to the proxy instead of being started by it. `endpoint.toml` holds the proxy host's IP address, from `--endpoint` (default `127.0.0.1`). Copy this folder to the host or container that runs the model. Its `README.md` gives the exact start command.

`fmugen init` and `fmugen.toml` are the same as for any other FMU. `--distributed` works with composite FMUs, `--vendor` and `--compile`:

| Build | Start the private side, in its folder | Needs there |
|---|---|---|
| default | `python main.py PORT` | Python with the model's packages, and `pip install fmugen[distributed]` (`toml`, `coloredlogs`, `colorama`, which `main_remote.py` imports) |
| `--vendor` | `python fmugen_launch.py PORT` | Python; the packages are installed from `wheels/` on the first run |
| `--compile` | `dist/main/main PORT` (`.\dist\main\main.exe PORT` on Windows) | nothing; same OS as the build |

**The port can't be set.** In UniFMU 0.14 the proxy port isn't configurable: `fmiapi/src/common/dispatcher/mod.rs` hardcodes `BackendSocket::create("tcp://0.0.0.0:0")`, so the OS assigns a new port on every run, and no environment variable or `launch.toml` field changes it. fmugen therefore has no port flag. The private side takes the port as its argument at start-up (or asks for it). When the importer instantiates the proxy, the proxy listens on all interfaces, logs

```
ACTION REQUIRED ~ Connect remote backend to dispatcher through port: PORT
```

under the `logUnifmuMessages` log category, and waits for the private side to connect. **The importer must have logging on, including that category**, or the port is never shown. A firewall between the two hosts must allow the port. See [unifmu-limitations.md §14](unifmu-limitations.md#14-the-distributed-proxys-port-cant-be-set).

With FMPy, start the private side from the logger callback:

```python
import re, subprocess
from fmpy import simulate_fmu

def logger(env, instance, status, category, message):   # FMI 2; FMI 3 has no `instance`
    found = re.search(rb"dispatcher through port:\s*(\d+)", message)
    if found:   # here on the same host; on another host, pass the port there
        subprocess.Popen(["python", "main.py", found.group(1).decode()], cwd="out/model_private")

simulate_fmu("out/model.fmu", stop_time=10, debug_logging=True, logger=logger)
```

---

## Simulating

```bash
fmpy simulate out/pid.fmu --stop-time 10 --start-values Kp 4 --output-file out/results.csv
```

FMPy simulates FMI 2 and FMI 3 FMUs the same way. In Python, `fmpy.simulate_fmu(..., event_mode_used=True)` also handles output clocks (`eventHandlingNeeded`) and `[events] terminate`. FMPy never ticks *input* clocks, though. FMUs whose code runs on input clocks need an importer that ticks them, or a script using `fmpy.fmi3.FMU3Slave` (see `tests/test_fmi3.py`).

Time-varying inputs come from a CSV whose first column is time and whose other columns are named like the FMU's inputs, e.g. a `measurements.csv` for the PID FMU:

```
time,input_
0,0.0
5,0.5
```

```bash
fmpy simulate out/pid.fmu --input-file measurements.csv --stop-time 10 --output-file out/pid.csv
```

Without `--output-file`, FMPy plots the result instead, which requires matplotlib (`pip install matplotlib`).

```bash
uv run python -m fmpy.gui
```

In the GUI, open the `.fmu`, set start values, press play, and tick outputs to plot.

![FMPy GUI example](../image.png)

---

## Project layout

| Path | Contents |
|---|---|
| `src/fmugen/config.py` | `fmugen.toml` loading, validation, normalisation into the interface spec (FMI 2 and 3) |
| `src/fmugen/interface.py` | `fmugen init`: inference orchestration (import, infer, rename, requirements) |
| `src/fmugen/arguments.py` | Argument processing and type/value utilities used by inference |
| `src/fmugen/probe.py` | Probing (calling the model once), output detection and state detection |
| `src/fmugen/toml_output.py` | Render an inferred config dict as a `fmugen.toml` file |
| `src/fmugen/description.py` | `modelDescription.xml` writers (FMI 2 and FMI 3) |
| `src/fmugen/templates/fmugen_runtime/` | The engine: split into `_constants`, `_helpers`, `_engine`, `_composite` for development; concatenated into one file at build time |
| `src/fmugen/templates/model_fmi2.py`, `model_fmi3.py` | The adapters copied into the FMU as `resources/model.py`: FMI calls → engine |
| `src/fmugen/__main__.py` | CLI (argparse and `main()`) |
| `src/fmugen/builder.py` | Build/init orchestration, packaging, UniFMU version check |
| `src/fmugen/distribute.py` | `--vendor` and `--compile` |
| `src/fmugen/templates/fmugen_launch.py` | The launcher of vendored FMUs: offline install into a cached environment |
| `tests/` | `uv run pytest` |

---

## UniFMU version

fmugen doesn't ship UniFMU. Every `fmugen build` runs the UniFMU CLI to generate the FMU skeleton (native binaries for Windows, Linux and macOS, plus the Python backend), so it must be installed:

- Download UniFMU **0.14.0** from https://github.com/INTO-CPS-Association/unifmu/releases.
- Put `unifmu` on `PATH`, or set `FMUGEN_UNIFMU` to the executable's path.

fmugen runs `unifmu --version` and refuses any other version, because the adapters (`resources/model.py`) are written against that version's `backend.py` and protobuf messages. The pinned version is `UNIFMU_VERSION` in `src/fmugen/builder.py`.

UniFMU 0.14 writes `fmiVersion="3.0-beta.4"` into the FMI 3 template's `modelDescription.xml`. That file is replaced on every build, so it doesn't matter.

`fmugen` replaces `resources/model.py` and `modelDescription.xml`, adds `fmugen_runtime.py`, `interface.json` and `fmugen_model/`, and rewrites `requirements.txt` and `launch.toml`. The backend and binaries are taken as-is.

To move to a newer UniFMU: change `UNIFMU_VERSION`, check the backend's command list in the generated `resources/backend.py` against [fmi.md](fmi.md) and the adapters, then run `uv run pytest`. If the new backend imports other packages, update `BACKEND_REQUIREMENTS` and fmugen's dependencies in `pyproject.toml`.
