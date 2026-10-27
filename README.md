# fmugen: Python models → UniFMU FMUs

`fmugen` turns an ordinary Python model into a [UniFMU](https://github.com/INTO-CPS-Association/unifmu) FMU (FMI 2.0 or FMI 3.0 Co-Simulation) that any FMI importer can run.

**You don't change your code.** A plain function, a class with a step method, a multi-file package, or a model from PyPI is packaged as it is. What fmugen needs to know (which function or class to call, which arguments are inputs or parameters, where outputs come from, units) lives in a separate `fmugen.toml`. `fmugen init` writes a first version of that file by inspecting your model.

---

## Installation

Install fmugen into the virtual environment of your model, the one that already has the model's packages:

fmugen is not on PyPI yet, so `pip install fmugen` doesn't work. Install it from a clone of this repository:

```bash
pip install path/to/2025_Inkindcontributions
```

(or `uv add path/to/2025_Inkindcontributions`). This also installs the two packages UniFMU's Python backend needs, `protobuf==5.27.3` and `pyzmq`. Then run the `fmugen` commands from that environment.

You also need [UniFMU](https://github.com/INTO-CPS-Association/unifmu/releases) **0.14.0**, exactly this version, for `fmugen build`. fmugen runs `unifmu generate` to create each FMU's native binaries and Python backend. Put the `unifmu` executable on `PATH`, or set `FMUGEN_UNIFMU` to its path.

By default the FMU runs your model with the Python of that environment, so it works on the machine where you built it. To run it on other machines:

- `--vendor` puts the packages inside the FMU as wheels. The target needs Python; the FMU installs them offline on its first run.
- `--compile pyinstaller` or `--compile nuitka` freezes the model, its packages and Python into an executable. The target needs nothing, and the FMU contains no source code, but it only runs on the OS it was built on.
- `--distributed` makes the FMU a proxy, and writes the model to a separate folder that runs on another host or container and connects to the proxy over TCP.

See [docs/packaging.md](docs/packaging.md#fmus-for-other-machines).

To validate and simulate FMUs as below, install [FMPy](https://github.com/CATIA-Systems/FMPy) too (`pip install fmpy`).

---

## Quick start

For your own model:

1. Let fmugen inspect the model and write `fmugen.toml` next to it:

   ```bash
   fmugen init path/to/your_model.py
   ```

   `init` reads your code and never calls the model, so it's safe for models that drive hardware. For ML models, compiled code or array results, add `--probe`: `init` then calls the model once to find array sizes and exact types.

2. Review the file. Check the start values, add units, and remove anything you don't want in the FMU. Commented lines mark what the code didn't show (array sizes, property types, `save_state`): complete them, or rerun with `--probe`. If the model rejects zeros or needs a setup call first, rerun `init` with `--start NAME=VALUE` / `--setup CALL` (see [docs/models.md](docs/models.md#what-fmugen-init-infers)).
3. Build:

   ```bash
   fmugen build path/to/fmugen.toml -o out/your_model.fmu
   ```

To try it on a published model, install the PID controller from [simple-pid](https://pypi.org/project/simple-pid/) next to fmugen:

```bash
pip install simple-pid
```

See what `init` infers for its `PID` class, printed to the terminal instead of written to a file:

```bash
fmugen init simple_pid:PID -o -
```

It finds `Kp`, `Ki`, `Kd` and `setpoint` as parameters, `input_` as an input, the control output as `y`, and binds `dt` to the step size. Write it to a file, set `setpoint = { start = 1.0 }`, then build, validate and simulate it:

```bash
mkdir pid
```

```bash
fmugen init simple_pid:PID -o pid/fmugen.toml
```

```bash
fmugen build pid -o out/pid.fmu
```

```bash
fmpy validate out/pid.fmu
```

```bash
fmpy simulate out/pid.fmu --stop-time 5 --output-file out/pid.csv
```

For a quick try you can skip the config and build straight from the model. fmugen then infers the config in memory, without units:

```bash
fmugen build simple_pid:PID -o out/quick.fmu
```

---

## What a model can look like

| Your code | How it runs in the FMU | Example |
|---|---|---|
| A function `f(a, b, ...)` returning a dict, tuple, number or object | Called once per step; outputs are read from the return value | `fluids.core:Reynolds` ([tested models](docs/tested-models.md)) |
| A class `C(params...)` with a method `step(inputs...)` / `__call__` | Built once at initialization; the method is called once per step | `simple_pid:PID` |
| A method that stores its results as attributes | Outputs are read from the object's attributes after each step | RC_BuildingSimulator's `Zone` ([tested models](docs/tested-models.md#github-repositories)) |
| A method that needs last step's value (`x_prev`) | A *state*: fed back from the result after each step | RC_BuildingSimulator's `Zone` |
| Code that needs the time or step size (`t`, `dt`) | The FMU passes its communication time / step size | `simple_pid:PID` |
| Several files, flat or package imports | Copied into the FMU with their layout | RC_BuildingSimulator's `rc_simulator/` |
| A package from PyPI | Installed in your environment; listed as a requirement | `simple_pid:PID` |
| A library that must be set up first (`SetUnitSystem(SI)`, `env.reset()`) | `[model] setup` calls run before the model is used | psychrolib, gymnasium ([tested models](docs/tested-models.md)) |
| A C extension with positional-only arguments | `to = "pos:N"` bindings | CoolProp ([tested models](docs/tested-models.md)) |
| Setter methods and properties (`lab.Q1(50)`, `lab.T1`) | `to = "call:Q1"` inputs; properties read like attributes | tclab ([tested models](docs/tested-models.md)) |
| A class whose constructor does all the work | `kind = "function"`: constructed with the inputs every step | iapws ([tested models](docs/tested-models.md)) |
| Matrices and vectors (lists, numpy), sized by constructor arguments | FMI 3 arrays and structural parameters | `filterpy.kalman:KalmanFilter` ([models.md](docs/models.md)) |
| A method that should run on events, not every step | FMI 3 clocks: periodic, triggered, or raised by the model | `simple_pid:PID`, `filterpy.kalman:KalmanFilter` ([models.md](docs/models.md#clocks-fmi-3)) |

## FMI 2 and FMI 3

FMUs are FMI 2.0 by default. Set `[model] fmi_version = 3` in `fmugen.toml`, or pass `--fmi 3` to `fmugen build`, for FMI 3.0. A config that doesn't use FMI 3-only features builds either version, with the same results.

Both versions support what UniFMU's Python backend supports for Co-Simulation:
- parameters (fixed and tunable) and calculated parameters
- inputs, outputs, locals and states
- variable or fixed step size, and a default experiment
- saving and restoring FMU state, which lets an importer roll back
- reset
- log messages from the model's `logging` calls, forwarded to the importer

FMI 2 has Real, Integer, Boolean, String and Enumeration variables. FMI 3 adds:
- Float32, Int8…UInt64 and Binary variables
- arrays, with sizes set by structural parameters (configuration mode)
- clocks, including periodic and triggered input clocks, output clocks raised by the model, and their intervals and shifts
- event mode, and letting the model stop the simulation

See [docs/fmi.md](docs/fmi.md) for what each FMI call does, function by function for both versions, and what isn't supported.

---

## Documentation

| Page | Contents |
|---|---|
| [docs/manual.md](docs/manual.md) | **The manual**: installation, every `fmugen init` / `fmugen build` option, recipes and troubleshooting |
| [docs/config.md](docs/config.md) | `fmugen.toml` reference: every key, its default, validation rules and a complete example |
| [docs/models.md](docs/models.md) | Writing models: the supported shapes with examples, what `fmugen init` infers and what to check |
| [docs/init-tutorial.md](docs/init-tutorial.md) | Tutorial: reading and editing the `fmugen.toml` that `fmugen init` writes, with a checklist and FAQ |
| [docs/tested-models.md](docs/tested-models.md) | Every published model fmugen was tried on, with its most recent result |
| [docs/fmi.md](docs/fmi.md) | FMI 2 and FMI 3 behaviour: function-by-function support, initialization, steps, clocks, state, logging |
| [docs/packaging.md](docs/packaging.md) | What `fmugen build` does, the generated FMU's layout, the FMU's Python environment, updating UniFMU |
| [docs/interface.md](docs/interface.md) | `resources/interface.json`, fmugen's description of the model inside each FMU: every key, with a JSON Schema |
| [docs/unifmu-limitations.md](docs/unifmu-limitations.md) | Bugs and limitations of UniFMU 0.14 that affect fmugen FMUs, and their workarounds |
| [docs/fmpy-limitations.md](docs/fmpy-limitations.md) | Bugs and limitations of FMPy 0.3.32, the importer fmugen is tested with, and their workarounds |

---

## CLI

```
fmugen init MODEL [-o fmugen.toml] [--call METHOD] [--fmi {2,3}] [--start NAME=VALUE ...] [--const NAME=VALUE ...] [--setup CALL ...] [--kind function] [--create CLASSMETHOD] [--probe] [--convert NAME=module:function ...] [--force]
fmugen build MODEL -o OUTPUT [--fmi {2,3}] [--format fmu|folder] [--name NAME] [--author AUTHOR] [--call METHOD] [--probe] [--vendor [--platform TAG] [--python-version X.Y] | --compile {pyinstaller,nuitka}] [--hf-weights | --no-hf-weights] [--capture-output] [--distributed [--endpoint IP] [--private-output PATH]]
```

`MODEL` can be any of these:
- for `build`: a `fmugen.toml`, or a directory containing one
- for both: `model.py`, `model.py:Name`, or `package.module:Name` for an installed module

All options are in the [manual](docs/manual.md#4-commands).

---

## Development and tests

From a clone of the repository:

```bash
uv sync
```

```bash
uv run pytest
```

That runs everything, including the `slow` tests that freeze FMUs with PyInstaller and Nuitka (minutes each). For a quick loop while editing, skip those and the `fmu` tests, which run built FMUs through UniFMU and an importer (about a minute):

```bash
uv run pytest -m "not slow and not fmu"
```

The tests cover:
- **Adapter:** the generated adapters' behaviour for every FMI 2 and FMI 3 call, including clocks, arrays and configuration mode.
- **Inference:** `fmugen init` on small models.
- **FMU validity:** `modelDescription.xml` is checked with FMPy's validator.
- **Real runs:** simulations through UniFMU for both versions. These cover every supported FMI 3 type, clocks, intervals and shifts, state save/restore, and changing a tunable parameter.

---

## References

- UniFMU: https://github.com/INTO-CPS-Association/unifmu
- FMI 2.0 standard: https://fmi-standard.org/
- FMPy: https://github.com/CATIA-Systems/FMPy
- Legaard, C. M., Tola, D., Schranz, T., Macedo, H. D., & Larsen, P. G. (2021). *A Universal Mechanism for Implementing Functional Mock-up Units*. SIMULTECH 2021, pp. 121–129. https://doi.org/10.5220/0010577601210129

## Credits

- Original psychrometry model and UniFMU workflow by Lucia Royo-Pascual, Ph.D. (EIUM).
- Models used in the docs: [simple-pid](https://github.com/m-lundberg/simple-pid) (MIT, Martin Lundberg), [RC_BuildingSimulator](https://github.com/architecture-building-systems/RC_BuildingSimulator) (MIT, Architecture and Building Systems, ETH Zürich) and [FilterPy](https://github.com/rlabbe/filterpy) (MIT, Roger R. Labbe).
