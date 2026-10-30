# fmugen: Python models → UniFMU FMUs

`fmugen` turns an ordinary Python model into a [UniFMU](https://github.com/INTO-CPS-Association/unifmu) FMU (FMI 2.0 or FMI 3.0 Co-Simulation) that any FMI importer can run.

**You don't change your code.** A plain function, a class with a step method, a multi-file package, or a model from PyPI is packaged as it is. What fmugen needs to know (which function or class to call, which arguments are inputs or parameters, where outputs come from, units) lives in a separate `fmugen.toml`. `fmugen init` writes a first version of that file by inspecting your model.

---

## Installation

Install fmugen into the virtual environment of your model, the one that already has the model's packages. fmugen is not on PyPI yet, so install it from a clone of this repository:

```bash
pip install path/to/2025_Inkindcontributions
```

`fmugen build` also needs [UniFMU](https://github.com/INTO-CPS-Association/unifmu/releases) **0.14.0**, exactly this version: put the `unifmu` executable on `PATH`, or set `FMUGEN_UNIFMU` to its path. To validate and simulate FMUs, install [FMPy](https://github.com/CATIA-Systems/FMPy) too (`pip install fmpy`). Details: [docs/getting-started.md](docs/getting-started.md#installation).

---

## Quick start

1. Let fmugen inspect the model and write `fmugen.toml` next to it:

   ```bash
   fmugen init path/to/your_model.py
   ```

   `init` reads your code and never calls the model, so it's safe for models that drive hardware. For ML models, compiled code or array results, add `--probe`: `init` then calls the model once to find array sizes and exact types.

2. Review the file: check the start values, add units, and remove anything you don't want in the FMU. Commented lines mark what the code didn't show ([how to read it](docs/init.md)).

3. Build and simulate:

   ```bash
   fmugen build path/to/fmugen.toml -o out/your_model.fmu
   ```

   ```bash
   fmpy simulate out/your_model.fmu --output-file out/result.csv
   ```

For a quick try, `fmugen build path/to/your_model.py -o out/your_model.fmu` skips the config: it is inferred in memory. [Getting started](docs/getting-started.md#example-a-published-pid-controller) does all of this with a published PID controller.

By default the FMU runs with the Python of your environment, so it works on the machine where you built it. For other machines, `--vendor` puts the packages inside the FMU, `--compile` freezes everything into an executable, and `--distributed` runs the model on another host ([packaging](docs/packaging.md#fmus-for-other-machines)).

---

## What a model can look like

Functions, classes with a step method, results stored as attributes, state fed back between steps, several files, packages from PyPI, libraries that need a setup call, C extensions, setter methods, arrays (FMI 3), clocks (FMI 3) and several models in one FMU. Each has a worked example in [docs/models.md](docs/models.md). fmugen has been tried on over 200 unmodified published models: [docs/testing/tested-models.md](docs/testing/tested-models.md).

FMUs are FMI 2.0 by default; `[model] fmi_version = 3` or `fmugen build --fmi 3` builds FMI 3.0, which adds more types, arrays, clocks and events. A config that doesn't use FMI 3-only features builds either version. See [docs/fmi.md](docs/fmi.md).

---

## Documentation

| Page | Contents |
|---|---|
| [Getting started](docs/getting-started.md) | Installation, from model to FMU, simulating |
| [CLI reference](docs/cli.md) | Every `fmugen init` / `fmugen build` option, exit codes, environment variables |
| [`fmugen init`](docs/init.md) | What `init` infers and how, and a walk through editing the config it writes, with a checklist and FAQ |
| [Writing models](docs/models.md) | Every supported kind of model, with examples |
| [`fmugen.toml` reference](docs/config.md) | Every key, its default, validation rules and a complete example |
| [FMI behaviour](docs/fmi.md) | FMI 2 and FMI 3, function by function: initialization, steps, clocks, state, logging |
| [Packaging and runtime](docs/packaging.md) | What `fmugen build` does, the FMU's layout and Python environment, FMUs for other machines |
| [Troubleshooting](docs/troubleshooting.md) | Error messages and their fixes, known problems |
| [UniFMU limitations](docs/limitations/unifmu.md), [FMPy limitations](docs/limitations/fmpy.md) | Bugs in UniFMU 0.14 and FMPy 0.3.32 that affect fmugen FMUs, and their workarounds |
| [`interface.json` reference](docs/interface.md) | fmugen's description of the model inside each FMU, with a JSON Schema |
| [Tested models](docs/testing/tested-models.md) | Every published model fmugen was tried on, with its result |
| [Development](docs/development.md) | Tests, project layout, updating UniFMU, open items |

---

## References

- UniFMU: https://github.com/INTO-CPS-Association/unifmu
- FMI standard: https://fmi-standard.org/
- FMPy: https://github.com/CATIA-Systems/FMPy
- Legaard, C. M., Tola, D., Schranz, T., Macedo, H. D., & Larsen, P. G. (2021). *A Universal Mechanism for Implementing Functional Mock-up Units*. SIMULTECH 2021, pp. 121–129. https://doi.org/10.5220/0010577601210129

## Credits

- Original psychrometry model and UniFMU workflow by Lucia Royo-Pascual, Ph.D. (EIUM).
- Models used in the docs: [simple-pid](https://github.com/m-lundberg/simple-pid) (MIT, Martin Lundberg), [RC_BuildingSimulator](https://github.com/architecture-building-systems/RC_BuildingSimulator) (MIT, Architecture and Building Systems, ETH Zürich) and [FilterPy](https://github.com/rlabbe/filterpy) (MIT, Roger R. Labbe).
