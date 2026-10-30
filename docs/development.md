# Development

How to work on fmugen itself: setup, tests, the source layout, updating UniFMU, and the open items.

- [Setup and tests](#setup-and-tests)
- [Project layout](#project-layout)
- [Updating UniFMU](#updating-unifmu)
- [Open items](#open-items)
- [Feature ideas](#feature-ideas)

---

## Setup and tests

From a clone of the repository:

```bash
uv sync
```

This installs fmugen in editable mode with the test dependencies. Use `uv run fmugen …` to run it.

```bash
uv run pytest
```

That runs everything, including the `slow` tests that freeze FMUs with PyInstaller and Nuitka (minutes each). For a quick loop while editing, skip those and the `fmu` tests, which run built FMUs through UniFMU and an importer (about a minute):

```bash
uv run pytest -m "not slow and not fmu"
```

The tests cover:
- **Adapter:** the generated adapters' behaviour for every FMI 2 and FMI 3 call, including clocks, arrays and configuration mode.
- **Inference:** `fmugen init` on small models, and model shapes found while testing published models.
- **FMU validity:** `modelDescription.xml` is checked with FMPy's validator, and `interface.json` against its JSON Schema.
- **Real runs:** simulations through UniFMU for both versions. These cover every supported FMI 3 type, clocks, intervals and shifts, state save/restore, and changing a tunable parameter.

The published models fmugen was tried on, and how, are in [testing/tested-models.md](testing/tested-models.md).

---

## Project layout

| Path | Contents |
|---|---|
| `src/fmugen/__main__.py` | CLI (argparse and `main()`) |
| `src/fmugen/builder.py` | Build/init orchestration, packaging, UniFMU version check |
| `src/fmugen/config.py` | `fmugen.toml` loading, validation, normalisation into the interface spec (FMI 2 and 3) |
| `src/fmugen/compose.py` | Composite FMUs: several configs in one FMU |
| `src/fmugen/description.py` | `modelDescription.xml` writers (FMI 2 and FMI 3) |
| `src/fmugen/units.py` | SI base units of unit strings, for `<BaseUnit>` |
| `src/fmugen/distribute.py` | `--vendor`, `--compile` and `--hf-weights` |
| `src/fmugen/interface.py` | `fmugen init`: inference orchestration (import, infer, rename, requirements) |
| `src/fmugen/static.py` | Reading the model's source code to find its results, without running it |
| `src/fmugen/examples.py` | Reading usage examples (doctests, README, `examples/`) for `init` |
| `src/fmugen/arguments.py` | Argument processing and type/value utilities used by inference |
| `src/fmugen/probe.py` | `init --probe`: calling the model once, output and state detection |
| `src/fmugen/toml_output.py` | Rendering an inferred config as a `fmugen.toml` file |
| `src/fmugen/interface.schema.json` | JSON Schema of `interface.json` ([interface.md](interface.md)) |
| `src/fmugen/templates/fmugen_runtime/` | The engine: split into `_constants`, `_helpers`, `_engine`, `_composite` for development; concatenated into one file at build time |
| `src/fmugen/templates/model_fmi2.py`, `model_fmi3.py` | The adapters copied into the FMU as `resources/model.py`: FMI calls → engine |
| `src/fmugen/templates/fmugen_launch.py` | The launcher of vendored FMUs: offline install into a cached environment |
| `tests/` | `uv run pytest` |
| `docs/testing/` | Records of the published-model test runs, with their scripts |

---

## Updating UniFMU

fmugen requires exactly UniFMU 0.14.0 (`UNIFMU_VERSION` in `src/fmugen/builder.py`) and refuses any other version, because the adapters are written against that version's `backend.py` and protobuf messages. `fmugen` replaces `resources/model.py` and `modelDescription.xml`, adds `fmugen_runtime.py`, `interface.json` and `fmugen_model/`, and rewrites `requirements.txt` and `launch.toml`; the backend and binaries are taken as they are.

To move to a newer UniFMU:

1. Change `UNIFMU_VERSION`.
2. Check the backend's command list in the generated `resources/backend.py` against [fmi.md](fmi.md) and the adapters.
3. If the new backend imports other packages, update `BACKEND_REQUIREMENTS` and fmugen's dependencies in `pyproject.toml`.
4. Recheck the workarounds in [limitations/unifmu.md](limitations/unifmu.md) (e.g. the FMI 2 adapter doubling `%` in log messages).
5. Run `uv run pytest`.

---

## Open items

### Before a public release
- **Package metadata:** no project URLs (no public repository yet), and a static version (consider `hatch-vcs` and a `CHANGELOG.md`). Check that the name `fmugen` is free on PyPI before the first upload. Once on PyPI, the docs can say `pip install fmugen` / `pipx install fmugen`.
- **Community files:** `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR templates, `CHANGELOG.md`, `CITATION.cff`.
- **`.gitignore`:** add `dist/`, `build/` and `*.egg-info`.
- **Repository size:** `.git` is about 75 MB, mostly early commits shared with `main` (generated `.fmu`/`.zip` files and a 71 MB `FMUs/ORIGINAL.fmu/binaries/linux64/unifmu.so`). Removing them means rewriting `main` too, or publishing from a fresh history.
- **Docs site:** MkDocs Material on GitHub Pages would add navigation and search.

### Portability
- **`--vendor` on Windows runs `python`** (`python3` elsewhere). On a clean Windows machine `python` is often the App Execution Alias that opens the Microsoft Store. Prefer the `py -3` launcher when it exists, and let users choose the interpreter at runtime (e.g. `FMUGEN_PYTHON`).
- **Multi-OS compiled FMUs.** `--compile` freezes for the OS it runs on. One FMU for several OSes needs one build per OS plus a step that merges their `resources/dist/` folders (`launch.toml` already has one entry per OS).
- **`--embed-python`** (opt-in): a python-build-standalone interpreter inside the FMU, for fully offline FMUs that keep their sources.
- **Python floor:** `requires-python = ">=3.13"` (decided 2026-10-14). The build side would need 3.11 and the runtime inside the FMU 3.10.

### Model coverage
- **Generators** (`yield`-based models driven with `next()`/`send()`): `build` refuses them with a hint, since no published generator model was found to design the inputs against. simpy models work through a factory calling `env.run(until=…)`.
- **Other model formats:** Jupyter notebooks, scikit-learn (`kind = "sklearn"` calling `predict` with the inputs as features) and standalone ONNX files.
- **Inference:** a `state` dict passed in and returned (`def step(u, state): …; return y, state`) isn't recognised as a state; it needs `to = "arg:state[key]"` / `next = "return:1.key"` by hand.
- **modelDescription:** FMI 3 terminals and icons aren't supported.

### Tests and quality
- **No test from an installed wheel.** That is how an earlier review found a wheel that couldn't build anything.
- **FMPy is the only importer used.** Add [fmusim from the Reference FMUs](https://github.com/modelica/Reference-FMUs) and, optionally, OMSimulator: importers call functions in different orders.
- **The published-model tests are manual.** An opt-in `pytest -m pypi` suite built from the configs in `models/r3/*/fmugen.toml` would keep them checked.
- **Property-based tests** (Hypothesis) for `render_toml` → `load_config` round trips and for `coerce` / `reshape` / `flatten`.
- **No linter, formatter or type checker.** Add `ruff` and `pre-commit`; `pyright` or `mypy` in basic mode, plus a `TypedDict` for the interface spec.

---

## Feature ideas

Ordered by value for users.

| Feature | Why |
|---|---|
| **`fmugen check MODEL`**: validate the config, import the model, print a table of the variables; `--run N` runs N steps in-process with `Engine`, without UniFMU | The fastest feedback loop; the runtime already works in-process |
| **`fmugen run FMU --stop-time … [--input csv] [--plot]`**: a thin FMPy wrapper with FMI logging on | Users get "failed with status 3" without a reason |
| **`fmugen inspect FMU`**: interface.json, the fmugen and UniFMU versions, the mode (venv / vendor / compile / distributed), platforms and requirements | Debugging FMUs received from others |
| **Record the fmugen version** in `interface.json` and `generationTool` | Nothing in an FMU says which fmugen built it |
| **Units from annotations** (`Annotated[float, "K"]`, pint annotations) | Fewer edits after `init` (numpydoc units are already read) |
| **JSON Schema for `fmugen.toml`** | Completion and validation in editors |
| **A documented Python API** (`fmugen.build(...)`, `fmugen.init(...)`) | Notebooks, scripts, other tools |
| **Merge compiled FMUs from several OSes** (`fmugen merge a.fmu b.fmu -o all.fmu`) | `--compile` is single-OS |
| **Watch mode** (`fmugen build --watch`) | A fast edit → build → simulate loop |
| **Plugin hooks** (entry points) for custom result and input types | Third parties can add types without changing fmugen |
