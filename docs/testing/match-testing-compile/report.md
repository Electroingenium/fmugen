# Compiled FMUs (PyInstaller): fmugen `dev` at `6dfc18c`

The 31 models of the compile test plan were compiled with `fmugen build --compile pyinstaller`. That is 33 builds, since psychrolib and dysts are compiled twice. Each compiled FMU was simulated and compared, output by output and step by step, with the uncompiled FMU of the same config from the last comparison run. Every build is in [results.csv](results.csv); every value is in [results.json](results.json).

## Summary

**No mismatch.** Every compiled FMU that simulates gives exactly the uncompiled FMU's values (largest relative difference 0). **13 compiled FMUs fail to simulate**, all GEKKO models. All 13 failures come from three fmugen problems with compiled FMUs plus two packaging issues, each confirmed on a copy with that cause undone ([below](#failures)).

| Verdict | Builds | Models |
|---|---|---|
| **match** | **20** | ambiance, commpy, impedance, pandapower, river, seirsplus, tellurium, uncertainties, cantera, PropsSI, units_Reynolds, units_head_from_P, metpy, pytci, psychrolib (without and **with numba**), pyquaternion, simpy, dysts Lorenz (`scipy:DOP853` FMI 3 and `rk4` FMI 2) |
| mismatch | 0 | |
| build fails | 0 | |
| timeout | 0 | |
| **FMU fails** | **13** | all 13 GEKKO models: gk_cftoc, gk_openhens, gk_distillation, gk_distcol_gym, gk_bioreactor, gk_debut_steady, gk_debut_reflux, gk_apsis_mpc, gk_fopdt, gk_mpc, gk_ethanol, gk_nhes, gk_cge (Windows) |

Per model (31): 18 match (psychrolib and dysts with both builds), 13 FMU fails.

The causes of the 13 failures:

| Cause | Where | Models | Each confirmed by |
|---|---|---|---|
| **A. Bundled executables lose their execute bit.** The FMU zip stores gekko's solver (`gekko/bin/apm`, `apm.exe`) with mode 755, but extracting the FMU (FMPy, `zipfile`) drops it. The launcher in `launch.toml` (`src/fmugen/distribute.py:378`) runs `chmod +x` only on `dist/main/main`, so gekko's `subprocess.Popen` fails: `PermissionError: [Errno 13] Permission denied: '…/dist/main/_internal/gekko/bin/apm.exe'` | fmugen, compiled FMUs on Linux and macOS | every GEKKO model on Linux (12) | `launch.toml` with `chmod -R +x dist`: the 8 models with no other problem match, every value identical |
| **B. `[model] cwd` data is kept where the compiled runtime doesn't look.** The build keeps `resources/fmugen_model/` (the data files of `[model] cwd`; `distribute.py:276`–`277`, `_remove_compiled`). But in a PyInstaller FMU, `RESOURCES_DIR` is `sys._MEIPASS` = `resources/dist/main/_internal` (`templates/model_fmi2.py:17`, `model_fmi3.py:17`), and the engine resolves `cwd` under it (`templates/fmugen_runtime/_engine.py:215`). Instantiation fails: `FileNotFoundError: … dist/main/_internal/fmugen_model` | fmugen, every compiled FMU whose config has `[model] cwd` | gk_nhes, gk_openhens, gk_cge | the folder also placed under `_internal/`: CGE matches (108 values) and NHES matches (4,892 values, with A) |
| **C. Editable installs aren't collected.** The debutanizer and OpenHENS projects are installed editable (as the last run did, `pip install -e .`), which setuptools does with an import hook (`__editable___…_finder.install()` in a `.pth`). PyInstaller's analysis doesn't follow it, so the package isn't in the FMU: `ModuleNotFoundError: No module named 'debutanizer'` | PyInstaller with setuptools editable installs | gk_debut_steady, gk_debut_reflux, gk_openhens | rebuilt from a venv copy with the project installed normally (`pip install --no-deps <copy>`): with A undone too, both debutanizer configs match (144 and 148 values) |
| **D. Pyomo's plugins aren't collected.** OpenHENS imports `pyomo.environ`, which imports its plugins by name (`pyomo.common.plugins`, …). Pyomo is only a dependency of the model's package, so fmugen gives PyInstaller `--hidden-import pyomo` but not `--collect-submodules pyomo` (`distribute.py:418`–`470`, `_packages`: only the entry's packages and `[model] requirements` are collected whole). Error: `ImportError: pyomo.environ failed to import pyomo.common.plugins: Original ModuleNotFoundError: No module named 'pyomo.common.plugins'` | packaging: the config needs `pyomo` in `requirements`, like skops needs scikit-learn | gk_openhens | the hand config with `requirements = ["pyomo==6.10.1"]`, built from the normal install, with A and B undone: matches (TAC 154853.8586) |

Without these, every model's compiled FMU equals its uncompiled one. The diagnosis copies only show that; the verdicts are those of the FMUs as built.

**Data files the compiled FMU can't find:** gk_nhes (`grid_data/more_data_scaled.csv`), gk_openhens (`examples/cases/…csv`) and gk_cge (`Data/*.csv`): all through cause B. Models that read files from their package (cantera's `gri30.yaml`, pvlib-style data) or by full path find them: cantera matches.

## Failures

The FMU's error for each build is in the `notes` column of results.csv, followed by "diagnosis copies:" with what each copy gave.

### Cause A only (8): gk_cftoc, gk_distillation, gk_distcol_gym (seeded), gk_bioreactor, gk_apsis_mpc, gk_fopdt, gk_mpc, gk_ethanol

```
PermissionError: [Errno 13] Permission denied:
'…/sim_unz/resources/dist/main/_internal/gekko/bin/apm.exe'
  File "gekko/gekko.py", line 2128, in solve
  File "subprocess.py", line 1992, in _execute_child
```

- gk_distillation fails at `fmi2ExitInitializationMode`, since its constructor solves; the others fail at the first `fmi2DoStep`.
- In the zip: `resources/dist/main/_internal/gekko/bin/apm` and `apm.exe` are `0o100755`. After extraction they're `-rw-r--r--`; only `dist/main/main` is `-rwxr-xr-x`, set by `launch.toml`'s `chmod +x dist/main/main`.
- It's `apm.exe` because gekko 1.3.2 deliberately runs the Windows binary under WSL (`"WSL" in os.uname().release`, gekko.py around line 2120). Elsewhere on Linux it runs `bin/apm`, which loses its execute bit the same way, so this isn't WSL-specific. On Windows there's no execute bit, which is why the docs' compiled CFTOC worked there.
- With `chmod -R +x dist`, all 8 match: 12, 28, 39, 20, 336, 24, 48 and 124 values respectively, all identical.
- **Fix (fmugen):** keep the execute bits of every bundled file (e.g. `chmod -R u+x dist`, or record the executables at build time and chmod them in `launch.toml`).

### gk_debut_steady, gk_debut_reflux: causes C and A

```
ModuleNotFoundError: No module named 'debutanizer'
  File "fmugen_runtime.py", line 139, in load_entry
```

- The backend dies at `fmi2Instantiate`. With `--capture-output` its traceback doesn't reach the importer: FMPy only shows "backend exited unexpectedly with exit status 1". The traceback above came from the same FMU with the backend's output sent to a file.
- Rebuilt with the project installed normally, the error becomes cause A's `PermissionError`. With both fixed, the two configs match (144 and 148 values).
- **Fix (setup):** install such projects normally (not `-e`) before `--compile`, or add the source folder to `[model] sources`. Possibly fmugen could resolve editable finders when collecting.

### gk_nhes: causes B and A

```
FileNotFoundError: [Errno 2] No such file or directory:
'…/resources/dist/main/_internal/fmugen_model'
  File "fmugen_runtime.py", line 1013, in __init__ / line 1964, in _in_model_dir
```

- The config has `cwd = "."` and `sources = ["grid_data/more_data_scaled.csv"]` (the synthetic data of the last run). The file is in the FMU at `resources/fmugen_model/grid_data/`.
- With the folder also under `_internal/`, the error becomes `LCOE is None` (gekko's solver can't run: A). With both fixed, it matches (4,892 values; `solve_time` excluded as in the last run).
- seaborn and plotly, which NHES imports without declaring, were already in the config's `requirements` and were collected; no change was needed.
- **Fix (fmugen):** in a PyInstaller FMU, resolve `[model] cwd` against the FMU's `resources/` folder (the executable's folder's parent's parent), not `sys._MEIPASS`. Or add the `cwd` data to the bundle under `fmugen_model/`.

### gk_openhens: causes C, D, B and A

- As built: the same `ModuleNotFoundError` for `openhens` (C).
- With a normal install: `No module named 'pyomo.common.plugins'` (D).
- With `pyomo` added to `requirements` too, and A and B undone: it matches, TAC 154853.85859989937 at every step.
- **couenne isn't in the compiled FMU.** For every simulation, couenne 0.5.8 (IDAES bundle 3.4.2) and its libraries were on `PATH` / `LD_LIBRARY_PATH`, as for the uncompiled FMU.

### gk_cge (Windows): cause B

- The compiled FMU builds on Windows (56 s, 66.8 MB); Smart App Control didn't block the compiled binary or UniFMU.
- The simulation crashes in UniFMU, before any FMI call returns:

```
thread '<unnamed>' panicked at …/zeromq-0.4.1/src/rep.rs:168:40:
not yet implemented
thread caused non-unwinding panic. aborting.
```

  (sim.py exits with 127; FMPy gets no result.)
- **It's cause B:** the config has `cwd = "."` (CGE reads `Data\\*.csv`). With `fmugen_model/` also placed under `dist/main/_internal/`, the same FMU simulates and matches all 108 values.
- On Windows, the backend's exit at instantiation shows up as this zeromq panic in UniFMU 0.14.0's reply socket, instead of the "backend exited unexpectedly" error the Linux build gives.

## Sizes and build times

The build time is `fmugen build … --compile pyinstaller` from start to end (including the uncompiled stage). Each model was built in its own venv with only its packages, fmugen, FMPy, psutil and PyInstaller.

| Model | Build | Size (MB) | Verdict |
|---|---|---|---|
| simpy | 8 s | 44.8 | match |
| GetHumRatioFromRelHum | 9 s | 44.8 | match |
| uncertainties | 9 s | 45.9 | match |
| pyquaternion | 12 s | 48.9 | match |
| PropsSI | 14 s | 56.7 | match |
| gk_debut_steady | 10 s | 59.1 | FMU fails |
| gk_debut_reflux | 11 s | 59.1 | FMU fails |
| gk_cftoc | 14 s | 62.1 | FMU fails |
| gk_distillation | 14 s | 62.1 | FMU fails |
| gk_apsis_mpc | 14 s | 62.1 | FMU fails |
| gk_mpc | 14 s | 62.1 | FMU fails |
| gk_ethanol | 13 s | 62.1 | FMU fails |
| gk_distcol_gym | 15 s | 63.1 | FMU fails |
| gk_cge (Windows) | 56 s | 66.8 | FMU fails |
| cantera | 12 s | 79.9 | match |
| ambiance | 18 s | 87.6 | match |
| pytci | 16 s | 87.6 | match |
| units_Reynolds | 18 s | 89.1 | match |
| units_head_from_P | 18 s | 89.1 | match |
| river | 20 s | 92.9 | match |
| seirsplus | 20 s | 94.8 | match |
| gk_bioreactor | 25 s | 97.8 | FMU fails |
| gk_fopdt | 24 s | 97.8 | FMU fails |
| GetHumRatioFromRelHum_numba | 18 s | 110.2 | match |
| gk_openhens | 33 s | 114.8 | FMU fails |
| pandapower | 34 s | 124.1 | match |
| commpy | 34 s | 126.0 | match |
| gk_nhes | 40 s | 128.5 | FMU fails |
| impedance | 33 s | 131.6 | match |
| metpy | 36 s | 149.9 | match |
| tellurium | 52 s | 211.1 | match |
| dysts_dop853 | 39 s | 267.5 | match |
| dysts_rk4 | 44 s | 267.5 | match |

- A GEKKO model compiles in 10–15 s to about 60 MB in a minimal venv. matplotlib (imported by the bioreactor, FOPDT and NHES files) adds about 35 MB.
- The largest are dysts (267.5 MB) and tellurium (211.1 MB).
- No build came near the 60-minute timeout.

## Notable results

- **numba inside a compiled FMU works:** psychrolib with numba (`GetHumRatioFromRelHum_numba`, numba 0.68.0) compiles, and the ufunc is JIT-compiled with LLVM at run time inside the frozen app: it matches (4 values).
- **dysts Lorenz** (docs/models.md's config: `call = "rhs"`, `[model] integrate`):
  - The last run had no reference rows for it, so its uncompiled FMUs were built and simulated here (dysts 0.96).
  - Both compiled variants equal them: `scipy:DOP853` (FMI 3) and `rk4` with 50 substeps (FMI 2), 12 values each.
- **tellurium:** `LD_LIBRARY_PATH` was set to uv Python's `lib` folder for every build and simulation (needed by tellurium, harmless for the rest). Whether the compiled FMU still needs it at run time wasn't tested.
- **pytci** matches the last run's uncompiled FMU, including its t = 0 value (the dose given at initialization).
- **gk_distcol_gym** was compiled from its seeded config; it fails only through cause A and matches with it undone.

## How it was run

- **Where:** WSL 2 (Ubuntu, Python 3.13 from uv), as the last run; CGE on Windows (its data paths are Windows paths). Smart App Control wasn't changed.
- **Code:** the clone at `~/cmp/repo` on `dev` at `6dfc18c` (the latest), FMPy 0.3.32, UniFMU 0.14.0, **PyInstaller 6.22.3** (the version of the last compile; it installed everywhere).
- **Reference values:** the uncompiled FMUs' `fmu_rows` of the last comparison run. The test plan names them `docs/match-testing-rerun/results.json`; that folder isn't in the repository, but is the last run's output, `rerun/results.json` in this work folder. Outputs that run treated as random are left out here too (NHES's `solve_time`).
- **Venvs:**
  - One minimal venv per model at `~/cmp/vc/<model>` (Windows: `C:\vc\gk_cge`). Each holds the model's packages, fmugen (editable from the clone), FMPy, psutil and PyInstaller.
  - Each is constrained to the versions of the venv the last run used (`uv pip freeze` of `~/cmp/v/<group>`), so the library versions match the reference values.
  - The GEKKO models, which shared the `gekko` venv last time, each have their own.
  - Debutanizer and OpenHENS are installed from their work copies with `pip install -e`, as last time (see cause C).
- **Configs:** those of the last run (`results.json`'s `toml` of the same model and config):
  - the Part B hand configs, the hand configs of tested-models.md / models.md, `gk_cftoc` and `gk_openhens`'s hand configs;
  - the `init --probe` configs, and gk_distcol_gym's seeded config;
  - dysts: docs/models.md's config, as `comparison/harness/hand/dysts_dop853` (`fmi_version = 3`) and `dysts_rk4` (`integrate = "rk4"`, `substeps = 50`, FMI 2).
- **Build:** `fmugen build <toml> -o <model>.compiled.fmu --capture-output --fmi <2, or 3 for dysts DOP853> --compile pyinstaller`, 60-minute hard timeout.
- **Simulation:** `fmpy.simulate_fmu(stop_time=3, output_interval=1)` with an instance the harness creates and frees itself (`comparison/harness/sim.py`), as for the uncompiled FMUs. Afterwards only that simulation's process tree is killed. On failure, it reran with debug logging to record the FMU's error.
  - `sim.py` now also reads FMI 3 array outputs (as `name[i]`, the FMI 2 names) and handles FMI 3's log callback; before, it failed on dysts' FMI 3 FMU.
- **Compared:** every output at t = 0, 1, 2, 3; relative difference ≤ 10⁻⁹, strings exactly.
- **Diagnosis:** for the 13 failures, the backend's traceback was taken from a copy of the FMU whose `launch.toml` sends the backend's output to a file. For OpenHENS that copy was rebuilt without `--capture-output`, which otherwise swallows it. Each suspected cause was then undone in a copy (`comparison/harness/diag_compiled.py`, `diag_run.py`): `launch.toml` with `chmod -R +x dist`; `fmugen_model/` also under `_internal/`; the project installed normally; `pyomo` in `requirements`. The copies are next to each FMU (`~/cmp/c/m/<model>/`).

## Nothing skipped

All 31 models (33 builds) were built and simulated. No published model was changed (the configs are those of the last run, with OpenHENS's `pyomo` variant a separate file in its work copy).

## Files

- [results.csv](results.csv): model, config used, library version, PyInstaller version, build ok?, build time, FMU size (MB), simulates?, outputs compared, max rel diff, verdict, notes.
- [results.json](results.json): the same, with the compiled and reference values and the diagnosis results.
- `comparison/harness/compile_all.py` (venvs, build, simulation, comparison, report), `diag_compiled.py`, `diag_run.py`. The FMUs and logs are in WSL at `~/cmp/c/m/<model>/`; CGE's are in `C:\cc\gk_cge`.
