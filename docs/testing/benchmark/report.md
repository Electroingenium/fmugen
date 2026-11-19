# Benchmark: `init --probe` and `build`

2026-11-19, at `b63e0a4` (fmugen 1.2.0), Windows 11, Python 3.13 in the repo's `.venv` (63 distributions, 15 123 files listed in their `RECORD`s). Timings are wall time; the profiles are cProfile.

Scripts, with the repo's `.venv` Python:

- [bench.py](bench.py) `PYTHON` (from this folder): `init` with and without `--probe`, median of 3 runs, one process each.
- [count.py](count.py) `TARGET [OPTIONS]` (from `models/`): `init TARGET --probe` in-process under cProfile; prints the wall time, the `_infer_config_once` calls, and the `importlib.metadata.packages_distributions` scans and their time.
- [patched.py](patched.py) `count.py TARGET [OPTIONS]` (from `models/`): the same, with `fmugen.examples` sent to the cached `fmugen.static.packages_distributions`.
- [build_bench.py](build_bench.py) `PYTHON` (from this folder): `build` variants, median of 3 runs, into `out/`.
- [first_read.py](first_read.py) `FOLDER`: runs `unifmu generate`, times the first and second open of each binary, and copies them to `FOLDER` (to read from another process).
- [models/](models/): `fn.py`, `tank.py` and `dictmod.py` (a function and a class that take a dict and an object); `tank/` and `kf/` hold the configs `init --probe` wrote for `tank.py:Tank` and `filterpy.kalman:KalmanFilter --call predict`.

## `init --probe`

### The probe call isn't what's slow

`--probe` adds under 0.05 s to `init` on every model tried: the model's own call is quick. `init` without `--probe` takes the same time, since both read the package's examples and work out the requirements.

| Model | `init` | `init --probe` |
|---|---|---|
| `fn.py` | 0.91 s | 0.91 s |
| `tank.py` | 0.91 s | 0.93 s |
| `filterpy.common:Q_discrete_white_noise` | 1.97 s | 1.93 s |
| `filterpy.kalman:KalmanFilter --call predict` | 2.29 s | 2.31 s |

Python starts in 0.09 s; importing `fmugen.interface` and `fmugen.probe` takes 0.49 s.

### Where the time goes: three scans of the installed packages

`importlib.metadata.packages_distributions()` reads the file list of every installed distribution and checks that each file exists (about 32 000 `stat` calls here). It runs three times per `init` of a package model:

| Model | Total | Scans | Share |
|---|---|---|---|
| `dictmod:Plant` (6-line class) | 1.95 s | 3, 1.78 s | 91 % |
| `filterpy.kalman:KalmanFilter --call predict` | 3.45 s | 3, 1.82 s | 53 % |
| `filterpy.kalman:rts_smoother` | 3.80 s | 3, 1.82 s | 48 % |
| `filterpy.kalman:ExtendedKalmanFilter --call predict_update` | 4.09 s | 3, 1.81 s | 44 % |
| `filterpy.monte_carlo:systematic_resample` | 3.26 s | 3, 1.82 s | 56 % |
| `filterpy.gh:GHFilter --call update` | 3.27 s | 3, 1.82 s | 56 % |
| `filterpy.common:Q_continuous_white_noise` | 2.64 s | 2, 1.21 s | 46 % |
| `filterpy.stats:gaussian` | 2.61 s | 2, 1.21 s | 46 % |
| `dictmod.py:Plant` (a file path) | 0.88 s | 1, 0.60 s | 68 % |
| `dictmod.py:mix` (a file path) | 1.26 s | 1, 0.61 s | 48 % |

(The totals are under cProfile, a little higher than in `bench.py`.)

A scan takes about 0.6 s with 63 distributions; its time grows with the files installed, so with torch, TensorFlow and the like each one takes seconds.

### Cause

[static.py](../../../src/fmugen/static.py) `packages_distributions()` caches the scan, keyed on the modification times of the `sys.path` folders that hold distributions. `_readmes` in [examples.py](../../../src/fmugen/examples.py) calls `importlib.metadata.packages_distributions()` directly, so it skips the cache: two full scans per inference of a package model (`pkg.module:Name`). A model given as a `.py` path doesn't call it, which is why those make one scan (`_add_requirements`, through the cache).

From the code, not measured: each rerun of `_infer_config_once` in `_infer_config`'s retries (a set-up the error asks for, shaped starts) reads the examples again, so it scans again. None of the models here triggered those retries.

### Fix

`_readmes` now calls `fmugen.static.packages_distributions` (cached). First tried in memory ([patched.py](patched.py)), then applied in `examples.py`; with it, `count.py` shows one scan per `init`:

| Model | Before | In memory | Applied |
|---|---|---|---|
| `dictmod:Plant` | 1.95 s | 0.73 s | 0.77 s |
| `filterpy.kalman:KalmanFilter --call predict` | 3.45 s | 2.20 s | 2.64 s |
| `filterpy.kalman:rts_smoother` | 3.80 s | 2.50 s | 2.50 s |

(One run each; the scans themselves went from 1.8 s to 0.6 s in every row, the rest varies between runs.) The `fmugen.toml` written for `KalmanFilter` is the same before and after. `tests/test_examples.py`, `test_inference.py`, `test_static.py` and `test_inferred_options.py`: 162 passed, 5 skipped.

### What's left after it

- The one cached scan, about 0.6 s; most of it is the existence check on every listed file. Reading `top_level.txt` or the `RECORD`s without the checks would be cheaper.
- Importing the model's package: filterpy imports `scipy.stats`, about 1.15 s. Not fmugen's.
- Importing fmugen, about 0.5 s: `fmugen.wizard`, `fmugen.config` and the runtime templates are imported up front.

## `build`

UniFMU 1.1.0 at `C:\Program Files\UniFMU`; Windows Defender real-time protection on.

### Timings

| Build | Median of 3 | FMU size |
|---|---|---|
| `models/tank/fmugen.toml` | 3.70 s | 6.8 MB |
| the same, `--fmi 3` | 3.59 s | 6.8 MB |
| the same, `--format folder` | 3.10 s | 19.6 MB unzipped |
| the same, `--distributed` | 3.67 s | 6.8 MB |
| `tank.py` (config inferred in memory) | 4.19 s | 6.8 MB |
| `tank.py --probe` | 4.19 s | 6.8 MB |
| `models/kf/fmugen.toml` (filterpy) | 4.52 s | 6.8 MB |

Run once each:

| Build | Time | FMU size |
|---|---|---|
| `tank --vendor`, first run (wheel cache cold) | 24.6 s | 7.9 MB |
| `tank --vendor`, again (profiled) | 5.9 s | 7.9 MB |
| `kf --vendor` | 14.1 s, then 12.1 s | 75 MB |
| `tank --compile pyinstaller` | 177 s | 94 MB |

The `.py` builds add about 0.5 s over the config, for the inference; the probe adds nothing. A `.py` build in a package model pays the metadata scans of `init` (see above).

### Where the time goes (`tank`, 3.05 s profiled in `build`)

The FMU is almost all UniFMU's binaries: `unifmu.dll` 6.8 MB, `unifmu.dylib` 5.9 MB and two `unifmu.so` 3.3 MB each, 19.4 MB of 19.6 MB. The model's own files are about 230 KB.

| Step | Time | What it is |
|---|---|---|
| `stamp_guid` | 1.26 s | Hashing is 0.02 s; the rest is Defender scanning the binaries the first time they are opened |
| `generate_boilerplate` | 1.03 s | `unifmu generate python out fmi2`, a new process: 0.71–0.93 s (cached only within one process) |
| `_write_zip` | 0.72 s | Deflating 19.4 MB of binaries at level 6 |
| copying the template | 0.22 s | `copytree`; the binaries are hard-linked |
| importing fmugen | about 0.5 s | `wizard`, `config`, the runtime templates |

### Defender scans every build's binaries again

`_template` runs `unifmu generate` into a new temporary folder in each process and removes it at exit, so each build writes the four binaries again, and Defender scans each new file on its first open ([first_read.py](first_read.py)):

| File | First open | Second open |
|---|---|---|
| `win64/unifmu.dll` | 0.735 s | 0.000 s |
| `darwin64/unifmu.dylib` | 0.100 s | 0.000 s |
| `linux64/unifmu.so` | 0.040 s | 0.000 s |
| `aarch64-linux/unifmu.so` | 0.012 s | 0.000 s |

`stamp_guid` is the first step that reads them, so it carries the scan. Copies of those files, made by one process, were scanned again in the next (new files: 0.73 s for the DLL), but not in a third, and new hard links to them weren't scanned either (0.005 s). So a template kept between runs (a user cache folder, keyed on the UniFMU executable, its version and the `generate` arguments), hard-linked into each FMU as now, would save the `unifmu generate` run and the scan: about 1.7 s of 3.7 s.

### Zipping

| Level | Time | Size |
|---|---|---|
| 6 (`zipfile` default, used now) | 0.67 s | 6.8 MB |
| 1 | 0.29 s | 7.5 MB |

Level 1 saves 0.4 s for 10 % more size. The binaries are the same in every build, so their compressed bytes could also be kept with the template and copied into the archive as they are, but `zipfile` has no public way to write already-compressed data.

### `--vendor` and `--compile`

- `--vendor` runs `uvx pip wheel` for the requirements plus the runtime's `protobuf==7.36.2`, `pyzmq` and `cloudpickle`: 2.5 s with a warm cache for `tank`. The first run, with a cold cache, took 24.6 s, most of it downloading. With filterpy (numpy, scipy, matplotlib) the wheels are 67 MB more and the build 12–14 s.
- `--compile pyinstaller` took 177 s for `tank`. Not profiled; PyInstaller's analysis and freezing are expected to take most of it.

## Heavier models

2026-11-19, a separate venv (uv, Python 3.13) with fmugen editable plus torch 2.15.0+cpu, gymnasium 1.4.0, pvlib 0.16.1 and scikit-learn 1.9.1: 36 distributions, 22 905 files (torch alone 12 358). Configs in [models/heavy/](models/heavy/), written by `init --probe`:

| Name | Target |
|---|---|
| `linear` | `torch.nn:Linear --start in_features=4 --start out_features=2` |
| `cartpole` | `gymnasium.envs.classic_control.cartpole:CartPoleEnv` |
| `solarpos` | `pvlib.solarposition:get_solarposition` |
| `ridge` | `sklearn.linear_model:Ridge` |
| `forest` | `sklearn.ensemble:RandomForestRegressor` |

Scripts: [timers.py](timers.py) `TARGET [OPTIONS]` times `init --probe` with plain timers around fmugen's steps (no profiler: cProfile made `written_globals` look 3.5× slower than it is); [torch_retry.py](torch_retry.py), [scan_after_import.py](scan_after_import.py) and [stat_per_package.py](stat_per_package.py) are described below.

**The machine's speed changed during the session.** The same runs took 2–3× longer in the second half (importing `sklearn.ensemble` alone: 4.5 s, later 8.9 s), while the repo's `.venv` kept its speed. The cause wasn't found. So only runs made next to each other (interleaved) are compared below, and absolute times are given for the quiet first round.

### `init --probe`, quiet round (two runs each)

| Model | `init --probe` | Importing the model's package alone | fmugen's import of it | Examples (≈ the scan) | Package scan | `globals` (modules parsed) | Probe calls |
|---|---|---|---|---|---|---|---|
| `linear` | 9.7–9.8 s | 4.1 s | 3.5 s | 1.0 s | 1.0 s | **3.5 s (783)** | 1.2 s (1) |
| `cartpole` | 3.0–3.1 s | 0.6 s | 0.4 s | 0.9 s | 0.9 s | 0.2 s (61) | 1.2 s (2) |
| `solarpos` | 4.4 s | 3.4 s | 3.0 s | 0.9 s | 0.9 s | 0.0 s (3) | 0.0 s (1) |
| `ridge` | 5.9–6.0 s | 4.3 s | 3.8 s | 0.9 s | 0.9 s | 0.5 s (132) | 0.1 s (6) |
| `forest` | 10.7–10.8 s | 4.5 s | 8.0 s | 1.1 s | 1.0 s | 0.6 s (150) | **3.8 s (30)** |

The columns overlap: the scan runs inside the examples step, and "fmugen's import" also counts what the probe imports. In the second, slow round the same models took 30.0, 5.7, 13.0, 18.4 and 34.0 s.

Apart from importing the model's own libraries (not fmugen's to save), three things cost time.

**1. `globals` reads all of torch.** `_add_globals` follows every module of the entry's package that the entry module reaches and parses each one's source with `static.written_globals`: 783 of torch's modules, 3.5 s (12.5 s under cProfile, almost all `ast.walk`). The result is also wrong for a model: the `linear` config lists 51 of torch's internal flags as `[model] globals` (`torch.cuda:_initialized`, `torch.autograd.forward_ad:_current_level`, …), state of the library, not of the model, saved and restored with every FMU state.

**2. The probe imports torch for models that don't use it.** When a probe call with an array input fails, `_probe_attempt` retries it with each converter of `ARRAY_RETRIES = ("numpy", "torch:tensor")` (`probe/_calling.py`) whose module is installed, so with torch installed every such retry imports torch (`forest`: its `predict` fails before `fit`, 30 probe calls). [torch_retry.py](torch_retry.py) compares that with the torch retry made only when torch is already imported, three times, interleaved (slow round):

| `forest` | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| as is | 33.6 s | 31.4 s | 33.5 s |
| torch retry only if torch is imported | 16.8 s | 17.4 s | 16.7 s |

It also changes the config: with torch imported, the `forest` config warns "--compile won't work: the model's libraries read their own source files by path (torch/_native/ops/topk/aot.py)", torch's own reads, not scikit-learn's. The `linear` config (torch imported by the model) is the same either way.

**3. The package scan is slower in a process that has imported a big library.** [scan_after_import.py](scan_after_import.py): `packages_distributions()` took 0.97 s after importing nothing, 3.5–3.7 s after `torch` or `sklearn.ensemble` (later 4.8–5.0 s), with the GC on or off and with `OMP_NUM_THREADS=1`: not garbage collection or thread pools. The same number of `stat` calls (about 5 800) each took about 4× longer. [stat_per_package.py](stat_per_package.py) then gave 0.13–0.6 ms per `stat` depending on the moment rather than on the import, so it looks like the file system being slow at times, not something fmugen does. A scan built without an existence check on every file (see "What's left after it" above) would avoid it either way.

`init` of `torch.nn:Linear` first failed before any of this: `static.written_globals` called `inspect.getsource` on `torch.classes`, a module whose `__getattr__` raises `RuntimeError` (not `AttributeError`) for `__wrapped__`, which `inspect` asks for. Fixed: `written_globals` now treats any exception there like an unreadable source (no globals); test `test_a_module_whose_getattr_raises_has_no_written_globals`.

### `build`

Default builds, median of 3, interleaved (slow round):

| Config | Build | FMU |
|---|---|---|
| `linear` | 18.5 s | 6.8 MB |
| `cartpole` | 5.4 s | 6.8 MB |
| `solarpos` | 13.6 s | 6.8 MB |
| `ridge` | 18.4 s | 6.8 MB |
| `forest` | 19.4 s | 6.8 MB |

`_prepare` imports the model (so a build fails where the FMU would): `torch.nn` 9.5 s, `sklearn.linear_model` 10.0 s, `pvlib.solarposition` 6.5 s, `gymnasium` 0.4 s in those runs. `normalize` and copying the sources took under 0.05 s. The rest is the fixed cost of the `tank` build above: `unifmu generate` 0.8 s, `stamp_guid` 1.3 s, zipping 1.9 s.

`--vendor`, the first run downloading the wheels, then again:

| Config | First | Again | FMU | Largest wheels |
|---|---|---|---|---|
| `ridge` | 60.7 s | 27.7 s | 62 MB | scipy 34 MB, numpy 11 MB, scikit-learn 7 MB |
| `linear` | 43.1 s | 29.5 s | 135 MB | torch 120 MB, sympy 6 MB, networkx 2 MB |

`linear --vendor`, again (27.9 s): `_prepare` 9.5 s, `uvx pip wheel` 4.3 s, `stamp_guid` 2.4 s, **`_write_zip` 10.3 s**. A wheel is already a zip archive; the same folder zipped with the `.whl` files stored instead of deflated took 4.0 s, for 138 MB instead of 135 MB.

`--compile pyinstaller` of `ridge`:

- From the benchmark venv it fails after 9 s: `FileNotFoundError: [WinError 206]` (file name too long). The PyInstaller command is 44 165 characters, over Windows' 32 767 limit: 42 738 of them are 194 `--add-binary` arguments from `_loaded_libraries`, one absolute path per `.pyd` in scikit-learn, scipy and numpy (`distribute.py`). The venv's path is long here (about 140 characters to `site-packages`), but the length grows with every native file and every character of path, so a shorter project path with more native packages (torch has hundreds of `.pyd`) can hit it too. A `.spec` file, or a hook in the `--additional-hooks-dir` that fmugen already writes, would keep them off the command line.
- From a venv at a short path (`%TEMP%\fb`, removed afterwards) it built in 257 s, 77 MB.

### What would help, by size

| Change | Saves | Where | Status |
|---|---|---|---|
| Probe's `torch:tensor` retry only when torch is already imported | 13–17 s (`forest`, slow round), and a wrong warning | `probe/_calling.py` | done |
| `globals`: don't parse an installed library's internals | 3.5 s for torch, and 51 wrong `globals` | `interface._add_globals` | done |
| `.whl` stored, not deflated, in the zip | 6 s for a torch `--vendor` | `builder._write_zip` | done |
| PyInstaller `--add-binary` list off the command line | a failure, not time | `distribute.compile_fmu` | done |
| UniFMU template kept between runs | 1.7 s per build | `builder._template` (see `build` above) | done |
| Package scan without existence checks | 0.6–4 s per `init` | `static.packages_distributions` | not done |

## The fixes

What changed:

- **`static.written_globals`** treats any exception from `inspect.getsource` as no source (torch's `torch.classes` raises `RuntimeError`), so `init` of `torch.nn:Linear` no longer fails.
- **Probe** (`probe/_calling.py`, `_array_retries`): a failing call with array inputs is retried as numpy arrays when numpy is installed, and as `torch:tensor` only when torch is already imported (by the model).
- **`[model] globals`** (`interface._add_globals`): in an installed library, a package's `__init__` other than the entry isn't followed, and private names (`_x`) are left out. The user's own project keeps both. pythermalcomfort still gets `PRE_SHIV` (checked with 4.8.1); the torch config gets none instead of 51. The tested tellurium configs listed two private plotting settings (`__save_plots_to_pdf`, `__default_plotting_engine`); a new `init` leaves them out.
- **`--compile pyinstaller`** (`distribute._binaries_hook`): the libraries `_loaded_libraries` finds go into `hook-fmugen_runtime.py` in the hooks folder, not on the command line as `--add-binary`. `ridge` now compiles from the long-path venv (928 s, 265 MB, while the full test suite ran; 265 MB rather than 77 MB because this venv also has torch and llvmlite, which PyInstaller collected); all 194 libraries are in the FMU, and it instantiates and initializes in FMPy. Its first step fails, as the config says it will: `predict()` needs the `fit()` the config leaves commented out for the user's data.
- **`.fmu` zip** (`builder._write_zip`): `.whl` and `.zip` files are stored, not deflated again.
- **UniFMU template** (`builder._template`): kept in `FMUGEN_TEMPLATE_DIR`, default `%LOCALAPPDATA%\fmugen\unifmu` (`~/Library/Caches/fmugen/unifmu`, `~/.cache/fmugen/unifmu`), keyed on the executable's path, size and modification time and the `generate` arguments, made in a temporary folder and renamed into place; one cut short is made again. Documented in [cli.md](../../cli.md).

Tests: `test_a_module_whose_getattr_raises_has_no_written_globals`, `test_globals_of_an_installed_package_skip_its_other_packages_and_private_names` (fails on the old code), `test_the_users_own_modules_keep_their_private_globals`, `test_the_probe_retries_with_torch_tensors_only_when_torch_is_already_imported`, `test_libraries_loaded_by_path_go_in_a_hook_not_the_command_line`, `test_wheels_are_stored_in_the_fmu_not_deflated_again`, `test_the_unifmu_template_is_kept_between_builds`. Full suite: 539 passed, 11 skipped (fmusim not on PATH).

### Before and after

[old_new.py](old_new.py) `OLD_SRC NEW_SRC ROUNDS`, from `models/heavy/` (plus `models/tank/`): each case with the commit before the fixes (`7c21ff4`, checked out apart and put first on `PYTHONPATH`), then with the fixes, in turn, in the heavy venv. The template cache was already made (by the test suite) for the new runs.

| Case | Before | After |
|---|---|---|
| `init forest --probe` | 13.2, 13.2, 13.3 s (one more run: 77.7 s) | 8.4–8.6 s (5 runs) |
| `init ridge --probe` | 7.9, 8.4 s | 7.5, 7.7 s |
| `init cartpole --probe` | 4.8, 6.5 s | 4.3, 4.4 s |
| `init linear --probe` | fails (`torch.classes`) | 8.2, 8.6 s; `globals` step 0.5 s for 106 modules (was 3.5 s for 783) |
| `build tank` | 3.7, 3.9 s | 1.6, 1.7 s |
| `build linear --vendor` (wheels cached) | 15.9, 19.5 s | 10.3, 10.4 s |

The package scan of `init` is still there (1–2 s here, more when the file system is slow): it's the next thing to take out.
