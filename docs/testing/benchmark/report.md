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
