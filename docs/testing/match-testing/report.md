# FMU vs direct run: fmugen `dev` (`adeebf3`)

Each model from the "Repository examples", "Python libraries" and "Neural networks" tables of `docs/testing/tested-models.md` was turned into FMUs by fmugen (`init --probe`, plain `init`, plus the repository config or the hand config where there is one). Each FMU was simulated with FMPy and compared, output by output and step by step, with the model called directly in plain Python. The direct run follows the config but doesn't use fmugen. The table of every config is in [results.csv](results.csv).

## Summary

**No mismatch.** Wherever both the FMU and the direct run produced values, they agree.

| | |
|---|---|
| Models | 120: every entry of the test plan's list (the FMI 3 repo configs of kalman and sampled_pid are the entries `ex_kalman_repo3` / `ex_sampled_pid_repo3`) |
| Configs tried | 253 |
| Configs compared value by value | 168, **230 341 values** |
| Largest relative difference | **1.3 × 10⁻¹⁶** (`examples/sampled_pid`, FMI 3); in every other matching config the values are identical |

Per model (a model counts as **match** if at least one config matches and none mismatches):

| Verdict | Models |
|---|---|
| **match** | **109** |
| mismatch | **0** |
| not comparable | 7: ambiance, impedance, pandapower, river, seirsplus, commpy, nashpy |
| FMU fails | 3: pyet, pyquaternion, simpy |
| direct fails | 0 |
| no config | 1: sgp4 |

Per config: 161 match, 0 mismatch, 33 not comparable, 53 FMU fails, 6 no config.

The 11 models that don't match are exactly the failures `tested-models.md` already lists: 10 of its 16 (*no useful outputs*: the FMU has no outputs, or no outputs a direct run can be compared with; *not supported*: pyet, pyquaternion, simpy, sgp4), and nashpy (a generator result). psychrolib, the 16th, matches here (see [Differences from the docs](#differences-from-tested-modelsmd)).

## Mismatches

**None.** Over the run, 3 apparent mismatches turned up. All 3 were in the comparison harness, not in fmugen, and were fixed and rerun:

| What showed up | Cause | Fix |
|---|---|---|
| pyfluids: `name` FMU 135, direct `'Water'`; `phase` FMU 1, direct 0 | The FMU reports an Enumeration as its 1-based item number, as `config.md` says. The direct runner reported the member's `.value`. | Direct runner maps enum members to their item number → match. |
| pyfluids: `coolprop_backend` FMU `"b'HEOS'"`, direct `'HEOS'` | FMPy's `getString` returns `bytes`; the harness stored their repr. | Decode → match. |
| tellurium (probe): `seed` FMU 1792578127752128, direct 1792578125591609 (rel. diff 1.2 × 10⁻⁹) | `seed` is RoadRunner's random seed, taken from the clock in microseconds when the model is loaded: every load, in or out of the FMU, gives a new one. The two direct runs differed too, but only by 5 × 10⁻¹⁰, below the tolerance, so the randomness check missed it. | Any output whose two direct runs differ at all is now excluded as random. `seed` is left out; the other 2 outputs match exactly. |

## Not comparable

- **The FMU has no outputs** (26 configs): ambiance, cantera (`init` configs; its hand config matches), impedance, pandapower, river, seirsplus, commpy, nashpy, PyTCI (`init --probe`; its hand config matches). These are the "no useful outputs" models of `tested-models.md`: `init` finds no outputs it can type. Also some plain configs whose model is a factory or returns arrays, so `init` without `--probe` lists no outputs (ex_kalman, Madgwick, sb3_ppo, simglucose, electricpy, obspy, opencv, pedon, pyxirr, scipysignal, shapely, skimage, transforms3d). The `--probe` config of each of these matches.
- **Random by design** (7 configs): the unseeded `init --probe` configs of CartPoleEnv, PendulumEnv and HighwayEnv (`reset()` without a seed), mesa BoltzmannWealth (unseeded scenario), and surfaces (a scikit-learn model with no `random_state`), plus surfaces' plain config and TCLabModel: its `tnow` and the temperatures follow the wall clock, so it can't be seeded. **Each seedable one was also run seeded and matches**: gym with `setup = ["reset(seed=0)"]`, mesa with `scenario = { call = "…:BoltzmannScenario(rng=42)" }`, surfaces with `setup = ["numpy.random:seed(0)"]` (these are the `seeded` rows).

## FMU fails

Every FMU failure is one `tested-models.md` already describes, or a plain `init` config failing where `--probe` succeeds:

- **pyet, pyquaternion, simpy** (both configs): the documented failures: pyet needs a pandas Series (`'float' object has no attribute 'time'`); pyquaternion's `*args` constructor gets the step's `--start` values (`Got: 2, Expected: 4`); simpy's `until` isn't bound to time.
- **PropsSI, `fluids.units` ×2, metpy** (`init` configs): documented: no signature / no pint quantities (`1.0 has no quantity`) / metpy's unit check. **Their hand configs match.**
- **pyloudnorm** (`init --probe`): documented (`blockwise_loudness has 0 values, expected 17`); its plain `init` config matches.
- **plain `init` configs (38)**: the result is an array, a tuple, a DataFrame, a dict or a string that code-only `init` types as one Real, and the FMU stops with fmugen's own message, e.g. `y is an array of shape [2], not a single value: … fmugen init --probe finds them`. The `--probe` config of each matches. The FMU's error is in the `notes` column of results.csv.

## Differences from `tested-models.md`

Not mismatches between FMU and direct run, but different from what the docs say:

- **psychrolib `GetHumRatioFromRelHum` matches**: docs list it as a failure (a numba ufunc rejects keyword arguments). With psychrolib 2.5.0 the `init --probe` config builds, simulates and matches, so the failure no longer reproduces.
- **Plain gym configs fail at initialization**: `np_random_seed` (a 128-bit seed) is typed Integer by code-only `init`, and the FMU stops (`… is out of range for Integer`). The `--probe` configs don't have this problem.
- **`fluids.units` hand configs: the `unit` must be written the way pint reads it**: `convert = "pint"` passes `unit` to pint as written, and pint rejects the Modelica spelling `config.md` uses (`kg/m3`, `Pa.s`, `m/s2`: `'m3' is not defined`). With `kg/m**3`, `Pa*s`, `m/s**2` both match (Reynolds 76401.3, head 1019.7 cm, the documented values). Those strings aren't Modelica units, so fmugen writes them without a `<BaseUnit>`.
- **`attr:` paths index lists but not numpy arrays**: `from = "attr:t_core.0"` on JOS3 fails (`'numpy.ndarray' object has no attribute '0'`), while the docs' OpenHENS example indexes a list (`attr:_best_solns.0.case.TAC`). The JOS3 hand config here reads the whole `t_core` array instead.

## How it was run

- **Where**: WSL 2 (Ubuntu, Python 3.13 from uv), not Windows. On this PC, Windows Smart App Control is in enforcement mode and blocks unsigned binaries with no reputation: CoolProp's, rdkit's and openturns' compiled extensions every time, and UniFMU's `unifmu.dll` and the uv venv launchers sometimes. The setting wasn't changed. Smoke tests on Windows (friction_factor, simple_pid, psychrometry, both FMI 3 repo configs) matched before switching.
- **Tools**: fmugen editable from the `dev` clone (`adeebf3`), FMPy 0.3.32, UniFMU 0.14.0 (Linux; Windows build for the smoke tests), one uv venv per library at `~/cmp/v/<name>`. Library versions: latest that installs on Python 3.13 (2026-10-21), in the `library version` column of results.csv.
- **Configs**: `init --probe` and plain `init` with the test plan's arguments; the committed `examples/*/fmugen.toml`; and the hand configs in `harness/hand/`: PropsSI, PyTCI and cantera as published in the docs (cantera built as FMI 2: nothing in it needs FMI 3), metpy's published tables plus `[model] entry`, and `fluids.units` ×2 and JOS3 written from the docs' description (units as above; JOS3 at 10 °C with the `attr:` inputs, the `options.update` setup and the `PRE_SHIV` global).
- **FMU**: `fmugen build <toml> --capture-output --fmi 2` (no `--compile`), `fmpy.simulate_fmu(stop_time=3, output_interval=1)` with an instance the harness creates and frees itself, after which it kills the UniFMU backends this simulation started (not all `main.py` processes, so runs in parallel don't kill each other's backends). `examples/rc_building`'s config has `fixed_step = 3600`: simulated with 3600 s steps. The FMI 3 configs `examples/kalman` and `examples/sampled_pid` were driven like their `run.py` (same seed, clock ticks and plant) and compared at every communication point: 350 and 82 values.
- **Direct run** (`harness/direct.py`): written from `docs/config.md`, not from fmugen's runtime, and doesn't import fmugen. It reads `[model]` (entry, kind, create, call, setup, constants, call_constants, init_call, cwd), `[parameters]`, `[inputs]`, `[states]`, `[time]` and each output's `from`; builds the object or calls the factory, applies `attr:`/`call:` bindings, makes the call at initialization when `init_call` is set (the FMU's t = 0 row), then makes 3 steps with the start values, reads every output by its `from` path and updates states from `next`. It ran twice per config, in separate processes, to detect random models.
- **Compared**: every FMU output at t = 0, 1, 2, 3 (t = 0 only where the model produces it there); numbers exactly or with relative difference ≤ 10⁻⁹, strings exactly (from t = 1: FMPy's result doesn't record strings, so they're read in `step_finished`), booleans as numbers.
- **Undeclared dependencies installed by hand**, besides the test plan's: `networkx` (mesa), `pandas` (impedance), `setuptools<81` (simglucose: `pkg_resources`), `surfaces[ml]` (scikit-learn); tellurium needed `LD_LIBRARY_PATH` set to uv Python's `lib` (`libpython3.13.so`).
- **Timeouts**: 15 min per init, build, simulation and direct run, with the whole process tree killed. None was hit.

## What couldn't be run

Nothing in the test plan was skipped. sgp4 has no config: `init` stops on C built-ins with no signature (documented; no hand config is published).

## Files

- `results.csv`: one row per model and config; `results.json`: the same, with every FMU and direct value.
- `harness/`: the scripts (`run_all.py` driver, `direct.py`, `sim.py`, `fmi3_examples.py`, `models.py` with the test plan's model list, `hand/` configs, `wsl_*.sh`). Work files (configs, FMUs, logs) are in WSL at `~/cmp/w/m/<model>/`.
