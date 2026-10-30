# FMU vs direct run, rerun: fmugen `dev` at `6dfc18c`

The models of `docs/testing/tested-models.md`, rebuilt into FMUs with the current `dev` branch and compared output by output, step by step, with the model run directly in plain Python (no fmugen). The last run (`docs/match-testing/`, at `adeebf3`) is the baseline. Every config is in [results.csv](results.csv); every FMU and direct value is in [results.json](results.json).

- **Part A**: the 13 models that failed or gave no useful outputs, re-tested.
- **Part B**: a hand config for each of the 10 models that simulate without useful outputs.
- **Part C**: every other model, as a regression check, plus the 13 GEKKO models the last run didn't cover.

## Summary

**No mismatch caused by the code changes; one regression by definition** (PyTCI's hand config at t = 0, from the intended setter change: [below](#regressions)). Wherever both the FMU and the direct run produced values, they agree, apart from that one point.

| | |
|---|---|
| Commit | `6dfc18c` (dev): `1a3a36c` setters once per call, `1ac9da6` `call_on_class`, and the other changes under test |
| Models | **134**: the 120 of the last run, psychrolib again with numba installed, and the 13 GEKKO models |
| Configs tried | 295 |
| Configs compared value by value | 194, **236,621 values** |
| Largest relative difference among matches | **1.3 × 10⁻¹⁶** (`examples/sampled_pid`, FMI 3, as before); every other match is identical |
| Direct values | 193 configs reused from the last run (config unchanged; each also rerun as a check, and all agree except HighwayEnv's unseeded probe, which is random), 34 new runs |

Per model (a model counts as **match** if at least one config matches and none mismatches):

| Verdict | Models (134) |
|---|---|
| **match** | **130** |
| mismatch | 1: pytci (t = 0 only, see below) |
| not comparable | 1: nashpy |
| FMU fails | 1: pyet |
| no config | 1: sgp4 |

Per config: 188 match, 1 mismatch, 38 not comparable, 60 FMU fails, 8 no config.

Per change (each config against its verdict in the last run's `results.csv`):

| Change | Configs |
|---|---|
| unchanged | 249 |
| **fixed** | 3: pyloudnorm, pyquaternion, simpy (`--probe`) |
| **regression** | 1: pytci (hand config) |
| new (not in the last run) | 42: the 13 GEKKO models (29 configs), psychrolib with numba (2), the 10 Part B hand configs, JOS3's `attr:t_core.0` config |

"Match" here is FMU = direct. For four Part A models (pymunk, pybamm, tellurium, uncertainties) the `--probe` config matches but only outputs settings or the object itself: by the test plan's rule that is still no useful output, so their hand configs are what counts ([Part B](#part-b-hand-configs)).

**Useful outputs now:** 8 of the 10 Part B models (ambiance, pandapower, seirsplus, impedance, commpy, river, tellurium, uncertainties) with a hand config; pyquaternion, simpy and pyloudnorm with `init --probe` (fixed); all 13 GEKKO models (CFTOC and OpenHENS with a hand config).

**Still need an fmugen feature:** pymunk (indexing a `dict_keys` view in a path), pybamm (item access by a string key on an object that isn't a Mapping), nashpy (a generator result), pyet (a pandas Series with a date index as input), sgp4 (C built-ins with no signature: only a hand config with `pos:` bindings, not written).

## Regressions

### pytci, hand config: t = 0 differs (intended change)

| t | x1 FMU | x1 direct (test plan loop) |
|---|---|---|
| 0 | 0.12531328320802004 | 0.0 |
| 1 | 0.12474310776942354 | 0.12474310776942354 |
| 2 | 0.24891885335561123 | 0.24891885335561123 |
| 3 | 0.37253003602056656 | 0.37253003602056656 |

(x2, x3 and xeo are equal at all four points: the bolus enters x1 first. x1 at t = 0 is the only value of the 16 that differs.)

- **Cause:** commit `1a3a36c` ("Run call: setters once per model call"). `src/fmugen/templates/fmugen_runtime/_engine.py:359` runs `_apply_attributes(clock=None, initial=True)` right after construction, so `give_drug(2.0)` is called at initialization. The first `doStep` then skips it, since the value hasn't changed (`_engine.py:1483`–`1488`, `_setter_fresh`). The t = 0 outputs therefore already include the first dose: x1 = 2 mg / V1 = 2 / (0.228 L/kg × 70 kg) = 0.1253.
- **Comparison:** the test plan's direct loop (`give_drug(dose); wait_time(time_seconds)` per step, no dose at initialization) gives that dose at the start of the first step. From t = 1 both sequences are the same: 3 doses in 3 steps, no extra one (the "61 doses for 60 steps" bug is gone). A direct run in the FMU's order (dose after construction, then wait; dose, wait; dose, wait) gives the FMU's x1 at all four points (checked).
- **Before:** the last run matched only because its direct runner also gave a dose after construction and again before every step, the same double first dose the old FMU gave.
- **Is it a bug?** It's how the new code is meant to work (the dose happens at time 0). It only differs from the test plan's loop in what the FMU reports at t = 0. If t = 0 should show the state before any dose, initialization must not call `call:` setters (or the outputs must be read before it does).

No other config that matched before fails or changed its outputs. Configs that changed (`config-diffs/`):
- **pyquaternion** `--probe`: `axis` now passes by keyword (`axis = { call = … }`) and outputs `y` with `dimensions = [3]`, plus the quaternion's properties. Previously it failed.
- **simpy** `--probe` and plain: `until` is now `[time] until = "end_time"`; previously a fixed input.
- **mendeleev** `--probe`: the same outputs, in a different order.

## Part A: the 13 models, rebuilt and compared

| Model | `--probe` | plain `init` | Outputs | Now |
|---|---|---|---|---|
| sgp4 | no config | no config | none | unchanged: `init` stops on `Satrec.twoline2rv`/`sgp4`, C built-ins with no signature |
| pyet | FMU fails | FMU fails | | unchanged: `'float' object has no attribute 'time'` (pm_fao56 needs a pandas Series with a date index) |
| nashpy | no outputs | FMU fails (`float() … not 'generator'`) | | unchanged |
| ambiance | no outputs | no outputs | | hand config: works |
| pandapower | no outputs | no outputs | | hand config: works (another entry) |
| pymunk | match (36 values) | match | settings only (`collision_bias`, `damping`, `iterations`, …) | hand config: not possible |
| seirsplus | no outputs | no outputs | | hand config: works |
| impedance | no outputs | FMU fails (array as one Real) | | hand config: works |
| commpy | no outputs | FMU fails (array as one Real) | | hand config: works |
| pybamm | match (11) | FMU fails (`not 'Solution'`) | solver flags only (`hermite_interpolation`, `observable`, `termination`) | hand config: not possible |
| river | no outputs | FMU fails (`max_cum_l1`) | | hand config: works |
| tellurium | match (4; `seed` excluded, clock-based) | match (3) | the step's return value (the time reached) and `seed` | hand config: works |
| uncertainties | match (4) | FMU fails | `s`, the `ufloat` itself (its string) | hand config: works |

All of these are the same as in the last run. The direct values for the `--probe` and plain configs were reused (configs unchanged). "No outputs" means the FMU has no output variables, so it's not comparable.

## Part B: hand configs

Each one starts from the model's `probe.toml` and uses only config features; no model was changed. They are also in [configs/](configs/). Direct values are always a new run.

| Model | Hand config | Verdict | Values compared |
|---|---|---|---|
| ambiance | works | match | 35 |
| pandapower | works (entry `run_contingency`) | match | 48 |
| pymunk | **not possible** | FMU fails (as expected) | |
| seirsplus | works | match | 24 |
| impedance | works | match | 32 |
| commpy | works | match | 16 |
| pybamm | **not possible** | FMU fails (as expected) | |
| river | works | match | 10 |
| tellurium | works | match | 12 |
| uncertainties | works | match | 8 |

### ambiance

**Works.** Each property of `Atmosphere(h)` is a 1-element numpy array; `.0` reads its element, now that paths index numpy arrays. At 1000 m: T 281.6510223716947 K, p 89876.27760234232 Pa, ρ 1.1116596736996904 kg/m³, a 336.43458210225776 m/s, layer `troposphere` (a String; FMPy records strings from t = 1). FMU = direct, all 35 values.

```toml
# ambiance: the `init --probe` config, with the atmosphere's properties as outputs.
# Each property of Atmosphere(h) is a 1-element numpy array: `.0` reads its element.
[model]
entry = "ambiance:Atmosphere"
kind = "function"
requirements = ["ambiance==1.3.1"]

[inputs]
h = { start = 1000.0, unit = "m" }
check_bounds = { start = true }

[outputs]
temperature = { from = "return:temperature.0", unit = "K" }
pressure = { from = "return:pressure.0", unit = "Pa" }
density = { from = "return:density.0", unit = "kg/m3" }
speed_of_sound = { from = "return:speed_of_sound.0", unit = "m/s" }
dynamic_viscosity = { from = "return:dynamic_viscosity.0", unit = "Pa.s" }
kinematic_viscosity = { from = "return:kinematic_viscosity.0", unit = "m2/s" }
thermal_conductivity = { from = "return:thermal_conductivity.0", unit = "W/(m.K)" }
grav_accel = { from = "return:grav_accel.0", unit = "m/s2" }
layer_name = { from = "return:layer_name.0", type = "String" }
```

### pandapower

**Works, with another entry.** `runpp(net)` returns None and writes its results onto `net`. A function entry's outputs can only read its return value, and `pandapowerNet` is a plain dict with no method to run as the step, so `runpp` itself can't be read back. `pandapower.contingency:run_contingency(net, nminus1_cases={})` runs the same `runpp` on `net` (its N-0 case, with runpp's default options) and returns the results. Bus voltages 1.02, 1.0208295, 1.0208295, 1.0245624, 1.0245624, 1.03, 1.0232053 pu, line loadings 8.598, 46.266, 0.059, 20.160 %, transformer 26.547 %: equal to `net.res_bus.vm_pu` / `net.res_line.loading_percent` after `runpp` (checked directly). FMU = direct, 48 values. Its other keys (`cause_index`) come from `np.empty_like` and are uninitialised, so they aren't read.

```toml
# pandapower: runpp(net) returns None and writes its results onto net, which a function entry
# can't read back. run_contingency(net, nminus1_cases={}) runs the same runpp on net (its N-0
# case, with runpp's default options) and returns the results as a dict of arrays, so it is the
# entry here. Each step runs the power flow again on the same net.
[model]
entry = "pandapower.contingency:run_contingency"
requirements = ["pandapower==3.5.6"]

[model.constants]
net = { call = "pandapower.networks:example_simple()" }
nminus1_cases = {}

[inputs]
write_to_net = { start = true }

[outputs]
vm_pu = { from = "return:bus.vm_pu", dimensions = [7] }                     # net.res_bus.vm_pu
line_loading_percent = { from = "return:line.loading_percent", dimensions = [4] }   # net.res_line.loading_percent
trafo_loading_percent = { from = "return:trafo.loading_percent.0" }          # net.res_trafo.loading_percent
```

### pymunk

**Not possible with a config.** `setup` can add gravity and a body (the table form with `{ call = "pymunk:Body(1.0, 10.0)" }` as the argument), but nothing can read it back: `Space.bodies`, `shapes` and `constraints` are `dict_keys` views, which a path can't index. The FMU stops: `x: cannot read attribute 'bodies.0.position.x' (AttributeError("'dict_keys' object has no attribute '0'"))`. A shape can't bring its body either (`add(shape)` asserts the body is already added, and two `setup` arguments can't refer to the same new object). **fmugen feature needed:** a path part that indexes any iterable (`list(obj)[i]` when `obj[i]` and `getattr` fail), or a way to name an object a `setup` call creates so that outputs can read it.

```toml
# pymunk: the `init --probe` config, with gravity and one falling body added in `setup`, and
# the body's position and velocity as outputs. Space.bodies is a dict_keys view, which a path
# can't index, so `attr:bodies.0...` is expected to fail.
[model]
entry = "pymunk:Space"
call = "step"
requirements = ["pymunk==7.3.1"]
setup = [
    { call = "__setattr__", args = ["gravity", { python = "(0.0, -9.81)" }] },
    { call = "add", args = [{ call = "pymunk:Body(1.0, 10.0)" }] },
]

[parameters]
threaded = { start = false }

[inputs]
dt = { start = 0.01 }

[outputs]
x = { from = "attr:bodies.0.position.x", unit = "m" }
y = { from = "attr:bodies.0.position.y", unit = "m" }
vy = { from = "attr:bodies.0.velocity.y", unit = "m/s" }
```

### seirsplus

**Works.** The latest compartment counts, `attr:numS.-1` and so on, instead of the growing arrays. `dt` is a fixed input 0.1 instead of the probe's `[time] dt = "step_size"`: `run_epoch(runtime, dt)` evaluates at `numpy.arange(t, t + runtime, dt)`, so with dt = runtime = 1 it keeps only the start point and `t` never advances (the model's own behaviour, checked directly). Each epoch then advances t by 0.9: S 99900 → 99887.15 → 99874.85 → 99862.85, I 100 → 94.81 → 91.71 → 90.22. FMU = direct, 24 values.

```toml
# seirsplus: the `init --probe` config, reading the latest compartment counts (the last element
# of each growing array) instead of the arrays.
# `dt` is a fixed 0.1 instead of the step size: run_epoch evaluates at arange(t, t + runtime, dt),
# so with dt = runtime = 1 it keeps only the start point and t never advances.
[model]
entry = "seirsplus.models:SEIRSModel"
call = "run_epoch"
requirements = ["seirsplus==1.0.9"]

[parameters]
initN = { start = 100000 }
beta = { start = 0.147 }
sigma = { start = 0.192 }
gamma = { start = 0.0714 }
xi = { start = 0 }
mu_I = { start = 0 }
mu_0 = { start = 0 }
nu = { start = 0 }
p = { start = 0 }
theta_E = { start = 0 }
theta_I = { start = 0 }
psi_E = { start = 0 }
psi_I = { start = 0 }
q = { start = 0 }
initE = { start = 0 }
initI = { start = 100 }
initQ_E = { start = 0 }
initQ_I = { start = 0 }
initR = { start = 0 }
initF = { start = 0 }

[inputs]
runtime = { start = 1.0 }
dt = { start = 0.1 }

[outputs]
t = { from = "attr:t" }
S = { from = "attr:numS.-1" }
E = { from = "attr:numE.-1" }
I = { from = "attr:numI.-1" }
R = { from = "attr:numR.-1" }
F = { from = "attr:numF.-1" }
```

### impedance

**Works.** `predict()` returns a complex array; `return:<i>.real` / `.imag` reads each element. Z(1 Hz) = 0.015019771 − 3.578×10⁻⁵ j Ω … Z(1 kHz) = 0.010459917 − 0.001445114 j Ω. FMU = direct, 32 values.

```toml
# impedance: the `init --probe` config, with the predicted impedance at each frequency as outputs:
# predict() returns a complex array, read as `.real` and `.imag` per element.
[model]
entry = "impedance.models.circuits:Randles"
call = "predict"
init_call = true
requirements = ["impedance==1.7.1"]

[parameters]
CPE = { start = false }
initial_guess = { dimensions = [5], start = [0.01, 0.005, 0.001, 200.0, 0.1] }

[inputs]
frequencies = { dimensions = [4], start = [1.0, 10.0, 100.0, 1000.0], unit = "Hz" }
use_initial = { start = false }

[outputs]
Z1_re = { from = "return:0.real", unit = "Ohm" }
Z1_im = { from = "return:0.imag", unit = "Ohm" }
Z2_re = { from = "return:1.real", unit = "Ohm" }
Z2_im = { from = "return:1.imag", unit = "Ohm" }
Z3_re = { from = "return:2.real", unit = "Ohm" }
Z3_im = { from = "return:2.imag", unit = "Ohm" }
Z4_re = { from = "return:3.real", unit = "Ohm" }
Z4_im = { from = "return:3.imag", unit = "Ohm" }
```

### commpy

**Works.** `modulate([0, 1, 1, 0])` gives two QPSK symbols, −1 + 1j and 1 − 1j, read as `.real` and `.imag`. FMU = direct, 16 values.

```toml
# commpy: the `init --probe` config, with the modulated symbols as outputs: modulate() returns a
# complex array (4 bits -> 2 QPSK symbols), read as `.real` and `.imag` per symbol.
[model]
entry = "commpy.modulation:QAMModem"
call = "modulate"
init_call = true
requirements = ["numpy==2.5.4", "scikit-commpy==0.8.0"]

[model.call_constants]
input_bits = { call = "numpy:array([0,1,1,0])" }

[parameters]
m = { start = 4 }

[outputs]
s1_re = { from = "return:0.real" }
s1_im = { from = "return:0.imag" }
s2_re = { from = "return:1.real" }
s2_im = { from = "return:1.imag" }
```

### pybamm

**Not possible with a config.** `Simulation.step()` returns a `Solution`, and the voltage is only reachable as `solution["Voltage [V]"]` (built on demand by `Solution.__getitem__`). `Solution` isn't a Mapping, so a path part `Voltage [V]` is looked up as an attribute: `voltage: cannot read the return value ['Voltage [V].entries.-1'] (AttributeError("'Solution' object has no attribute 'Voltage [V]'"))`. `solution.data` is a dict, but only of variables already built, and after `step()` it is empty, also with `output_variables=["Voltage [V]"]` and the IDAKLU solver (checked). Directly, the voltage would be 3.760580, 3.748689, 3.737863 V after 100, 200, 300 s. **fmugen feature needed:** item access by key on objects that aren't Mappings but have `__getitem__` (e.g. `get_path` trying `obj[part]` when `getattr` fails, or a `["…"]` path syntax).

```toml
# pybamm: the `init --probe` config, with the terminal voltage read from the solution step()
# returns. The Solution is not a Mapping: "Voltage [V]" is only reachable as solution["Voltage [V]"]
# (built on demand), so the path below is expected to fail. The time of the solution's last point
# is read too.
[model]
entry = "pybamm:Simulation"
call = "step"
init_call = true
requirements = ["pybamm==26.10.1.1"]

[model.constants]
model = { call = "pybamm.lithium_ion:SPM()" }

[parameters]
cache_esoh = { start = true }
experiment_model_mode = { start = "legacy" }

[inputs]
dt = { start = 100.0, unit = "s" }
save = { start = true }

[outputs]
t = { from = "return:t.-1", unit = "s" }
voltage = { from = "return:Voltage [V].entries.-1", unit = "V" }
```

### river

**Works.** `learn_one` returns None; the learned weights are a dict on the model (`attr:weights.a`) and the intercept an attribute. Weights 0.06 → 0.1128 → 0.159264 (a), 0.12 → 0.2256 → 0.318528 (b). FMU = direct, 10 values.

```toml
# river: the `init --probe` config, with the learned weights and intercept as outputs.
# learn_one returns None; the weights are a dict on the model (`weights.a`).
[model]
entry = "river.linear_model:LinearRegression"
call = "learn_one"
requirements = ["river==0.26.1"]

[parameters]
l2 = { start = 0.0 }
l1 = { start = 0.0 }
intercept_init = { start = 0.0 }
intercept_lr = { start = 0.01 }
clip_gradient = { start = 1000000000000.0 }

[inputs]
a = { start = 1.0, to = "arg:x[a]" }
b = { start = 2.0, to = "arg:x[b]" }
y = { start = 3.0 }
w = { start = 1.0 }

[outputs]
weight_a = { from = "attr:weights.a" }
weight_b = { from = "attr:weights.b" }
intercept = { from = "attr:intercept" }
```

### tellurium

**Works.** The species concentrations from the RoadRunner object after `oneStep` (`attr:S1`, `attr:S2`): S1 9.9005 → 9.8020 → 9.7045 → 9.6079 (from the initialization call on), S2 the complement. `seed` (clock-based) is left out. FMU = direct, 12 values. As in the probe config, `currentTime` is a fixed 0.0, so `oneStep` returns 0.1 every step; bind it to `[time] currentTime = "time"` to follow the simulation clock.

```toml
# tellurium: the `init --probe` config, with the species concentrations read from the
# RoadRunner object after oneStep (`r.S1`, `r.S2`). `seed` (RoadRunner's clock-based random seed)
# is left out.
[model]
entry = "tellurium:loada"
call = "oneStep"
init_call = true
globals = ["tellurium.tellurium:__default_plotting_engine", "tellurium.tellurium:__save_plots_to_pdf"]
requirements = ["tellurium==2.2.13.1"]

[parameters]
ant = { start = "S1 -> S2; k1*S1; S1 = 10; k1 = 0.1" }

[inputs]
currentTime = { start = 0.0 }
stepSize = { start = 0.1 }
reset = { start = true }

[outputs]
y = { from = "return" }
S1 = { from = "attr:S1" }
S2 = { from = "attr:S2" }
```

### uncertainties

**Works.** `return:nominal_value` and `return:std_dev` instead of the `ufloat` object: 1.0 and 0.1. FMU = direct, 8 values.

```toml
# uncertainties: the `init --probe` config, reading the ufloat's nominal value and standard
# deviation instead of the object itself.
[model]
entry = "uncertainties:ufloat"
requirements = ["uncertainties==3.2.3"]

[inputs]
nominal_value = { start = 1.0 }
std_dev = { start = 0.1 }

[outputs]
n = { from = "return:nominal_value" }
s = { from = "return:std_dev" }
```


## Part C: regression check

Every model of the last run was rerun the same way (`init --probe`, plain `init`, repository and hand configs; the FMI 3 repository configs of kalman and sampled_pid driven by their `run.py`). Apart from PyTCI (above) and the 3 fixed configs, **every verdict is the one of the last run**: 249 configs unchanged. Notable results:

- **Expected changes, all confirmed:**
  - simpy's `--probe` config runs without `--start until` (bound to the step's end time) and matches: `now` = 1, 2, 3.
  - pyquaternion's `--probe` config passes `axis` and `angle` by keyword and matches: y = (0.87758, 0.47943, 0), (1, 0, 0) rotated by 0.5 rad about z, and the quaternion's properties.
  - pyloudnorm's `--probe` config simulates with `blockwise_loudness` and matches. Its config is the same as last run's; the runtime now skips the array at initialization while it's still empty.
  - psychrolib matches with numba installed (`GetHumRatioFromRelHum_numba`, psychrolib 2.5.0 with numba 0.68.0): the ufunc gets its arguments by position (`to = "pos:N"` in the config).
  - PyTCI's hand config gives one dose per step (above).
- **JOS3 `from = "attr:t_core.0"`** now works: `t_core_head` = the head's core temperature, 37.2795 → 37.2287 °C, equal to element 1 of the whole-array config and to the direct run (16 values).
- **Random by design**, compared seeded as before: the unseeded `--probe` configs of CartPole, Pendulum and HighwayEnv, mesa, surfaces, and TCLab's `--probe` config (wall clock) are not comparable. In this rerun's raw records, CartPole/Pendulum/TCLab showed "match" on their non-random outputs, and HighwayEnv "mismatch": `reset()` places the ego car in a random lane, and the two direct runs happened to get the same lane while the FMU got another (y0[1,3] 0.25 against 0.75). Their seeded configs match.
- **Direct values:** reused for the 193 configs whose config is unchanged (same parsed TOML, `requirements` aside), with the same library versions as the last run (the same venvs). Each was also rerun as a check, and all agree with the reused values except HighwayEnv's unseeded `--probe` config (random).
- Windows smoke test before WSL: friction_factor, simple_pid and psychrometry (`--probe`, plain, repo) and both FMI 3 repository configs match on Windows too.

## GEKKO models

Each source repository at the commit the table names (GEKKO's own example at `master` `b18dcb7`, since the table names none), entry and `init` options as the table gives them, FMI 2, gekko 1.3.2 with its local solver. Renames, wrappers and synthetic data are in copies (`comparison/harness/gekko/prepare.sh`); the published repositories weren't changed. There are no old direct values for these, so all were run directly, and the FMU values were checked against the values the table's notes give.

| Model | `--probe` | plain | Hand | Against the docs' values |
|---|---|---|---|---|
| Distillation column (GEKKO's script wrapped) | match (21) | match (21) | | **same**: condenser 0.785721, 0.837391, 0.867374; reboiler 0.190012, 0.145341, 0.120733 |
| Gym distillation column | random, not comparable; **seeded: match (33)** | FMU fails (and direct fails: `can only concatenate list (not "float") to list`, plain `init` types `Action` wrongly) | | D, B random: D = 1.086 here against the docs' 1.133 (another random draw); seeded with `random:seed(0)`, `numpy.random:seed(0)` before construction |
| Batch bioreactor | match (15) | no outputs | | X 0.045100, 0.045050, 0.045001 g/L over 3 steps, the docs say 0.0451 → 0.0444; same start (0.0451), smaller decrease, cause not found (`time_arr`/`simulation_input` are the notebook's first case; the docs' exact starts aren't given) |
| CGE (renamed `Code/model.py`, pandas 2.3.3) | match (81), **on Windows** | FMU fails (`y is None`: `results()` returns `print(...)`) | | **same**: 27 outputs, first step x3 supply 72.076, price 1.119. On Linux it can't run: its data paths are Windows paths (`'Data\\production_structure.csv'`) |
| CFTOC | no config | no config | match (9) | not checked: the table's starts are elided (`--start calculate_optimal_actuation_x0=…`), and the repo test's values give y = (−1.7×10⁻⁷, −7.7×10⁻⁴, 0), not the docs' (−0.09546, −0.02481, 0) |
| Debutanizer, steady state | match (144) | FMU fails (dataclass result as one Real) | | **same**: C4 0.96999957519, vapor 133.0 kmol/h, duty 0.849722 MW |
| Debutanizer, reflux optimum | match (148) | FMU fails | | **same**: reflux 2.7602061381, purity 0.991, vapor 142.8878332478 |
| Linear MPC (apsis) | match (336) | FMU fails | | 21 + 42 + 21 outputs (time, states, inputs) as the docs say |
| 2×2 FOPDT process | match (18) | no outputs | | **same**: y = (0.3333, 0.0833), (0.6111, 0.1528), (0.8426, 0.2106) |
| MPC for that process | match (36) | no outputs | | runs; with no setpoints the moves stay near 0, as the docs say |
| Ethanol bioreactor (APMonitor's script wrapped) | match (93) | FMU fails | | **same**: Xv 2.9955, Xt 11.1237, S 0.5443, P 72.8812 g/L, T 35.4157 °C, Vl 1485 L |
| NHES (synthetic data) | match (4,892; `solve_time` excluded) | FMU fails (dict result) | | different data: LCOE 52.372103389 $/MWh here, 56.134985404 in the docs; the docs' synthetic series isn't published, so this one is generated from their description (`gekko/make_nhes_data.py`) |
| OpenHENS (+ couenne) | no outputs (as the docs say) | FMU fails | match (3) | **same** as the docs' FMU value: TAC 154853.8586 (docs: 154853.8586 FMU, 154853.8602 direct) |

Details:
- **Distillation column wrapper** (`comparison/harness/gekko/distillation_column.py`): the script's model code is unchanged. The constructor makes its steady-state solve at reflux 0.7; each step makes one interval of its dynamic simulation (`imode 4`) at the step's `reflux_ratio` (default 3.0, `dt` = the step size) and returns the 7 compositions the script plots. A first version kept them as attributes, which `--probe` classes as `[locals]` ("changed by step()"), not outputs. It uses `GEKKO(remote=False)`, the local solver; the script itself uses GEKKO's server.
- **Ethanol wrapper** (`gekko/ethanol_run.py`): `run()` holds the script from APMonitor's page (fetched 2026-10-21) up to its `solve()`, unchanged and without the plots. It then keeps each GEKKO variable's end-of-batch value as an attribute; `--probe` finds 31 outputs.
- **CGE:**
  - Renamed `Code/model.py` → `Code/cge_model.py`. The model reads `Data\\…csv` relative to the repository root, so `init` ran there with the config at the root, next to `Data/`. `init` then writes `sources` and `cwd = "."`.
  - On Linux the FMU can't be made: those paths are file names containing a backslash, and FMPy refuses the zip (`Illegal path resources/fmugen_model/Data\consumption_structure.csv … All slashes must be forward slashes`). So CGE was run on Windows (Smart App Control didn't block gekko's solver there). That is a property of the model (hardcoded Windows paths), not of fmugen.
- **CFTOC:**
  - **`init` stops** on the table's own syntax with a `call:` value: `fmugen: error: --start names that are not arguments of the model: ['calculate_optimal_actuation_x0']`. The same happens with `--convert calculate_optimal_actuation_Xm=numpy`, and for the docs' own example of this feature, radioactivedecay's `--start decay_units=call:builtins:str('y')`.
  - A literal works (`--start decay_units=y`), but the CFTOC method indexes `Xm[i, t]`, which needs a numpy array. A `call:` start becomes a constant stored under the argument's own name (`x0`), so the unused-starts check (`src/fmugen/interface.py:290`–`310`) doesn't match it to the prefixed name; `--convert` is checked against argument names too (`interface.py:287`–`289`).
  - **Not a regression:** `adeebf3` fails the same way (checked).
  - The hand config ([configs/gk_cftoc.toml](configs/gk_cftoc.toml)) passes every argument as a constant / call constant, with the repo test's values: A = I, B = 0.5 I, x0 = (1, 0, 0), Molly's expected path at rest (zeros), N = 50, umax = 0.2, l = 1, the barrier y ≤ 0.5. FMU = direct.
- **Debutanizer, OpenHENS:** both are installable projects (`pyproject.toml` / `setup.py`; OpenHENS's README says `pip install -e .`). Without that, their FMUs fail at instantiation (the backend can't import the package), so each work copy was installed into its venv (`--no-deps`).
- **OpenHENS:**
  - couenne 0.5.8 from the IDAES solver bundle 3.4.2 (Linux), with libgfortran/LAPACK/BLAS unpacked from Ubuntu packages (no root on this PC). About 6.7 s per solve here.
  - **The FMU = direct comparison needs the same files:** the direct run was made in a folder holding only the config's `sources` (the case CSV), which is what `cwd = "."` means. In the repository copy itself, OpenHENS also reads the solutions the repository publishes in its `output_folder` (`examples/results/Four-stream-…/N best.pkl`) and gives TAC 159038.72 instead (2.6 % higher, both direct runs equal). The FMU doesn't contain that folder, since `init` lists only files the model reads by a path in its code or starts.
- **NHES** also needs plotly and seaborn, which it imports without declaring.
- **Plain `init` configs** of GEKKO models fail or give no outputs, as the docs say: GEKKO results are GEKKO variables, which only `--probe` finds.

## How it was run

- **Where:** WSL 2 (Ubuntu, Python 3.13 from uv), as last time. Only CGE ran on Windows (above), plus the Windows smoke test. Windows Smart App Control wasn't changed.
- **Code:** the clone at `6dfc18c` (`~/cmp/repo` in WSL; `repo/` on Windows), fmugen editable in every venv; FMPy 0.3.32, UniFMU 0.14.0.
- **Venvs:** the last run's (`~/cmp/v/<name>`), so the library versions are the same as in `docs/match-testing/results.json`. New: `psychrolib_numba` (psychrolib 2.5.0 + numba), `gekko` (gekko 1.3.2, pandas 2.3.3, seaborn, plotly, gym 0.26.2), `openhens` (gekko 1.3.2 + its `requirements.txt`).
- **Work files:** in `~/cmp/w2/m/<model>/`; the last run's `~/cmp/w` was kept for the diffs.
- **Configs:**
  - `init --probe` and plain `init` with the test plan's arguments; simpy without `--start until=1.0`, as the test plan updates it.
  - The committed `examples/*/fmugen.toml`; the last run's hand configs; the 10 Part B hand configs; JOS3's `attr:t_core.0` variant; CFTOC's and OpenHENS's hand configs.
- **FMU:** `fmugen build <toml> --capture-output --fmi 2` (no `--compile`), then `fmpy.simulate_fmu(stop_time=3, output_interval=1)` with an instance the harness frees itself. After that, it kills only that simulation's process tree. When a simulation failed, it was rerun with `debug_logging=True` and the FMU's log recorded; the error is in the `notes` column.
- **Direct run** (`comparison/harness/direct.py`): written from `docs/config.md` without importing fmugen, and run twice per config in separate processes to detect random models. Updated for this rerun's documented changes:
  - `call:` setters run once before each model call and not after construction;
  - `call_on_class` calls the step method on the class;
  - ufuncs get `to = "pos:N"` from `init`, which the runner already followed.
- **Compared:** every output at t = 0, 1, 2, 3; relative difference ≤ 10⁻⁹, strings exactly. Outputs whose two direct runs differ at all are excluded as random.
- **Timeouts:** 15 min per init, build, simulation and direct run, with the process tree killed. None was hit.

## Files

- [results.csv](results.csv): the last run's columns plus `part`, `direct source`, `hand config` and `change`. [results.json](results.json): the same, with every FMU and direct value (`direct_rows`, and for reused rows `reuse_check` against the new direct run).
- [configs/](configs/): each hand config (`<model>.toml`).
- [config-diffs/](config-diffs/): each config whose text differs from the last run's, comment lines dropped.
- `comparison/harness/`: the scripts. New for this run: `wsl_run2.sh`, `rerun_report.py`, `gekko/`, `hand/<Part B model>/`, `assemble_report.py`, `table2.py`.

## Every model

The verdict of each config, with the number of values compared, what changed against the last run, and where the direct values came from.

| Model | Part | Configs (verdict, values compared) | Change | Direct values |
|---|---|---|---|---|
| ex_psychrometry | C | probe: match (24); plain: match (24); repo: match (24) | unchanged | reused |
| ex_simple_pid | C | probe: match (27); plain: match (3); repo: match (3) | unchanged | reused |
| ex_rc_building | C | probe: match (87); plain: match (72); repo: match (15) | unchanged | reused |
| ex_kalman | C | probe: match (12); plain: not comp. | unchanged | reused |
| ex_sampled_pid | C | probe: match (27); plain: match (3) | unchanged | reused |
| pvwatts_dc | C | probe: match (4); plain: match (4) | unchanged | reused |
| sapm_cell | C | probe: match (4); plain: match (4) | unchanged | reused |
| alt2pres | C | probe: match (4); plain: match (4) | unchanged | reused |
| aoi | C | probe: match (4); plain: match (4) | unchanged | reused |
| effectiveness_from_NTU | C | probe: match (4); plain: match (4) | unchanged | reused |
| Antoine | C | probe: match (4); plain: match (4) | unchanged | reused |
| friction_factor | C | probe: match (4); plain: match (4) | unchanged | reused |
| Reynolds | C | probe: match (4); plain: match (4) | unchanged | reused |
| Nu_conv_internal | C | probe: match (4); plain: match (4) | unchanged | reused |
| TSat_P | C | probe: match (4); plain: match (4) | unchanged | reused |
| Q_discrete_white_noise | C | probe: match (12); plain: FMU fails | unchanged | reused |
| GetHumRatioFromRelHum | C | probe: match (4); plain: match (4) | unchanged | reused |
| GetHumRatioFromRelHum_numba | C | probe: match (4); plain: match (4) | new: probe, plain | new run |
| PropsSI | C | probe: FMU fails; plain: FMU fails; hand: match (4) | unchanged | reused |
| IAPWS97 | C | probe: match (213); plain: FMU fails | unchanged | reused |
| GHFilter | C | probe: match (6); plain: match (6) | unchanged | reused |
| TCLabModel | C | probe: not comp. (16); plain: match (3) | unchanged | reused |
| CartPoleEnv | C | probe: not comp. (9); plain: FMU fails; seeded: match (25) | unchanged | reused |
| PendulumEnv | C | probe: not comp. (9); plain: FMU fails; seeded: match (25) | unchanged | reused |
| Madgwick | C | probe: match (16); plain: not comp. | unchanged | reused |
| Mahony | C | probe: match (12); plain: FMU fails | unchanged | reused |
| units_Reynolds | C | probe: FMU fails; plain: FMU fails; hand: match (4) | unchanged | reused |
| units_head_from_P | C | probe: FMU fails; plain: FMU fails; hand: match (4) | unchanged | reused |
| HighwayEnv | C | probe: not comp. (66); plain: FMU fails; seeded: match (105) | unchanged | reused |
| get_solarposition | C | probe: match (24); plain: FMU fails | unchanged | reused |
| sports_heat_stress_risk | C | probe: match (19); plain: FMU fails | unchanged | reused |
| read_tmy3 | C | probe: match (26); plain: FMU fails | unchanged | reused |
| JOS3 | C | probe: match (1379); plain: match (12); hand: match (80); hand:JOS3_tcore0: match (16) | new: hand:JOS3_tcore0 | new run, reused |
| chronos | C | probe: match (2348); plain: match (8) | unchanged | reused |
| silero_vad | C | probe: match (4); plain: no config | unchanged | reused |
| granite_ttm | C | probe: match (12680); plain: no config | unchanged | reused |
| sb3_ppo | C | probe: match (3); plain: not comp. | unchanged | reused |
| surfaces | C | probe: not comp.; plain: not comp.; seeded: match (3) | unchanged | reused |
| torchani | C | probe: match (24); plain: FMU fails | unchanged | reused |
| ambiance | A/B | probe: not comp.; plain: not comp.; hand: match (35) | new: hand | new run, reused |
| astral | C | probe: match (4); plain: match (4) | unchanged | reused |
| cantera | C | probe: not comp.; plain: not comp.; hand: match (16) | unchanged | reused |
| colour | C | probe: match (8); plain: FMU fails | unchanged | reused |
| control | C | probe: match (21); plain: FMU fails | unchanged | reused |
| erfa | C | probe: match (4); plain: match (4) | unchanged | reused |
| gsw | C | probe: match (4); plain: match (4) | unchanged | reused |
| hplib | C | probe: match (32); plain: match (24) | unchanged | reused |
| impedance | A/B | probe: not comp.; plain: FMU fails; hand: match (32) | new: hand | new run, reused |
| mesa | C | probe: not comp.; plain: FMU fails; seeded: match (4) | unchanged | reused |
| metpy | C | probe: FMU fails; plain: FMU fails; hand: match (4) | unchanged | reused |
| neurokit2 | C | probe: match (400); plain: FMU fails | unchanged | reused |
| pandapower | A/B | probe: not comp.; plain: not comp.; hand: match (48) | new: hand | new run, reused |
| ppigrf | C | probe: match (12); plain: match (12) | unchanged | reused |
| pybamm | A/B | probe: match (11); plain: FMU fails; hand: FMU fails | new: hand | reused |
| pyet | A | probe: FMU fails; plain: FMU fails | unchanged | - |
| pymsis | C | probe: match (44); plain: FMU fails | unchanged | reused |
| pyproj | C | probe: match (29); plain: FMU fails | unchanged | reused |
| pyromat | C | probe: match (4); plain: no config | unchanged | reused |
| pysolar | C | probe: match (4); plain: FMU fails | unchanged | reused |
| pytci | C | probe: not comp.; plain: match (16); hand: **mismatch** (16) | regression: hand | new run, reused |
| river | A/B | probe: not comp.; plain: FMU fails; hand: match (10) | new: hand | new run, reused |
| rtb | C | probe: match (482); plain: FMU fails | unchanged | reused |
| seirsplus | A/B | probe: not comp.; plain: not comp.; hand: match (24) | new: hand | new run, reused |
| sgp4 | A | probe: no config; plain: no config | unchanged | - |
| simglucose | C | probe: match (64); plain: not comp. | unchanged | reused |
| skops | C | probe: match (3); plain: no config | unchanged | reused |
| thermo | C | probe: match (823); plain: FMU fails | unchanged | reused |
| vollib | C | probe: match (4); plain: match (4) | unchanged | reused |
| windpowerlib | C | probe: match (4); plain: match (4) | unchanged | reused |
| anastruct | C | probe: match (209); plain: FMU fails | unchanged | reused |
| antropy2 | C | probe: match (4); plain: match (4) | unchanged | reused |
| astropy | C | probe: match (12); plain: match (12) | unchanged | reused |
| biopython | C | probe: match (4); plain: match (4) | unchanged | reused |
| blackscholes | C | probe: match (4); plain: match (3) | unchanged | reused |
| commpy | A/B | probe: not comp.; plain: FMU fails; hand: match (16) | new: hand | new run, reused |
| electricpy | C | probe: match (16); plain: not comp. | unchanged | reused |
| haversine | C | probe: match (4); plain: match (4) | unchanged | reused |
| heartpy | C | probe: match (208484); plain: FMU fails | unchanged | reused |
| librosa | C | probe: match (4); plain: match (4) | unchanged | reused |
| mendeleev | C | probe: match (291); plain: match (4) | unchanged | reused |
| meteocalc | C | probe: match (4); plain: match (4) | unchanged | reused |
| nashpy | A | probe: not comp.; plain: FMU fails | unchanged | reused |
| npf | C | probe: match (4); plain: match (4) | unchanged | reused |
| numdifftools | C | probe: match (7); plain: match (7) | unchanged | reused |
| obspy | C | probe: match (12); plain: not comp. | unchanged | reused |
| openap | C | probe: match (4); plain: match (3) | unchanged | reused |
| opencv | C | probe: match (8); plain: not comp. | unchanged | reused |
| openturns | C | probe: match (8); plain: FMU fails | unchanged | reused |
| pedon | C | probe: match (4); plain: not comp. | unchanged | reused |
| pyco2sys | C | probe: match (360); plain: FMU fails | unchanged | reused |
| pyfluids | C | probe: match (111); plain: FMU fails | unchanged | reused |
| pygfunction | C | probe: match (4); plain: match (4) | unchanged | reused |
| pykalman | C | probe: match (8); plain: match (6) | unchanged | reused |
| pyloudnorm | C | probe: match (54); plain: match (3) | fixed: probe | new run, reused |
| pymap3d | C | probe: match (12); plain: FMU fails | unchanged | reused |
| pymunk | A/B | probe: match (36); plain: match (36); hand: FMU fails | new: hand | reused |
| pyorbital | C | probe: match (4); plain: match (4) | unchanged | reused |
| pyquaternion | C | probe: match (241); plain: FMU fails | fixed: probe | new run |
| pyteomics | C | probe: match (4); plain: match (4) | unchanged | reused |
| pythermalcomfort | C | probe: match (11); plain: FMU fails | unchanged | reused |
| pywt | C | probe: match (72); plain: FMU fails | unchanged | reused |
| pyxirr | C | probe: match (4); plain: not comp. | unchanged | reused |
| quantecon | C | probe: match (45); plain: match (30) | unchanged | reused |
| quantlib | C | probe: match (4); plain: match (4) | unchanged | reused |
| radioactivedecay | C | probe: match (8); plain: FMU fails | unchanged | reused |
| rainflow | C | probe: match (32); plain: FMU fails | unchanged | reused |
| rdkit | C | probe: match (4); plain: FMU fails | unchanged | reused |
| scipysignal | C | probe: match (16); plain: not comp. | unchanged | reused |
| shapely | C | probe: match (68); plain: not comp. | unchanged | reused |
| simpy | C | probe: match (4); plain: FMU fails | fixed: probe | new run |
| skfuzzy | C | probe: match (4); plain: match (4) | unchanged | reused |
| skimage | C | probe: match (12); plain: not comp. | unchanged | reused |
| statsmodels | C | probe: match (8); plain: match (8) | unchanged | reused |
| tellurium | A/B | probe: match (4); plain: match (3); hand: match (12) | new: hand | new run, reused |
| thermofeel | C | probe: match (4); plain: FMU fails | unchanged | reused |
| timezonefinder | C | probe: match (18); plain: match (18) | unchanged | reused |
| tmm2 | C | probe: match (60); plain: FMU fails | unchanged | reused |
| transforms3d | C | probe: match (16); plain: not comp. | unchanged | reused |
| uncertainties | A/B | probe: match (4); plain: FMU fails; hand: match (8) | new: hand | new run, reused |
| ex_kalman_repo3 | C | repo (FMI 3, run.py): match (350) | unchanged | reused |
| ex_sampled_pid_repo3 | C | repo (FMI 3, run.py): match (82) | unchanged | reused |
| gk_distillation | C | probe: match (21); plain: match (21) | new: probe, plain | new run |
| gk_distcol_gym | C | probe: not comp. (15); plain: FMU fails; seeded: match (33) | new: probe, plain, seeded | new run |
| gk_bioreactor | C | probe: match (15); plain: not comp. | new: probe, plain | new run |
| gk_cge | C | probe: match (81); plain: FMU fails | new: probe, plain | new run |
| gk_cftoc | C | probe: no config; plain: no config; hand: match (9) | new: probe, plain, hand | new run |
| gk_debut_steady | C | probe: match (144); plain: FMU fails | new: probe, plain | new run |
| gk_debut_reflux | C | probe: match (148); plain: FMU fails | new: probe, plain | new run |
| gk_apsis_mpc | C | probe: match (336); plain: FMU fails | new: probe, plain | new run |
| gk_fopdt | C | probe: match (18); plain: not comp. | new: probe, plain | new run |
| gk_mpc | C | probe: match (36); plain: not comp. | new: probe, plain | new run |
| gk_ethanol | C | probe: match (93); plain: FMU fails | new: probe, plain | new run |
| gk_nhes | C | probe: match (4892); plain: FMU fails | new: probe, plain | new run |
| gk_openhens | C | probe: not comp.; plain: FMU fails; hand: match (3) | new: probe, plain, hand | new run |
