# Tested models

fmugen has been tried on 208 unmodified Python models (three needed a file renamed or a script wrapped in a class, and one synthetic input data; see their notes): models from about 132 libraries on PyPI, 44 neural networks, 13 models built on [GEKKO](https://github.com/BYU-PRISM/GEKKO), one model published as source on GitHub and one unpublished model. Each model is listed once, with its most recent result. 207 simulate with useful results; 3 fail because the library has no function or class that is the model. 20 libraries tested at first turned out not to be models: 17 are tools for building one, and 3 (pymunk, PyNite, PyPSA) are empty containers that the user fills with bodies, nodes or buses; they are listed in [discarded-models.md](discarded-models.md).

- [How the models were tested](#how-the-models-were-tested)
- [Summary](#summary)
- [GitHub repositories](#github-repositories)
- [Unpublished models](#unpublished-models)
- [Python libraries](#python-libraries)
- [Neural networks](#neural-networks)
- [GEKKO models](#gekko-models)
- [Hand configs](#hand-configs)
- [Known problems](#known-problems)
- [Versions](#versions)
- [Reproducing a run](#reproducing-a-run)

---

## How the models were tested

Each model went through:

1. `fmugen init` (code only)
2. `fmugen init --probe`
3. `fmugen build --capture-output` on each config `init` wrote (and on the hand configs), then `fmpy.simulate_fmu` for 3 s with 1 s outputs
4. `fmugen build --capture-output --compile pyinstaller` on the best config (the `--probe` one when it simulated), then the same simulation

**FMI 2 by default.** A model was run with FMI 3 only when no FMI 2 config simulated; the results below combine both.

Setup: fmugen at `6dfc18c` (2026-10-22), WSL 2 (Ubuntu), Python 3.13, `uv` venvs, FMPy 0.3.32, UniFMU 0.14, gekko 1.3.2 with its local solver. CGE ran on Windows 11. Every FMU was compared value by value with the model run directly ([report](match-testing-rerun/report.md)). The **Compiled** column: PyInstaller 6.22.3 in WSL 2 (Linux), each compiled FMU compared with the uncompiled one of the same config, value by value ([report](match-testing-compile-rest/report.md), [GEKKO and hand configs](match-testing-compile/report.md), [GEKKO retest](match-testing-compile/retest.md)). Sizes are of the Linux FMUs, except CGE's (Windows only). Compile in a venv with only what the model needs: extra packages make the FMU larger and the build slower.

82 of the models (those whose links point into [pending-100/](pending-100/)) were tested later, on 2026-10-23, the same way, each config also run directly from the same config (twice, to find models random by design; those were compared seeded) and each compiled FMU compared with its uncompiled one. Setup: fmugen at `0a7c525` (before the fixes below), WSL 2 (Ubuntu 26.04), Python 3.13, one `uv` venv per model, torch 2.14.1 CPU build, FMPy 0.3.32, UniFMU 0.14, PyInstaller 6.22.3. Inputs: ultralytics' `bus.jpg` and `zidane.jpg`, the ONNX model zoo's MNIST, 5 s of whisper's `jfk.flac`, EPANET's `Net1.inp`, pyswmm's `model_weir_setting.inp`. TabPFN was left out: it downloads its weights only after a license is accepted interactively, so it can't run unattended. The scripts, wrappers, hand configs and results are in [pending-100/](pending-100/): `run100.py` runs every step, `results.json` has every model's verdicts and versions.

**Models that need a wrapper count as failures.** 3 libraries have a model but no single function or class that runs it: PVWatts is built by a C built-in and needs weather data set on it, Wav2Vec2 needs CTC decoding and Moirai GluonTS data preparation. `init` has nothing to point at. Each was also tried through a short file written for this test, in [pending-100/wrappers](pending-100/wrappers/), calling the library as its documentation does; the notes say what the wrapper did and whether fmugen handled it. 7 more needed one at first (neuralforecast, floris, pandapipes, sectionproperties, vqf, imufusion, mujoco); they run without one, with hand configs in [pending-100/hand](pending-100/hand/), using `get:` outputs, `attr:` paths through keys, a step that is a function taking the object and `substeps` without `integrate`, all added for them.

Columns: **`init` / `init --probe`**: given only the model and `--start` values (and `--fmi`), ✓ if the config of either one builds, simulates and outputs the model's results; otherwise how they failed (✗ no useful outputs: it runs, but only outputs parameters, settings or constants); "needs options" when the run that passed needed other options. `--start` only gives start values of FMU variables (numbers, text, dates, arrays), so it doesn't count as an option; how many in brackets in the next column. **… with other options**: the same, for an `init` run given other options (`--call`, `--create`, `--setup`, `--const`, `--kind`, `--convert`), which ones in brackets (`×2`: given twice); the config is used as `init` wrote it, unedited; — when it needed none. The runs used `--start` for everything, before `--const` existed: their `call:` values, dicts and strings that name what's loaded or configure the model (files, model names and weights, devices, a TLE, a coordinate system, a chemical) are counted as `--const`; a `call:` that only builds an array from literals (`numpy:array([…])`, `torch:tensor([…])`) counts as a `--start` array where giving it as a list was checked to work with the same outputs (18 models, and 25 that still need other options: for scikit-commpy, nashpy, pykalman, quantecon, simdkalman, statsforecast, reservoirpy, GPy, pysindy, pmdarima, pyquaternion and arch only the `--call` remains; the probe passes lists as numpy arrays or torch tensors where the model needs them). A `--const` that only sets how the model runs (`device`, `default_dtype`) counts as a start value, and a `--call` naming what `init` picks by itself doesn't count: since 2026-10-27 that includes the only method the `--start` names fit (JOS3's `simulate`, scikit-fem's `refined`, the FOPDT process model's `run`) the only method whose required arguments the `--start` names cover, counting names the constructor also takes optionally (ahrs' `Madgwick`: `updateIMU`, not `updateMARG`, which also needs `mag`), or that takes the most of them (pyquaternion's `rotate`), and the `__call__` of a model a function returns (silero's `load_silero_vad`), since 2026-10-28 the method `init` picks when it isn't sure, by its name (`predict`, `forecast`, `simulate`, `solve`, `run`, …; `fit`, `reset`, `save` last; in a terminal it asks, otherwise it warns with the candidates), a `--setup` that only trains the model (`fit(...)` on the user's data: fitting is the user's part, not fmugen's; since 2026-10-28, and only where the fitted model then runs), and `--kind function` for a class with no method to run, only properties (ambiance's `Atmosphere`), checked to give the same outputs. **Hand config**: a config written by hand, ✓ if it builds, simulates and outputs the model's results, with a link to it; — when the model needs none: ✓, ✗ no config (`init` stopped), ✗ build fails, ✗ simulate fails, or — (not run). **Compiled**: the best config compiled with PyInstaller, built and simulated. **Notes**: what was checked, or the error of each config that failed ("`init`:", "`init --probe`:").

---

## Summary

208 models: 193 from libraries and neural networks, 13 built on GEKKO, one from a GitHub repository and one unpublished. FMI 2 and FMI 3 results are combined: a model counts as passing if it passes with either. The 82 tested later ran on fmugen before the fixes below; the 19 that failed on fmugen problems were run again with the fixes, and those results are counted.

| | Models (208) |
|---|---|
| `init` writes a config | 186 |
| `init --probe` writes a config | 192 |
| **Simulates with useful results** (a config from `init` or `init --probe`) | **195** |
| …of which given only the model and `--start` values | **104** |
| …of which given other options (`--call`, `--const`, `--setup`, …) | **91** |
| …or with a hand-written config | **205** |
| Fails | **3**: all need a wrapper |
| **Compiled FMU simulates** | **207**; py-pde simulates uncompiled only |

A config only counts as a success if the FMU simulates **and** gives the model's results. An FMU that runs but only outputs its own parameters, settings or constants counts as a failure.

How far each step gets:

| Config | Models | Success |
|---|---|---|
| `init` (code only), given other options where needed | 55 of 208 | 26 % |
| `init --probe` (or `init`, when it already works), given only the model and `--start` values | 104 of 208 | 50 % |
| `init --probe` (or `init`), given other options where needed (`--call`, `--const`, `--setup`, …) | 195 of 208 | 94 % |
| `init --probe` + a hand-written config | 205 of 208 | 99 % |

Without the 3 models that need a wrapper, `init --probe` + a hand config gets all 205.

The `init` config fails mostly on models that return arrays, tuples, strings or objects, which only `--probe` types correctly; the error message points to it. Of the 67 `init` configs of the first group that simulate, 23 only output parameters, settings or constants. GEKKO models keep their results in GEKKO variables, which only `--probe` finds. The hand configs add `PropsSI`, cantera, PyTCI and the kalman example ([Hand configs](#hand-configs)), pandapower, seirsplus, tellurium, the GEKKO model OpenHENS, sgp4 and pybamm ([configs](match-testing-rerun/configs/)).

Later, `init --probe` got 9 more without a hand config, each built, compiled with PyInstaller 6.22.3 and simulated in WSL 2; the compiled FMUs give exactly the values of the model called directly (157 values, t = 0–3; [probe-9](probe-9/CONCLUSION.md)): commpy and impedance (complex results split into real and imaginary parts), uncertainties and ambiance (a returned object's properties, one-element arrays read as their element), `fluids.units` ×2 and metpy (inputs retried as pint quantities, with units read from the errors) CFTOC (a `call:` start for the step method's copy of a shared argument is no longer refused) and river (when the probe finds no outputs, what the step changes becomes the outputs). seirsplus can't be done by `init`, an FMI limitation: its counts are arrays that grow by one element each step, and an FMI variable has a fixed size.

On 2026-10-27, `init --probe` got 10 more without a hand config, each built and compared in WSL 2 with the model called directly, with the same inputs, at every step: all identical (8,916 values). ephem, sgp4, vqf, pinocchio and imufusion: C built-ins without a signature take the `--start` values by position, and the probe gives the step method as many as it takes (ephem's date goes to `compute()`, not `Sun()`); imufusion's step returns its object, now read as returning nothing, so its getters are outputs. simdkalman and YOLO: objects inside the result are read (`observations.mean`, `y0.boxes.xyxy`). neuralforecast: a table's numeric columns are outputs. sympy's `lambdify`: the generated function's own arguments are inputs. pandapipes: `--call "pandapipes:pipeflow(self)"` is accepted, and result tables on the object give their columns. Also new: a history that grows each step gives its last entry (seirsplus' counts, pybamm's `t`), but seirsplus still needs its hand config: `init` binds `dt` to the step size, and with `dt` = 1 `run_epoch` doesn't advance. pandapower's `runpp` returns None; `init` now suggests making `net` the model (`--call "pandapower:runpp(self)"`). tellurium's species are still not found: RoadRunner doesn't list them.

On 2026-10-28, `init` stopped refusing a class whose step method it can't tell: in a terminal it asks which one, the candidates numbered, best first; otherwise it picks the first and prints a warning with the candidates. The 184 recorded commands, run in WSL 2 with the old and the new code, give identical configs (151; 6 fail with both, 27 not run: they need data files). Run without their `--call`, 20 models give the same outputs with the method picked: cv2's KalmanFilter, ephem, GPy, pysindy, statsforecast, lifelines and timezonefinder now need only `init --probe`; anastruct, catboost, hapsira, openap, pastas, py-pde, prophet, pyloudnorm, chronos, tabnet, skorch, radioactivedecay and scikit-rf no longer need `--call`, only their other options. The ranking puts `fit`, `train` and `update` last only when the class also has a run-like method (`predict`, `solve`, …), so lifelines' `fit` is its step; and a method named after another one plus other words comes after it (`timezone_at` before `certain_timezone_at`, `decay` before `cumulative_decays`). On the remote host, 20 of the 27 recorded commands that need data files or cloned sources also give identical configs (5 fail with the old code too; mtcnn and YOLO ran out of memory), and 3 more no longer need `--call`: the GEKKO ethanol bioreactor and Transformer_MPC's `MPCModel` need only `init --probe`, and wntr only its `--const`; the RC building zone picked `calc_energy_demand` instead of `solve_energy`, and pyswmm `close` instead of `__next__`. Since then, a name starting with a run-like word ranks right after it (`solve_energy` after `solve`), and clean-up names (`close`, `stop`, `shutdown`, …) rank last: the RC building zone picks `solve_energy` and its probe succeeds (not yet compared with its `--call` run); pyswmm still needs `--call`, since `__next__` isn't a candidate. The 184 local commands give the same results as before. On the local PC, 10 still get a wrong method and need `--call`: osqp and pyfluids (`update` is taken as the step before any ranking), astropy, blackscholes, chaospy, openturns, nashpy, pedon, quantecon and matgl.

On 2026-10-29, the ranking stopped putting last a method whose required arguments the `--start` names don't all cover: a method taking some of the names the constructor doesn't take now comes before those taking none, so a partial set of values points at the step (`solve(a, b)` given only `a`, before `summary()`). The 184 recorded commands give the same configs, and the same results without `--call` (47 with the recorded outputs). Run again with `--call` dropped and only one `--start` kept, each in turn (126 runs of the 42 classes with two or more), plain `init` picks the recorded method in 44 runs (39 before), none worse: lifelines (`fit`, not `plot`), and pyproj, JOS3 and timezonefinder, which `init` refused before (`--start names that are not arguments of …`).

Also on 2026-10-29, the start made up for an argument with no default (and no `--start`) is `1.0` (`1` for an Integer), not `0.0`, which models refuse when they check it's positive or divide by it. When the step fails, the probe tries the made-up inputs as `0.0`, then as arrays of ones as before; when the step fails on an object built with made-up parameters, it builds it again with them as `0.0` (quantecon's `LinearStateSpace(A=1.0)`: `I - A` is singular). The 184 recorded commands give the same configs and the same results without `--call` (47). Of the 151 run with no values, none got worse: metpy's `dewpoint_from_relative_humidity` and pythermalcomfort's `sports_heat_stress_risk` now give the recorded outputs, and filterpy's GHFilter the recorded ones instead of others; pyloudnorm now gives the recorded outputs but for their array sizes, and simdkalman other outputs: `update` is the step before any ranking (`smooth` is recorded), with the matrices made up as scalars. pedon still fails: the made-up `1.0` for both `theta_s` and `theta_r` divides by their difference, and `0.0` fails the constructor (`1 - 1/n`); it also picks `h` (recorded: `theta`). With only `--call` kept: filterpy the recorded outputs, pyloudnorm, simdkalman and statsforecast the recorded ones but for their array sizes; quantecon as before. With the fit set-ups: tabnet the recorded outputs, and NBEATS outputs that differ only in `time_index`, again a size: with the recorded forecast length `n = 4` it's an Integer array of 4 (`return:time_index`); with the made-up `n = 1`, a one-element array, read as its element (`return:time_index.0`, Real). simple-pid's `Kp` is now found tunable: with `input_ = 0.0`, changing it changed nothing.

Also on 2026-10-29, `update` is taken as the step before any ranking only when the class has no run-like method (`predict`, `solve`, …): it then sets the model up. The 184 recorded commands give the same configs; without `--call`, 48 give the recorded outputs (47): osqp picks `solve`. simdkalman, refused before without `--call` (`data` is an argument of several methods), now picks `em` (recorded: `smooth`); pyfluids still picks `update` (recorded: `with_state`), as it has no run-like method. With no values, osqp and simdkalman now fail instead of giving other outputs from `update`: osqp's `solve` needs `setup` first, and simdkalman's `em` fails on a made-up float where it needs an integer (`'float' object cannot be interpreted as an integer`).

Also on 2026-10-29, when the probe step fails, `init --probe` looks for a set-up it needs: a public method the error message asks to call first ("call compile() first", "not fitted yet. Call 'fit'"), else one that assigns an attribute the step reads (`self._solver`) that is None or missing on the object. It's run in `[model] setup`, with no arguments or its required ones made up as `1.0`, and kept when the step then works; a training method (`fit`, `train`, `learn`, `fit_…`, …) is the user's part, so it's never made up, only named. Otherwise the probe's failure comment names it: `solve() seems to need setup() first (it sets _solver, which solve() uses): add --setup "setup(...)"`. The 184 recorded commands give the same configs and the same results without `--call` (48). With no values, no config changed; 9 failing probes now name the set-up: osqp's `setup`, timesfm's `compile` (both what the recorded commands run), and the training of catboost, prophet, statsforecast, pyod, skorch, river (`learn_many`) and lifelines (`fit_interval_censoring`; recorded: `fit` as the step). None named a wrong method once the message had to ask for the call (before, torch modules got `forward()` from `forward() missing 1 required argument`, and `float()` from `'float' object`).

Also on 2026-10-29, `init` reads more than the signatures:
- **The package's usage examples:** `>>>` docstring examples, the README's code blocks (from the package metadata) and `examples/` scripts. The first that builds the class gives the options not passed: the step, the classmethod, the set-up calls, and the values of arguments with no default, including values computed by an import, as a `{ call = ... }` constant, only with `--probe` and only when they compute. Some example steps are not taken:
  - a method's own docstring calling it;
  - a run-like name when an earlier one exists (pastas' README calls `solve`, `simulate` runs it);
  - training when the class has a run-like method (GPy's `optimize`).

  When the example's values fail the probe, they're dropped.
- **Better starts for arguments with no default:**
  - a documented default (`default 1e-6`);
  - a typical value for a docstring unit (`[K]`: 293.15);
  - a value the code's `if x < 3: raise` checks accept.
- **With `--probe`:**
  - after a failing constructor, classmethods taking no required argument (`default()`, `from_...()`);
  - after a guessed step that fails or gives no outputs, the next two candidates;
  - an iterator with no run-like method steps with `__next__`.

The 184 recorded commands give the same configs. Without `--call`, 50 give the recorded outputs (48): pydmd and simpy, which failed before.

With no values, none got worse but two:
- skimage's `rgb2hsv` takes its docstring's `astronaut()` image, so its outputs differ only in size (512×512×3);
- sympy's `lambdify` gives no outputs instead of failing.

These now give the recorded outputs, from failing:
- Bio's `gc_fraction`, filterpy's `Q_discrete_white_noise`, fluids' `head_from_P`, ht's `Nu_conv_internal`, py_vollib's `black_scholes`, slycot's `sb02md`, astropy's `FlatLambdaCDM` (with the recorded options), and simpy (with none);
- pythermalcomfort's `pmv_ppd_iso`, from other outputs.

14 more work, with other outputs than recorded, from failing: mendeleev, openturns, pedon, pydmd, pymoo, pymsis, pyproj, pyromat, tabnet, pywt, radioactivedecay, simdkalman, statsforecast and thermo. The fit set-ups are unchanged.

Of those 14, compared with the recorded configs:
- 5 differ only in array sizes (pydmd, pywt, statsforecast, pymoo with the README's 30-variable problem, pymsis with its docstring's 336 dates).
- 3 pick another valid choice:
  - mendeleev, the docstring's Si (an element's outputs are the properties it has: Fe 82, Si 76, H 70, U 69; the others are commented out);
  - pedon, `s()` instead of `theta()`;
  - thermo, `calculate(T, P)` instead of rebuilding the object every step.
- 6 were wrong; four of them are fixed:
  - **openturns:** the docstring's `getSample(5)` returns no FMI value, and SWIG's `thisown` flag passed for an output. Wrapper flags are no longer outputs, and an example whose step gives no outputs is dropped. It fails again, as before, on its C methods.
  - **pyproj and radioactivedecay:** the step a method's own docstring shows, `transform()` and `decay()`, was dropped. It's now dropped only when the method is a variant of another (timezonefinder's `timezones_at`, of `timezone_at`). The probing had then picked `to_json()`; methods named like set-up, save or conversion (`to_…`) are no longer probed.
  - **tabnet:** `predict()` failed before `fit()` (`'TabNetRegressor' object has no attribute 'network'`), and the probing picked `predict_func()`. When the guessed step needs a set-up, only run-like names (`forecast`) may replace it, so statsforecast still picks `forecast()`.

  Still wrong:
  - **pyromat's `get`:** it returns the model object; the step is its `cp()`.
  - **simdkalman:** made-up scalars where matrices are needed; it picks `predict_next()`.

After these fixes, the 184 recorded commands give the same configs but openturns', without `thisown`. Without `--call`, 50 give the recorded outputs. With no values, 10 more give the recorded outputs than before; stable-baselines3's SAC is new among them. 12 work with other outputs than recorded, 2 of them wrong.

Also on 2026-10-29, sizes and torch inputs:
- An unannotated argument named like a count or size (`n_…`, `num_…`, `…size`, `…_channels`, `…_features`, `units`; not `step_size`) is made up as the Integer `1`, as is one annotated `int | …` without float.
- A torch module's `__call__(*args, **kwargs)` takes the arguments of its `forward()`.
- When the probe fails with the made-up values, it also tries them as torch tensors of ones (for a torch module), then as plain lists of two ones, of ints for sizes.
- Building methods (`add_…`, `remove_…`) aren't step candidates (anastruct's `add_element`).

The 184 recorded commands give the same configs. Without `--call`, 50 give the recorded outputs; momentfm and torchani now take `__call__`, which runs the recorded `forward`, where they failed before. With no values:
- haversine gives the recorded outputs (its points as lists of two floats);
- ncps' CfC, norse's LIFCell, snntorch's Leaky and pytorch-tcn work, with other outputs than recorded, from made-up sizes;
- FNO and neuromancer's MLP now build, but fail on their input's shape;
- chgnet, moment and sentence-transformers need inputs no made-up value gives (graphs, a batch, tokenised text).

Also on 2026-10-29, a made-up input of an object with gymnasium-style spaces is first tried as a value of its space: `observation_space` for an argument named like an observation (stable-baselines3's `predict(observation)`), else `action_space` (an environment's `step(action)`). That's the start of a `Discrete` space, or the middle of a `Box`. The `reset()` the probe tries after a failing step is checked with these retries, so CartPole's `step(1.0)` no longer makes it give up. The 184 recorded commands give the same configs; without `--call`, 50 give the recorded outputs. With no values, these now give the recorded outputs, whether or not the other options are kept:
- gymnasium's CartPole and Pendulum (`reset` found, action `0` and `[0.0]`);
- stable-baselines3's PPO and sb3-contrib's RecurrentPPO (`load` and `predict` found from the weights path, observation zeros).

Also on 2026-10-29, when the probe call still fails, numeric arguments whose default is None are made up as `1.0`, one alone and then two together (up to 40 sets); the smallest set that works becomes inputs, with a comment. That covers models needing some of several optional arguments: fluids' `Reynolds` needs `rho` and `mu`, or `nu`. The 184 recorded commands give the same configs; without `--call`, 50 give the recorded outputs. With no values, these now give the recorded outputs, from failing:
- fluids' `Reynolds` (`nu`);
- electricpy's `powerset` (`P` and `Q`);
- uncertainties' `ufloat` (`std_dev`).

pyet's `pm_fao56` still fails: it also needs its inputs as dated Series, a retry this one doesn't combine with.

Also on 2026-10-29, two more retries when the probe call fails:
- **Text:** when the error asks for a string ("Expected one of: str, …", "text input must be of type `str`"), the made-up inputs are tried as a sentence.
- **Positional numbers:** for a callable without a readable signature, the probe makes up as many numbers as its error says it takes, as inputs `x0`, `x1`, … passed by position. The count comes from "takes exactly 2 arguments (0 given)", or from the fewest numbers of a SWIG prototype that takes only numbers, such as `computePDF(OT::Scalar const)`.

The 184 recorded commands give the same configs. With no values but `--call`, 28 of 80 give the recorded outputs (24): sentence-transformers' `encode`, transformers' `pipeline`, sgp4's `sgp4` and openturns' `computePDF`. With the other options kept too, 71 of 151 (68). With no options at all, openturns works, with other outputs.

Also on 2026-10-29, the made-up inputs' array retries first try the shapes the error message names ("cannot reshape array of size 1 into shape (7)"). roboticstoolbox's `Panda.fkine` now gives the recorded outputs with no values, with `q` as 7 ones, whether `--call` is kept or not. With no values but `--call`, 29 of 80 give the recorded outputs; with the other options kept too, 72 of 151. The 184 recorded commands give the same configs.

Also on 2026-10-29, when the step needs training first (`fit`, `train`, `learn`), the probe trains a new object on made-up data, only to find the step's inputs and outputs. The data are 64 deterministic random samples of 4 features; `y` is tried as a vector, a column and class labels, as float64 then float32, and last `fit(X)` alone. The config gets the training only as a commented-out line, `# setup = ["fit(...)"]`, to fill with the user's data, so no FMU is trained on made-up data. The class's own `fit` is tried too when the step fails on an attribute its object doesn't have yet (tabnet's `network`, set in a helper `fit` calls). The 184 recorded commands give the same configs.

With no values but `--call`, 32 of 80 give the recorded outputs (29): catboost, tabnet and pyod, with `X` sized from the fitted model. With no options at all, catboost and tabnet give the recorded outputs, and pyod other outputs. Then the probe also trains on a made-up dated table (`ds`, `y`, 64 days) when pandas is loaded, and on a made-up darts `TimeSeries` when darts is. "Not having fit the model" and "not trained" now count as asking for a set-up. prophet and darts' NBEATS now work, with no values whether `--call` is kept or not. Their outputs follow the made-up training data (64 days, not the recorded 24), so they count as other outputs. skorch still fails: its network, `module`, is a made-up 1.0 without its `--const`.

Also on 2026-10-29, a factory function returns the model in two more cases:
- The `--start` names it doesn't declare are arguments of calling what it returns, even when it takes `**kwargs`: transformers' `pipeline(task, model, **kwargs)` and its `__call__(inputs)`.
- With no names left over, it returns a function, which is then called on every step: sympy's `lambdify`.

Without `--call`, 54 of the recorded commands give the recorded outputs (52), and none got worse. mace's `mace_mp` now picks `calculate`, where it picked no method (recorded: `get_potential_energy`). tsfm's `get_model` now picks `__call__`, which runs the recorded `forward`.

Also on 2026-10-29:
- A set-up's required arguments annotated with a class are made up as an instance of it built with no arguments, not 1.0: timesfm's `compile(ForecastConfig())`.
- The step after a set-up is checked with the made-up inputs' retries.
- The None-default retry also runs when building the object fails, with counts and sizes as `1` (reservoirpy's `Reservoir(units=None)`).

None of the recorded commands changed. reservoirpy and timesfm get further but still fail: a reservoir of 1 unit is too small for its eigenvalue solver, and timesfm's default `ForecastConfig()` has `max_horizon = 0`.

Fixed in the runtime: setting a path through a pandas label (`loads.p_set.l`) set an attribute on a copy of the column under pandas 3's copy-on-write. It now sets the table's cell. The test was skipped before, as pandas wasn't installed.

Also on 2026-10-29, three fixes to reading usage examples:
- **Fences:** Markdown fences are paired in order. The closing fence of a ```` ```bash ```` block no longer opens a Python block of the prose after it, which had hidden nashpy's README examples.
- **Loop headers:** a method iterated in a `for` loop's header counts as a call in a loop: nashpy's `for eq in game.support_enumeration():`.
- **Several queries:** of the calls an example makes, the step is a run-like name (by its order in that list: simpy's `run` before `process`), then one in a loop, then one whose result is used, then the first, not the last. A README showing several queries in turn leads with the main one: blackscholes' `price()`, then `delta()`, … `charm()`.

The 184 recorded commands give the same configs. Without `--call`, 57 give the recorded outputs (54): blackscholes, nashpy, chgnet and roboticstoolbox are new. py-pde now picks `solve`, what its README runs (recorded: `evolution_rate`). With no options at all, 58 give the recorded outputs (56): blackscholes and roboticstoolbox, and py-pde works.

Also on 2026-10-29: an example's value with arithmetic, which a config call can't hold (pysindy's `numpy.stack((3 * numpy.exp(...), ...))`), is computed with `--probe` and written as its numbers (at most 2000). A `(*args)` constructor whose error says nothing of its arguments gets a 2 x 2 identity matrix (nashpy's `Game`). An empty-list default, its base classes' too, gets as many ones as the error asks for (impedance's `Randles`: "circuit length (5)"). With only `--call` kept and no values, 44 of 80 give the recorded outputs and 11 fail (anastruct and impedance now work, with other outputs); with the other options kept, 84 of 151; with none, 63. The 184 recorded commands give the same configs.

Also on 2026-10-29: an ASE calculator's `atoms=None`, when the error is about it ("'NoneType' object has no attribute 'pbc'"), is made up as a water molecule (`ase.build.molecule('H2O', vacuum=5.0)`): mace and matgl. A network argument (`module`, `vector_field`, or an error asking for a callable or a Module) is made up as `torch.nn.Linear(4, 1)` or `(4, 4)`, and a made-up input is also tried as a float32 tensor `(1, 4)`, as wide as the made-up training data: skorch (trained on made-up data for the probe) and torchdyn. With only `--call` kept and no values, 43 of 80 give the recorded outputs and 14 fail; with the other options kept, 83 of 151. The 184 recorded commands give the same configs.

Also on 2026-10-29, more made-up values with `--probe`, so fewer options are needed:
- **Values from other calls in the package's examples:** an argument no use of the model shows takes the value the examples pass to an argument of the same name of anything else in the package (openap's README builds `FuelFlow("A320")`; `Thrust` takes the aircraft as `ac` too). When `--call` names another method than the example's, its arguments of the same name are taken (py-pde's `state`); factory functions read their examples too (pyromat's `get('ig.N2')`).
- **Other numbers:** made-up values the model refuses as 1, 0, arrays and lists are tried as 4, 16 and 100 (commpy's `QAMModem(m)`), integer arrays (its bits), and Integer 1 when the error asks for an integer (nengo's `run_steps(steps)`). When the step fails, the object is rebuilt with those numbers, and with None-default sizes (reservoirpy's `units`).
- **Arguments an error names:** "Provide `context_length` and `prediction_length`" (granite's `get_model`) makes them up through `**kwargs`. Tensors are then also tried in the batch shapes the model's config gives (`(1, context_length, channels)`), and arguments annotated `torch.Tensor` as tensors (chronos' `predict(inputs)`).
- **Shapes an error asks for**, by running the inference again: a dict of two features (river's "not an instance of 'dict'"), a dict or `**kwargs` with the keys a `KeyError` names (surfaces' hyperparameters), an object with the attributes an `AttributeError` names, a dated pandas Series when the error is about a series (pastas' `Model(oseries)`), a class of the package named as the argument built with no arguments (nengo's `Simulator(network)`: `nengo.Network()`), and for a `(*args)` constructor the number of arguments its error expects (control's `StateSpace`: "Expected 1, 4, or 5 arguments", 4 named by its docstring).

The 184 recorded commands give the same configs. With no values and only `--call` kept, 40 of 80 give the recorded outputs (32) and 18 fail (31): chronos, commpy, nengo, openap, pastas, pyromat, river and surfaces give the recorded outputs; control, py-pde, reservoirpy, sympy and granite now work with other outputs. With no values and the other options kept, 80 of 151 (72); with no options at all, 62 (58). pyfluids, simglucose and thermo still fail, at another point.

Also on 2026-10-28: how far `init --probe` gets with no values at all, on the 151 recorded commands that run on the local PC: every `--start` and every `--const` dropped, except those naming a file, weights or a pretrained model (`<data>/speech.npy`, `yolo11n.pt`, `amazon/chronos-bolt-tiny`, `hf_hub_download(...)`), and every other option dropped too (or only `--call` kept). Arguments with no default then get made-up values. Since this date, when the probe fails with the made-up `0.0`, it retries: arguments the numpydoc docstring or type hint calls arrays as arrays of ones in their documented shape, then every made-up value as `1.0`, then as arrays of ones of a few shapes. With these retries, 69 of the 151 give a config with outputs (56 before), 52 with the same outputs as the recorded command (counting outputs whose array sizes differ, since sizes follow the input values: tirex's forecast is as long as its `prediction_length`; 34 before, counted without sizes): ahrs ×2, antropy, blackscholes, GPy, iapws, pedon, psychrolib, scikit-image, pyod and SAC now give the same outputs with no values; none got worse. Training a model (`--setup fit(...)`) is the user's part, so it was kept for the 8 models that need it: with it, and with the probe sizing `predict`'s made-up `X` from the fitted model's feature count (`n_features_in_`, tabnet's `input_dim`), catboost, tabnet, prophet and pyod also work: 73 of the 151 give outputs, 56 the recorded ones. With `--call` where the recorded command has one and nothing else (no values, data files kept), 62 give outputs and 52 the recorded ones (66 and 56 with the fit set-ups). The other outputs differ because the model took another path with made-up values (iapws has no liquid phase at the made-up state; PyCO2SYS computes only what its inputs allow) or returned its results in another form (a tuple instead of one value). skorch (its network) and NBEATS (its window lengths) still need values, and lightgbm and xgboost fail with their recorded commands on the local PC. The recorded commands themselves give the same configs as before.

Of the 131 models tested first, none fails. The last 4 needed fmugen features, added after the main run and checked on these 4 models only (built, simulated, compiled and compared with a direct run; the other models weren't rerun with them):

| Model | What was missing | Feature |
|---|---|---|
| pymunk | `Space.bodies` is a `dict_keys` view, which a path couldn't index | A number in a path picks an element of any iterable, where there's no such attribute ([config.md](config.md)) |
| pybamm | The voltage is `solution["Voltage [V]"]` on an object that isn't a mapping | A path part is tried as a key on objects that take keys, where there's no such attribute |
| nashpy | `support_enumeration` returns a generator | Generator results are read into a tuple; `init --probe` splits nested tuples into `y0_0`, `y0_1` |
| pyet | Its inputs must be pandas Series with a date index | `convert = "series"` with an `epoch`, which `init --probe` writes when plain numbers and arrays fail |

**FMU vs direct run** ([report](match-testing-rerun/report.md)): every model was built into FMUs (`init --probe`, plain `init`, and the hand configs; no `--compile`) and compared, output by output and step by step, with the model called directly in plain Python without fmugen. 130 of 134 match (236,621 values; largest relative difference 1.3 × 10⁻¹⁶, otherwise identical), counting psychrolib with numba and the two FMI 3 hand configs (filterpy's KalmanFilter, simple-pid's PID with clocks) as separate entries. Models random by design were compared seeded. The other 4 are nashpy (no outputs to compare), pyet (the FMU fails), sgp4 (no config) and PyTCI's hand config, which differs from a per-step loop only at t = 0 ([PyTCI](#pytci)).

Checked later, after the features above: sgp4's hand config, pymunk, pybamm, nashpy and pyet all match the direct run with no difference, and so do their compiled FMUs. The `init --probe` configs of the 9 models added later (CFTOC, commpy, impedance, uncertainties, ambiance, `fluids.units` ×2, metpy, river) match it too, compiled and with no difference: 157 values over t = 0–3 ([probe-9](probe-9/CONCLUSION.md)).

CGE counts as a success after renaming its `model.py`, NHES with synthetic input data, OpenHENS with a hand-added output and the COIN-OR couenne solver.

The 82 models tested later: every matching config is identical to the model run directly (51,974 values compared, largest relative difference 0), and every compiled FMU to its uncompiled one.

**fmugen problems found.** 19 models failed on one of these in the first run (some models that also need a wrapper hit one too). fmugen has since been fixed for each, with tests, and the 19 were run again with the fixes: 18 match the direct run and compile (their rows say what failed before). The 10 whose compiled FMU failed to instantiate also compile and simulate with the fixes. pykan still failed then: its `init --probe` config failed at initialization (`missing input data x`); it passes since `8390062`, which sets `init_call` when a property fails before the step.

- **`build` refuses a config `init --probe` wrote** (10 models: nbeats, timesfm, skorch, tabnet, pyod, osqp, prophet, xgboost, lightgbm, catboost): a `setup` call with a call as argument (`fit(numpy.array(...))`) runs in the probe, but `build` accepts only literals and names.
- **Lists of integers are typed Real** (5: deepxde, fno, tcn, neuromancer, kan): torch refuses float sizes (`layer_sizes=[1, 16, 16, 1]`). An int scalar default is typed Integer.
- **`init --probe` finds no outputs** (3) for `bytes` (rembg; FMI 3 has Binary), a pandas Series indexed by dates (pastas), numpy arrays of `dtype=object` holding numbers (mtcnn).
- **`init --probe` writes invalid TOML** when a comment quotes a multi-line error (why the object can't be pickled): wav2vec2 fails, do-mpc works through its plain `init` config.
- **An output named `time` is hidden in FMI 2** (nengo): FMPy's results name the simulation time `time`, so the simulator's clock shows 1.0 instead of 0.01. FMI 3 configs rename it `model_time`.
- **Compiled FMUs fail to instantiate without an error message** (12): the backend exits and `--capture-output` keeps its traceback in the capture file, so UniFMU reports "no output". Underneath: torchvision's `_C_stable.so`, loaded by path, isn't collected (timm, ultralytics, easyocr); code-only package folders don't exist in the frozen app, so paths through them fail (chgnet's `model/../pretrained`, speechbrain); TorchScript needs the `.py` sources of the functions it compiles (mace, norse, matgl); ray imports its `_private` modules by name (neuralforecast).

Others found with earlier start values, which the configs above avoid: a `call:` value with text after its `)` (`pathlib:Path('x.png').read_bytes()`) silently calls `Path()` (rembg and easyocr got a `PosixPath`); `init` crashed on a class as an argument default (`TypeError: '>' not supported between 'getset_descriptor' and 'int'`: numpy's `default_rng` with a `size` start, reservoirpy with a 2-D input); a factory annotated with a base class crashes plain `init` (tirex: `AttributeError: forecast`); a `--start` named like the `**kwargs` parameter is refused (MOMENT's `model_kwargs`). These are fixed too.

---

## GitHub repositories

Models published as source on GitHub, not as a package.

| Model | `init` / `init --probe` | … with other options | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|---|
| `building_physics.py:Zone` ([architecture-building-systems/RC_BuildingSimulator](https://github.com/architecture-building-systems/RC_BuildingSimulator) `97c1cdd`) | needs options | ✓ (`--call`) | ✓† | ✓ (`init --probe` config, 108.8 MB) | ISO 13790 5R1C thermal model of a building zone, `--call solve_energy`. The hand config feeds `t_m_next` back as `t_m_prev` (`[states]`) and runs two winter days of hourly weather. †It fixes `step_size = 3600`, so the generic 1 s run is refused (correct); simulated at 3600 s steps. |

---

## Unpublished models

Models with no public source: they aren't counted as published models.

| Model | `init` / `init --probe` | … with other options | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|---|
| `psychrometry.py:compute_balances_simplified` (Lucia Royo-Pascual, EIUM) | ✓ | — | ✓ | ✓ (`init --probe` config, 108.7 MB) | Mass and energy balance of an air-based drying process: a function of 18 temperatures, humidities and flow rates returning a dict. |

---

## Python libraries

Errors such as `y is an array of shape […], not a single value` in the `init` config are expected: `--probe` finds the shape.

| Model | `init` / `init --probe` | … with other options | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|---|
| `ahrs.filters:Madgwick` | ✓ | — (`--call updateIMU`: what init picks by itself; `--start` ×3) | — | ✓ (`init --probe` config, 50.3 MB) |  |
| `ahrs.filters:Mahony` | ✓ | — (`--call updateIMU`: what init picks by itself; `--start` ×3) | — | ✓ (`init --probe` config, 50.3 MB) |  |
| `ambiance:Atmosphere` | ✓ | — (`--kind function`: what init picks by itself; `--start`) | ✓ ([config](match-testing-rerun/configs/ambiance.toml)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | `init --probe` reads each one-element array property as its element (`return:temperature.0`, …): 281.65 K, 89 876 Pa at 1000 m. Before: The `init` configs output only the inputs: the results are properties returning 1-element arrays. The hand config reads element 0 of each (`return:temperature.0`, …). |
| `anastruct:SystemElements` | needs options | ✓ (`--setup` ×4; `--call solve`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 88.5 MB) |  |
| `antropy:perm_entropy` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 164.9 MB) |  |
| `arch:arch_model` | needs options | ✓ (`--call`, `--start`) | — | ✓ (`init --probe` config, 122.9 MB) | `arch_model(y)`, `--call fit`: the GARCH fit's parameters. |
| `astral.sun:elevation` | ✓ | — (`--start` ×3) | — | ✓ (`init --probe` config, 46.2 MB) |  |
| `astropy.cosmology:FlatLambdaCDM` | needs options | ✓ (`--call`, `--start` ×3) | — | ✓ (`init --probe` config, 65.2 MB) |  |
| `Bio.SeqUtils:gc_fraction` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 54.3 MB) |  |
| `blackscholes:BlackScholesCall` | needs options | ✓ (`--call`, `--start` ×5) | — | ✓ (`init --probe` config, 45.9 MB) |  |
| `cantera:Solution` | — | ✗ no useful outputs (`--call`, `--const`, `--start`) | ✓ ([config](#cantera)) | ✓ (hand config, 79.9 MB) | Outputs are only solver settings (`rtol`, `max_steps`): the results are Cython properties, and the gas state can't be set from inputs. |
| `catboost:CatBoostRegressor` | ✓ | — (`--setup fit(...)`: training the model on the user's data, the user's part, not counted as an option since 2026-10-28; `--call predict`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start` ×5) | — | ✓ (`init --probe` config, 247 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). `fit` in `setup`, `--call predict`. |
| `chaospy:Normal` | needs options | ✓ (`--call`, `--start` ×3) | — | ✓ (`init --probe` config, 92.0 MB) | `Normal(0, 1)`, `--call pdf`. |
| `chemicals.vapor_pressure:Antoine` | ✓ | — | — | ✓ (`init --probe` config, 126.5 MB) |  |
| `colour.temperature:CCT_to_xy_CIE_D` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 56.4 MB) |  |
| `commpy.modulation:QAMModem` | ✓ | — (`--call modulate`: what init picks by itself; `--start` ×2) | ✓ ([config](match-testing-rerun/configs/commpy.toml)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | `init --probe` splits the complex symbols into `y_real` / `y_imag` (`return:real`, `return:imag`): (−1+1j, 1−1j), as `modulate` called directly. Before: `init`: only `m = 4`; the modulated symbols are complex, so they're left out. The hand config reads `.real` and `.imag` per symbol. |
| `control:StateSpace` | needs options | ✓ (`--call`, `--start` ×7) | — | ✓ (`init --probe` config, 124.6 MB) | Needs `--start x=[1.0, 0.0] --start u=[0.0]`. |
| `CoolProp.CoolProp:PropsSI` | ✗ build fails | — | ✓ ([config](models.md#positional-only-arguments)) | ✓ (hand config, 56.7 MB) | C extension with no signature: no variables inferred. |
| `cv2:KalmanFilter` | ✓ | — (`--call predict`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start` ×2) | — | ✓ (`init --probe` config, 163.5 MB) |  |
| `dysts:Lorenz` (`call = "rhs"`, `[model] integrate`) | — | — | ✓ | ✓ (hand config, 267.5 MB, both integrators) | Integrated by the FMU with `[model] integrate`: `"scipy:DOP853"` (FMI 3) or `"rk4"`, 50 substeps (FMI 2). |
| `electricpy:powerset` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 123.8 MB) |  |
| `ephem:Sun` | ✓ | — (`--call compute`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start`) | ✓ ([config](pending-100/hand/ephem/fmugen.toml)) | ✓ (hand config, 46.6 MB) | `Sun`, `--call compute`: the position is in C attributes that exist only after `compute()`; the hand config reads them, with `init_call = true`. **2026-10-27:** `init --probe` alone, `--start date_or_observer=…`: the date now goes to `compute()` (the step), not `Sun(date)`, so the 18 C attributes compute sets are outputs, with `init_call`; FMU identical to the direct run (54 values). |
| `erfa:gmst06` | ✓ | — (`--start` ×4) | — | ✓ (`init --probe` config, 46.7 MB) |  |
| `filterpy.common:Q_discrete_white_noise` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 114.8 MB) |  |
| `filterpy.kalman:KalmanFilter` (hand config FMI 3: structural parameters, clock) | — | ✗ no useful outputs (`--call`, `--start` ×2) | ✓ | ✓ (hand config (FMI 3), 114.8 MB) | The `init` configs output only filter settings and likelihoods (`dim_x`, …, `alpha`, `likelihood`), not the state estimate. The hand config was driven by a script feeding it measurements on a triggered clock. |
| `filterpy.gh:GHFilter` | ✓ | — (`--call update`: what init picks by itself; `--start` ×5) | — | ✓ (`init --probe` config, 114.8 MB) | `update(z, g=None, h=None)`'s `h` is a gain: this `init` config bound it to the step size. `init` no longer does: rerun, `h` is a parameter (start 0.2) and there's no `[time]`; both configs match the direct run and the compiled FMU its uncompiled one. |
| `floris:FlorisModel` | needs options | ✓ (`--call`, `--const`) | ✓ [config](pending-100/hand/floris_lib/fmugen.toml) | ✓ 157 MB, identical | Since `8390062`: `init --probe` alone works: the step returns nothing, so it reads the getters, `get:get_farm_power`, `get:get_turbine_powers` and 10 more `get:` outputs; matches the direct run (120 values). Before: no useful outputs. The hand config sets the conditions in `setup`. |
| `fluids.core:Reynolds` | ✓ | — (`--start` ×4) | — | ✓ (`init --probe` config, 104.0 MB) |  |
| `fluids.friction:friction_factor` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 104.0 MB) |  |
| `fluids.units:head_from_P` | ✓ | — (`--start` ×2) | ✓ ([config](#fluidsunits)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | `init --probe` retries with pint quantities and takes each input's unit from the error that names it (`convert = "pint"`, `unit = "Pa"`, …): 10.197 m. Before: TypeError: 100000.0 has no quantity |
| `fluids.units:Reynolds` | ✓ | — (`--start` ×4) | ✓ ([config](#fluidsunits)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | `init --probe` retries with pint quantities, units from the errors: Re = 38 200.66. Before: TypeError: 1.0 has no quantity |
| `imufusion:Ahrs` (FMI 3) | needs options | ✓ (`init --probe`, since 2026-10-27) (`--call`, `--start` ×2) | ✓ [config](pending-100/hand/imufusion/fmugen.toml) | ✓ 45 MB | `substeps = 100`, `get:get_quaternion`. Needed a wrapper before `substeps` without `integrate` and `get:`. **2026-10-27:** `init --probe` alone with `--setup "set_sample_period(0.01)"` and gyroscope/accelerometer starts (passed by position: no signature): `update_no_magnetometer` returns the object, now read as returning nothing, so the getters (`get_quaternion`, …) are outputs; FMU identical to the direct run at 100 Hz over 6 s, past the 3 s start-up (7,800 values). |
| `GPy.models:GPRegression` | ✓ | — (`--call predict`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start` ×3) | — | ✓ (`init --probe` config, 119.4 MB) | GP regression, `--call predict`: mean and variance. **Needs matplotlib installed by hand.** |
| `gsw:rho` | ✓ | — (`--start` ×3) | — | ✓ (`init --probe` config, 48.3 MB) |  |
| `gymnasium.envs.classic_control.cartpole:CartPoleEnv` | ✓ | — (`--call step`, `--setup reset`: what init picks by itself; `--start`) | — | ✓ (seeded config, 104.0 MB) |  |
| `gymnasium.envs.classic_control.pendulum:PendulumEnv` | ✓ | — (`--call step`, `--setup reset`: what init picks by itself; `--start`) | — | ✓ (seeded config, 104.0 MB) |  |
| `hapsira.twobody:Orbit` | needs options | ✓ (`--create`, `--const` ×8; `--call propagate`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 237.2 MB) | `Orbit.from_classical` (astropy units), `--call propagate` 600 s. **Needs numpy < 2.4** (`numpy.in1d`) **and astropy < 7** (`matrix_product`). |
| `harold:lqr` (FMI 3) | needs options | ✓ (`--const`, `--start` ×2) | — | ✓ (`init --probe` config, 116.5 MB) | LQR of a 2-state system: gain K (1×2), Riccati solution X (2×2), closed-loop eigenvalues (complex, split). **Needs numpy < 2.3** (harold imports `numpy.linalg.linalg`, removed in 2.3; FMPy needs ≥ 2.1.3). |
| `haversine:haversine` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 44.8 MB) |  |
| `heartpy:process` | needs options | ✓ (`--const`, `--start`) | — | ✓ (`init --probe` config, 119.6 MB) | **Needs `setuptools<81` installed by hand** (heartpy imports `pkg_resources` without declaring it), and its `data/data.csv` given by full path. |
| `highway_env.envs.highway_env:HighwayEnv` | ✓ | — (`--call step`, `--setup reset`: what init picks by itself; `--start`) | — | ✓ (seeded config, 104.2 MB) |  |
| `hplib.hplib:HeatPump` | needs options | ✓ (`--const`, `--start` ×3) | — | ✓ (`init --probe` config, 102.7 MB) |  |
| `ht.conv_internal:Nu_conv_internal` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 104.8 MB) |  |
| `ht.hx:effectiveness_from_NTU` | ✓ | — | — | ✓ (`init --probe` config, 104.8 MB) |  |
| `iapws.iapws97:_TSat_P` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 87.8 MB) |  |
| `iapws:IAPWS97` | needs options | ✓ (`--start` ×2, `--kind`) | — | ✓ (`init --probe` config, 87.8 MB) |  |
| `impedance.models.circuits:Randles` | ✓ | — (`--call predict`: what init picks by itself; `--start` ×2) | ✓ ([config](match-testing-rerun/configs/impedance.toml)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | `init --probe` splits the complex prediction into `y_real` / `y_imag` at each frequency, identical to `predict` called directly. Needs pandas installed by hand (impedance imports it without declaring it). Before: `init`: outputs only echo `initial_guess`; the complex prediction is left out. The hand config reads `.real` and `.imag` at each frequency. |
| `joblib:load` | needs options | ✓ (`--call`, `--const`, `--start`) | — | ✓ (`init --probe` config, 112.8 MB) | **Compiled: needs `scikit-learn` added to `[model] requirements` by hand**: the model is unpickled, so nothing imports scikit-learn by name. Uncompiled, it needs nothing extra. |
| `librosa:hz_to_mel` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 167.7 MB) |  |
| `lifelines:KaplanMeierFitter` | ✓ | — (`--call fit`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start` ×2) | — | ✓ (`init --probe` config, 142.5 MB) | Kaplan–Meier, `--call fit`: the survival function. |
| `lightgbm:LGBMRegressor` | needs options | ✓ (`--call`, `--setup`, `--start` ×5) | — | ✓ (`init --probe` config, 106 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). `fit` in `setup`, `--call predict`. **Needs scikit-learn installed by hand.** |
| `mendeleev:element` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 75.7 MB) |  |
| `mesa.examples:BoltzmannWealth` | ✓ | — (`--call step`: what init picks by itself) | — | ✓ (seeded config, 105.2 MB) |  |
| `meteocalc:heat_index` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 44.8 MB) |  |
| `metpy.calc:dewpoint_from_relative_humidity` | ✓ | — (`--start` ×2) | ✓ ([config](#metpy)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | `init --probe` retries with pint quantities; metpy names only the dimension, so the input gets its SI unit (`temperature` in K): set the unit your start is in (the hand config's `degC`). Before: ValueError: This function changed in version(s) ['1.0']--double check that the function is being called properly. |
| `mujoco:MjData` (FMI 3) | ✗ no config (C built-in) | — | ✓ [config](pending-100/hand/mujoco/fmugen.toml) | ✓ 84 MB | `call = "mujoco:mj_step(self.model, self)"`, `substeps = 500`. Needed a wrapper before a step could be a function taking the object. |
| `nashpy:Game` | needs options | ✓ (`--call`, `--start`) | — | ✓ (`init --probe` config, 94.9 MB) | `--call support_enumeration`, `--start A=… --start B=…` (a 2×2 game with one mixed equilibrium). The method returns a generator: its items are read into a tuple, and `--probe` writes `y0_0` and `y0_1`, the two players' strategies in the first equilibrium. Plain `init` fails (`float() argument must be … not 'generator'`). 0.0.43 has no `is_zero_sum`. |
| `neurokit2:ecg_simulate` | ✓ | — (`--start` ×4) | — | ✓ (`init --probe` config, 150.4 MB) |  |
| `numdifftools:Derivative` | needs options | ✓ (`--const`, `--start`) | — | ✓ (`init --probe` config, 90.8 MB) |  |
| `numpy.random:default_rng` (FMI 3) | needs options | ✓ (`--call`, `--start`) | — | ✓ (`init --probe` config, 48.8 MB) | `default_rng(seed=12345678901234567890)`, `--call random`: the seed is a UInt64 parameter (FMI 3). |
| `numpy_financial:pmt` | ✓ | — (`--start` ×3) | — | ✓ (`init --probe` config, 46.0 MB) |  |
| `obspy.geodetics:gps2dist_azimuth` | ✓ | — (`--start` ×4) | — | ✓ (`init --probe` config, 145.5 MB) |  |
| `openap:Thrust` | needs options | ✓ (`--const`, `--start` ×3; `--call climb`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 106.7 MB) |  |
| `openturns:Normal` | needs options | ✓ (`--call`, `--start` ×3) | — | ✓ (`init --probe` config, 152.2 MB) |  |
| `osqp:OSQP` (FMI 3) | needs options | ✓ (`--call`, `--setup`) | — | ✓ (`init --probe` config, 91 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). `--setup "setup(scipy.sparse.csc_matrix(...), ...)"`, `--call solve`: the QP's solution and info. Not on a clock. |
| `pandapower:runpp` | — | ✗ no useful outputs (`--const`) | ✓ ([config](match-testing-rerun/configs/pandapower.toml), another entry) | ✓ (hand config, 124.1 MB) | `runpp(net)` returns None and writes its results onto `net`, which a function entry can't read back. The hand config uses `pandapower.contingency:run_contingency(net, nminus1_cases={})`, which runs the same `runpp` and returns its results (bus voltages, line and transformer loadings). |
| `pastas:Model` | needs options | ✓ (`--const`; `--call simulate`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 215 MB) | Failed before the fixes: `init --probe` finds no output: the result is a pandas Series indexed by dates. `Model(oseries)`, `--call simulate`. |
| `pde:DiffusionPDE` (FMI 3) | needs options | ✓ (`--const`, `--start`; `--call evolution_rate`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✗ FMU fails | `DiffusionPDE`, `--call evolution_rate` on a 16-cell scalar field. **Compiled FMU fails**: py-pde loads `pde/backends/*/config.py` by path, and a compiled FMU ships no `.py` files. |
| `pedon:Genuchten` | needs options | ✓ (`--call`, `--start` ×6) | — | ✓ (`init --probe` config, 129.0 MB) |  |
| `PIL.ImageOps:grayscale` (FMI 3) | needs options | ✓ (`--const`) | — | ✓ (`init --probe` config, 57.6 MB) | A 64×64 image to grayscale: the image's properties. |
| `pinocchio:rnea` | needs options | ✓ (`init --probe`, since 2026-10-27) (`--const` ×2, `--start` ×3) | ✓ ([config](pending-100/hand/pinocchio/fmugen.toml)) | ✓ (hand config, 90.0 MB) | Inverse dynamics (`rnea`) of the sample manipulator: Boost.Python functions with no signature, so `init` stops. The hand config passes model and data as positional constants and q, v, a by position. **2026-10-27:** `init --probe` alone: the `--start` values go by position (no signature); FMU identical to the direct run (18 values). |
| `pandapipes.networks:gas_meshed_delta` | needs options | ✓ (`init --probe`, since 2026-10-27) (`--call`) | ✓ [config](pending-100/hand/pandapipes_lib/fmugen.toml) | ✓ 159 MB | Factory plus `call = "pandapipes:pipeflow(self)"`, results from `attr:res_junction.p_bar` etc. Needed a wrapper before a step could be a function taking the object. **2026-10-27:** `init --probe` alone with `--call "pandapipes:pipeflow(self)"`: the result tables' columns are outputs (`res_junction.p_bar`, …, 21 in all); FMU identical to the direct run (174 values). |
| `pmdarima:auto_arima` | needs options | ✓ (`--call`, `--start` ×2) | — | ✓ (`init --probe` config, 135.8 MB) | `auto_arima`, `--call predict`. |
| `ppigrf:igrf` | ✓ | — (`--start` ×3) | — | ✓ (`init --probe` config, 60.8 MB) |  |
| `prophet:Prophet` | needs options | ✓ (`--setup`, `--const`; `--call predict`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 119 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). `fit` on 24 days in `setup` (a DataFrame built by a setup call), `--call predict` for 2 days. |
| `psychrolib:GetHumRatioFromRelHum` | needs options | ✓ (`--setup`, `--start` ×3) | — | ✓ (`init --probe` config, 44.8 MB; 110.2 MB with numba) | Also with numba installed, which makes psychrolib's functions ufuncs: `init` passes their arguments by position. |
| pvlib `get_total_irradiance` → `sapm_cell` → `pvwatts_dc` (**composite FMU**) | ✓ (each part) | — | ✓ (compiled on Windows) | The three parts made with `init --probe`. |
| `pvlib.atmosphere:alt2pres` | ✓ | — | — | ✓ (`init --probe` config, 132.8 MB) |  |
| `pvlib.iotools:read_tmy3` | needs options | ✓ (`--const`) | — | ✓ (`init --probe` config, 132.8 MB) |  |
| `pvlib.irradiance:aoi` | ✓ | — | — | ✓ (`init --probe` config, 132.8 MB) |  |
| `pvlib.pvsystem:pvwatts_dc` | ✓ | — | — | ✓ (`init --probe` config, 132.8 MB) |  |
| `pvlib.solarposition:get_solarposition` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 132.8 MB) |  |
| `pvlib.temperature:sapm_cell` | ✓ | — | — | ✓ (`init --probe` config, 132.8 MB) |  |
| `pvwatts.py:pvwatts` (nrel-pysam) | ✗ no model to point `init` at | — | — | ✗ (wrapper only) | **Fails: the library has no function or class that is the model**, so `init` has nothing to point at. Tried through a wrapper written for this test: Wrapper ([pvwatts.py](pending-100/wrappers/pvwatts.py)): `Pvwattsv8.default("PVWattsNone")` is a C built-in, and the model needs weather data set as an attribute. A 4 kW system, a synthetic year. |
| `py_vollib.black_scholes:black_scholes` | ✓ | — (`--start` ×6) | — | ✓ (`init --probe` config, 102.7 MB) |  |
| `pybamm:Simulation` | — | ✗ no useful outputs (`--call`, `--const`, `--start`) | ✓ ([config](match-testing-rerun/configs/pybamm.toml)) | ✓ (hand config, 265.2 MB) | `init`: outputs are only solver flags. The hand config reads the voltage as `return:Voltage [V].entries.-1`: `solution["Voltage [V]"]` on the `Solution` that `step()` returns. Without `init_call`: it would run one extra `step()` before t = 0. |
| `PyCO2SYS:sys` | ✓ | — (`--start` ×4) | — | ✓ (`init --probe` config, 46.1 MB) |  |
| `pydmd:DMD` (FMI 3) | needs options | ✓ (`--call`, `--const`, `--start`) | ✓ ([config](pending-100/hand/pydmd/fmugen.toml)) | ✓ 154 MB, identical (`init --probe` config) | DMD with 2 modes of 8 snapshots. Since `8390062`: the probe sees DMD's properties fail before `fit()` and sets `init_call = true`; matches the direct run (832 values). Before, the `init --probe` config read them at initialization, before `fit()`, and failed; the hand config does the same with `init_call = true`. |
| `pyet:pm_fao56` | ✓ | — (`--start` ×6) | — | ✓ (`init --probe` config, 72.2 MB) | The inputs must be pandas Series with a date index (the day of the year comes from `rs.index`). With plain numbers the call fails (`'float' object has no attribute 'time'`), so `--probe` writes `convert = "series"` with an `epoch` for each number input. Checked over 5 days with varying `rs` and `tmean`. |
| `pyfluids:Fluid` | needs options | ✓ (`--call`, `--const` ×3) | — | ✓ (`init --probe` config, 56.7 MB) |  |
| `pygfunction.heat_transfer:finite_line_source` | needs options | ✓ (`--const` ×2, `--start` ×2) | — | ✓ (`init --probe` config, 88.2 MB) |  |
| `pykalman:KalmanFilter` | ✓ | — (`--call filter_update`: what init picks by itself; `--start` ×3) | — | ✓ (`init --probe` config, 92.0 MB) |  |
| `pyloudnorm:Meter` | needs options | ✓ (`--const`, `--start`; `--call integrated_loudness`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 91.2 MB) |  |
| `pymap3d:geodetic2enu` | ✓ | — (`--start` ×6) | — | ✓ (`init --probe` config, 46.0 MB) |  |
| `pymoo.optimize:minimize` | needs options | ✓ (`--const` ×2, `--start` ×2) | — | ✓ (`init --probe` config, 134.0 MB) | NSGA-II on ZDT1, 5 generations, seed 1: the Pareto set and front. |
| `pymsis:calculate` | ✓ | — (`--start` ×4) | — | ✓ (`init --probe` config, 48.8 MB) |  |
| `pyorbital.astronomy:sun_zenith_angle` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 107.7 MB) |  |
| `pyproj:Transformer` | needs options | ✓ (`--create`, `--const` ×2, `--start` ×3) | — | ✓ (`init --probe` config, 61.2 MB) |  |
| `pyquaternion:Quaternion` | ✓ | — (`--call rotate`: what init picks by itself; `--start` ×3) | — | ✓ (`init --probe` config, 48.9 MB) | `--call rotate`, axis, angle, vector. Plain `init` fails: it passes axis and angle to the constructor by position. |
| `pyromat:get` | needs options | ✓ (`--call`, `--const`, `--start`) | — | ✓ (`init --probe` config, 47.0 MB) |  |
| `pysindy:SINDy` (FMI 3) | ✓ | — (`--call fit`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start` ×2) | — | ✓ (`init --probe` config, 99.9 MB) | SINDy fitted to two decaying exponentials: the coefficient matrix (2×10). |
| `pysolar.solar:get_altitude` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 45.9 MB) |  |
| `pyswmm:Simulation` | needs options | ✓ (`--call`, `--const`) | — | ✓ (`init` config, 46.1 MB) | `Simulation(inputfile)`, `--call __next__`: steps pyswmm's `model_weir_setting.inp`. Plain `init` works. The `init --probe` config failed (it reads the current time before the simulation starts); since `8390062` the probe sets `init_call = true` and it matches the direct run (47 values) and compiles identically (46 MB). |
| `PyTCI.models.propofol:Marsh` | — | ✗ no useful outputs (`--call`, `--start` ×2) | ✓ ([config](#pytci)) | ✓ (hand config, 87.6 MB) | Outputs only echo the parameters (`weight`, `time_seconds`): the state is in a nested function. |
| `pyteomics.mass:calculate_mass` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 51.6 MB) |  |
| `pythermalcomfort.models.jos3:JOS3` | ✓ | — (`--call simulate`: what init picks by itself; `--start` ×2) | ✓ ([config](#jos3)) | ✓ (`init --probe` config, 153.8 MB) |  |
| `pythermalcomfort.models:pmv_ppd_iso` | ✓ | — (`--start` ×6) | — | ✓ (`init --probe` config, 153.8 MB) |  |
| `pythermalcomfort.models:sports_heat_stress_risk` | ✓ | — (`--start` ×8) | — | ✓ (`init --probe` config, 153.8 MB) |  |
| `python_speech_features:mfcc` (FMI 3) | needs options | ✓ (`--const`, `--start`) | — | ✓ (`init --probe` config, 87.0 MB) | MFCC matrix of 5 s of speech: 499 × 13 (FMI 3 2-D array). |
| `pywt:dwt` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 53.3 MB) |  |
| `pyxirr:npv` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 45.3 MB) |  |
| `pyzbar.pyzbar:decode` (FMI 3) | needs options | ✓ (`--const`) | — | ✓ (`init --probe` config, 59.6 MB) | Decodes the QR image: `fmugen`. **Needs the zbar library** (`libzbar0t64`, here extracted without root and found through `LD_LIBRARY_PATH`). |
| `qrcode:make` (FMI 3) | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 62.6 MB) | `make("fmugen")`: the QR image's properties (size, box size, …). |
| `quantecon:LinearStateSpace` | needs options | ✓ (`--call`, `--start` ×3) | — | ✓ (`init --probe` config, 156.1 MB) |  |
| `QuantLib:blackFormula` | ✓ | — (`--start` ×4) | — | ✓ (`init --probe` config, 78.8 MB) |  |
| `radioactivedecay:Inventory` | needs options | ✓ (`--const`, `--start`; `--call decay`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 135.5 MB) |  |
| `rainflow:count_cycles` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 48.8 MB) |  |
| `rdkit.Chem.Descriptors:MolWt` | needs options | ✓ (`--const`) | — | ✓ (`init --probe` config, 110.7 MB) |  |
| `river.linear_model:LinearRegression` | ✓ | — (`--call learn_one`: what init picks by itself; `--start` ×2) | ✓ ([config](match-testing-rerun/configs/river.toml)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | `init --probe` outputs the learned `intercept` (0.06 after the first step, as `learn_one` called directly): with no other outputs found, the attributes the step changes become outputs instead of locals. The weights (a dict) still need the hand config. Before: `init`: only hyperparameters (`l1`, `l2`, …). The hand config reads the learned weights (`attr:weights.a`, …) and the intercept. |
| `roboticstoolbox.models.DH:Panda` | needs options | ✓ (`--call`, `--start`) | — | ✓ (`init --probe` config, 172.7 MB) |  |
| `scipy.signal:lfilter` | ✓ | — (`--start` ×3) | — | ✓ (`init --probe` config, 98.3 MB) |  |
| `sectionproperties.analysis:Section` | needs options | ✓ (`--call`, `--const`) | ✓ [config](pending-100/hand/section_lib/fmugen.toml) | ✓ 141 MB, identical | Since `8390062`: `init --probe` alone works: `get:get_area`, `get_ic`, `get_c` and 10 more `get:` outputs; matches the direct run (81 values). Before: no outputs. The hand config passes the meshed geometry as a computed constant. |
| `seirsplus.models:SEIRSModel` | — | ✗ not possible (FMI limitation) (`--call`, `--start` ×6) | ✓ ([config](match-testing-rerun/configs/seirsplus.toml)) | ✓ (hand config, 94.8 MB) | **Not possible with `init` (FMI limitation):** the counts (`numS`, `numI`, …) are arrays that grow by one element each step, and an FMI variable has a fixed size ([Limitations](#known-problems)). `init`: outputs only echo the parameters; the state arrays grow each step and are left out. The hand config reads the latest counts (`attr:numS.-1`, …) and sets `dt = 0.1`: with `dt` = the step size (1), `run_epoch` keeps only the start point and `t` never advances. |
| `sgp4.api:Satrec` | needs options | ✓ (`init --probe`, since 2026-10-27) (`--call`, `--create`, `--const` ×2, `--start` ×2) | ✓ ([config](match-testing-rerun/configs/sgp4.toml)) | ✓ (hand config, 44.9 MB) | `init`: C built-ins with no signature: `--start fr, jd, line1, line2: cannot tell which arguments these are …`. The hand config builds the satellite with `create = "twoline2rv"` from the two TLE lines and binds `sgp4(jd, fr)` by position (`to = "pos:N"`); outputs the error code, position (km) and velocity (km/s). Checked against `Satrec.sgp4` called directly: identical; the compiled FMU (PyInstaller 6.22.3, WSL 2) is identical to the uncompiled one. **2026-10-27:** `init --probe` alone, same `--create` and `--start` values: they go by position (no signature), line1/line2 to `twoline2rv` and jd/fr to `sgp4` (the probe's split); outputs the error code, position and velocity; FMU identical to the direct run (21 values). |
| `shapely:Point` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 50.3 MB) |  |
| `simdkalman:KalmanFilter` (FMI 3) | needs options | ✓ (`init --probe`, since 2026-10-27) (`--call`, `--start` ×5) | ✓ ([config](pending-100/hand/simdkalman/fmugen.toml)) | ✓ (hand config, 48.9 MB) | Kalman smoother of a 24-point series. `smooth` returns a `Result` whose estimates are two levels down (`result.observations.mean`): the hand config reads them (24 levels, 24×2 states). **2026-10-27:** `init --probe` alone: objects inside the result are now read, giving `states.mean`, `states.cov`, `observations.mean`, `observations.cov`; FMU identical to the direct run (576 values). |
| `simglucose.patient.t1dpatient:T1DPatient` | needs options | ✓ (`--const` ×2) | — | ✓ (`init --probe` config, 134.0 MB) |  |
| `simple_pid:PID` | ✓ | — (`--start`) | ✓ | ✓ (`init --probe` config, 111.0 MB) |  |
| `simple_pid:PID` (hand config FMI 3: clocks) | ✓ | — (`--start`) | ✓ | ✓ (hand config (FMI 3), 111.1 MB) | A sampled-data controller: the hand config runs it on a clock ticked every 0.1 s, driven by a script with a first-order plant. |
| `simpy:Environment` | needs options | ✓ (`--call`) | — | ✓ (`init --probe` config, 44.8 MB) | `--call run`: `until` is bound to the step's end time. |
| `skfem:MeshTri` (FMI 3) | ✓ | — (`--call refined`: what init picks by itself; `--start`) | — | ✓ (`init --probe` config, 90.5 MB) | `MeshTri().refined(2)`: the refined mesh's nodes and elements (2-D arrays). |
| `skfuzzy:interp_membership` | ✓ | — (`--start` ×3) | — | ✓ (`init --probe` config, 94.5 MB) | **Needs scipy, packaging and networkx installed by hand**: scikit-fuzzy imports them without declaring them. |
| `skimage.color:rgb2hsv` | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 127.1 MB) |  |
| `skrf.media:DefinedGammaZ0` | needs options | ✓ (`--const`, `--start` ×2; `--call line`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 107.0 MB) | A 90° line (`DefinedGammaZ0.line`) at 3 frequencies: S-parameters, complex, split. |
| `slycot:sb02md` (FMI 3) | ✓ | — (`--start` ×5) | — | ✓ (`init --probe` config, 69.7 MB) | `sb02md`, the continuous Riccati equation (SLICOT, Fortran): X, eigenvalues, condition. |
| `ssqueezepy:ssq_cwt` (FMI 3) | ✓ | — (`--start`) | ✓ ([config](pending-100/hand/ssq_cwt/fmugen.toml)) | ✓ 158 MB, identical (`init --probe` config) | Synchrosqueezed CWT of a 64-sample sine. Since `8390062`: the probe fails with the FMU time as `t` (the signal's time vector, default None) and works with the default, so `t` isn't bound: outputs the two transforms (162×64, real and imaginary parts), the frequencies and the scales; matches the direct run (167,184 values). Before, `init` bound `t` to the FMU time and the model got a float. |
| `statsforecast.models:AutoARIMA` | ✓ | — (`--call forecast`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start` ×2) | — | ✓ (`init --probe` config, 225.3 MB) | AutoARIMA, `--call forecast`, h = 4. |
| `statsmodels.stats.proportion:proportion_confint` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 121.1 MB) |  |
| `sympy:lambdify` | needs options | ✓ (`init --probe`, since 2026-10-27) (`--call`, `--const` ×2, `--start` ×2) | ✓ ([config](pending-100/hand/sympy_lambdify/fmugen.toml)) | ✓ (hand config, 64.9 MB) | `lambdify` makes a function of x and y by position, which `init` can't see; the hand config binds them with `to = "pos:N"`. **2026-10-27:** `init --probe` alone with `--call __call__`: the generated function's own arguments x, y are inputs, its result `result`; FMU identical to the direct run with x, y set at simulation time (2.5, 0.7). |
| `tclab:TCLabModel` | ✓ | — | — | ✓ (`init` config, 44.9 MB) |  |
| `tellurium:loada` | — | ✗ no useful outputs (`--call`, `--const`, `--start` ×2) | ✓ ([config](match-testing-rerun/configs/tellurium.toml)) | ✓ (hand config, 211.1 MB) | `init`: only `oneStep`'s return (the time reached). The hand config reads the species from the RoadRunner object (`attr:S1`, `attr:S2`). `currentTime` is a fixed input there; bind it with `[time] currentTime = "time"` to follow the simulation clock. On Linux it needs `LD_LIBRARY_PATH` set to Python's `lib` folder. |
| `thermo:Chemical` | needs options | ✓ (`--const`, `--start` ×2, `--kind`) | — | ✓ (`init --probe` config, 133.1 MB) |  |
| `thermofeel:calculate_heat_index_simplified` | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 48.9 MB) |  |
| `timezonefinder:TimezoneFinder` | ✓ | — (`--call timezone_at`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start` ×2) | — | ✓ (`init --probe` config, 80.6 MB) |  |
| `tmm:coh_tmm` | ✓ | — (`--start` ×5) | — | ✓ (`init --probe` config, 49.1 MB) |  |
| `transforms3d.euler:euler2quat` | ✓ | — (`--start` ×3) | — | ✓ (`init --probe` config, 47.3 MB) |  |
| `uncertainties:ufloat` | ✓ | — (`--start` ×2) | ✓ ([config](match-testing-rerun/configs/uncertainties.toml)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | `init --probe` reads the returned object's properties; those named like an argument get the result's name: `y_nominal_value`, `y_std_dev`. Before: `init`: only `s`, the `ufloat` itself. The hand config reads `return:nominal_value` and `return:std_dev`. |
| `vqf:VQF` (FMI 3) | needs options | ✓ (`init --probe`, since 2026-10-27) (`--call`, `--const` ×2, `--start`) | ✓ [config](pending-100/hand/vqf/fmugen.toml) | ✓ 48 MB | `substeps = 100`, `get:getQuat6D`. Needed a wrapper before `substeps` without `integrate` and `get:`. **2026-10-27:** `init --probe` alone, `--start gyrTs=0.01` (by position: no signature): the getters (`getQuat6D`, …) are outputs; FMU identical to the direct run (57 values). |
| `windpowerlib.wind_speed:logarithmic_profile` | ✓ | — (`--start` ×4) | — | ✓ (`init --probe` config, 62.9 MB) |  |
| `wntr.sim:EpanetSimulator` (FMI 3) | needs options | ✓ (`--const`; `--call run_sim`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 146.0 MB) | EPANET simulation of `Net1.inp`: pressures at the network's junctions (2-D, time × node). |
| `xgboost:XGBRegressor` | needs options | ✓ (`--call`, `--setup`, `--start` ×3) | — | ✓ (`init --probe` config, 470 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). `fit` in `setup`, `--call predict`. **Compiled: needs hypothesis installed**: PyInstaller imports every xgboost submodule, and `xgboost.testing` calls `pytest.importorskip("hypothesis")`, whose exception stops PyInstaller. **Needs scikit-learn installed by hand.** |

---

## Neural networks

Each was installed as published (`pip install <package>`) into its own venv together with fmugen. Neural networks need `--probe`: their result sizes depend on the weights, and their code is mostly compiled or framework code. Weights from Hugging Face are downloaded on first use, unless the FMU is built with `--hf-weights` (automatic with `--vendor` / `--compile`). `--capture-output` is needed whenever a model prints while loading, which hangs UniFMU 0.14 otherwise. Compiled on Linux, their FMUs are 4.3–4.6 GB: torch from PyPI brings the NVIDIA CUDA libraries and triton (about 4 GB installed), and fmugen collects every dependency's binaries. Installing a CPU-only torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) avoids them; Windows' torch has no CUDA, and compiled Windows FMUs were about 300 MB. The models tested later used the CPU-only torch, so their compiled Linux FMUs are 0.3–2.9 GB.

| Model | `init` / `init --probe` | … with other options | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|---|
| `chgnet.model:CHGNet` | needs options | ✓ (`--call`, `--create`, `--const`) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message: chgnet opens `chgnet/model/../pretrained/…`, and `model/` doesn't exist in the frozen app (code-only folder). `--create load --call predict_structure` on fcc aluminium (pymatgen): energy, forces, stress, magnetic moments. |
| `chronos:ChronosBoltPipeline` | needs options | ✓ (`--const`, `--start`; `--call predict`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 4.6 GB) |  |
| `d3rlpy.algos:CQLConfig` | ✓ | — (`--call create`: what init picks by itself; `--start`) | — | ✓ (`init --probe` config, 346.4 MB) | `CQLConfig(...).create()`: the CQL algorithm object's settings. |
| `darts.models:NBEATSModel` | ✓ | — (`--setup fit(...)`: training the model on the user's data, the user's part, not counted as an option since 2026-10-28; `--start` ×5) | — | ✓ (`init --probe` config, 473 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). `--setup "fit(darts.TimeSeries.from_values(numpy.array([...])))"`, `--call predict`: 2 epochs, seeded. **Needs `darts[torch]`**: plain `darts` gives placeholders for its torch models. |
| `deepxde.nn:FNN` | needs options | ✓ (seeded) (`--const` ×2, `--start` ×2) | — | ✓ (seeded config, 368 MB) | Failed before the fixes: the integer list start is typed Real, so the model gets float sizes (`empty(): argument 'size' must be tuple of ints`). A 1-16-16-1 tanh network (PyTorch backend, `DDE_BACKEND=pytorch`). Random weights: compared seeded (`torch:manual_seed(0)` in `setup`). |
| `easyocr:Reader` (FMI 3) | needs options | ✓ (`--call`, `--const`, `--start` ×4) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message: torchvision's `_C_stable.so`, loaded by path, isn't collected. OCR of a rendered `FMUGEN 2026` image (bytes): the text. |
| `equinox.nn:MLP` | needs options | ✓ (`--const`, `--start` ×5) | — | ✓ (`init --probe` config, 273.1 MB) | A 2-8-8-1 MLP, seeded by its `PRNGKey(0)`. |
| `kan:KAN` | ✓ | — (`--start` ×6) | — | ✓ 365 MB, identical (`init --probe` config) | Since `8390062`: the `init --probe` config sets `init_call = true` and matches the direct run (56 values). Before, it failed at initialization (`missing input data x`); before the earlier fixes, `width=[2, 3, 1]` was typed Real (`'float' object is not subscriptable`). A 2-3-1 KAN (grid 3, k 3, seed 0). **Needs scikit-learn, pandas, matplotlib, pyyaml and tqdm installed by hand.** |
| `keras.applications:MobileNetV2` | needs options | ✓ (`--call`, `--const` ×2, `--start`) | — | ✓ (`init --probe` config, 283.0 MB) | MobileNetV2 (ImageNet), `--call predict`, **Keras on its JAX backend** (`KERAS_BACKEND=jax`): TensorFlow needs protobuf ≥ 5.28 and fmugen pins 5.27.3 (UniFMU 0.14's). |
| `mace.calculators:mace_mp` | needs options | ✓ (`--call`, `--const` ×2, `--start` ×2) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message: TorchScript needs the `.py` sources of the functions it compiles at import. MACE-MP-0 small, `--call get_potential_energy` on ASE's fcc copper. |
| `matgl.ext.ase:PESCalculator` | needs options | ✓ (`--call`, `--const` ×2) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message: TorchScript needs the `.py` sources of the functions it compiles at import. TensorNet-PES-MatPES-PBE-2025.2 through its ASE calculator (`matgl.ext.ase:PESCalculator`, `--call get_potential_energy`) on fcc aluminium: −3.7124 eV. `load_model` returns a `Potential` with no `predict_structure`. |
| `moirai_fc.py:MoiraiForecaster` (uni2ts) | — | ✗ no model to point `init` at (`--call`) | — | ✗ (wrapper only) | **Fails: the library has no function or class that is the model**, so `init` has nothing to point at. Tried through a wrapper written for this test: Wrapper ([moirai_fc.py](pending-100/wrappers/moirai_fc.py)): Moirai's `forward` takes GluonTS-prepared tensors; its documented use is `create_predictor`. Moirai 1.0 R small, zero-shot, median of 20 seeded samples. **uni2ts pins numpy 1.26**, which has no Python 3.13 wheels: installed with `--no-deps`. |
| `momentfm:MOMENTPipeline` | needs options | ✓ (`--call`, `--setup`, `--const` ×2, `--start`) | — | ✓ (`init --probe` config, 1.2 GB) | MOMENT-1-small, `--create from_pretrained --call forward`, `model_kwargs={'task_name': 'reconstruction'}`, `--setup init`: the reconstruction of a 512-point input. **momentfm pins numpy 1.25**: installed with `--no-deps`. |
| `ncps.torch:CfC` | ✓ (seeded) | — (`--start` ×3) | — | ✓ (seeded config, 283.0 MB) | Closed-form continuous-time network, 8 units, on a 4-step sequence. Random weights: compared seeded. |
| `nengo:Simulator` | needs options | ✓ (`--call`, `--const`, `--start` ×2) | — | ✓ (`init --probe` config, 48 MB) | Failed before the fixes: the output `time` (the simulator's clock) shows the simulation time in FMPy's results, which name the simulation time `time`: 1.0 instead of 0.01. `nengo.Simulator` of an empty network, `--call run_steps`. |
| `neuralop.models:FNO` | ✓ (seeded) | — (`--start` ×5) | — | ✓ (seeded config, 405 MB) | Failed before the fixes: the integer list start is typed Real, so the model gets float sizes (`empty(): argument 'size' must be tuple of ints`). Fourier neural operator, 8 modes, on a 16-point input. Random weights: compared seeded. |
| `neuromancer.modules.blocks:MLP` | ✓ (seeded) | — (`--start` ×4) | — | ✓ (seeded config, 681 MB) | Failed before the fixes: the integer list start is typed Real, so the model gets float sizes (`empty(): argument 'size' must be tuple of ints`). A 2-8-1 MLP block. Random weights: compared seeded. **Needs wandb installed by hand.** |
| `neuralforecast:NeuralForecast` | needs options | ✓ (`init --probe`, since 2026-10-27) (`--call`, `--setup`, `--const` ×3) | ✓ [config](pending-100/hand/nhits_lib/fmugen.toml) | ✓ 552 MB | NHITS as a computed constant, `fit(df=…)` in `setup`, `predict` each step. `init` bound `predict`'s `h` (the forecast horizon) to the step size, fixed since: rerun, `h` stays a constant with its default and there's no `[time]` (`init` checked only, not built); the hand config leaves `[time]` out. **2026-10-27:** `init --probe` alone: a table's numeric columns are now outputs, `NHITS` (4 values); FMU identical to the direct run (12 values). |
| `norse.torch:LIFCell` | ✓ | — (`--start`) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message: TorchScript needs the `.py` sources of the functions it compiles at import. LIF cell, `--call forward` with `input_tensor`: spikes, voltage and current. |
| `onnxruntime:InferenceSession` | needs options | ✓ (`--call`, `--const` ×3) | — | ✓ (`init --probe` config, 76.5 MB) | The ONNX model zoo's MNIST network, `--call run` with `input_feed=call:builtins:dict(Input3=numpy.zeros(...))`: 10 logits. |
| `pyod.models.auto_encoder:AutoEncoder` | ✓ | — (`--setup fit(...)`: training the model on the user's data, the user's part, not counted as an option since 2026-10-28; `--start` ×5) | — | ✓ (`init --probe` config, 432 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). Autoencoder detector, `fit` in `setup` on 40 rows, `--call decision_function`. With 8 rows its training loop fails (`loss` unbound). **Needs tqdm installed by hand.** |
| `pytorch_tabnet.tab_model:TabNetRegressor` | ✓ | — (`--setup fit(...)`: training the model on the user's data, the user's part, not counted as an option since 2026-10-28; `--call predict`: what init picks by itself since 2026-10-28, with a warning listing the candidates; `--start` ×3) | — | ✓ (`init --probe` config, 347 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). `fit` in `setup` (3 epochs, seeded), `--call predict`. **Needs pandas installed by hand.** |
| `pytorch_tcn:TCN` | ✓ (seeded) | — (`--start` ×4) | — | ✓ (seeded config, 283 MB) | Failed before the fixes: the integer list start is typed Real, so the model gets float sizes (`empty(): argument 'size' must be tuple of ints`). Causal TCN with channels `[4, 4]` on a 16-point input. Random weights: compared seeded. |
| `rembg:remove` (FMI 3) | needs options | ✓ (`--const`) | — | ✓ (`init --probe` config, 208 MB) | Failed before the fixes: `init --probe` finds no output: the result is `bytes`, which could be an FMI 3 Binary output. U²-Net background removal: PNG bytes in, PNG bytes out. |
| `reservoirpy.nodes:Reservoir` | needs options | ✓ (`--call`, `--start` ×3) | — | ✓ (`init --probe` config, 91.1 MB) | A 10-unit reservoir, `--call step` on a 1-D input. |
| `sb3_contrib:RecurrentPPO` | needs options | ✓ (`--const`, `--start` ×2) | — | ✓ (`init --probe` config, 291.9 MB) | LSTM PPO on CartPole (`sb3/ppo_lstm-CartPoleNoVel-v1`), `--create load --call predict`: action and LSTM states, carried between steps. **Needs gym and shimmy.** |
| `sentence_transformers:SentenceTransformer` | needs options | ✓ (`--call`, `--const`, `--start`) | — | ✓ (`init --probe` config, 2.9 GB) | `all-MiniLM-L6-v2`, `--call encode`: the 384-value embedding. |
| `silero_vad:load_silero_vad` | ✓ | — (`--call __call__`: what init picks by itself; `--start` ×3) | — | ✓ (`init --probe` config, 4.5 GB) |  |
| `skorch:NeuralNetRegressor` | needs options | ✓ (`--setup` ×2, `--const`, `--start` ×3; `--call predict`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 338 MB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). A `torch.nn.Linear(4, 1)` behind `NeuralNetRegressor`, `fit` in `setup`, `--call predict`. |
| `snntorch:Leaky` | ✓ | — (`--start` ×3) | — | ✓ (`init --probe` config, 280.4 MB) | A leaky integrate-and-fire neuron with its hidden state: spikes and membrane potential between steps. |
| `speechbrain.inference.speaker:EncoderClassifier` | needs options | ✓ (`--call`, `--create`, `--const` ×2) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message: speechbrain lists its `utils/` folder, which doesn't exist in the frozen app. ECAPA speaker embedding, `--create from_hparams --call encode_batch`: 192 values. |
| `stable_baselines3:PPO` | needs options | ✓ (`--const`, `--start` ×2) | — | ✓ (`init --probe` config, 4.5 GB) |  |
| `stable_baselines3:SAC` | needs options | ✓ (`--const`, `--start` ×2) | — | ✓ (`init --probe` config, 295.5 MB) | SAC on Pendulum from the Hugging Face hub (`sb3/sac-Pendulum-v1`), `--create load --call predict`. **Needs gym and shimmy**: the saved model unpickles with them. |
| `surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.regression.test_functions.gradient_boosting_regressor:GradientBoostingRegressorFunction` | ✓ | — (`--call __call__`: what init picks by itself; `--start` ×3) | — | ✓ (seeded config, 613.9 MB) |  |
| `timesfm:TimesFM_2p5_200M_torch` | needs options | ✓ (`--create`, `--setup`, `--const`, `--start` ×2) | — | ✓ (`init --probe` config, 2.9 GB) | Failed before the fixes: `build` refuses the config `init --probe` wrote: its `setup` has a call as an argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or names`). TimesFM 2.5 (200M), `--create from_pretrained --call forecast` with `--setup "compile(timesfm.ForecastConfig(max_context=64, max_horizon=8))"` (a setup call with a call as argument). Point and quantile forecasts of 4 values. |
| `timm:create_model` | needs options | ✓ (`--call`, `--const` ×2, `--start`) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message: torchvision's `_C_stable.so`, loaded by path, isn't collected. `resnet18`, pretrained, on a 224×224 image tensor: 1000 logits. |
| `tirex:load_model` | needs options | ✓ (`--call`, `--const`, `--start` ×3) | — | ✓ (`init --probe` config, 1.1 GB) | `load_model("NX-AI/TiRex")`, `--call forecast` on a 24-point series: quantiles and mean of 4 values. |
| `torchani.models:ANI2x` | needs options | ✓ (`--call`, `--start` ×2) | — | ✓ (`init --probe` config, 4.4 GB) |  |
| `torchdyn.core:NeuralODE` | needs options | ✓ (`--const` ×3) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message. `NeuralODE` around a `torch.nn.Linear(2, 2)` vector field. Random weights: compared seeded. |
| `torchvision.models:resnet18` | needs options | ✓ (`--call`, `--const` ×2) | — | ✓ (`init --probe` config, 298.1 MB) | ImageNet weights on a 224×224 image tensor: 1000 logits. |
| `transformers:pipeline` | needs options | ✓ (`--call`, `--const` ×2, `--start`) | — | ✓ (`init --probe` config) | Before the fixes, the compiled FMU failed to instantiate with no error message. `"sentiment-analysis"` (DistilBERT SST-2), `--call __call__`: label and score. Plain `init` stops: the object the factory returns is only known by running it. |
| `tsfm_public.toolkit.get_model:get_model` | needs options | ✓ (`--call`, `--const`, `--start` ×3) | — | ✓ (`init --probe` config, 4.3 GB) |  |
| `w2v_asr.py:Wav2Vec2ASR` (torchaudio) | — | ✗ no model to point `init` at (`--call`, `--const`) | — | ✗ (wrapper only) | **Fails: the library has no function or class that is the model**, so `init` has nothing to point at. Tried through a wrapper written for this test: **Fails: `init --probe` writes invalid TOML**: a comment quotes a multi-line error (why the object can't be pickled). Plain `init` types the transcript as Real. Passes with the fixes. Wrapper ([w2v_asr.py](pending-100/wrappers/w2v_asr.py)): the pretrained model is a bundle constant's `get_model()`, and CTC decoding is code. The transcript of the speech clip. |
| `whisper:load_model` | needs options | ✓ (`--call`, `--const` ×2, `--start` ×2) | — | ✓ (`init --probe` config, 735.9 MB) | `tiny`, `--call transcribe` on 5 s of whisper's `jfk.flac` (16 kHz numpy array): the text. A file path needs ffmpeg. |

---

## GEKKO models

Dynamic optimization and simulation models built on GEKKO, FMI 2, checked against a direct run and against each model's published results where it has them. GEKKO's distillation script and APMonitor's ethanol script have nothing to call, so their code is wrapped in a class, unchanged.

| Model | `init` / `init --probe` | … with other options | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|---|
| `distillation_column.py:DistillationColumn` (GEKKO [`examples/distillation.py`](https://github.com/BYU-PRISM/GEKKO/blob/master/examples/distillation.py)) | ✓ | — (`--call step`: what init picks by itself) | — | ✓ (`init --probe` config, 62.1 MB) | Distillation column (GEKKO's example, wrapped in a class with its equations unchanged). `--call step`. |
| `DistColEnvDynInc_altrewards.py:DistColEnvDynInc` ([Kalpesh-RL/DistillationColumn](https://github.com/Kalpesh-RL/DistillationColumn) `f2b9966`) | needs options | ✓ (`--setup`, `--start`) | — | ✓ (seeded config, 63.1 MB) | Distillation column as a Gym environment. `--start "Action=[0.0, 0.0, 0.0]" --setup reset`. Random by design: seed it (`random:seed(0)`, `numpy.random:seed(0)` before construction) to reproduce a run. |
| `Bioreactor.py:SimpleBioreactor` ([Bartosz1280/Bioreactor-modeling](https://github.com/Bartosz1280/Bioreactor-modeling) `3112aba`) | needs options | ✓ (`--call`, `--const`, `--start`) | — | ✓ (`init --probe` config, 97.8 MB) | Batch bioreactor (Monod growth). `--call solve`, `time_arr` and `simulation_input` from the repo notebook. |
| `Code/model.py:CGE` ([konradarchicinski/cge_model_python](https://github.com/konradarchicinski/cge_model_python) `60d611d`) | needs options | ✓ (`--call`, `--start` ×2) | — | ✓ (`init --probe` config, 64 MB, Windows) | Computable general equilibrium economy. `--call results --start capital=0.1 --start labour=0.0`. **Needs its `Code/model.py` renamed** (any other name): `build` refuses a model file named `model.py`, which clashes with UniFMU's backend module. **Needs pandas 2**: with pandas 3 the model itself fails (its chained `df[a][b] = …` assignments don't write under copy-on-write). **Windows only**: its data paths are hardcoded with backslashes (`Data\production_structure.csv`), which on Linux are file names that FMPy refuses in an FMU. |
| `CFTOCSolverV2.py:CFTOCSolverV2` ([ollyme102b/ros_workspace](https://github.com/ollyme102b/ros_workspace) `03f5938`) | needs options | ✓ (`--call`, `--const` ×5, `--start` ×5) | ✓ ([config](match-testing-rerun/configs/gk_cftoc.toml)) | ✓ (`init --probe` config, [identical to the direct run](probe-9/CONCLUSION.md)) | Constrained finite-time optimal control. `--call calculate_optimal_actuation`. `init --probe` with the repo test's values as `--start` (`x0`, `Xm` and the method's `calculate_optimal_actuation_x0`, `calculate_optimal_actuation_Xm` as `call:numpy:…`) writes the same config as the hand one; y = (−1.7×10⁻⁷, −7.7×10⁻⁴, 0) at t = 1, as the hand config. Before the fix, `init` refused the method's prefixed `call:` starts. |
| `debutanizer.model:solve_steady_state` ([Arushan45/Debutanizer-Project](https://github.com/Arushan45/Debutanizer-Project) `f6fe813`) | ✓ | — (`--start` ×2) | — | ✓ (`init --probe` config, 57 MB) | Debutanizer column, steady state (30 stages). `--start reflux_ratio=2.5 --start alpha=1.9717789099`. **The repository must be installed** (`pip install -e .`): the entry is a package imported by name. |
| `debutanizer.model:optimize_reflux` ([Arushan45/Debutanizer-Project](https://github.com/Arushan45/Debutanizer-Project) `f6fe813`) | ✓ | — (`--start`) | — | ✓ (`init --probe` config, 57 MB) | Debutanizer reflux optimization. `--start alpha=1.9717789099`. **The repository must be installed**, as for the steady state. |
| `apsis/mpc.py:solve_mpc` ([dhruvhaldar/apsis](https://github.com/dhruvhaldar/apsis) `a171cf1`) | ✓ | — (`--start` ×9) | — | ✓ (`init --probe` config, 62.1 MB) | Linear MPC. The app's defaults. |
| `process_fopdt.py:ProcessModel` ([BYU-PRISM/Transformer_MPC](https://github.com/BYU-PRISM/Transformer_MPC) `7686735`) | ✓ | — (`--call run`: what init picks by itself; `--start`) | — | ✓ (`init --probe` config, 130 MB with matplotlib and scipy) | 2×2 FOPDT process (a `GEKKO` subclass). `--call run --start "u=[1.0, 0.5]"`. |
| `mpc_gekko.py:MPCModel` ([BYU-PRISM/Transformer_MPC](https://github.com/BYU-PRISM/Transformer_MPC) `7686735`) | ✓ | — (`--call solve`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 62.1 MB) | MPC controller for that process (a `GEKKO` subclass). `--call solve`. Its setpoints are attributes (`y1.SP`): bind an input with `to = "attr:y1.SP"` by hand. |
| `ethanol_run.py:EthanolBioreactor` ([APMonitor](https://apmonitor.com/do/index.php/Main/EthanolBioreactor)) | ✓ | — (`--call run`: what init picks by itself since 2026-10-28, with a warning listing the candidates) | — | ✓ (`init --probe` config, 62.1 MB) | Ethanol fed-batch bioreactor (APMonitor's script, wrapped in a class: `run()` holds the script's code up to its `solve()`, unchanged, without the plots). `--call run`. The script's model can only be solved once, so `run()` builds it again each step. |
| `simple_nhes.py:model` ([BYU-PRISM/NHES-opt-sensitivity](https://github.com/BYU-PRISM/NHES-opt-sensitivity) `5e9a6f2`) | needs options | ✓ (`--const`) | — | ✓ (`init --probe` config, 128.5 MB) | Nuclear-renewable hybrid energy system. `--probe --start "params=call:simple_nhes:nom_params"`. **Needs input data that isn't published** (`grid_data/more_data_scaled.csv`): it was run on a generated series of hourly `Load`, `Wind` and `Solar` in MW. Also needs seaborn and plotly, which it imports without declaring. |
| `openhens:OpenHENS` ([waikato-ahuora-smart-energy-systems/OpenHENS](https://github.com/waikato-ahuora-smart-energy-systems/OpenHENS) `5081bab`) | — | ✗ no useful outputs (`--call`, `--const` ×2, `--start` ×7) | ✓ ([config](match-testing-rerun/configs/gk_openhens.toml): one output added, `TAC = { from = "attr:_best_solns.0.case.TAC" }`) | ✓ (hand config, 118 MB; couenne must be on PATH) | Heat exchanger network synthesis. `--call solve`, the README's options. **Needs the COIN-OR `couenne` executable on PATH** (from the [IDAES solver bundle](https://github.com/IDAES/idaes-ext/releases/tag/3.4.2)) and **the repository installed** (`pip install -e .`), as its README says. `init --probe` finds no outputs: the solutions are kept in a private list; the hand config adds the total annual cost. It also reads the solutions saved in its `output_folder`, which `init` doesn't copy into the FMU, so its result differs from running it in the repository. |

**Compiling GEKKO models:** with PyInstaller, in a venv with only the model's packages, fmugen, PyInstaller and FMPy, a GEKKO model compiles in 10–25 s to 57–130 MB (matplotlib, which some model files import, adds about 35 MB). In a shared venv with many other packages, a compile took 19 min and made a 195 MB FMU: fmugen collects every gekko submodule, and some of them import scikit-learn, Pyomo and IPython when installed. Nuitka builds with gekko and numpy took over 30 minutes and were stopped. The compiled FMU contains gekko's own solver; a solver the model runs from PATH (OpenHENS's couenne) must be installed where the FMU runs.

---

## Hand configs

Configs written by hand where `init` can't infer everything. They are linked from the **Hand config** column. The ones not shown here are in [match-testing-rerun/configs](match-testing-rerun/configs/), each with a comment saying what it changes and why. `PropsSI`'s is [shown in models.md](models.md#positional-only-arguments).

### fluids.units

`Reynolds` and `head_from_P` take and return pint quantities: each input gets `convert = "pint"` and a `unit`, and `head_from_P`'s output `unit = "cm"`. Write the units the way pint reads them (`kg/m**3`, `Pa*s`, `m/s**2`): pint rejects the Modelica spelling (`kg/m3`, `Pa.s`), and fmugen then writes no `<BaseUnit>` for them. Checked: `Reynolds` gives 76401.3 and `head_from_P` 1019.7 cm, the values of `fluids.units` called directly.

### JOS3

Inputs bound to attributes (`to = "attr:tdb"`, …) and `setup = ["options.update({'limit_dshiv/dt': True})"]`; `init` writes `globals` for the module state `PRE_SHIV`. Checked: 40 one-minute steps at 10 °C give the core temperature of JOS3 run directly, also after a rollback.

### PyTCI

```toml
[model]
entry = "PyTCI.models.propofol:Marsh"
fmi_version = 2
call = "wait_time"

[parameters]
weight = { start = 70.0 }

[inputs]
time_seconds = { start = 1, type = "Integer" }
dose = { start = 2.0, unit = "mg", to = "call:give_drug" }

[outputs]
x1 = {}
x2 = {}
x3 = {}
xeo = {}
```

The model advances by `time_seconds`, not by the FMI step size. `give_drug` adds a bolus each time it's called, and a `to = "call:…"` setter runs once per step: 60 steps give 60 doses.

The first dose is given at initialization, so the outputs at t = 0 already include it: x1 = 2 mg / (0.228 L/kg × 70 kg) = 0.1253. The first step doesn't give it again. A loop that gives each dose at the start of its step (`give_drug(dose); wait_time(1)`) shows x1 = 0 at t = 0; from t = 1 the two are equal.

### Cantera

```toml
[model]
entry = "cantera:Solution"
fmi_version = 3
call = "equilibrate"
setup = ["__setattr__('TPX', (300.0, 101325.0, 'CH4:1, O2:2, N2:7.52'))"]

[parameters]
infile = { start = "gri30.yaml" }

[inputs]
XY = { start = "HP" }

[outputs]
T = { unit = "K" }
P = { unit = "Pa" }
density = {}
mean_molecular_weight = {}
```

The fuel/air state can only be set in `setup`: Cantera's `TPX` is a tuple property, and there is no way to bind inputs to it.

### metpy

```toml
[inputs]
temperature = { start = 25.0, unit = "degC", convert = "pint" }
relative_humidity = { start = 0.5, unit = "dimensionless", convert = "pint" }

[outputs]
y = { from = "return", unit = "degC" }
```

---

## Known problems

- **The backend outlives a failed FMU.** FMPy's `simulate_fmu` raises without calling `freeInstance`, so the UniFMU backend (`main.py`, 2 processes) keeps running and holds the importer's stdout/stderr; anything reading it through a pipe hangs. Not fmugen's: [fmpy-limitations.md §1](fmpy-limitations.md#1-simulate_fmu-doesnt-free-the-instance-after-an-fmu-error), and UniFMU's part in [unifmu-limitations.md §2](unifmu-limitations.md#2-the-backend-keeps-running-after-an-fmu-error). Workaround: free the instance yourself (`simulate_fmu(…, fmu_instance=fmu)`, then `fmu.freeInstance()` in a `finally`).
- **`fmpy simulate` shows no reason for a failure**, only the failed FMI call. The traceback shows with `simulate_fmu(…, debug_logging=True, logger=print)`. With FMI 2, a log message containing `%` (a URL, a traceback, Pyomo's log) can then crash FMPy: [unifmu-limitations.md §12](unifmu-limitations.md#12-fmi-2-log-messages-are-used-as-printf-format-strings).
- **Module-level scripts can't be models**: a script that solves at import time has nothing to call (GEKKO's distillation example, the ethanol bioreactor). Wrapping its code in a class works.
- **Windows long paths:** cantera and pybamm failed to import from a venv in a deeply nested folder. The FMU runs with the build venv, so deeply nested project folders may hit this.
- **`--convert` is refused for a step method's argument given by its prefixed name** (`--convert calculate_optimal_actuation_x0=…`): `--convert names that are not arguments of the model`. A `call:` start (`--start calculate_optimal_actuation_x0=call:numpy:array(…)`) works instead.
- **A local package imported by name isn't put in the FMU.** An entry like `debutanizer.model:solve_steady_state` or `openhens:OpenHENS`, from a cloned repository that isn't installed, is found by `init` only because it runs in that folder. The FMU's backend runs elsewhere and fails to import it. Installing the repository (`pip install -e .`) works; `init` gives no warning. Packages installed from PyPI aren't affected.
- **Files the model reads from a folder it computes aren't put in the FMU.** `init` copies the files the code or the starts name by a path. OpenHENS also reads saved solutions from its `output_folder`, so its FMU gives a different TAC than running it in the repository (154853.86 against 159038.72).

- **A compiled FMU has no `.py` files**, so a library that reads its own source files by path fails compiled: py-pde loads `pde/backends/*/config.py` (it works uncompiled).
- **PyInstaller stops on xgboost** unless hypothesis is installed: it imports every xgboost submodule, and `xgboost.testing` calls `pytest.importorskip("hypothesis")`, whose exception isn't an `Exception`.
- **`init` binds an argument named `t` to the FMU time**, also when it is a time vector (ssqueezepy's `ssq_cwt(x, t=None)`): remove `[time]` from the config.

Limitations (not bugs):

- **An output's size can't change during the simulation.** FMI variables have fixed sizes (FMI 3 arrays resize only through structural parameters), so an output that becomes `None` or changes length fails the step instead of giving wrong values. `init --probe` comments out the ones it sees change, except a history that grows by one entry each step (seirsplus' counts, pybamm's `t`): that one becomes its last entry (`attr:numS.-1`).
- **Model files named `model.py`** are refused by `build`: the name clashes with UniFMU's backend module. Rename the file; `init` warns about it.

---

## Versions

The versions are in [match-testing-rerun/results.json](match-testing-rerun/results.json) (`versions`), and for the 82 models tested later in [pending-100/results.json](pending-100/results.json): the latest on PyPI for Python 3.13 when the venvs were made. Pinned where it mattered: pandas 2.3.3 for the CGE model, `setuptools<81` for heartpy.

None of the models is copied into this repository: they are installed from PyPI or cloned from their repositories when needed.

---

## Reproducing a run

1. Install fmugen in an environment together with the model's library.
2. Run `init` with the options from the table.
3. Build and check the FMU.

For example, for `fluids`:

```bash
fmugen init fluids.friction:friction_factor --start Re=1e5 -o -
```

Write the config with `-o path/to/fmugen.toml` instead of `-o -`, then:

```bash
fmugen build path/to/fmugen.toml -o out/friction.fmu
```

Add `--fmi 3` to `init` or `build` for FMI 3.

Run these from an environment with fmugen and `fluids` installed (`pip install path/to/2025_Inkindcontributions fluids`; fmugen is not on PyPI yet). The FMU runs with that environment (see [packaging.md](packaging.md#the-fmus-python-environment)).
