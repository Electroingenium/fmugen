# Tested models

fmugen has been tried on **291 unmodified Python models**: about 187 PyPI libraries, 66 neural networks, 13 GEKKO models, 5 GitHub repositories and one unpublished model. Three needed a file renamed or a script wrapped in a class, and some needed synthetic input or training data (see their notes). 33 more libraries turned out not to be models, and 6 couldn't be installed or run unattended: see [discarded-models.md](discarded-models.md).

## Results

| Config | Models | Success |
|---|---|---|
| `init` (code only), given other options where needed | 59 | 20 % |
| `init --probe`, given only the model and `--start` values | 126 | 43 % |
| …given also `--call` where needed, no other option | 170 | 58 % |
| …given also `--const` only for the data it runs on or the assets it loads | 224 | 77 % |
| …given other options where needed (`--const`, `--setup`, `--create`, `--read`, …) | 269 | 92 % |
| …or with a hand-written config | 280 | 96 % |
| **Compiled FMU (PyInstaller) simulates** | 267 | 92 % |

The data-and-assets row adds 54 models whose only other flag is a `--const` giving their input (an image, audio, a signal, a data file, a series) or what they load (pretrained weights, a saved agent, a model name, a TLE, a CRS); `init --probe` warns about missing data and suggests the `--const`. Training on the user's data (`--setup fit(...)`) counts as data there, as catboost's, darts' and pyod's rows already count it: lightgbm, xgboost, mapie and prophet need nothing else. Not counted there: `--const` objects that define the system (atoms, a potential, a network, an estimator, a problem), a choice of function, set-ups that define it (anastruct's elements, osqp's problem, timesfm's forecast config), and values `init` should make up itself (CFTOC's arrays).

A model counts only if its FMU simulates **and** outputs the model's results, with FMI 2 or FMI 3; an FMU that only outputs parameters or constants is a failure. Every FMU was compared value by value with the model run directly, and every compiled FMU with its uncompiled one: all matching configs are identical ([report](match-testing-rerun/report.md), [compiled](match-testing-compile-rest/report.md)).

**11 fail:** 3 libraries have no function or class that is the model (PVWatts, Wav2Vec2, Moirai: they need a wrapper), and 8 fail for the reasons in their notes. Not compiled or not compiling: py-pde, pysr, SevenNet and JSBSim simulate uncompiled only; 10 models (orb, piper, skyfield, kokoro, spaCy, KeyBERT, faster-whisper, equilibrator, depth estimation, flopy) have not been compiled.

**Still open:** what the other models need is mostly their own data or the model they run, given as `--const` (open_clip's and TrOCR's image tensors, rrmpg's forcing data, skyfield's Time, vosk's audio and acoustic model, openmdao's Group: `init --probe` warns about each one left out and suggests the `--const`); mapie's interval bounds need `--call predict_interval` (without it, `init` steps `predict`). Results reached only by a path the probe doesn't search (openmdao's `prob['y1']`, JSBSim's property tree, flopy's heads, pybamm's voltage, OpenHENS's cost) are given with `init --read` since 2026-11-04; JSBSim also needs `--const root_dir`, two `--setup` and `--call`, and its data folder isn't found in a compiled FMU. Since 2026-11-04 `init` infers what geopy, thermopack, cobra, pylife, vosk, flopy and the others needed by hand: see their notes. Rechecked the same day without the flag rules now infer: 13 more need it no longer (psychrolib's `--setup`, the `--create` of pyproj, sgp4, chgnet, speechbrain, DETR, ESM, SAM and GLiNER, gplearn's tuple `--const`, the `--setup` of mpe2, metaworld and pybullet-drones, and its `--convert`); thermo's `--kind` is still needed (75 properties without it, instead of 731). Later the same day `init --probe` learned to infer what 7 more needed: iapws' `--kind function` (stepping it finds no number, building it does), the distillation column's `--setup reset` (an environment is reset before its first step), mpe2's `actions` (one per agent, from each action space), onnxruntime's `input_feed` (from the session's inputs), simglucose's `action` (from the fields of the namedtuple it's named after), botorch's FMU (a tensor kept as a torch buffer was typed as a single value) and tellurium's species (properties roadrunner adds to its class at run time); also hapsira's and timesfm's `--create`, though both still need other options. Rechecked: momentfm and equinox need no option but their input. Users' workarounds are in [troubleshooting.md](../troubleshooting.md#known-problems); tool limitations in [limitations/unifmu.md](../limitations/unifmu.md) and [limitations/fmpy.md](../limitations/fmpy.md).

### Failure causes

The models whose row in the per-model pages has no ✓ in the `init` / `init --probe` or "… with other options" column, each under one cause: 26 when this table was made, 21 since `--read` (10 pass with a hand config, 11 fail outright). "Unblocks" counts the models the candidate fix would take to the "other options" row, from their notes and hand configs; only the first fix is implemented.

| Cause | Models | Candidate fix in fmugen | Unblocks |
|---|---|---|---|
| The results are reached only by a path `--probe` doesn't search: a private attribute, a method chain, a string key on an object that isn't a Mapping | OpenHENS (`_best_solns.0.case.TAC`), flopy (`get_model().output.head().get_data()`), JSBSim (`position/h-sl-ft`, …), openmdao (`prob['y1']`), pybamm (`solution['Voltage [V]']`) | **Done:** `init --read NAME=PATH` writes the `attr:` / `return:` path the hand config adds, checked by the probe. All five rerun on 2026-11-04 and matching their direct runs | 5 (JSBSim uncompiled only: its data folder isn't collected) |
| Too many output values: a whole series is output | demandlib (a year of 15-minute values), demucs (about 530,000 samples) | `--probe` keeps a few elements of an array above a size limit and warns, or `--output` picks them | 2 (demucs' `shifts` is random: also a seed) |
| An input goes in through a second method call on each step | PyTCI (`give_drug(dose)` beside `wait_time`), filterpy's `KalmanFilter` (`update(z)` beside `predict`) | `--probe` binds an input to a one-argument method it names (`to = "call:give_drug"`), as the hand config does | 2 (filterpy's hand config used FMI 3 clocks: not checked this way) |
| C built-ins with no signature | CoolProp (`PropsSI`), mujoco (the step is the function `mj_step(model, data)`) | Read the signature from the docstring's first line (pybind11, Cython `embedsignature`); let `--call` be a module function taking the object (`mujoco:mj_step(self.model, self)`) | 2 |
| A function returns None and writes its results onto an argument | pandapower (`runpp(net)` fills `net.res_bus`, …) | When a function returns None, `--probe` reads the object arguments after the call | 1 |
| The results are extension-type properties, and the state is a tuple property | cantera (`T`, `P`, `density`; `TPX`) | `--probe` reads the class's Cython properties (getset descriptors); the state set by `--setup "__setattr__('TPX', …)"` as in the hand config | 1 (not rechecked since `--probe` learned to read run-time properties) |
| Arrays that grow by one element each step | seirsplus (`numS`, `numI`, …) | `--probe` sees an array grow between steps and outputs its last element (`numS.-1`) | 1 (also `--start dt=0.1`) |
| The model is an ODE right-hand side | dysts (`Lorenz.rhs`) | `init` writes `[model] integrate` when the step returns the derivative of the state it's given | 1 |
| No function or class is the model | PVWatts, Moirai, Wav2Vec2 | None: what they need is code (weather set as an attribute, GluonTS preparation, CTC decoding); they need a wrapper | 0 |
| The model fails outside fmugen: run directly, or in this environment | feedinlib (pandas' `clip_lower`), myokit (SUNDIALS headers), NeuralProphet (Lightning's tuner), OCHRE (`KeyError: 'key_0'`), pymgrid (`Bad action []`), PySB (BioNetGen), fasttext (numpy 2), mediapipe (`libGLESv2.so.2`) | None in fmugen | 0 |

**nashpy is not a failure.** `nashpy:Game` with `--call support_enumeration` passes with the `--probe` config, compiled too ([results-libraries.md](results-libraries.md)): the generator is read into a tuple (`results_of` in `fmugen_runtime/_helpers.py`, up to `GENERATOR_LIMIT` items) and `--probe` outputs the first equilibrium. The reports that list it as failing ([match-testing](match-testing/report.md), [match-testing-rerun](match-testing-rerun/report.md): `float() … not 'generator'`, [match-testing-compile-rest](match-testing-compile-rest/report.md): left out) date from 2026-10-30, before generator results were read (commit `2fd2e79`, 2026-11-03). Plain `init` doesn't run the model, so it can't know the generator yields pairs of arrays and types the result as one value: only the `--probe` config works.

## Per-model results

| Page | Models |
|---|---|
| [results-libraries.md](results-libraries.md) | Python libraries from PyPI |
| [results-neural-networks.md](results-neural-networks.md) | Neural networks (torch, JAX, Keras, ONNX, Hugging Face) |
| [results-gekko.md](results-gekko.md) | GEKKO models, and compiling them |
| [results-repositories.md](results-repositories.md) | GitHub source repositories and the unpublished model |
| [hand-configs.md](hand-configs.md) | Configs written by hand where `init` can't infer everything |

## Background

- [methodology.md](methodology.md): the steps each model went through, the setups of each round, what the table columns mean, and package versions.
- [run-pending-models.md](run-pending-models.md): how to test a new batch of models. Scripts and raw results: [pending-100/](pending-100/), [pending-r4/](pending-r4/).

## Reproducing a run

Install fmugen with the model's library, run `init` with the options from the table, then build and simulate as in [getting-started.md](../getting-started.md):

```bash
pip install path/to/2025_Inkindcontributions fluids
```

```bash
fmugen init fluids.friction:friction_factor --start Re=1e5 -o friction/fmugen.toml
```

```bash
fmugen build friction -o out/friction.fmu
```

None of the models is copied into this repository: they are installed from PyPI or cloned when needed.
