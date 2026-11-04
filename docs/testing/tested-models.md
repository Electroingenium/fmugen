# Tested models

fmugen has been tried on **291 unmodified Python models**: about 187 PyPI libraries, 66 neural networks, 13 GEKKO models, 5 GitHub repositories and one unpublished model. Three needed a file renamed or a script wrapped in a class, and some needed synthetic input or training data (see their notes). 33 more libraries turned out not to be models, and 6 couldn't be installed or run unattended: see [discarded-models.md](discarded-models.md).

## Results

| Config | Models | Success |
|---|---|---|
| `init` (code only), given other options where needed | 59 | 20 % |
| `init --probe`, given only the model and `--start` values | 122 | 42 % |
| …given also `--call` where needed, no other option | 160 | 55 % |
| …given other options where needed (`--const`, `--setup`, `--create`, …) | 262 | 90 % |
| …or with a hand-written config | 280 | 96 % |
| **Compiled FMU (PyInstaller) simulates** | 267 | 92 % |

A model counts only if its FMU simulates **and** outputs the model's results, with FMI 2 or FMI 3; an FMU that only outputs parameters or constants is a failure. Every FMU was compared value by value with the model run directly, and every compiled FMU with its uncompiled one: all matching configs are identical ([report](match-testing-rerun/report.md), [compiled](match-testing-compile-rest/report.md)).

**11 fail:** 3 libraries have no function or class that is the model (PVWatts, Wav2Vec2, Moirai: they need a wrapper), and 8 fail for the reasons in their notes. Not compiled or not compiling: py-pde, pysr, SevenNet and JSBSim simulate uncompiled only; 10 models (orb, piper, skyfield, kokoro, spaCy, KeyBERT, faster-whisper, equilibrator, depth estimation, flopy) have not been compiled.

**Still open:** what the other models need is mostly their own data or the model they run, given as `--const` (open_clip's and TrOCR's image tensors, rrmpg's forcing data, skyfield's Time, vosk's audio and acoustic model, openmdao's Group: `init --probe` warns about each one left out and suggests the `--const`); mapie's interval bounds need `--call predict_interval` (without it, `init` steps `predict`). Results reached only by name need a hand config or one added line (openmdao's `prob['y1']`, JSBSim's property tree, flopy's heads); JSBSim also needs `--const root_dir`, two `--setup` and `--call`, and its data folder isn't found in a compiled FMU. Since 2026-11-04 `init` infers what geopy, thermopack, cobra, pylife, vosk, flopy and the others needed by hand: see their notes. Users' workarounds are in [troubleshooting.md](../troubleshooting.md#known-problems); tool limitations in [limitations/unifmu.md](../limitations/unifmu.md) and [limitations/fmpy.md](../limitations/fmpy.md).

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
