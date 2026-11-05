# Pending models run (2026-11-03): conclusions

The 100 models of `tested-models.md`'s Pending tables were run following `run-pending-models.md`.
Setup: fmugen `cc07fc3` (unchanged during the run), WSL 2 (Ubuntu), Python 3.13.16, one `uv` venv
per model (torch CPU build where needed), FMPy 0.3.32, UniFMU 0.14.0 (the Windows `unifmu.exe`
through WSL interop; its FMUs carry the Linux binary), PyInstaller 6.22.3. The host has no C++
compiler and no root: packages without a Python 3.13 wheel were built with zig (`ziglang`) and Ninja.

## Contents of this folder

- `runr4.py` (drives `pending-100`'s harness), `models_r4.py` (every model, its
  packages and `init` options), `prep_data.py` (inputs), `hand/` (11 hand configs),
  `results.json` / `results.csv` (every verdict and version).
- The results (rows, counts, setup, versions) are in [tested-models.md](../tested-models.md) and
  [discarded-models.md](../discarded-models.md).

## Results

| | Models |
|---|---|
| Tested | 100 |
| Are models | 81 |
| Simulate with useful results, matching the model run directly | **63** (52 with an `init` / `init --probe` config, 11 with a hand config) |
| Fail | 18 |
| Compiled FMU simulates | 55 |
| Moved to discarded-models.md | 19: 8 frameworks, 5 empty containers, 5 not installable for Python 3.13, sinergym (needs EnergyPlus) |

New totals in tested-models.md: 291 models, 270 with useful results, 262 compiled.

## Problems found in fmugen (not fixed: a separate change)

- `init --probe` stops when reading a property raises NotImplementedError (pysr, scikit-survival).
- A tuple default becomes an FMU array passed back as a list, which the model refuses (gplearn,
  minigrid); worked around with `--const NAME=call:builtins:tuple([...])`.
- A `call:` value is one call: it can't chain or unpack (skyfield, equilibrator, orb fail).
- Generators can't be stepped (kokoro, piper).
- `init` maps galpy's integrator `dt` to the FMU step size, and sends a class's `**kwargs` to the
  step when the constructor also takes them (feedinlib).
- `init` refuses a factory's step argument (`pipeline(...)` with `--start inputs=…`, depth estimation).
- Results inside xarray Datasets, DataFrames or reached through methods aren't output (py_wake,
  pycalphad, demandlib, flopy, openmdao, jsbsim, vosk): hand configs read them where paths reach.
- Compiled FMUs miss what PyInstaller doesn't see (PettingZoo's Atari file, mne's lazy submodules,
  a mypyc module, JSBSim's data, TorchScript source, Hugging Face weights); galpy's and interpret's
  (joblib's loky re-running the frozen executable with `-c`) hang.

## Problems in the libraries or the host

- fasttext: `np.array(copy=False)`, an error with numpy 2.
- neuralprophet: 0.9.0 needs numpy < 2; 0.5.0's training fails in current PyTorch Lightning.
- OCHRE: its bundled sample house fails to load (`KeyError: 'key_0'`).
- pymgrid: `step` fails on its own scenario with its own sample action.
- feedinlib: PyPI's 0.0.12 uses pandas' removed `clip_lower`; GitHub's needs PostgreSQL to install.
- mediapipe needs `libGLESv2` (not on the host); myokit needs SUNDIALS and pysb BioNetGen (not on PyPI).
- Undeclared dependencies installed: six (imodels), flask (deepface), peakutils (biosppy),
  python-dateutil (pysb), statsmodels (sktime). Pins: `pandas<3`, `holidays<0.25`, `av<16`,
  `pint<0.25`, `dm-tree>=0.1.9` where noted.

## To reproduce

Paths in `models_r4.py` and the hand configs point to `~/cmp/...`, deleted after the run: run
`prep_data.py`, then `python runr4.py venvs`, `init`, `run`, `compile`, `report`.
