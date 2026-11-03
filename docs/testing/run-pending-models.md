# Running the pending models

**Done on 2026-11-03:** the results are in [tested-models.md](tested-models.md) and [discarded-models.md](discarded-models.md), the run in [pending-r4/](pending-r4/CONCLUSIONS.md). Kept as the procedure for the next batch of pending models.

Instructions for running the 100 models that were in tested-models.md's Pending models tables on a remote Linux host. Follow the procedure in [tested-models.md, How the models were tested](tested-models.md#how-the-models-were-tested); this file summarizes it and adds what's needed to run unattended.

## Rules

- **Don't change the models.** fmugen's goal is that users never edit their Python: use the published model as installed from PyPI, Hugging Face or GitHub, configured only through `fmugen init` options and, where those can't do it, a hand-written config. A model that needs a file written for the test (a wrapper) counts as a failure: record it in [discarded-models.md](discarded-models.md).
- **Don't make up models or data.** Use the published model and, as inputs, published example files (the library's own examples or test data, or the files already used: ultralytics' `bus.jpg`, whisper's `jfk.flac`, …). Training data for a model that needs fitting may be synthetic; say so in the notes.
- **Find the real error.** When an FMU fails, read the backend's own output (`--capture-output`) before blaming UniFMU, FMPy or fmugen. Only list something as a UniFMU limitation after checking it fails the same way outside fmugen.
- **Don't fix fmugen during the run** unless told to. Record what fails and why; fixes are a separate change, after which the affected models are rerun.
- **Don't attribute anything to Claude**: no `Co-Authored-By` or "Generated with" lines in commits, PRs or issues.
- **Unattended only.** A model that needs a license accepted interactively, a login or a paid API key goes in the "Impractical to run" table of [discarded-models.md](discarded-models.md), as TabPFN did.

## Setup

- Linux (Ubuntu), Python 3.13, one `uv` venv per model, with fmugen installed from this repository's checkout (`uv pip install path/to/fmugen <model's package>`), the torch CPU build where torch is needed.
- FMPy and UniFMU at the versions listed in [tested-models.md](tested-models.md#how-the-models-were-tested) unless newer ones are agreed; PyInstaller for the compiled step.
- Record the fmugen commit (`git rev-parse --short HEAD`) and every package version per model.
- Reuse the scripts in [pending-100/](pending-100/): `run100.py` (init, run, compile, report), `venvs.py`, `harness/`, `prep_data.py`. Add the new models to a new list next to `models100.py` (same tuple format) rather than editing the old one, and write the results to a new folder (for example `pending-r4/`), leaving `pending-100/` as it is.
- Entry points in the pending table are a first guess. If the named class or function isn't the model, find the one that is from the library's documentation and note the change.

## Per model

1. `fmugen init <entry point>` (code only), given only the model and `--start` values (and `--fmi`).
2. `fmugen init --probe`, the same way.
3. If neither gives the model's results, retry with other options (`--call`, `--create`, `--setup`, `--const`, `--kind`, `--convert`), unedited configs; then, if still needed, a hand config in the results folder's `hand/<name>/fmugen.toml`.
4. `fmugen build --capture-output` on each config, then `fmpy.simulate_fmu` for 3 s with 1 s outputs.
5. Run the model directly from the same config (the harness's `direct.py`), twice: if the two runs differ, the model is random by design; seed it and compare seeded. Compare the FMU's values with the direct run's, value by value.
6. `fmugen build --capture-output --compile pyinstaller` on the best config (the `--probe` one when it simulated), simulate it the same way and compare it value by value with the uncompiled FMU. Record the FMU size.
7. **FMI 2 by default.** Use `--fmi 3` only when no FMI 2 config simulates.

Use long timeouts for builds (an hour) and download weights before the run, so a slow download isn't recorded as a failure.

## Recording results

- Move each model from the Pending models tables of [tested-models.md](tested-models.md) to its section (Python libraries, Neural networks or GitHub repositories), in the same columns: **`init` / `init --probe`**, **… with other options**, **Hand config**, **Compiled (PyInstaller)**, **Notes**, following the column rules under "How the models were tested" (what counts as `--start`, as `--const`, as an option).
- Notes: what was checked (inputs and the outputs it gave), the error of each config that failed (`init`:, `init --probe`:), and anything that needed a hand config.
- Models that turn out not to be models (frameworks, empty containers, models that need a wrapper) go to [discarded-models.md](discarded-models.md), in the matching table.
- Update the counts at the top of [tested-models.md](tested-models.md) and the Summary, and add the new setup (fmugen commit, date, versions) to "How the models were tested" and "Versions".
- Keep `results.json` (every model's verdicts and versions) and `results.csv` in the results folder.
- Commit on `dev` with a message that says what ran and the totals; push only when told to.
