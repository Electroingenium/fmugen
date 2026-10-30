# FMPy: bugs and limitations

Problems that come from [FMPy](https://github.com/CATIA-Systems/FMPy) **0.3.32** (the importer fmugen is tested with), not from fmugen or UniFMU. Each entry says where it was found, its effect on fmugen FMUs, and the workaround, if any. For problems in UniFMU, see [unifmu-limitations.md](unifmu.md).

| # | Problem | Kind | Workaround |
|---|---|---|---|
| 1 | [`simulate_fmu` doesn't free the instance after an FMU error](#1-simulate_fmu-doesnt-free-the-instance-after-an-fmu-error) | Bug | Free the instance yourself, or send the output to a file and kill leftover `main.py` processes |
| 2 | [The result array has no strings](#2-the-result-array-has-no-strings) | Limitation | Read String outputs with `getString` |

---

## 1. `simulate_fmu` doesn't free the instance after an FMU error

When an FMU call returns an error status (`exitInitializationMode`, `doStep`), `fmpy.simulate_fmu` raises without calling `freeInstance`. FMPy then exits, but a UniFMU FMU's backend is never told to stop: its 2 processes (`venv\Scripts\python.exe main.py` and the base `python.exe main.py`) stay alive. `fmpy simulate` uses `simulate_fmu`, so it does the same.

**Cause:** in `fmpy/simulation.py`, `simulate_fmu` calls `simulateCS` (or `simulateME`) and then `fmu.freeInstance()`, with no `try/finally`. `simulateCS` re-raises the `FMICallException` of a failed step, so `freeInstance` is skipped. The FMI standard asks the importer to call `fmi2FreeInstance` or `fmi2Reset` after `fmi2Error` (likewise for FMI 3).

**Not fmugen or UniFMU:** checked on 2026-10-13 with nashpy's `Game.support_enumeration` (whose step fails: it returns a generator), driven with `FMU2Slave` / `FMU3Slave`. After the failing `doStep`, `freeInstance` removed both backend processes within 3 s, in FMI 2 and FMI 3, as for an FMU that passes (mendeleev). UniFMU's part is only that its backend can't tell when the importer has gone, so it waits for the free forever: [unifmu-limitations.md §2](unifmu.md#2-the-backend-keeps-running-after-an-fmu-error).

Effects:
- The orphaned backend holds the importer's stdout/stderr. Anything that reads them through a pipe (`subprocess.run(capture_output=True)`, `fmpy simulate x.fmu | grep …`) waits forever.
- The orphans pile up: 31 pairs in the first round-2 run, 14 in its rerun, 2 in two nashpy runs on 2026-10-13. One held a lock on the FMU's work folder until killed.

Found while [testing published models](../testing/tested-models.md#known-problems); the cause was found on 2026-10-13. Not reported upstream yet.

**Workaround:** instantiate the FMU yourself and free it in a `finally`: either with `FMU2Slave` / `FMU3Slave`, or by passing `fmu_instance=` to `simulate_fmu`, which then leaves the free to the caller:

```python
from fmpy import extract, instantiate_fmu, read_model_description, simulate_fmu

md = read_model_description("model.fmu")
unzipdir = extract("model.fmu")
fmu = instantiate_fmu(unzipdir, md)
try:
    result = simulate_fmu(unzipdir, model_description=md, fmu_instance=fmu, stop_time=3)
finally:
    fmu.freeInstance()
```

With `fmpy simulate`, send the output to a file, not a pipe, and kill leftover `main.py` processes.

## 2. The result array has no strings

`simulate_fmu` records only numbers and Booleans (FMI 2: Real, Integer, Boolean; FMI 3: the float, int and Boolean types), so String outputs (timezonefinder's `Europe/Madrid`, mendeleev's `cas`) don't appear in the result, nor do Binary ones in FMI 3.

Found in [tested models](../testing/tested-models.md#python-libraries) (timezonefinder).

**Workaround:** read them with `fmu.getString([value_reference])` on an instance you drive yourself, or use another importer.
